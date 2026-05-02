use crate::{
    animation::{
        AnimationChannel, AnimationChannelType, AnimationData, MeshKey, MorphKey, PositionKey,
        RotationKey, ScalingKey,
    },
    materials::MaterialData,
};
use asset_importer::{
    Scene, mesh::Mesh, node::Node, postprocess::PostProcessSteps,
    importer::{PropertyStore, import_properties},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use half::f16;
use std::{collections::HashMap, path::Path};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniforms {
    pub model: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Vertex types
// ---------------------------------------------------------------------------

/// Compressed vertex: 32 bytes (down from 92).
/// Normal: oct-encoded Snorm8x2. Texcoord: Float16x2. Tangent: Snorm8x4 (xyz + w handedness).
/// Bitangent: reconstructed in shader as cross(normal, tangent.xyz) * sign(tangent.w).
/// Bone indices: Uint8x4. Bone weights: Unorm8x4.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],    // 12 bytes, offset 0
    pub texcoord: [u16; 2],    // 4 bytes, offset 12 (Float16x2, raw bits)
    pub normal: [i8; 2],       // 2 bytes, offset 16 (Snorm8x2, oct-encoded)
    pub _pad: [u8; 2],        // 2 bytes, offset 18 (alignment padding)
    pub tangent: [i8; 4],      // 4 bytes, offset 20 (Snorm8x4, xyz + w sign)
    pub node_indices: [u8; 4], // 4 bytes, offset 24 (Uint8x4)
    pub node_weights: [u8; 4], // 4 bytes, offset 28 (Unorm8x4)
}

impl ModelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3,  offset: 0,  shader_location: 0 }, // position
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm8x2,   offset: 16, shader_location: 1 }, // normal (oct)
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float16x2,  offset: 12, shader_location: 2 }, // texcoord
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm8x4,   offset: 20, shader_location: 3 }, // tangent (xyz+w)
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint8x4,    offset: 24, shader_location: 4 }, // node_indices
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4,   offset: 28, shader_location: 5 }, // node_weights
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ---------------------------------------------------------------------------
// Vertex packing helpers
// ---------------------------------------------------------------------------

/// Octahedral encode: unit vec3 → [i8; 2] (Snorm8)
fn oct_encode_snorm8(n: Vec3) -> [i8; 2] {
    let n = n.normalize();
    let sum = n.x.abs() + n.y.abs() + n.z.abs();
    let mut p = [n.x / sum, n.y / sum];
    if n.z < 0.0 {
        let px = p[0];
        let py = p[1];
        p[0] = (1.0 - py.abs()) * px.signum();
        p[1] = (1.0 - px.abs()) * py.signum();
    }
    [
        (p[0].clamp(-1.0, 1.0) * 127.0).round() as i8,
        (p[1].clamp(-1.0, 1.0) * 127.0).round() as i8,
    ]
}

/// Pack tangent xyz + bitangent handedness sign into [i8; 4]
fn pack_tangent_with_sign(tangent: Vec3, normal: Vec3, bitangent: Vec3) -> [i8; 4] {
    let t = tangent.normalize_or_zero();
    let sign = normal.cross(t).dot(bitangent);
    [
        (t.x.clamp(-1.0, 1.0) * 127.0).round() as i8,
        (t.y.clamp(-1.0, 1.0) * 127.0).round() as i8,
        (t.z.clamp(-1.0, 1.0) * 127.0).round() as i8,
        if sign >= 0.0 { 127i8 } else { -127i8 },
    ]
}

/// Pack bone weights to Unorm8, normalized so they sum to 255
fn pack_weights_unorm8(weights: [f32; 4]) -> [u8; 4] {
    let sum: f32 = weights.iter().sum();
    if sum < 0.0001 {
        return [0; 4];
    }
    let normalized = weights.map(|w| w / sum);
    let mut packed: [u8; 4] = normalized.map(|w| (w * 255.0).round() as u8);
    // Fix rounding error so they sum to 255
    let packed_sum: u16 = packed.iter().map(|&b| b as u16).sum();
    if packed_sum > 0 && packed_sum != 255 {
        let diff = 255i16 - packed_sum as i16;
        // Apply correction to largest weight
        let max_idx = packed.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap().0;
        packed[max_idx] = (packed[max_idx] as i16 + diff).clamp(0, 255) as u8;
    }
    packed
}

// ---------------------------------------------------------------------------
// Model data types
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct ModelMeshPart {
    pub material_index: usize,
    pub index_start: u32,
    pub index_count: u32,
}

pub struct ModelMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<ModelMeshPart>,
}

pub struct ModelNode {
    pub name: String,
    pub parent_index: i32,
    pub first_child_index: i32,
    pub next_sibling_index: i32,
    pub offset_matrix: Mat4,
    pub default_transform: Mat4,
    /// Cached decomposition of default_transform: (scale, rotation, translation)
    pub default_decomposed: (Vec3, Quat, Vec3),
}

pub struct ModelMarker {
    pub name: String,
    pub node_index: i32,
    pub position: Vec3,
    pub rotation: Vec3,
}

pub struct ModelData {
    pub materials: Vec<MaterialData>,
    pub meshes: Vec<ModelMesh>,
    pub nodes: Vec<ModelNode>,
    pub markers: Vec<ModelMarker>,
    pub animations: Vec<AnimationData>,
    pub root_inverse_transform: Mat4,
}

// ---------------------------------------------------------------------------
// ModelData helpers
// ---------------------------------------------------------------------------

impl ModelData {
    pub fn get_root_node(&self) -> Option<usize> {
        self.nodes.iter().position(|n| n.parent_index == -1)
    }

    pub fn find_node_by_name(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    pub fn find_marker_by_name(&self, name: &str) -> Option<usize> {
        self.markers.iter().position(|m| m.name == name)
    }

    pub fn find_animation_by_name(&self, name: &str) -> Option<usize> {
        self.animations.iter().position(|a| a.name == name)
    }

    pub fn add_child_node(&mut self, parent_index: i32, node: ModelNode) -> usize {
        let new_index = self.nodes.len();
        self.nodes.push(node);
        self.nodes[new_index].parent_index = parent_index;
        self.nodes[new_index].first_child_index = -1;
        self.nodes[new_index].next_sibling_index = -1;

        if parent_index >= 0 {
            let parent = parent_index as usize;

            if self.nodes[parent].first_child_index == -1 {
                self.nodes[parent].first_child_index = new_index as i32;
            } else {
                // Walk to last sibling
                let mut sibling = self.nodes[parent].first_child_index as usize;

                while self.nodes[sibling].next_sibling_index != -1 {
                    sibling = self.nodes[sibling].next_sibling_index as usize;
                }

                self.nodes[sibling].next_sibling_index = new_index as i32;
            }
        }

        new_index
    }
}

// ---------------------------------------------------------------------------
// Matrix conversion helpers
// ---------------------------------------------------------------------------

fn mat4_from_assimp(m: &asset_importer::Matrix4x4) -> Mat4 {
    // asset-importer Matrix4x4 is column-major with x/y/z/w_axis columns,
    // matching glam's from_cols_array layout directly.
    Mat4::from_cols_array(&[
        m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w, m.y_axis.x, m.y_axis.y, m.y_axis.z,
        m.y_axis.w, m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w, m.w_axis.x, m.w_axis.y,
        m.w_axis.z, m.w_axis.w,
    ])
}

// ---------------------------------------------------------------------------
// Skeleton hierarchy import
// ---------------------------------------------------------------------------

/// Collect bone names and offset matrices from all meshes in the scene graph.
fn collect_bone_info_recursive(scene: &Scene, node: &Node, info: &mut HashMap<String, Mat4>) {
    for mesh_idx in node.mesh_indices_iter() {
        if let Some(mesh) = scene.mesh(mesh_idx) {
            if mesh.has_bones() {
                for bone in mesh.bones() {
                    let name = bone.name();
                    info.entry(name)
                        .or_insert_with(|| mat4_from_assimp(&bone.offset_matrix()));
                }
            }
        }
    }
    for child in node.children() {
        collect_bone_info_recursive(scene, &child, info);
    }
}

/// Check if any descendant of this node (inclusive) is a bone.
fn has_bone_descendant(node: &Node, bone_info: &HashMap<String, Mat4>) -> bool {
    if bone_info.contains_key(&node.name()) {
        return true;
    }
    for child in node.children() {
        if has_bone_descendant(&child, bone_info) {
            return true;
        }
    }
    false
}

/// Walk scene graph depth-first, creating model nodes only for actual bones.
/// Non-bone ancestors (scene root, armature) are traversed but not included,
/// preserving the same hierarchy the old code built while guaranteeing correct
/// parent-child ordering (parents are always created before children).
fn import_skeleton(
    node: &Node,
    model: &mut ModelData,
    bone_info: &HashMap<String, Mat4>,
    parent_model_index: i32,
) {
    let name = node.name();
    let is_bone = bone_info.contains_key(&name);

    let current_parent = if is_bone {
        let default_transform = mat4_from_assimp(&node.transformation());
        let offset_matrix = bone_info[&name];
        let default_decomposed = decompose_transform(&default_transform);

        let new_node = ModelNode {
            name: name.clone(),
            parent_index: -1,
            first_child_index: -1,
            next_sibling_index: -1,
            offset_matrix,
            default_transform,
            default_decomposed,
        };

        model.add_child_node(parent_model_index, new_node) as i32
    } else {
        // Not a bone — traverse through without creating a node
        parent_model_index
    };

    for child in node.children() {
        if has_bone_descendant(&child, bone_info) {
            import_skeleton(&child, model, bone_info, current_parent);
        }
    }
}

/// Strip `$AssimpFbx$` suffixes from node names.
/// Assimp may generate these when `preserve_pivots` is true, and they can
/// persist in animation channel names even when `preserve_pivots` is false.
fn strip_assimp_fbx_suffix(name: &str) -> &str {
    match name.find("_$AssimpFbx$") {
        Some(pos) => &name[..pos],
        None => name,
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

impl ModelData {
    pub fn from_file(path: &str) -> Self {
        Self::from_file_with_uv_scale(path, 1.0)
    }

    pub fn from_file_with_uv_scale(path: &str, uv_scale: f32) -> Self {
        // Disable preserve_pivots to prevent Assimp from creating $AssimpFbx$
        // intermediate nodes that break bone hierarchy parent chains.
        let mut props = PropertyStore::new();
        props.set_bool(import_properties::FBX_PRESERVE_PIVOTS, false);

        let scene = Scene::from_file_with_props(
            path,
            PostProcessSteps::TRIANGULATE
                | PostProcessSteps::CALC_TANGENT_SPACE
                | PostProcessSteps::JOIN_IDENTICAL_VERTICES
                | PostProcessSteps::SORT_BY_PTYPE
                | PostProcessSteps::FIND_DEGENERATES
                | PostProcessSteps::FIND_INVALID_DATA
                | PostProcessSteps::VALIDATE_DATA_STRUCTURE,
            &props,
        )
        .expect("Failed to load model");

        let model_dir = Path::new(path).parent().unwrap_or(Path::new("."));

        // Extract materials
        let materials: Vec<MaterialData> = scene
            .materials()
            .map(|mat| MaterialData::from_assimp(&mat, model_dir, &resolve_texture_path))
            .collect();

        let mut model = ModelData {
            materials,
            meshes: Vec::new(),
            nodes: Vec::new(),
            markers: Vec::new(),
            animations: Vec::new(),
            root_inverse_transform: Mat4::IDENTITY,
        };

        if let Some(root) = scene.root_node() {
            // Phase 1: Collect bone names + offset matrices from all meshes
            let mut bone_info = HashMap::new();
            collect_bone_info_recursive(&scene, &root, &mut bone_info);

            // Phase 2: Build skeleton hierarchy by walking the scene graph depth-first.
            // This guarantees parents are created before children.
            import_skeleton(&root, &mut model, &bone_info, -1);

            // Phase 3: Import mesh geometry and bone weights
            import_node(&scene, &root, &mut model, uv_scale);

            // Import markers (# prefixed meshes)
            import_markers(&scene, &root, &mut model);
        }

        // Import animations (after nodes exist for name lookups)
        for anim in scene.animations() {
            import_animation(&anim, &mut model);
        }

        model
    }
}

fn resolve_texture_path(model_dir: &Path, raw_path: &str) -> Option<String> {
    let stripped = raw_path
        .strip_prefix("/Users/camden/Source/c-language-prototypes/")
        .unwrap_or(raw_path);

    let assets_dir = crate::assets_dir();

    let candidate = assets_dir.join(stripped);

    if candidate.exists() {
        return Some(candidate.to_string_lossy().into_owned());
    }

    if Path::new(stripped).exists() {
        return Some(stripped.to_string());
    }

    let candidate = model_dir.join(stripped);

    if candidate.exists() {
        return Some(candidate.to_string_lossy().into_owned());
    }

    if let Some(filename) = Path::new(stripped).file_name() {
        if let Some(found) = find_file_recursive(&assets_dir.join("textures"), filename.to_str()?) {
            return Some(found);
        }
    }

    None
}

fn find_file_recursive(dir: &Path, target_filename: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, target_filename) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == target_filename) {
            return Some(path.to_string_lossy().into_owned());
        }
    }

    None
}

fn import_node(
    scene: &Scene,
    node: &Node,
    model: &mut ModelData,
    uv_scale: f32,
) {
    let mesh_indices: Vec<usize> = node.mesh_indices_iter().collect();

    let mut mesh = ModelMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
    };

    for &mesh_idx in &mesh_indices {
        let assimp_mesh = scene.mesh(mesh_idx).unwrap();
        import_mesh(&assimp_mesh, &mut mesh, model, uv_scale);
    }

    if !mesh.vertices.is_empty() {
        model.meshes.push(mesh);
    }

    for child in node.children() {
        import_node(scene, &child, model, uv_scale);
    }
}

/// Decompose a transformation matrix into (scale, rotation, translation).
/// Uses the same algorithm as cglm's glm_decompose + glm_mat4_quat:
/// - Translation from column 3
/// - Scale from column norms of the upper-left 3x3
/// - Rotation matrix from normalized columns
/// - Quaternion via Shepperd method
fn decompose_transform(m: &Mat4) -> (Vec3, Quat, Vec3) {
    let translation = Vec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z);

    let col0 = Vec3::new(m.x_axis.x, m.x_axis.y, m.x_axis.z);
    let col1 = Vec3::new(m.y_axis.x, m.y_axis.y, m.y_axis.z);
    let col2 = Vec3::new(m.z_axis.x, m.z_axis.y, m.z_axis.z);

    let sx = col0.length();
    let sy = col1.length();
    let sz = col2.length();
    let scale = Vec3::new(sx, sy, sz);

    let inv_sx = if sx > 0.0 { 1.0 / sx } else { 0.0 };
    let inv_sy = if sy > 0.0 { 1.0 / sy } else { 0.0 };
    let inv_sz = if sz > 0.0 { 1.0 / sz } else { 0.0 };

    // Normalized rotation matrix elements: r[row][col]
    let r00 = col0.x * inv_sx;
    let r10 = col0.y * inv_sx;
    let r20 = col0.z * inv_sx;
    let r01 = col1.x * inv_sy;
    let r11 = col1.y * inv_sy;
    let r21 = col1.z * inv_sy;
    let r02 = col2.x * inv_sz;
    let r12 = col2.y * inv_sz;
    let r22 = col2.z * inv_sz;

    // Quaternion from rotation matrix (Shepperd method, matching cglm's glm_mat4_quat)
    let trace = r00 + r11 + r22;

    let rotation = if trace >= 0.0 {
        let r = (1.0 + trace).sqrt();
        let rinv = 0.5 / r;
        Quat::from_xyzw(
            rinv * (r21 - r12),
            rinv * (r02 - r20),
            rinv * (r10 - r01),
            r * 0.5,
        )
    } else if r00 >= r11 && r00 >= r22 {
        let r = (1.0 - r11 - r22 + r00).sqrt();
        let rinv = 0.5 / r;
        Quat::from_xyzw(
            r * 0.5,
            rinv * (r10 + r01),
            rinv * (r20 + r02),
            rinv * (r21 - r12),
        )
    } else if r11 >= r22 {
        let r = (1.0 - r00 - r22 + r11).sqrt();
        let rinv = 0.5 / r;
        Quat::from_xyzw(
            rinv * (r10 + r01),
            r * 0.5,
            rinv * (r21 + r12),
            rinv * (r02 - r20),
        )
    } else {
        let r = (1.0 - r00 - r11 + r22).sqrt();
        let rinv = 0.5 / r;
        Quat::from_xyzw(
            rinv * (r20 + r02),
            rinv * (r21 + r12),
            r * 0.5,
            rinv * (r10 - r01),
        )
    };

    (scale, rotation, translation)
}

fn import_mesh(
    assimp_mesh: &Mesh,
    out_mesh: &mut ModelMesh,
    model: &mut ModelData,
    uv_scale: f32,
) {
    let mesh_name = assimp_mesh.name();

    if mesh_name.starts_with('#') {
        return;
    }

    let num_vertices = assimp_mesh.num_vertices();

    let vertex_start = out_mesh.vertices.len() as u32;

    let index_start = out_mesh.indices.len() as u32;

    let vertices = assimp_mesh.vertices();
    let normals = assimp_mesh.normals();
    let tangents = assimp_mesh.tangents();
    let bitangents = assimp_mesh.bitangents();
    let uvs = assimp_mesh.texture_coords(0);

    // Extract bone data if present — nodes already exist from import_skeleton
    let mut per_vertex_indices = vec![[0u32; 4]; num_vertices];
    let mut per_vertex_weights = vec![[0.0f32; 4]; num_vertices];

    if assimp_mesh.has_bones() {
        for bone in assimp_mesh.bones() {
            let bone_name = bone.name();

            let node_index = match model.find_node_by_name(&bone_name) {
                Some(idx) => idx,
                None => {
                    eprintln!(
                        "Warning: bone '{}' not found in skeleton hierarchy",
                        bone_name
                    );
                    continue;
                }
            };

            // Assign bone weights to vertices
            for weight in bone.weights_iter() {
                let vid = weight.vertex_id as usize;

                if vid >= num_vertices {
                    continue;
                }

                for slot in 0..4 {
                    if per_vertex_indices[vid][slot] == node_index as u32
                        && per_vertex_weights[vid][slot] > 0.0
                    {
                        // Same bone, keep max weight
                        per_vertex_weights[vid][slot] =
                            per_vertex_weights[vid][slot].max(weight.weight);
                        break;
                    }

                    if per_vertex_weights[vid][slot] == 0.0 {
                        per_vertex_indices[vid][slot] = node_index as u32;
                        per_vertex_weights[vid][slot] = weight.weight;
                        break;
                    }
                }
            }
        }
    }

    // Create compressed vertices
    for i in 0..num_vertices {
        let pos = &vertices[i];
        let normal = normals.as_ref().and_then(|v| v.get(i));
        let uv = uvs.as_ref().and_then(|v| v.get(i));
        let tangent = tangents.as_ref().and_then(|v| v.get(i));
        let bitangent = bitangents.as_ref().and_then(|v| v.get(i));

        let position = [pos.x, pos.y, pos.z];

        let normal_vec = Vec3::new(
            normal.map_or(0.0, |v| v.x),
            normal.map_or(0.0, |v| v.y),
            normal.map_or(1.0, |v| v.z),
        );

        let u = uv.map_or(0.0, |v| v.x) * uv_scale;
        let v_coord = -uv.map_or(0.0, |v| v.y) * uv_scale;
        let texcoord = [f16::from_f32(u).to_bits(), f16::from_f32(v_coord).to_bits()];

        let tangent_vec = Vec3::new(
            tangent.map_or(1.0, |v| v.x),
            tangent.map_or(0.0, |v| v.y),
            tangent.map_or(0.0, |v| v.z),
        );
        let bitangent_vec = Vec3::new(
            bitangent.map_or(0.0, |v| v.x),
            bitangent.map_or(1.0, |v| v.y),
            bitangent.map_or(0.0, |v| v.z),
        );

        let packed_normal = oct_encode_snorm8(normal_vec);
        let packed_tangent = pack_tangent_with_sign(tangent_vec, normal_vec, bitangent_vec);
        let packed_indices = [
            per_vertex_indices[i][0] as u8,
            per_vertex_indices[i][1] as u8,
            per_vertex_indices[i][2] as u8,
            per_vertex_indices[i][3] as u8,
        ];
        let packed_weights = pack_weights_unorm8(per_vertex_weights[i]);

        out_mesh.vertices.push(ModelVertex {
            position,
            texcoord,
            normal: packed_normal,
            _pad: [0; 2],
            tangent: packed_tangent,
            node_indices: packed_indices,
            node_weights: packed_weights,
        });
    }

    // Create indices
    let mut index_count = 0u32;

    for face in assimp_mesh.faces() {
        for &idx in face.indices() {
            out_mesh.indices.push(vertex_start + idx);
            index_count += 1;
        }
    }

    out_mesh.parts.push(ModelMeshPart {
        material_index: assimp_mesh.material_index(),
        index_start,
        index_count,
    });
}

// ---------------------------------------------------------------------------
// Marker import
// ---------------------------------------------------------------------------

fn import_markers(scene: &Scene, node: &Node, model: &mut ModelData) {
    for mesh_idx in node.mesh_indices_iter() {
        let assimp_mesh = scene.mesh(mesh_idx).unwrap();
        let mesh_name = assimp_mesh.name();

        if mesh_name.starts_with('#') {
            let marker_name = &mesh_name[1..];

            if model.find_marker_by_name(marker_name).is_none() {
                let parent_name = node.parent().map(|p| p.name());

                let node_index = parent_name
                    .as_deref()
                    .and_then(|name| model.find_node_by_name(name))
                    .map(|idx| idx as i32)
                    .unwrap_or(-1);

                model.markers.push(ModelMarker {
                    name: marker_name.to_string(),
                    node_index,
                    position: Vec3::ZERO,
                    rotation: Vec3::ZERO,
                });
            }
        }
    }

    for child in node.children() {
        import_markers(scene, &child, model);
    }
}

// ---------------------------------------------------------------------------
// Animation import
// ---------------------------------------------------------------------------

fn import_animation(anim: &asset_importer::Animation, model: &mut ModelData) {
    let mut name = anim.name();

    // Blender hack: strip "Armature|" prefix
    if let Some(stripped) = name.strip_prefix("Armature|") {
        name = stripped.to_string();
    }

    let mut animation = AnimationData {
        name,
        duration: anim.duration() as f32,
        ticks_per_second: anim.ticks_per_second() as f32,
        channels: Vec::new(),
        node_channel_map: Vec::new(),
    };

    // Node channels
    for node_anim in anim.channels() {
        let raw_name = node_anim.node_name();
        // Strip $AssimpFbx$ suffixes that may persist from Assimp's FBX pivot handling
        let node_name = strip_assimp_fbx_suffix(&raw_name);

        let node_index = model
            .find_node_by_name(node_name)
            .map(|i| i as i32)
            .unwrap_or(-1);

        if node_index == -1 {
            // eprintln!(
            //     "Warning: animation channel references unknown node '{}'",
            //     node_name
            // );
            continue;
        }

        let position_keys: Vec<PositionKey> = node_anim
            .position_keys()
            .iter()
            .map(|k| PositionKey {
                time: k.time as f32,
                position: Vec3::new(k.value.x, k.value.y, k.value.z),
            })
            .collect();

        let rotation_keys: Vec<RotationKey> = node_anim
            .rotation_keys()
            .iter()
            .map(|k| RotationKey {
                time: k.time as f32,
                // asset-importer Quaternion: (x, y, z, w) — same order as glam
                rotation: Quat::from_xyzw(k.value.x, k.value.y, k.value.z, k.value.w),
            })
            .collect();

        let scaling_keys: Vec<ScalingKey> = node_anim
            .scaling_keys()
            .iter()
            .map(|k| ScalingKey {
                time: k.time as f32,
                scaling: Vec3::new(k.value.x, k.value.y, k.value.z),
            })
            .collect();

        let position_times: Vec<f32> = position_keys.iter().map(|k| k.time).collect();
        let rotation_times: Vec<f32> = rotation_keys.iter().map(|k| k.time).collect();
        let scaling_times: Vec<f32> = scaling_keys.iter().map(|k| k.time).collect();
        animation.channels.push(AnimationChannel {
            channel_type: AnimationChannelType::Node,
            target_index: node_index,
            position_keys,
            rotation_keys,
            scaling_keys,
            mesh_keys: Vec::new(),
            morph_keys: Vec::new(),
            position_times,
            rotation_times,
            scaling_times,
        });
    }

    // Mesh channels
    for mesh_anim in anim.mesh_channels() {
        let mesh_keys: Vec<MeshKey> = mesh_anim
            .keys()
            .iter()
            .map(|k| MeshKey {
                time: k.time as f32,
                mesh_index: k.value as i32,
            })
            .collect();

        animation.channels.push(AnimationChannel {
            channel_type: AnimationChannelType::Mesh,
            target_index: -1,
            position_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scaling_keys: Vec::new(),
            mesh_keys,
            morph_keys: Vec::new(),
            position_times: Vec::new(),
            rotation_times: Vec::new(),
            scaling_times: Vec::new(),
        });
    }

    // Morph mesh channels
    for morph_anim in anim.morph_mesh_channels() {
        let morph_keys: Vec<MorphKey> = (0..morph_anim.num_keys())
            .filter_map(|i| morph_anim.key(i))
            .map(|k| MorphKey {
                time: k.time() as f32,
                values: k.values().iter().map(|&v| v as i32).collect(),
                weights: k.weights().iter().map(|&w| w as f32).collect(),
            })
            .collect();

        animation.channels.push(AnimationChannel {
            channel_type: AnimationChannelType::Morph,
            target_index: -1,
            position_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scaling_keys: Vec::new(),
            mesh_keys: Vec::new(),
            morph_keys,
            position_times: Vec::new(),
            rotation_times: Vec::new(),
            scaling_times: Vec::new(),
        });
    }

    // Build node_index → [channel indices] lookup
    let num_nodes = model.nodes.len();
    let mut node_channel_map: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for (ch_idx, channel) in animation.channels.iter().enumerate() {
        if channel.channel_type == AnimationChannelType::Node && channel.target_index >= 0 {
            let ni = channel.target_index as usize;
            if ni < num_nodes {
                node_channel_map[ni].push(ch_idx);
            }
        }
    }
    animation.node_channel_map = node_channel_map;

    model.animations.push(animation);
}
