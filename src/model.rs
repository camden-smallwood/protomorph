use crate::{
    animation::{
        AnimationChannel, AnimationChannelType, AnimationData, MeshKey, MorphKey, PositionKey,
        RotationKey, ScalingKey,
    },
    materials::MaterialData,
};
use asset_importer::{Scene, mesh::Mesh, node::Node, postprocess::PostProcessSteps};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::{collections::HashMap, path::Path};

// ---------------------------------------------------------------------------
// Vertex types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexType {
    Rigid,
    Skinned,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VertexRigid {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl VertexRigid {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // texcoord
        3 => Float32x3,  // tangent
        4 => Float32x3,  // bitangent
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<VertexRigid>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VertexSkinned {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    pub node_indices: [u32; 4],
    pub node_weights: [f32; 4],
}

impl VertexSkinned {
    const ATTRIBS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // texcoord
        3 => Float32x3,  // tangent
        4 => Float32x3,  // bitangent
        5 => Uint32x4,   // node_indices
        6 => Float32x4,  // node_weights
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<VertexSkinned>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
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
    pub vertex_type: VertexType,
    pub rigid_vertices: Vec<VertexRigid>,
    pub skinned_vertices: Vec<VertexSkinned>,
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
// Scene node tree walking (for bone parent resolution)
// ---------------------------------------------------------------------------

/// Build a map from node name -> parent node name by walking the scene node tree.
fn build_node_parent_map(
    node: &Node,
    parent_name: Option<&str>,
    map: &mut HashMap<String, String>,
) {
    let name = node.name();

    if let Some(pname) = parent_name {
        map.insert(name.clone(), pname.to_string());
    }

    for child in node.children() {
        build_node_parent_map(&child, Some(&name), map);
    }
}

/// Build a map from node name -> node transformation by walking the scene node tree.
fn build_node_transform_map(node: &Node, map: &mut HashMap<String, Mat4>) {
    let name = node.name();
    
    map.insert(name.clone(), mat4_from_assimp(&node.transformation()));

    for child in node.children() {
        build_node_transform_map(&child, map);
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

impl ModelData {
    pub fn from_file(path: &str) -> Self {
        let scene = Scene::from_file_with_flags(
            path,
            PostProcessSteps::TRIANGULATE
                | PostProcessSteps::CALC_TANGENT_SPACE
                | PostProcessSteps::JOIN_IDENTICAL_VERTICES
                | PostProcessSteps::SORT_BY_PTYPE
                | PostProcessSteps::FIND_DEGENERATES
                | PostProcessSteps::FIND_INVALID_DATA
                | PostProcessSteps::VALIDATE_DATA_STRUCTURE,
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
        };

        // Build helper maps from the scene node tree for bone parent/transform resolution
        let mut node_parent_map = HashMap::new();
        let mut node_transform_map = HashMap::new();

        if let Some(root) = scene.root_node() {
            build_node_parent_map(&root, None, &mut node_parent_map);
            build_node_transform_map(&root, &mut node_transform_map);

            // Walk scene graph: import meshes + bones
            import_node(
                &scene,
                &root,
                &mut model,
                &node_parent_map,
                &node_transform_map,
            );

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

    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");

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
    node_parent_map: &HashMap<String, String>,
    node_transform_map: &HashMap<String, Mat4>,
) {
    // Collect mesh indices so we can iterate twice (has_bones check + import)
    let mesh_indices: Vec<usize> = node.mesh_indices_iter().collect();

    // Determine if any mesh in this node has bones
    let has_bones = mesh_indices
        .iter()
        .any(|&idx| scene.mesh(idx).map_or(false, |m| m.has_bones()));

    let vertex_type = if has_bones {
        VertexType::Skinned
    } else {
        VertexType::Rigid
    };

    let mut mesh = ModelMesh {
        vertex_type,
        rigid_vertices: Vec::new(),
        skinned_vertices: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
    };

    for &mesh_idx in &mesh_indices {
        let assimp_mesh = scene.mesh(mesh_idx).unwrap();

        import_mesh(
            &assimp_mesh,
            &mut mesh,
            model,
            node_parent_map,
            node_transform_map,
        );
    }

    let has_verts = match vertex_type {
        VertexType::Rigid => !mesh.rigid_vertices.is_empty(),
        VertexType::Skinned => !mesh.skinned_vertices.is_empty(),
    };

    if has_verts {
        model.meshes.push(mesh);
    }

    for child in node.children() {
        import_node(scene, &child, model, node_parent_map, node_transform_map);
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
    node_parent_map: &HashMap<String, String>,
    node_transform_map: &HashMap<String, Mat4>,
) {
    let mesh_name = assimp_mesh.name();

    if mesh_name.starts_with('#') {
        return;
    }

    let num_vertices = assimp_mesh.num_vertices();

    let vertex_start = match out_mesh.vertex_type {
        VertexType::Rigid => out_mesh.rigid_vertices.len() as u32,
        VertexType::Skinned => out_mesh.skinned_vertices.len() as u32,
    };

    let index_start = out_mesh.indices.len() as u32;

    let vertices = assimp_mesh.vertices();
    let normals = assimp_mesh.normals();
    let tangents = assimp_mesh.tangents();
    let bitangents = assimp_mesh.bitangents();
    let uvs = assimp_mesh.texture_coords(0);

    // Extract bone data if present
    let mut per_vertex_indices = vec![[0u32; 4]; num_vertices];
    let mut per_vertex_weights = vec![[0.0f32; 4]; num_vertices];

    if out_mesh.vertex_type == VertexType::Skinned {
        for bone in assimp_mesh.bones() {
            let bone_name = bone.name();

            // Skip "Armature" bone (Blender hack)
            if bone_name == "Armature" {
                continue;
            }

            let node_index = match model.find_node_by_name(&bone_name) {
                Some(idx) => idx,
                None => {
                    // Create new model node for this bone
                    let default_transform = node_transform_map
                        .get(&bone_name)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);

                    let offset_matrix = mat4_from_assimp(&bone.offset_matrix());

                    // Find parent node index by walking the scene node tree
                    let parent_node_index = node_parent_map
                        .get(&bone_name)
                        .and_then(|parent_name| {
                            // Skip "Armature" as parent (Blender hack)
                            if parent_name == "Armature" {
                                return None;
                            }
                            model.find_node_by_name(parent_name)
                        })
                        .map(|idx| idx as i32)
                        .unwrap_or(-1);

                    let default_decomposed = decompose_transform(&default_transform);

                    let new_node = ModelNode {
                        name: bone_name.clone(),
                        parent_index: -1,
                        first_child_index: -1,
                        next_sibling_index: -1,
                        offset_matrix,
                        default_transform,
                        default_decomposed,
                    };

                    model.add_child_node(parent_node_index, new_node)
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
                        per_vertex_weights[vid][slot] = per_vertex_weights[vid][slot].max(weight.weight);
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

    // Create vertices
    for i in 0..num_vertices {
        let pos = &vertices[i];
        let normal = normals.as_ref().and_then(|v| v.get(i));
        let uv = uvs.as_ref().and_then(|v| v.get(i));
        let tangent = tangents.as_ref().and_then(|v| v.get(i));
        let bitangent = bitangents.as_ref().and_then(|v| v.get(i));

        let position = [pos.x, pos.y, pos.z];

        let normal_arr = [
            normal.map_or(0.0, |v| v.x),
            normal.map_or(0.0, |v| v.y),
            normal.map_or(0.0, |v| v.z),
        ];

        let texcoord = [uv.map_or(0.0, |v| v.x), -uv.map_or(0.0, |v| v.y)];

        let tangent_arr = [
            tangent.map_or(0.0, |v| v.x),
            tangent.map_or(0.0, |v| v.y),
            tangent.map_or(0.0, |v| v.z),
        ];

        let bitangent_arr = [
            bitangent.map_or(0.0, |v| v.x),
            bitangent.map_or(0.0, |v| v.y),
            bitangent.map_or(0.0, |v| v.z),
        ];

        match out_mesh.vertex_type {
            VertexType::Rigid => {
                out_mesh.rigid_vertices.push(VertexRigid {
                    position,
                    normal: normal_arr,
                    texcoord,
                    tangent: tangent_arr,
                    bitangent: bitangent_arr,
                });
            }

            VertexType::Skinned => {
                out_mesh.skinned_vertices.push(VertexSkinned {
                    position,
                    normal: normal_arr,
                    texcoord,
                    tangent: tangent_arr,
                    bitangent: bitangent_arr,
                    node_indices: per_vertex_indices[i],
                    node_weights: per_vertex_weights[i],
                });
            }
        }
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
    };

    // Node channels
    for node_anim in anim.channels() {
        let node_name = node_anim.node_name();

        // Blender hack: skip "Armature" channel
        if node_name == "Armature" {
            continue;
        }

        let node_index = model
            .find_node_by_name(&node_name)
            .map(|i| i as i32)
            .unwrap_or(-1);

        if node_index == -1 {
            eprintln!(
                "Warning: animation channel references unknown node '{}'",
                node_name
            );
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

        animation.channels.push(AnimationChannel {
            channel_type: AnimationChannelType::Node,
            target_index: node_index,
            position_keys,
            rotation_keys,
            scaling_keys,
            mesh_keys: Vec::new(),
            morph_keys: Vec::new(),
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
        });
    }

    model.animations.push(animation);
}
