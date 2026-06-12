pub mod geometry_sampling;

use crate::halo::render_methods::materials::MaterialData;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

/// Maximum per-object node (bone) count — sizes the GPU
/// `node_matrices` storage buffer (one `Mat4` slot per node). Mirrors
/// the engine's fixed `MAXIMUM_NODES_PER_MODEL`. Relocated here from
/// the (now-deleted) animations module since it's purely a geometry/GPU
/// buffer-sizing constant.
pub const MAXIMUM_NUMBER_OF_MODEL_NODES: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniforms {
    pub model: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Vertex types
// ---------------------------------------------------------------------------

/// Compressed vertex: 36 bytes (down from 92, +4 for lightmap UV).
/// Normal: oct-encoded Snorm8x2. Texcoord: Float16x2. Tangent: Snorm8x4 (xyz + w handedness).
/// Lightmap texcoord: Float16x2 (zero on objects/instances/scenery; populated for
/// BSP cluster meshes from the per-BSP lightmap tag's `imported geometry`).
/// Bitangent: reconstructed in shader as cross(normal, tangent.xyz) * sign(tangent.w).
/// Bone indices: Uint8x4. Bone weights: Unorm8x4.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],         // 12 bytes, offset 0
    pub texcoord: [u16; 2],         // 4 bytes,  offset 12 (Float16x2, raw bits)
    pub normal: [i8; 2],            // 2 bytes,  offset 16 (Snorm8x2, oct-encoded)
    pub _pad: [u8; 2],              // 2 bytes,  offset 18 (alignment padding)
    pub tangent: [i8; 4],           // 4 bytes,  offset 20 (Snorm8x4, xyz + w sign)
    pub node_indices: [u8; 4],      // 4 bytes,  offset 24 (Uint8x4)
    pub node_weights: [u8; 4],      // 4 bytes,  offset 28 (Unorm8x4)
    pub lightmap_texcoord: [u16; 2],// 4 bytes,  offset 32 (Float16x2, raw bits)
}

impl ModelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 7] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 }, // position
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm8x2,  offset: 16, shader_location: 1 }, // normal (oct)
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float16x2, offset: 12, shader_location: 2 }, // texcoord
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm8x4,  offset: 20, shader_location: 3 }, // tangent (xyz+w)
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint8x4,   offset: 24, shader_location: 4 }, // node_indices
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4,  offset: 28, shader_location: 5 }, // node_weights
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float16x2, offset: 32, shader_location: 6 }, // lightmap_texcoord
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Per-vertex SH probe — one entry per vertex, parallel to the
/// primary vertex buffer's stream. Built at scenario load from
/// `LightmapPerVertexBlock::lightprobe_data[v_idx]` for clusters /
/// instances with `pervertex_block_index >= 0`. Bound as a SECONDARY
/// vertex buffer at draw time; the rasterizer interpolates to the
/// fragment, and the PS evaluates `ravi_order_2` at the fragment
/// normal — engine-faithful per-vertex SH path mirrors what dllcache
/// does for `pervertex_block_index >= 0` instances/clusters (see
/// `reference_bsp_per_vertex_sh_stream.md`).
///
/// SH packing matches `bake.rs::ravi_eval_order2`: a vec4 per channel
/// holding `(DC, X, Y, -Z)` so the WGSL helper can do
/// `0.886 × dc + (-1.023) × dot(normal, l1)`.
///
/// 64 bytes / vertex (4 × vec4). Larger maps with many per-vertex-lit
/// instances pay roughly 2× the vertex memory of the primary buffer;
/// guardian is ~30 KB extra (~426 instances × ~hundreds of vertices).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct PerVertexShVertex {
    /// `(DC, X, Y, -Z)` for the R channel. Z is negated to match
    /// Halo's L1 cbuffer convention (`reference_halo_sh_packing.md`)
    /// so the same eval helper works for both vertex-attrib and
    /// cbuffer-bound SH.
    pub sh_r: [f32; 4],
    pub sh_g: [f32; 4],
    pub sh_b: [f32; 4],
    /// `(R, G, B, _)` — the per-vertex probe's authored dominant
    /// light intensity. Per-vertex probes don't carry a direction;
    /// the L1 bands give it implicitly via the SH eval.
    pub dominant_intensity: [f32; 4],
}

impl PerVertexShVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset:  0, shader_location: 7 },  // sh_r
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 8 },  // sh_g
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 9 },  // sh_b
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 10 }, // dominant_intensity
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<PerVertexShVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    /// Pack a dequantized per-vertex probe into the vertex-attrib form.
    pub fn from_probe(
        probe: &blam_tags::scenario_lightmap::DequantizedPerVertexProbe,
    ) -> Self {
        // Halo per-vertex SH: red_terms[0]=DC, [1]=Y, [2]=Z, [3]=X.
        // Pack as (DC, X, Y, -Z) — `-Z` matches the cbuffer convention
        // (`reference_halo_sh_packing.md`: "Z component is NEGATED at
        // upload"). The shared `ravi_eval_order2` helper then works on
        // both attrib- and cbuffer-bound forms.
        let pack = |c: &[f32; 4]| [c[0], c[3], c[1], -c[2]];
        Self {
            sh_r: pack(&probe.red_terms),
            sh_g: pack(&probe.green_terms),
            sh_b: pack(&probe.blue_terms),
            dominant_intensity: [
                probe.dominant_light_intensity[0],
                probe.dominant_light_intensity[1],
                probe.dominant_light_intensity[2],
                0.0,
            ],
        }
    }
}

/// Per-vertex PRT Ambient transfer scalar — one `f32` per vertex,
/// baked from `per_mesh_prt_data[i].mesh pca data` at scenario load
/// (see `blam-tags/src/render_model.rs` and Reach
/// `create_prt_vertex_buffer @ 0x82E080F0`). MCC PC vertex declaration
/// is `R32_FLOAT BLENDWEIGHT1` slot 2 (Ares
/// `rasterizer_resource_definitions.cpp:46`); we mirror format + count
/// in WGSL at `entry_static_prt_ambient.wgsl`'s `@location(11)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct PrtAmbientVertex {
    pub transfer: f32,
}

impl PrtAmbientVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 0, shader_location: 11 },
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<PrtAmbientVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Per-vertex baked sky color — engine
/// `_vertex_buffer_usage_vert_color` stream (`mesh.vertex_buffer_indices
/// [2]`). Consumed by the sky pipeline (`entry_sky_dome_simple.wgsl`)
/// which mirrors the engine sky shader `out_color = vert_color *
/// g_exposure.rrr + sun_glare`. Padded to vec4 because Metal aligns
/// 3-component vertex attributes to 16B.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct SkyVertColorVertex {
    pub color: [f32; 4],
}

impl SkyVertColorVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 12 },
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<SkyVertColorVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ---------------------------------------------------------------------------
// Vertex packing helpers (used by halo.rs to import render_model vertices)
// ---------------------------------------------------------------------------

/// Octahedral encode: unit vec3 → [i8; 2] (Snorm8)
pub(crate) fn oct_encode_snorm8(n: Vec3) -> [i8; 2] {
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
pub(crate) fn pack_tangent_with_sign(tangent: Vec3, normal: Vec3, bitangent: Vec3) -> [i8; 4] {
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
pub(crate) fn pack_weights_unorm8(weights: [f32; 4]) -> [u8; 4] {
    let sum: f32 = weights.iter().sum();
    if sum < 0.0001 {
        return [0; 4];
    }
    let normalized = weights.map(|w| w / sum);
    let mut packed: [u8; 4] = normalized.map(|w| (w * 255.0).round() as u8);
    let packed_sum: u16 = packed.iter().map(|&b| b as u16).sum();
    if packed_sum > 0 && packed_sum != 255 {
        let diff = 255i16 - packed_sum as i16;
        let max_idx = packed.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap().0;
        packed[max_idx] = (packed[max_idx] as i16 + diff).clamp(0, 255) as u8;
    }
    packed
}

/// Decompose a 4×4 transform into (scale, rotation, translation).
#[allow(dead_code)]
pub(crate) fn decompose_transform(m: &Mat4) -> (Vec3, Quat, Vec3) {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    (scale, rotation, translation)
}

// ---------------------------------------------------------------------------
// Model data types
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct ModelMeshPart {
    pub material_index: usize,
    pub index_start: u32,
    pub index_count: u32,
    /// [`blam_tags::render_model::GeometryPartType`] (Ares
    /// `e_geometry_part_type`).
    pub part_type: blam_tags::render_model::GeometryPartType,
    /// Model-space transparent sort centroid — `RenderMeshPart.sort_position
    /// .position` (the part's authored `SortingPosition`). `None` for parts
    /// with no authored sort position (opaque / `transparent_sorting_index<0`).
    /// The transparent dispatch transforms this by the part's world matrix and
    /// derives `z_sort` (camera distance) for the back-to-front sort — engine
    /// `c_transparency_renderer::add_element`.
    pub sort_centroid: Option<[f32; 3]>,
}

pub struct ModelMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<ModelMeshPart>,
    /// Per-vertex PRT Ambient transfer scalars (one `f32` per vertex,
    /// grayscale 0..1). Sourced from `RenderMesh::prt_ambient_stream`;
    /// empty when the mesh's `per_mesh_prt_data[i].mesh pca data` blob
    /// was absent or size-mismatched. Uploaded as a slot-1 vertex
    /// buffer for `HaloEntryPoint::StaticPrtAmbient` draws.
    pub prt_ambient_stream: Vec<f32>,
    /// Per-vertex baked sky color (RGB, one entry per vertex).
    /// Populated for `_mesh_has_vertex_color_bit` meshes (engine
    /// `RenderMesh::has_vertex_color`). Drives the sky pipeline path
    /// in `submit_and_render_sky(1, ...)`. Engine equivalent: the
    /// `_vertex_buffer_usage_vert_color` stream consumed by
    /// `_entry_point_vertex_color_lighting` (idx=14).
    pub vert_color_stream: Vec<[f32; 3]>,
    /// `MeshFlags::UseRegionIndexForSorting` (`mesh->flags & 2`). When set,
    /// transparent parts of this mesh sort by `region_index` (constant
    /// `global_origin3d` centroid + `region*0.1` offset) per engine
    /// `submit_object_mesh_parts`, so co-located transparent layers (waterfall
    /// liquid/sheet/mist) hold a fixed composite order under camera motion.
    pub use_region_index_for_sorting: bool,
    /// The render-model region this mesh belongs to (the engine's
    /// `region_index` sort key). Assigned during `select_meshes`.
    pub region_index: u8,
}

pub struct ModelData {
    pub materials: Vec<MaterialData>,
    /// GPU mesh data (`ModelVertex` format).
    pub meshes: Vec<ModelMesh>,
    /// Object-local node-**world** bind-pose matrices (one per node).
    /// Precomputed at load from `render_model.nodes` (parent chain of
    /// `default_transform`s). The renderer uploads these as the
    /// per-object `node_matrices` (`model_u × node × vertex`). Static
    /// per model until the real animation system lands.
    pub node_local_matrices: Vec<Mat4>,
    /// Marker name → object-local marker **world** matrix
    /// (`node_world[marker.node] × marker.local`). The attach point for
    /// parented children (engine
    /// `internal_object_get_markers_by_string_id`).
    pub marker_transforms: Vec<(String, Mat4)>,
    /// Authored model-space bounding sphere `(center.xyz, radius)`,
    /// pulled from the object tag's `bounding offset` + `bounding
    /// radius` fields (or the .model's `model object data[0]` as
    /// fallback). `None` when neither source authored a usable value
    /// — caller should fall back to vertex-walk autogen. Engine
    /// equivalent: `object_get_bounding_sphere @ 0x1802473A0`
    /// reads the runtime-cached form of these.
    pub authored_bounding_sphere: Option<glam::Vec4>,
    /// `true` if this object's tag is eligible to cast a runtime
    /// lightmap shadow per `render_object_has_lightmap_shadow @
    /// 0x180696EE0`. Bakes the engine's per-object shadow gate at
    /// load time:
    ///   - false if `obj_def.flags & does_not_cast_shadow` (bit 0)
    ///   - false if `lightmap_shadow_mode == never (1)`
    ///   - if `lightmap_shadow_mode == default (0)`: true only for
    ///     types {biped, vehicle, weapon}
    ///   - if `lightmap_shadow_mode == always (2) | blur (3)`: true
    ///     only for types {terminal, scenery, machine, control,
    ///     sound_scenery}
    ///
    /// Most static scenery on a typical riverworld-style map evaluates
    /// to `false` here — it's lightmap-baked, not runtime-shadowed.
    /// Engine consults this via the pre-baked
    /// `cluster_partition.object_visibility_cull_flags` byte (bit 1).
    pub casts_shadow: bool,
}

impl ModelData {
    /// Object-local world matrix of a named marker (precomputed at load
    /// into `marker_transforms`). Compose with the object's world matrix
    /// to get the marker's world transform — the attach point for
    /// parented children (engine
    /// `internal_object_get_markers_by_string_id`). `None` if no marker
    /// matches `name`.
    pub fn marker_transform(&self, name: &str) -> Option<Mat4> {
        self.marker_transforms
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| *m)
    }

    /// ALL markers matching `name`, in declaration order. A marker name can
    /// repeat (e.g. the guardian minilift has three `fx_minilift_pad` markers,
    /// one per pad), and the engine emits an effect at each — so a location
    /// must resolve to every match, not just the first.
    pub fn marker_transforms_all(&self, name: &str) -> Vec<Mat4> {
        self.marker_transforms
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, m)| *m)
            .collect()
    }
}
