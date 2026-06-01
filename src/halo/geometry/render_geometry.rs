//! Mirror of `Ares/source/geometry/geometry_definitions.h` +
//! `geometry_tag_definitions.h`.
//!
//! Halo's render geometry is a tag-resident layout that the runtime
//! decodes into vertex/index buffers + per-mesh metadata. The render
//! path reads:
//!
//!   - `s_render_geometry` — top-level (132B). One per render_model
//!     or per-bsp. Owns blocks of `s_mesh`, compression infos,
//!     positioning, raw mesh data, MOPP, node maps, prt data, etc.
//!   - `s_mesh` (76B) — one mesh's metadata. Carries vertex buffer
//!     indices (8 slots for the different `e_vertex_buffer_usage`
//!     types), index buffer index, type/flags, instance buckets.
//!   - `s_part` (16B) — a draw range within a mesh. References a
//!     render_method index. Carries transparent_sorting_index,
//!     budget_vertex_count.
//!   - `s_subpart` (8B) — sub-ranges inside a part for shadowing /
//!     occlusion culling.
//!   - `s_raw_vertex` (96B) — full-precision tag-side vertex used
//!     by the lightmapper / build pipeline (runtime uses the
//!     compressed `s_render_vertex`).
//!   - `s_compression_info` (44B) — bounds for vertex-position +
//!     UV decompression.
//!   - `s_per_mesh_raw_data` (44B) — uncompressed verts+indices
//!     (only present in tags built with raw data retained).
//!   - `s_per_instance_lightmap_texcoords` (16B) — per-placement
//!     lightmap UV stream (lightmap tag's
//!     `imported_geometry/per_instance_lightmap_texcoords`).
//!
//! This module mirrors the Ares structs verbatim. Loaders that
//! decode tags into runtime GPU resources continue to live in
//! `halo/scenario/` + `halo/structures/`; this file only carries the
//! type shapes so c_object_renderer / c_structure_renderer can refer
//! to them by their canonical names.

use glam::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// `e_mesh_flags` (geometry_definitions.h:13-21).
pub mod mesh_flags {
    pub const HAS_VERTEX_COLOR: u32 = 1 << 0;
    pub const USE_REGION_INDEX_FOR_TRANSPARENT_SORTING: u32 = 1 << 1;
    pub const USE_VERTEX_BUFFERS_FOR_INDICES: u32 = 1 << 2;
    pub const HAS_PER_INSTANCE_LIGHTING: u32 = 1 << 3;
    pub const IS_UNINDEXED: u32 = 1 << 4;
}

/// `e_vertex_buffer_usage` (geometry_definitions.h:23-34). The 8 slots
/// `s_mesh::vertex_buffer_indices[8]` map to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VertexBufferUsage {
    Standard = 0,
    LightmapUv = 1,
    VertColor = 2,
    Prt = 3,
    Instances = 4,
    Indices = 5,
    WaterIndices = 6,
    WaterVertex = 7,
}

/// `e_per_mesh_raw_data_flags` (geometry_definitions.h:36-42).
pub mod per_mesh_raw_data_flags {
    pub const INDICES_ARE_TRIANGLE_STRIPS: u32 = 1 << 0;
    pub const INDICES_ARE_TRIANGLE_LISTS: u32 = 1 << 1;
    pub const INDICES_ARE_QUAD_LISTS: u32 = 1 << 2;
}

/// `e_part_flags` (geometry_definitions.h:44-51).
pub mod part_flags {
    pub const DISLIKES_PHOTONS: u8 = 1 << 0;
    pub const IGNORED_BY_LIGHTMAPPER: u8 = 1 << 1;
    pub const HAS_TRANSPARENT_SORTING_PLANE: u8 = 1 << 2;
    pub const IS_WATER_SURFACE: u8 = 1 << 3;
}

/// `e_render_geometry_user_data_type` (geometry_definitions.h:53-57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RenderGeometryUserDataType {
    PrtInfo = 0,
}

// ---------------------------------------------------------------------------
// Mesh / part / subpart / raw_vertex / compression_info
// ---------------------------------------------------------------------------

/// `s_mesh` (geometry_definitions.h:76-91, 76B).
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub parts: Vec<Part>,                    // 0x0   (s_tag_block parts)
    pub subparts: Vec<Subpart>,               // 0xC   (s_tag_block subparts)
    pub vertex_buffer_indices: [u16; 8],      // 0x18  (one per VertexBufferUsage)
    pub index_buffer_index: u16,              // 0x28
    pub index_buffer_index_tessellation: u16, // 0x2A
    pub flags: u8,                            // 0x2C  (e_mesh_flags)
    pub rigid_node_index: u8,                 // 0x2D
    /// `e_vertex_type` selector (rigid / skinned / ambient_prt /
    /// linear_prt / quadratic_prt / static_prt / etc.).
    pub vertex_type: u8,                      // 0x2E
    pub transfer_vector_vertex_type: u8,      // 0x2F
    /// `0` = triangle strip, `3` = triangle list (per existing
    /// `BspMeshMetadata::index_buffer_type` doc).
    pub index_buffer_type: u8,                // 0x30
    // 3-byte pad at 0x31
    pub instance_buckets: Vec<InstanceBucket>, // 0x34
    pub water_indices_start: Vec<u16>,        // 0x40
}

/// `s_part` (geometry_definitions.h:151-163, 16B).
#[derive(Debug, Clone, Copy, Default)]
pub struct Part {
    pub render_method_index: i16,             // 0x0
    pub transparent_sorting_index: i16,       // 0x2
    pub index_start: u16,                     // 0x4
    pub index_count: u16,                     // 0x6
    pub subpart_start: i16,                   // 0x8
    pub subpart_count: i16,                   // 0xA
    pub part_type: u8,                        // 0xC
    /// `e_part_flags` bits.
    pub flags: u8,                            // 0xD
    pub budget_vertex_count: u16,             // 0xE
}

/// `s_subpart` (geometry_definitions.h:165-172, 8B).
#[derive(Debug, Clone, Copy, Default)]
pub struct Subpart {
    pub index_start: u16,                     // 0x0
    pub index_count: u16,                     // 0x2
    pub part_index: u16,                      // 0x4
    pub budget_vertex_count: u16,             // 0x6
}

/// `s_raw_vertex` (geometry_definitions.h:174-186, 96B).
///
/// Lightmapper / build-pipeline vertex (full-precision). Runtime
/// uses the compressed `s_render_vertex` from the rasterizer vertex
/// buffer; this shape only matters when re-decoding raw verts (e.g.
/// our lightmap geometry path that reads from
/// `imported_geometry.per_mesh_temporary[i].raw_vertices`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RawVertex {
    pub position: Vec3,            // 0x0
    pub texcoord: Vec2,            // 0xC
    pub normal: Vec3,              // 0x14
    pub binormal: Vec3,            // 0x20
    pub tangent: Vec3,             // 0x2C
    pub lightmap_texcoord: Vec2,   // 0x38
    pub node_indices: [u8; 4],     // 0x40
    pub node_weights: [f32; 4],    // 0x44
    pub vert_color: Vec3,          // 0x54
}

/// `s_compression_info` (geometry_definitions.h:188-195, 44B).
#[derive(Debug, Clone, Copy, Default)]
pub struct CompressionInfo {
    pub compression_flags: u16,            // 0x0
    pub _pad: u16,                          // 0x2
    /// `real_rectangle3d position_bounds` — (x_min, x_max, y_min,
    /// y_max, z_min, z_max).
    pub position_bounds: [f32; 6],          // 0x4
    /// `real_rectangle2d texcoord_bounds` — (u_min, u_max, v_min,
    /// v_max).
    pub texcoord_bounds: [f32; 4],          // 0x1C
}

// ---------------------------------------------------------------------------
// Auxiliary blocks
// ---------------------------------------------------------------------------

/// `s_per_mesh_raw_data` (geometry_definitions.h:94-103, 44B).
#[derive(Debug, Clone, Default)]
pub struct PerMeshRawData {
    pub raw_vertices: Vec<RawVertex>,       // 0x0
    pub raw_indices: Vec<u16>,              // 0xC
    pub raw_water_data: Vec<u8>,            // 0x18 (opaque blob)
    pub parameterized_texture_width: i16,   // 0x24
    pub parameterized_texture_height: i16,  // 0x26
    /// `e_per_mesh_raw_data_flags` bits.
    pub flags: u32,                         // 0x28
}

/// `s_per_mesh_mopp` (geometry_definitions.h:105-110, 32B). MOPP =
/// Memory-Optimized-Partial-Polytope, Halo's spatial-acceleration
/// blob for lightmap occlusion / AABB tree.
#[derive(Debug, Clone, Default)]
pub struct PerMeshMopp {
    pub mopp_code: Vec<u8>,
    pub mopp_reorder_table: Vec<u16>,
}

/// `s_per_mesh_node_map` (geometry_definitions.h:112-115, 12B).
/// Per-mesh remap from local node indices to skeleton bone indices.
#[derive(Debug, Clone, Default)]
pub struct PerMeshNodeMap {
    pub node_map: Vec<u16>,
}

/// `s_per_instance_prt_data` (`scenario_lightmap_bsp_data.json`
/// `per_instance_prt_data_block`, 20B). Per-instance PCA codebook
/// override — empty on every MCC H3 tag sampled (instances use the
/// per-mesh codebook).
#[derive(Debug, Clone, Default)]
pub struct PerInstancePrtData {
    /// Author-time PCA codebook (`3 floats × vertex_count` for PRT
    /// Ambient). Reach `create_prt_vertex_buffer @ 0x82E080F0` averages
    /// these to a 1-byte UNORM stream and resizes this blob to 0; MCC
    /// tags sometimes still carry it.
    pub mesh_pca_data: Vec<u8>,
}

/// `s_per_mesh_prt_data` (`scenario_lightmap_bsp_data.json`
/// `per_mesh_prt_data_block`, 32B). One entry per mesh. Carries the
/// PCA codebook from which the offline cache builder produces the
/// per-vertex PRT vertex stream (slot 3, declaration 9, 1 byte/vertex
/// grayscale transfer coefficient — see Reach
/// `create_prt_vertex_buffer @ 0x82E080F0`).
///
/// Runtime never reads this block directly — the baked PRT vertex
/// buffer lives in `api_resource.pc_vertex_buffers[mesh.vertex_buffer_indices[3]]`.
/// Project memory `project_research_per_mesh_prt_2026_05_11.md` has the
/// full bake + runtime trace.
#[derive(Debug, Clone, Default)]
pub struct PerMeshPrtData {
    /// Author-time PCA codebook (`3 floats × vertex_count` of RGB
    /// transfer coefficients for PRT Ambient). The bake averages RGB
    /// to grayscale and quantizes 0..255 into the runtime vertex
    /// buffer.
    pub mesh_pca_data: Vec<u8>,
    /// Per-instance codebook overrides. Empty across the entire MCC H3
    /// corpus — instances reuse the per-mesh codebook.
    pub per_instance_prt_data: Vec<PerInstancePrtData>,
}

/// `s_per_instance_lightmap_texcoords` (geometry_definitions.h:118-124,
/// 16B). Per-placement lightmap UV stream from the lightmap tag's
/// `imported_geometry/per_instance_lightmap_texcoords` block. Already
/// loaded ad-hoc in protomorph; this struct defines the canonical
/// shape.
#[derive(Debug, Clone, Default)]
pub struct PerInstanceLightmapTexcoords {
    pub texture_coordinates: Vec<Vec2>, // 0x0
    pub vertex_buffer_index: i16,       // 0xC
    pub _pad: i16,
}

/// `s_instance_bucket` (geometry_definitions.h:222-228, 16B).
/// Groups instances of the same mesh+definition for instanced
/// drawing.
#[derive(Debug, Clone, Default)]
pub struct InstanceBucket {
    pub mesh_index: u16,        // 0x0
    pub definition_index: u16,  // 0x2
    pub instances: Vec<u16>,    // 0x4 (instance indices)
}

// ---------------------------------------------------------------------------
// Buffer info + access
// ---------------------------------------------------------------------------

/// `s_vertex_buffer_info` (geometry_definitions.h:230-235, 12B).
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexBufferInfo {
    pub vertex_count: i32,
    pub vertex_stride: i32,
    pub vertex_declaration: i32,
}

/// `s_index_buffer_info` (geometry_definitions.h:238-243, 8B).
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexBufferInfo {
    pub index_count: i32,
    pub primitive_type: i32,
}

/// `s_render_geometry_api_resource` (geometry_definitions.h:66-72,
/// 48B). The four `s_tag_block` slots are populated post-tag-load
/// with handles to the actual GPU vertex/index buffers (PC: D3D11
/// buffers; Xenon: tile-resource buffers).
#[derive(Debug, Clone, Default)]
pub struct RenderGeometryApiResource {
    pub pc_vertex_buffers: Vec<VertexBufferInfo>,
    pub pc_index_buffers: Vec<IndexBufferInfo>,
    /// Halo also stores Xenon-specific buffer descriptors. We don't
    /// load the Xenon path; field omitted.
    pub _xenon_omitted: (),
}

// ---------------------------------------------------------------------------
// Top-level: s_render_geometry (132B)
// ---------------------------------------------------------------------------

/// `s_render_geometry` (geometry_tag_definitions.h:14-28, 132B).
#[derive(Debug, Clone, Default)]
pub struct RenderGeometry {
    pub flags: u32,                                          // 0x0
    pub meshes: Vec<Mesh>,                                    // 0x4
    pub compression_infos: Vec<CompressionInfo>,              // 0x10
    /// `s_positioning` block — per-mesh per-permutation bounding
    /// sphere + plane for occlusion culling.
    pub positioning: Vec<Positioning>,                        // 0x1C
    pub user_data: Vec<RenderGeometryUserData>,               // 0x28
    pub per_mesh_raw_data: Vec<PerMeshRawData>,               // 0x34
    pub per_mesh_mopp: Vec<PerMeshMopp>,                      // 0x40
    pub per_mesh_node_map: Vec<PerMeshNodeMap>,               // 0x4C
    pub per_mesh_subpart_visibility: Vec<u32>,                // 0x58 (bit-packed)
    pub per_mesh_prt_data: Vec<PerMeshPrtData>,               // 0x64
    pub per_instance_lightmap_texcoords: Vec<PerInstanceLightmapTexcoords>, // 0x70
    pub api_resource: RenderGeometryApiResource,              // 0x7C
}

/// `s_positioning` (geometry_definitions.h:212-220, 48B).
#[derive(Debug, Clone, Copy, Default)]
pub struct Positioning {
    pub plane: [f32; 4],         // 0x0  (real_plane3d)
    pub position: Vec3,          // 0x10
    pub radius: f32,             // 0x1C
    pub node_indices: [u8; 4],   // 0x20
    pub node_weights: [f32; 3],  // 0x24
}

/// `s_render_geometry_user_data` (geometry_definitions.h:205-209, 24B).
#[derive(Debug, Clone, Default)]
pub struct RenderGeometryUserData {
    pub user_data_type: u8,
    pub user_data_count: u8,
    pub user_data_size: u16,
    pub data: Vec<u8>,
}
