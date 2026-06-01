//! Engine struct mirrors for the decal mesh-fragment builder.
//!
//! Anchored against Ares `effects/decals.cpp` + dllcache:
//! - `s_fold`                                   — `effects/decals.cpp:395`
//! - `s_decal_projection_builder`               — `effects/decals.cpp:523`
//! - `c_decal::s_working_vertex`                — `effects/decals.cpp:192`
//! - `c_decal::s_fragment`                      — `effects/decals.cpp:201`
//! - `s_decal_mesh_builder`                     — `effects/decals.cpp:579`
//! - `s_decal_mesh_fragment_builder`            — `effects/decals.cpp:605`
//! - `s_decal_mesh_fragment_builder::s_neighbor_surface` — `effects/decals.cpp:611`
//! - `c_collision_surface_edge_iterator`        — `effects/decals.cpp:488`
//!
//! Pointer fields in the C++ types are mirrored as indices (typed
//! handles into the collision_bsp / per-decal seed arrays) — engine
//! stack-allocates the targets so the "pointer" is functionally an
//! offset into a known array. The Rust port chooses indices both for
//! safety and because the algorithm only ever needs the index back.

use blam_tags::math::RealMatrix4x3;
use blam_tags::math::{RealPlane3d, RealPoint2d, RealPoint3d, RealVector3d};

use super::writer::RasterizerVertexWorld;



/// `s_decal_mesh_builder::k_max_vertices`.
pub const K_MAX_WORK_VERTICES: usize = 1024;
/// `s_decal_mesh_builder::k_max_indices`.
pub const K_MAX_WORK_INDICES: usize = 1024;
/// `s_decal_mesh_fragment_builder::k_max_surfaces` — BFS queue cap.
pub const K_MAX_NEIGHBOR_SURFACES: usize = 128;

// =============================================================================
// `s_fold` — `effects/decals.cpp:395`, sizeof == 48
// =============================================================================

/// Describes a seed surface in the BFS — the "fold" between the
/// incoming projection direction and this surface's plane.
///
/// `bsp_index` / `instance_definition_index` identify which collision
/// BSP this surface lives in (main BSP when `instance_definition_index
/// == -1`, otherwise the instance geometry definition's own BSP).
/// `origin` is the entry point on the surface plane; `axis` is the
/// edge axis around which the projection folds; `normal` is the
/// surface plane normal.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Fold {
    /// Index into `scenario.structure_bsps[]`.
    pub bsp_index: i32,
    /// Index into the BSP's `instanced_geometry_definitions[]`, or
    /// `-1` for the main BSP.
    pub instance_definition_index: i32,
    /// Index into `collision_bsp.surfaces[]`.
    pub surface_index: i32,
    /// Surface plane normal at the fold point.
    pub normal: RealVector3d,
    /// World-space fold origin (point on the shared edge between this
    /// and the previous surface in the BFS).
    pub origin: RealPoint3d,
    /// Fold rotation axis (the shared edge direction; the projection
    /// rotates about this axis when stepping to this surface).
    pub axis: RealVector3d,
}

// =============================================================================
// `s_decal_projection_builder` — `effects/decals.cpp:523`, sizeof == 72
// =============================================================================

/// Flag bits on [`DecalProjectionBuilder::flags`]. Engine enum
/// `_*_bit` values from `effects/decals.cpp:525-531`.
pub mod flag {
    /// Bit 0 — projection matrix + angles have been populated.
    pub const INITIALIZED: u32 = 1 << 0;
    /// Bit 1 — `build_projection` has run for the current fold.
    pub const BUILT: u32 = 1 << 1;
    /// Bit 2 — projection has been folded (i.e. derived from a parent
    /// projection across a `Fold`).
    pub const FOLDED: u32 = 1 << 2;
    /// Bit 3 — left-handed coordinate frame.
    pub const LEFT_HANDED: u32 = 1 << 3;
    /// Bit 4 — projection basis needs renormalization before use
    /// (set after composition; cleared by the build step).
    pub const NEEDS_RENORMALIZE: u32 = 1 << 4;
}

/// Per-surface projection volume + clip-angle gates. The decal's
/// `cull_angle` is the hard exclusion cone (surfaces past this are
/// dropped); `clamp_angle` is the softer falloff threshold used by
/// the alpha shader.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DecalProjectionBuilder {
    /// Local-to-world (or parent-to-this) projection transform.
    pub projection: RealMatrix4x3,
    /// Authored cull angle in radians (from `c_decal_definition`).
    pub cull_angle_radians: f32,
    /// Precomputed `cosf(cull_angle_radians)`.
    pub cull_angle_cos: f32,
    /// Authored clamp angle in radians.
    pub clamp_angle_radians: f32,
    /// Precomputed `cosf(clamp_angle_radians)`.
    pub clamp_angle_cos: f32,
    /// Bit set of [`flag::*`].
    pub flags: u32,
}

// =============================================================================
// `c_decal::s_working_vertex` — `effects/decals.cpp:192`, sizeof == 40
// =============================================================================

/// One vertex in the per-decal work buffer. `position` is the engine's
/// `*const real_point3d` mirrored as an index into the owning
/// `collision_bsp.vertices[]` (the polygon ring is sourced from
/// collision surface vertices, not render-mesh vertices).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorkingVertex {
    /// Index into the producing `collision_bsp.vertices[]` (engine
    /// stores a pointer at offset 0; we mirror as an index). Ignored
    /// when `world_position_override` is `Some`.
    pub position: i32,
    /// 2D texcoord in projection-space, computed during clip.
    pub texcoord: RealPoint2d,
    /// Tangent-frame normal carried through `smooth_mesh_fragment`.
    pub normal: RealVector3d,
    /// Tangent-frame binormal (cross of normal × tangent at write).
    pub binormal: RealVector3d,
    /// `Some(world_pos)` for floating-quad fragments — engine
    /// `c_decal::build_floating_quad @ 0x18039C010` precomputes 4
    /// world-space vertex positions and writes them into
    /// `c_decal::m_floating_vertices[4]` (a `rasterizer_vertex_world[4]`
    /// runtime field). Protomorph keeps everything in the work buffer
    /// for uniform smoother+writer dispatch, so we mirror the
    /// "world-pos already known" state via this override. The writer
    /// uses it directly and skips both the `bsp.vertices[position]`
    /// lookup and any `instance_local_to_world` transform (since the
    /// quad emitter already applied both).
    pub world_position_override: Option<RealPoint3d>,
}

// =============================================================================
// `c_decal::s_fragment` — `effects/decals.cpp:201`, sizeof == 56
// =============================================================================

/// Per-fragment vertex/index interval pair — one for the work buffer
/// (pre-clip) and one for the output buffer (post-dedupe).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FragmentBufferIntervals {
    pub starting_vertex: u32,
    pub vertex_count: u32,
    pub starting_index: u32,
    pub index_count: u32,
}

/// Per-collision-seed mesh fragment. Engine fields are pointers; the
/// Rust port mirrors them as indices into per-decal-orchestration
/// arrays (matrix slot, collision seed slot).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecalFragment {
    /// Index of the local-to-world matrix in the per-decal matrix
    /// scratch (engine field `*const real_matrix4x3 m_local_to_world`).
    pub local_to_world: i32,
    /// Index into the per-decal collision-seed array (0..16)
    /// (engine field `*const s_decal_collision_result m_collision_result`).
    pub collision_result: i32,
    /// Per-`s_decal_definition` Z-bias gate; flipped on by the
    /// per-decal orchestrator for floating decals (see
    /// `c_decal::x_apply_floating_z_bias_always`).
    pub requires_floating_z_bias: bool,
    /// Pre-dedupe vertex/index range in [`DecalMeshBuilder`].
    pub working_intervals: FragmentBufferIntervals,
    /// Post-dedupe vertex/index range — written by sort + collapse.
    pub output_intervals: FragmentBufferIntervals,
    /// Engine field `*const real_matrix4x3 m_local_to_world` resolved to
    /// the bsp instance's matrix (`Some(M_inst)`) when the seed was an
    /// instanced-geometry hit. `None` for main-BSP seeds. The writer
    /// uses this to push instance-local BFS output vertices back to
    /// world space (`c_decal::build_tangent_frame @ 0x18039BD40`'s
    /// rotation block). Engine line: `decal_fragment->m_local_to_world
    /// = v11` at `c_decal::build_mesh_fragment @ 0x18039CCC0:189`.
    pub instance_local_to_world: Option<RealMatrix4x3>,
    /// Floating-quad output — engine `c_decal::m_floating_vertices[4]`.
    /// `Some` when `build_mesh_fragment` chose the floating-quad fast
    /// path (`can_use_quad && fragment_builder.can_be_quad`). In that
    /// case the BFS contribution to the working buffer is rewound and
    /// `working_intervals.{vertex,index}_count` are zero; the renderer
    /// drains this array as a 4-vert triangle strip. Engine call site:
    /// `c_decal::build_mesh_fragment @ 0x18039CCC0:226-239` invokes
    /// `build_floating_quad @ 0x18039C010` then rewinds the working
    /// cursors. Engine renders via `draw_primitive_up(triangle_strip,
    /// 2, m_floating_vertices, 0x2Cu)` in `c_decal::render @
    /// 0x18039B100`.
    pub floating_quad: Option<[RasterizerVertexWorld; 4]>,
}

// =============================================================================
// `s_decal_mesh_builder` — `effects/decals.cpp:579`, sizeof == 51240
// =============================================================================

/// Per-decal work + output buffers shared across all fragments of a
/// single decal. Engine stack-allocates this 51 KB struct; the Rust
/// port heap-allocates via `Box<DecalMeshBuilder>` to avoid stack
/// overflows in async runtimes.
///
/// Layout mirrors the engine; field order is preserved for grep-
/// against-Ares parity.
pub struct DecalMeshBuilder {
    /// Pre-clip work vertices. Filled by `build_mesh_fragment`,
    /// consumed by the sorter for dedupe.
    pub work_vertex_buffer: Box<[WorkingVertex; K_MAX_WORK_VERTICES]>,
    /// Pre-clip work indices — `unsigned short` per engine.
    pub work_index_buffer: Box<[u16; K_MAX_WORK_INDICES]>,
    /// Number of valid entries in `work_vertex_buffer`.
    pub working_vertex_count: u32,
    /// Number of valid entries in `work_index_buffer`.
    pub working_index_count: u32,
    /// Number of unique output vertices after sort + collapse.
    pub output_vertex_count: u32,
    /// Number of remapped output indices after collapse.
    pub output_index_count: u32,
    /// Sort-order array — engine's `c_decal_sorter::m_order`. Holds
    /// 16-bit indices into `work_vertex_buffer` arranged so that
    /// duplicate vertices land adjacent for `collapser` to merge.
    pub sorter_order: Box<[u16; K_MAX_WORK_VERTICES]>,
    /// Number of vertices currently registered in the sorter.
    pub sorter_count: u16,
    /// Sort start offset (engine `c_decal_sorter::m_start`).
    pub sorter_start: u16,
    /// Per-work-vertex remap: `output_index = collapser[input_index]`.
    pub collapser: Box<[u16; K_MAX_WORK_VERTICES]>,
    /// Per-output-vertex grouping intervals — `(start_run, run_count)`
    /// pairs after collapse.
    pub grouper: Box<[FragmentBufferIntervals; K_MAX_WORK_VERTICES]>,
}

impl DecalMeshBuilder {
    /// Allocate a fresh builder with zeroed buffers.
    pub fn new() -> Self {
        Self {
            work_vertex_buffer: Box::new([WorkingVertex::default(); K_MAX_WORK_VERTICES]),
            work_index_buffer: Box::new([0u16; K_MAX_WORK_INDICES]),
            working_vertex_count: 0,
            working_index_count: 0,
            output_vertex_count: 0,
            output_index_count: 0,
            sorter_order: Box::new([0u16; K_MAX_WORK_VERTICES]),
            sorter_count: 0,
            sorter_start: 0,
            collapser: Box::new([0u16; K_MAX_WORK_VERTICES]),
            grouper: Box::new([FragmentBufferIntervals::default(); K_MAX_WORK_VERTICES]),
        }
    }
}

impl Default for DecalMeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// `s_decal_mesh_fragment_builder::s_neighbor_surface` — sizeof == 124
// =============================================================================

/// One entry in the BFS queue. The parent index lets the algorithm
/// reconstruct the projection-fold chain for tangent-frame transport
/// across surface seams.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NeighborSurface {
    /// Index of the parent neighbor in
    /// [`DecalMeshFragmentBuilder::neighbor_surfaces`], or `-1` for
    /// the seed surface.
    pub parent_index: i32,
    /// Fold describing this surface relative to the parent.
    pub fold: Fold,
    /// Projection volume to clip this surface against.
    pub projection_builder: DecalProjectionBuilder,
}

// =============================================================================
// `s_decal_mesh_fragment_builder` — `effects/decals.cpp:605`, sizeof == 15884
// =============================================================================

/// Per-mesh-fragment BFS state. Holds up to 128 surfaces in a queue;
/// the recursive walker pushes neighbors as they pass the clip-cone
/// + cull-angle gates and pops in FIFO order to emit polygon
/// fragments.
pub struct DecalMeshFragmentBuilder {
    /// Originating instance-geometry index (`-1` for main BSP).
    /// Engine field 0x0.
    pub instanced_geometry_index: i32,
    /// Set when the entire fragment can be emitted as a single quad
    /// (planar fragment, no folds). Engine field 0x4.
    pub can_be_quad: bool,
    /// Current valid entries in `neighbor_surfaces`. Engine field 0x8.
    pub neighbor_surface_count: i32,
    /// BFS queue (FIFO). Engine field 0xC.
    pub neighbor_surfaces: Box<[NeighborSurface; K_MAX_NEIGHBOR_SURFACES]>,
}

impl DecalMeshFragmentBuilder {
    pub fn new() -> Self {
        Self {
            instanced_geometry_index: -1,
            can_be_quad: false,
            neighbor_surface_count: 0,
            neighbor_surfaces: Box::new(
                [NeighborSurface::default(); K_MAX_NEIGHBOR_SURFACES],
            ),
        }
    }
}

impl Default for DecalMeshFragmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// `c_collision_surface_edge_iterator` — `effects/decals.cpp:488`, sizeof == 64
// =============================================================================

/// Walks the edges of a collision surface to find opposing surfaces
/// for the BFS step. Engine's `get_opposing_surface_fold` returns the
/// neighbor's [`Fold`] when one shares the iterated edge.
///
/// Pointer fields in the engine type are mirrored as indices: the
/// collision BSP is identified by `(m_bsp_index, m_instance_definition_index)`
/// and looked up through the scenario-load loaded-BSP slot.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CollisionSurfaceEdgeIterator {
    /// Index into `scenario.structure_bsps[]` (engine field offset 0x0
    /// holds a `structure_bsp const*`; we mirror as the index).
    pub bsp_index: i32,
    /// Instance-definition index, or `-1` for the main BSP (engine
    /// field offset 0x8 holds a `collision_bsp const*`).
    pub instance_definition_index: i32,
    /// Surface index within the resolved collision BSP (engine offset
    /// 0x18).
    pub surface_index: i32,
    /// Current edge index in the surface's edge ring (engine offset
    /// 0x38).
    pub surface_edge_index: i32,
    /// Cached plane for the iterated surface (engine offset 0x28
    /// holds a `real_plane3d const`).
    pub plane: RealPlane3d,
}
