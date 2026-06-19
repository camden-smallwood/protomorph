//! Mirror of `Ares/source/visibility/visibility.h` — core types of
//! the visibility subsystem.
//!
//! These are pure runtime types, never serialized. Layout matches the
//! engine 1:1 so future cross-references against IDA/dllcache stay
//! cheap. `static_assert(sizeof(T) == N)` from Ares is reproduced as
//! `const _: () = assert!(size_of::<T>() == N);` — a build break here
//! means the layout drifted.
//!
//! See [`crate::halo::visibility`] for the consumer-side overview.

use blam_tags::math::{RealMatrix4x3, RealPlane3d, RealPoint2d, RealPoint3d, RealVector3d};

// Re-export the math types pulled from blam_tags::math so existing callers
// importing them from `crate::halo::visibility` keep compiling. Per engine
// source layout, all four (`RealMatrix4x3`, `RealRectangle2d`,
// `RealRectangle3d`, plus the basic point/vector types) live in
// `Ares/source/math/real_math.h`; we host them in `blam_tags::math` for
// cross-crate sharing. `RealMatrix4x3` is re-exported below alongside its
// doc-link comment.
pub use blam_tags::math::{RealRectangle2d, RealRectangle3d};

use crate::halo::structures::clusters::PackedClusterReference;

// =============================================================================
// Constants — Ares `visibility_constants.h` and inline `visibility.h` enums
// =============================================================================

pub const MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION: usize = 6;
pub const MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION: usize = 128;
pub const MAXIMUM_VOLUMES_PER_VISIBILITY_REGION: usize = 512;

// =============================================================================
// Visibility primitives — Ares `visibility.h:17-87`
// =============================================================================

/// `visibility_volume` (visibility.h:17-32, 368B).
///
/// One frustum-clipped 3D region — six SSE planes for fast
/// box/sphere/triangle tests, plus a duplicate set of world-space
/// planes/edges for portal-poly clipping in the `from_projections`
/// walker.
///
/// `vector_planes` packs each plane as `(plane.n.i, plane.n.j,
/// plane.n.k, plane.d)` — engine reads via `__m128`, we mirror as
/// `[f32; 4]` (4-byte aligned; SIMD intrinsics are not required for
/// engine fidelity since the math is the same scalar dot-product).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VisibilityVolume {
    pub vector_planes: [[f32; 4]; 6],         // 0x0   (96B)
    pub projection_index: i16,                // 0x60
    pub region_cluster_index: i16,            // 0x62
    pub plane_index: i32,                     // 0x64
    pub plane_bsp_index: i16,                 // 0x68
    pub parent_volume_index: i16,             // 0x6A
    pub frustum_bounds: RealRectangle2d,      // 0x6C   (16B)
    pub portal_nearest_z: f32,                // 0x7C
    pub world_vertices: [RealPoint3d; 5],     // 0x80   (60B)
    pub world_bounds: RealRectangle3d,        // 0xBC   (24B)
    pub world_planes: [RealPlane3d; 6],       // 0xD4   (96B)
    pub world_edge_vectors: [RealVector3d; 4], // 0x134  (48B)
    _pad0: [u8; 12],                          // 0x164
}

const _: () = assert!(std::mem::size_of::<VisibilityVolume>() == 368);

impl Default for VisibilityVolume {
    fn default() -> Self {
        Self {
            vector_planes: [[0.0; 4]; 6],
            projection_index: 0,
            region_cluster_index: 0,
            plane_index: 0,
            plane_bsp_index: 0,
            parent_volume_index: 0,
            frustum_bounds: RealRectangle2d::default(),
            portal_nearest_z: 0.0,
            world_vertices: [RealPoint3d::default(); 5],
            world_bounds: RealRectangle3d::default(),
            world_planes: [RealPlane3d::default(); 6],
            world_edge_vectors: [RealVector3d::default(); 4],
            _pad0: [0; 12],
        }
    }
}

/// `visibility_projection` (visibility.h:35-56, 560B).
///
/// One camera-like projection — basis transforms + near/far + the
/// volume that defines its viewable region. Carries `volume` directly
/// (the seed volume for the from_projections walker; copied into
/// `s_visibility_region.volumes[0]` per cluster). Multiple projections
/// per region support cube-map shadow setups (up to 6 faces).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VisibilityProjection {
    pub parallel_projection_flag: bool,       // 0x0
    _pad0: [u8; 3],                           // 0x1
    pub world_to_basis: RealMatrix4x3,        // 0x4   (52B)
    pub basis_to_world: RealMatrix4x3,        // 0x38  (52B → ends at 0x6C)
    pub near_bounded_flag: bool,              // 0x6C
    _pad1: [u8; 3],                           // 0x6D
    pub near_distance: f32,                   // 0x70
    pub near_plane: RealPlane3d,              // 0x74  (16B → ends at 0x84)
    pub far_bounded_flag: bool,               // 0x84
    pub far_bounded_spherical_flag: bool,     // 0x85
    _pad2: [u8; 2],                           // 0x86
    pub far_distance: f32,                    // 0x88
    pub volume_bounded_flag: bool,            // 0x8C
    _pad3: [u8; 3],                           // 0x8D
    pub volume: VisibilityVolume,             // 0x90  (368B → ends at 0x200)
    pub convex_hull_point_count: i32,         // 0x200
    pub convex_hull_points: [RealPoint2d; 4], // 0x204 (32B → ends at 0x224)
    _pad4: [u8; 12],                          // 0x224
}

const _: () = assert!(std::mem::size_of::<VisibilityProjection>() == 560);

impl Default for VisibilityProjection {
    fn default() -> Self {
        Self {
            parallel_projection_flag: false,
            _pad0: [0; 3],
            world_to_basis: RealMatrix4x3::default(),
            basis_to_world: RealMatrix4x3::default(),
            near_bounded_flag: false,
            _pad1: [0; 3],
            near_distance: 0.0,
            near_plane: RealPlane3d::default(),
            far_bounded_flag: false,
            far_bounded_spherical_flag: false,
            _pad2: [0; 2],
            far_distance: 0.0,
            volume_bounded_flag: false,
            _pad3: [0; 3],
            volume: VisibilityVolume::default(),
            convex_hull_point_count: 0,
            convex_hull_points: [RealPoint2d::default(); 4],
            _pad4: [0; 12],
        }
    }
}

/// `visibility_cluster` (visibility.h:58-64, 26B). One cluster slot
/// inside `s_visibility_region.clusters[]` — the per-projection volume
/// counts + first-volume-indices link the cluster to its
/// portal-clipped volumes in `volumes[]`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VisibilityCluster {
    pub cluster_reference: PackedClusterReference, // 0x0  (2B)
    pub volume_counts: [i16; 6],                   // 0x2  (12B)
    pub first_volume_indices: [i16; 6],            // 0xE  (12B)
}

const _: () = assert!(std::mem::size_of::<VisibilityCluster>() == 26);

/// `visibility_volume_intersection` (visibility.h:66-74, 156B). The
/// 3D intersection of two volumes (typically light × camera region) —
/// used by `c_visibility_collection::intersect_light_region_with_camera_region`
/// to crop dynamic-light influence to the camera's PVS.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VisibilityVolumeIntersection {
    pub intersection_bounds: RealRectangle3d,        // 0x0  (24B)
    pub lit_bounds: RealRectangle3d,                 // 0x18 (24B)
    pub project_points_convex_hull: [RealPoint2d; 12], // 0x30 (96B)
    pub projected_points_convex_hull_count: i16,     // 0x90
    _pad0: [u8; 2],                                  // 0x92 (alignment to next i32)
    pub light_volume_index: i32,                     // 0x94
    pub light_projection_index: i32,                 // 0x98
}

const _: () = assert!(std::mem::size_of::<VisibilityVolumeIntersection>() == 156);

impl Default for VisibilityVolumeIntersection {
    fn default() -> Self {
        Self {
            intersection_bounds: RealRectangle3d::default(),
            lit_bounds: RealRectangle3d::default(),
            project_points_convex_hull: [RealPoint2d::default(); 12],
            projected_points_convex_hull_count: 0,
            _pad0: [0; 2],
            light_volume_index: 0,
            light_projection_index: 0,
        }
    }
}

// =============================================================================
// `s_visibility_region` (visibility.h:77-87, 195_648B) — top-level
// =============================================================================

/// `s_visibility_region` (195,648B).
///
/// The output of one `c_visibility_collection::compute()` call —
/// projection list + cluster list + per-projection-per-cluster
/// volumes. Lives inside [`SVisibilityInput`] (Phase A3); per-frame
/// scratch ~196KB so the engine stack-allocates per collection (4
/// active collections per frame: camera + lightmap_shadows + lights ×
/// per-light + occlusion).
///
/// The runtime layout has projection_count at 0x0 followed directly by
/// `projections[6]` at 0x10 (16-byte alignment for SIMD copy). We
/// reproduce the gap explicitly via padding for engine-exact size.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SVisibilityRegion {
    pub projection_count: i16,                                                          // 0x0
    _pad0: [u8; 14],                                                                    // 0x2  (align projections to 0x10)
    pub projections: [VisibilityProjection; MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION], // 0x10 (3360B → ends at 0xD30)
    pub cluster_count: i16,                                                             // 0xD30
    pub clusters: [VisibilityCluster; MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION],          // 0xD32 (3328B → ends at 0x1A32)
    _pad1: [u8; 2],                                                                     // 0x1A32 (align cluster_remap_table to 0x1A34)
    pub cluster_remap_table: [i32; MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION],             // 0x1A34 (512B → ends at 0x1C34)
    pub volume_count: i16,                                                              // 0x1C34
    _pad2: [u8; 10],                                                                    // 0x1C36 (align volumes to 0x1C40)
    pub volumes: [VisibilityVolume; MAXIMUM_VOLUMES_PER_VISIBILITY_REGION],             // 0x1C40 (188_416B → ends at 0x2FC40)
}

const _: () = assert!(std::mem::size_of::<SVisibilityRegion>() == 195_648);

impl Default for SVisibilityRegion {
    fn default() -> Self {
        Self {
            projection_count: 0,
            _pad0: [0; 14],
            projections: [VisibilityProjection::default(); MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION],
            cluster_count: 0,
            clusters: [VisibilityCluster::default(); MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION],
            _pad1: [0; 2],
            cluster_remap_table: [0; MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION],
            volume_count: 0,
            _pad2: [0; 10],
            volumes: [VisibilityVolume::default(); MAXIMUM_VOLUMES_PER_VISIBILITY_REGION],
        }
    }
}

