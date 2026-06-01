//! Mirror of `s_visibility_input` (Ares
//! `source/visibility/visibility_collection.h:172-188`, 200,304B).
//!
//! Per-frame scratch storage owned by `c_visibility_collection`.
//! Holds the [`SVisibilityRegion`] output along with per-collection
//! parameters (camera vs light vs shadow type), the per-frame
//! visible-cluster bitvectors, and the (region cluster index → BSP
//! cluster index) lookup populated during `prepare_collection_for_build`.

use blam_tags::math::RealPoint3d;

use crate::halo::structures::clusters::PackedClusterReference;
use crate::halo::visibility::SVisibilityRegion;

// =============================================================================
// `e_collection_type` / `e_collection_shape` / `e_visibility_collection_flags`
// — Ares `visibility_collection.h:14-43`
// =============================================================================

/// `e_collection_type` (visibility_collection.h:28-35). Picked by the
/// producer entry point — each of the 4 view types passes its own
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ECollectionType {
    Camera = 0,
    Light = 1,
    LightmapShadow = 2,
    CinematicShadow = 3,
}

impl Default for ECollectionType {
    fn default() -> Self {
        Self::Camera
    }
}

/// `e_collection_shape` (visibility_collection.h:37-43). Output of
/// `prepare_collection_for_build` — picks which region builder runs
/// (`setup_projections_region` / `setup_sphere_region` /
/// `visibility_build_region_from_pvs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ECollectionShape {
    Projections = 0,
    Sphere = 1,
    Pvs = 2,
}

impl Default for ECollectionShape {
    fn default() -> Self {
        Self::Projections
    }
}

/// `visibility_collection_flags` (visibility_collection.h:14-26). Bit
/// indices; engine ORs them into the `flags: long` field of
/// `s_visibility_input`.
pub mod visibility_collection_flags {
    pub const USING_REGION_INTERSECTION: u32 = 1 << 0;
    pub const RENDER_DEBUG: u32 = 1 << 1;
    pub const IGNORE_PORTAL_VOLUMES: u32 = 1 << 2;
    pub const IGNORE_LIGHTS: u32 = 1 << 3;
    pub const IGNORE_STRUCTURE: u32 = 1 << 4;
    pub const IGNORE_OBJECTS: u32 = 1 << 5;
    /// Bit 6 (= 0x40). Set by `render_visibility_camera_collection_compute`
    /// when `debug_pvs_render_all` is on; switches camera path from
    /// projections walker to PVS-only.
    pub const USE_PVS_VISIBILITY: u32 = 1 << 6;
    pub const FORCE_FRUSTUM_VISIBILITY: u32 = 1 << 7;
    /// Bit 8 (= 0x100). Set by `render_visibility_camera_collection_compute`
    /// when `single_cluster_only` is true (debug + special view modes).
    pub const SINGLE_CLUSTER_ONLY: u32 = 1 << 8;
}

// =============================================================================
// `s_visibility_input` (200,304B)
// =============================================================================

/// `s_visibility_input` (visibility_collection.h:172-188, 200,304B).
///
/// Owns the [`SVisibilityRegion`] (195KB) plus per-frame scratch:
/// camera cluster, user/window indices, flags, sphere shape input
/// (used when `collection_shape == Sphere`), the visible-cluster
/// bitvector accumulated by `collect_*_from_regions`, and the
/// per-region cluster-index lookup table built by
/// `prepare_collection_for_build`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SVisibilityInput {
    pub region: SVisibilityRegion,                          // 0x0      (195,648B)
    pub cluster_reference: PackedClusterReference,          // 0x2FC40  (2B)
    _pad0: [u8; 2],                                         // 0x2FC42  (align user_index to 0x2FC44)
    pub user_index: i32,                                    // 0x2FC44
    pub player_window_index: i32,                           // 0x2FC48
    pub flags: i32,                                         // 0x2FC4C
    pub projection_count: i16,                              // 0x2FC50
    _pad1: [u8; 2],                                         // 0x2FC52  (align sphere_center to 0x2FC54)
    pub sphere_center: RealPoint3d,                         // 0x2FC54  (12B)
    pub sphere_radius: f32,                                 // 0x2FC60
    pub collection_type: ECollectionType,                   // 0x2FC64  (4B)
    pub collection_shape: ECollectionShape,                 // 0x2FC68  (4B)
    pub visible_items_marker_index: i32,                    // 0x2FC6C
    /// `unsigned long visible_cluster_bitvector[16][8]` — per-BSP
    /// (16) × per-cluster-bit-block (8 u32 = 256 bits) bitvector
    /// accumulated as collect_* iterates the region.
    pub visible_cluster_bitvector: [[u32; 8]; 16],          // 0x2FC70  (512B → ends 0x2FE70)
    /// `cluster_to_visibility_cluster_lookup[16][256]` — per-BSP
    /// (16) × per-cluster-index (256) lookup mapping BSP cluster
    /// index → region cluster index (or -1 if not in region).
    pub cluster_to_visibility_cluster_lookup: [[i8; 256]; 16], // 0x2FE70 (4,096B → ends 0x30E70)
}

const _: () = assert!(std::mem::size_of::<SVisibilityInput>() == 200_304);

impl SVisibilityInput {
    /// Heap-allocated zeroed input — 200KB stack pressure makes
    /// boxing the right default.
    pub fn new_zeroed() -> Box<Self> {
        Box::new(Self {
            region: SVisibilityRegion::default(),
            cluster_reference: PackedClusterReference::NONE,
            _pad0: [0; 2],
            user_index: -1,
            player_window_index: -1,
            flags: 0,
            projection_count: 0,
            _pad1: [0; 2],
            sphere_center: RealPoint3d::default(),
            sphere_radius: 0.0,
            collection_type: ECollectionType::Camera,
            collection_shape: ECollectionShape::Projections,
            visible_items_marker_index: -1,
            visible_cluster_bitvector: [[0; 8]; 16],
            cluster_to_visibility_cluster_lookup: [[-1; 256]; 16],
        })
    }
}
