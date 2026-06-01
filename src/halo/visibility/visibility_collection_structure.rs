//! Mirror of `Ares/source/visibility/visibility_collection_structure.h`.
//!
//! Per-frame structure-side visible-items rows: clusters, instance
//! lists (groups of instanced-geometry instances per cluster), and
//! individual instances. Filled by `c_visibility_collection::collect_structure_elements_from_regions`,
//! consumed by `c_structure_renderer::submit_visibility`.

use crate::halo::structures::clusters::PackedClusterReference;

/// `s_zone_cluster` (visibility_collection_structure.h:13-17, 4B).
/// Cluster reference + the zone (designer-zone-set) it belongs to.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SZoneCluster {
    pub cluster_reference: PackedClusterReference, // 0x0  (2B)
    pub zone_index: i16,                           // 0x2
}

const _: () = assert!(std::mem::size_of::<SZoneCluster>() == 4);

/// `s_visible_clusters` (visibility_collection_structure.h:20-27, 24B).
/// One row per visible BSP cluster. `part_bitvector` points into the
/// shared `c_visible_items::m_items.visible_subpart_bitvector` (one
/// bit per subpart of the cluster's mesh — `1` = part is visible from
/// the camera and should be drawn).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SVisibleClusters {
    pub flags: i16,                  // 0x0
    pub cluster: SZoneCluster,       // 0x2  (4B)
    pub region_cluster_index: u16,   // 0x6
    pub mesh_index: u16,             // 0x8
    _pad0: [u8; 6],                  // 0xA — align ptr to 0x10
    pub part_bitvector: u64,         // 0x10 — opaque pointer slot
}

const _: () = assert!(std::mem::size_of::<SVisibleClusters>() == 24);

/// `s_visible_instance_list` (visibility_collection_structure.h:30-37,
/// 12B). Groups visible instanced-geometry instances by cluster +
/// mesh_index. Engine-side iteration: per cluster, walk the
/// `instance_list` rows that belong to it; each row points at a
/// contiguous span of `s_visible_instances` rows (via
/// `first_visible_instance_list_instance_index` + count).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SVisibleInstanceList {
    pub cluster: SZoneCluster,                                // 0x0  (4B)
    pub instance_list_index: u16,                             // 0x4
    pub first_visible_instance_list_instance_index: u16,      // 0x6
    pub visible_instance_list_count: u16,                     // 0x8
    pub mesh_index: u16,                                      // 0xA
}

const _: () = assert!(std::mem::size_of::<SVisibleInstanceList>() == 12);

/// `s_visible_instances` (visibility_collection_structure.h:40-48, 24B).
/// One row per visible instanced-geometry instance. `part_bitvector`
/// is the per-subpart visibility bitset (same scheme as
/// `s_visible_clusters`).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SVisibleInstances {
    pub flags: i16,                                  // 0x0
    pub structure_bsp_instance_index: u16,           // 0x2
    pub structure_bsp_index: i16,                    // 0x4
    pub region_cluster_bitvector_start_index: i16,   // 0x6
    pub alpha_byte: u8,                              // 0x8
    _pad0: [u8; 7],                                  // 0x9 — align ptr to 0x10
    pub part_bitvector: u64,                         // 0x10 — opaque pointer slot
}

const _: () = assert!(std::mem::size_of::<SVisibleInstances>() == 24);
