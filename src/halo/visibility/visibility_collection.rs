//! Mirror of `Ares/source/visibility/visibility_collection.h`.
//!
//! `c_visibility_collection` orchestrator + its supporting global
//! storage `c_visible_items::m_items` (157,112B) — the per-frame
//! "what to draw" produced by each of the 4 view types
//! (camera/lights/lightmap_shadows/occlusion).
//!
//! Phase A2 lands the storage shapes only — `c_simple_list<T,K>`
//! template + the `s_visible_items` aggregate. The class itself
//! (compute, prepare_collection_for_build, collect_*) ships in
//! Phase F.

use crate::halo::visibility::visibility_collection_structure::{
    SVisibleClusters, SVisibleInstanceList, SVisibleInstances,
};
use crate::halo::visibility::{VisibleObjectHierarchy, VisibleObjectRenderVisibility};

// =============================================================================
// Constants — Ares `visibility_collection.h:54-59`
// =============================================================================

/// Per-object subpart bitvector capacity (16,384 bits = 2,048 bytes
/// = 512 u32 words). One contiguous arena shared across all visible
/// objects this frame; entries point at slices of it.
pub const K_SUBPART_BITVECTOR_SIZE_IN_BITS: usize = 16384;
pub const K_SUBPART_BITVECTOR_SIZE_IN_U32: usize = K_SUBPART_BITVECTOR_SIZE_IN_BITS / 32;
pub const K_MAXIMUM_REGION_MEMORY_ITEMS: usize = 512;
pub const K_MAXIMUM_ITEM_MARKERS: usize = 6;
pub const K_MAXIMUM_SKY_ITEMS: usize = 6;

// =============================================================================
// `c_simple_list<T, K>` — Ares `visibility_collection.h:73-90`
//
// Engine layout:
//     unsigned short m_maximum_count; // 0x0
//     unsigned short m_count;         // 0x2
//     T data[k_count];                // 0x4
//
// We mirror with const generics. Capacity is set at construction time
// from the template parameter (engine writes `m_maximum_count` in
// each ctor).
// =============================================================================

/// `c_simple_list<T, k_count>` (visibility_collection.h:73-90).
///
/// Fixed-capacity vec-like container. Header is 4B (max + count u16
/// each); data follows inline. Append-only (`add()` returns the new
/// element index or -1 on overflow); `clear()` resets `m_count` to 0.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CSimpleList<T: Copy + Default, const K: usize> {
    pub m_maximum_count: u16, // 0x0
    pub m_count: u16,         // 0x2
    pub data: [T; K],         // 0x4
}

impl<T: Copy + Default, const K: usize> CSimpleList<T, K> {
    /// `c_simple_list<T,K>::c_simple_list()` — sets max from template
    /// arg, count to zero.
    pub fn new() -> Self {
        Self {
            m_maximum_count: K as u16,
            m_count: 0,
            data: [T::default(); K],
        }
    }

    /// `c_simple_list::add @ 0x18050FC90/etc` — returns the new index
    /// or -1 on overflow.
    pub fn add(&mut self) -> i32 {
        if self.m_count >= self.m_maximum_count {
            return -1;
        }
        let idx = self.m_count as i32;
        self.m_count += 1;
        idx
    }

    /// `c_simple_list::clear @ 0x180393280/etc`.
    pub fn clear(&mut self) {
        self.m_count = 0;
    }

    pub fn get_count(&self) -> i32 {
        self.m_count as i32
    }

    pub fn set_count(&mut self, count: u16) {
        // Engine asserts `count <= m_maximum_count` — match.
        debug_assert!(count <= self.m_maximum_count, "set_count overflow");
        self.m_count = count.min(self.m_maximum_count);
    }

    pub fn valid(&self, index: i32) -> bool {
        index >= 0 && (index as u16) < self.m_count
    }
}

impl<T: Copy + Default, const K: usize> Default for CSimpleList<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const K: usize> std::ops::Index<i32> for CSimpleList<T, K> {
    type Output = T;
    fn index(&self, index: i32) -> &T {
        debug_assert!(self.valid(index), "CSimpleList index OOB");
        &self.data[index as usize]
    }
}

impl<T: Copy + Default, const K: usize> std::ops::IndexMut<i32> for CSimpleList<T, K> {
    fn index_mut(&mut self, index: i32) -> &mut T {
        debug_assert!(self.valid(index), "CSimpleList index OOB");
        &mut self.data[index as usize]
    }
}

// Concrete instantiation size checks (from Ares `visibility_collection.h`)
const _: () = assert!(std::mem::size_of::<CSimpleList<i32, 50>>() == 204);
const _: () = assert!(std::mem::size_of::<CSimpleList<i32, 6>>() == 28);

// =============================================================================
// `s_visible_items` aggregate — Ares `visibility_collection.h:114-127`,
// 157,112B
// =============================================================================

/// `s_visible_items` (157,112B). The per-frame visibility output the
/// 4 submitters consume. Sub-lists:
///
/// - `root_objects` (768) — one row per root object (parent of an
///   attachment hierarchy)
/// - `objects` (896) — one row per visible object instance
///   (root + children expanded)
/// - `instance_list` (1024) — per-cluster groupings of BSP instanced
///   geometry instances
/// - `instances` (1152) — individual BSP instanced-geometry instances
/// - `clusters` (348) — one row per visible BSP cluster
/// - `lights` (50) — visible dynamic light indices
/// - `skies` (6) — visible sky placeholders
/// - `visible_subpart_bitvector[512]` — shared u32 arena holding
///   per-object/per-cluster/per-instance subpart visibility bits;
///   pointer fields in the rows above slice into this arena
/// - `visibility_region_cluster_bitvector[512][4]` — per-region cluster
///   visibility bitvectors (4 u32 = 128 cluster bits per region; up to
///   512 regions tracked across the frame)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SVisibleItems {
    pub root_objects: CSimpleList<VisibleObjectHierarchy, 768>, // 0x0     (12,292B)
    pub objects: CSimpleList<VisibleObjectRenderVisibility, 896>, // 0x3008  (86,020B)
    pub instance_list: CSimpleList<SVisibleInstanceList, 1024>, // 0x18010 (12,292B)
    pub instances: CSimpleList<SVisibleInstances, 1152>,        // 0x1B018 (27,652B)
    pub clusters: CSimpleList<SVisibleClusters, 348>,           // 0x21C20 (8,356B)
    pub lights: CSimpleList<i32, 50>,                           // 0x23CC8 (204B)
    pub skies: CSimpleList<i32, K_MAXIMUM_SKY_ITEMS>,           // 0x23D94 (28B)
    pub visible_subpart_bitvector: [u32; 512],                  // 0x23DB0 (2,048B)
    pub visible_subpart_bitvector_count: u16,                   // 0x245B0
    pub visiblity_region_cluster_bitvector_count: u16,          // 0x245B2
    pub visibility_region_cluster_bitvector: [[u32; 4]; 512],   // 0x245B4 (8,192B)
}

const _: () = assert!(std::mem::size_of::<SVisibleItems>() == 157_112);

impl SVisibleItems {
    pub fn new_zeroed() -> Box<Self> {
        // 157KB — heap-box to avoid stack-overflow risk.
        Box::new(Self {
            root_objects: CSimpleList::new(),
            objects: CSimpleList::new(),
            instance_list: CSimpleList::new(),
            instances: CSimpleList::new(),
            clusters: CSimpleList::new(),
            lights: CSimpleList::new(),
            skies: CSimpleList::new(),
            visible_subpart_bitvector: [0; 512],
            visible_subpart_bitvector_count: 0,
            visiblity_region_cluster_bitvector_count: 0,
            visibility_region_cluster_bitvector: [[0; 4]; 512],
        })
    }

    /// `c_visible_items::clear @ 0x????????` — resets per-list
    /// counters + bitvector counts. Storage is not zeroed (engine
    /// behavior — entries get overwritten on next add).
    pub fn clear(&mut self) {
        self.root_objects.clear();
        self.objects.clear();
        self.instance_list.clear();
        self.instances.clear();
        self.clusters.clear();
        self.lights.clear();
        self.skies.clear();
        self.visible_subpart_bitvector_count = 0;
        self.visiblity_region_cluster_bitvector_count = 0;
    }
}

// =============================================================================
// `c_visible_items` static container — Ares `visibility_collection.h:130-169`
// =============================================================================

/// `c_visible_items::s_marker_indices` (visibility_collection.h:155-167,
/// 18B). Snapshot of per-list counts at marker push time, used to
/// rewind during `pop_marker`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CVisibleItemsMarkerIndices {
    pub root_objects_count: u16,                    // 0x0
    pub objects_count: u16,                         // 0x2
    pub instance_list_count: u16,                   // 0x4
    pub instances_count: u16,                       // 0x6
    pub cluster_count: u16,                         // 0x8
    pub light_count: u16,                           // 0xA
    pub visible_subpart_bitvector_count: u16,       // 0xC
    pub visiblity_region_cluster_bitvector_count: u16, // 0xE
    pub sky_count: u16,                             // 0x10
}

const _: () = assert!(std::mem::size_of::<CVisibleItemsMarkerIndices>() == 18);

/// `c_visible_items` (visibility_collection.h:130-169). All fields
/// static in the engine — `m_items` is the global storage,
/// `m_marker_count` + `m_markers[6]` track nested push/pop.
///
/// We mirror as a struct holding the same statics; access is via
/// [`global_visible_items()`] / [`global_visible_items_mut()`].
pub struct CVisibleItems {
    pub items: Box<SVisibleItems>,
    pub marker_count: i32,
    pub markers: [CVisibleItemsMarkerIndices; K_MAXIMUM_ITEM_MARKERS],
}

impl CVisibleItems {
    pub fn new() -> Self {
        Self {
            items: SVisibleItems::new_zeroed(),
            marker_count: 0,
            markers: [CVisibleItemsMarkerIndices::default(); K_MAXIMUM_ITEM_MARKERS],
        }
    }

    /// `c_visible_items::push_marker @ 0x????????`. Snapshot per-list
    /// counts so a later `pop_marker()` rewinds them.
    pub fn push_marker(&mut self) {
        debug_assert!(
            (self.marker_count as usize) < K_MAXIMUM_ITEM_MARKERS,
            "marker stack overflow"
        );
        let m = &mut self.markers[self.marker_count as usize];
        m.root_objects_count = self.items.root_objects.m_count;
        m.objects_count = self.items.objects.m_count;
        m.instance_list_count = self.items.instance_list.m_count;
        m.instances_count = self.items.instances.m_count;
        m.cluster_count = self.items.clusters.m_count;
        m.light_count = self.items.lights.m_count;
        m.sky_count = self.items.skies.m_count;
        m.visible_subpart_bitvector_count = self.items.visible_subpart_bitvector_count;
        m.visiblity_region_cluster_bitvector_count =
            self.items.visiblity_region_cluster_bitvector_count;
        self.marker_count += 1;
    }

    /// `c_visible_items::pop_marker @ 0x????????`. Rewind to the
    /// most-recent push.
    pub fn pop_marker(&mut self) {
        debug_assert!(self.marker_count > 0, "marker stack underflow");
        self.marker_count -= 1;
        let m = &self.markers[self.marker_count as usize];
        self.items.root_objects.set_count(m.root_objects_count);
        self.items.objects.set_count(m.objects_count);
        self.items.instance_list.set_count(m.instance_list_count);
        self.items.instances.set_count(m.instances_count);
        self.items.clusters.set_count(m.cluster_count);
        self.items.lights.set_count(m.light_count);
        self.items.skies.set_count(m.sky_count);
        self.items.visible_subpart_bitvector_count = m.visible_subpart_bitvector_count;
        self.items.visiblity_region_cluster_bitvector_count =
            m.visiblity_region_cluster_bitvector_count;
    }

    /// `c_visible_items::clear_all @ 0x180391D00` — resets everything,
    /// including marker stack.
    pub fn clear_all(&mut self) {
        self.items.clear();
        self.marker_count = 0;
    }

    /// `c_visible_items::clear` — resets list contents but NOT the
    /// marker stack. Used between collections within the same marker
    /// scope.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn marker_index(&self) -> i32 {
        self.marker_count
    }
}

impl Default for CVisibleItems {
    fn default() -> Self {
        Self::new()
    }
}
