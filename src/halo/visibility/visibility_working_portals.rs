//! Mirror of the working-portal container types from
//! `Ares/source/visibility/visibility_portal_traversal.cpp:83-146`.
//!
//! - [`SWorkingPortal`] (56B in engine) — one BFS-frontier portal:
//!   parent + portal_reference + cluster_reference + hull + bookkeeping.
//! - [`SWorkingPortalStack`] (55328B in engine) — append-only stack
//!   of working portals (max 256), with a shared 2D-point allocator
//!   for the hulls.
//! - [`SClusterWorkingPortals`] (9216B in engine) — per-(bsp,cluster)
//!   linked list of working portals targeting that cluster (head +
//!   per-portal next-index chain).
//! - [`SStaticIndexQueue`] (520B in engine) — circular FIFO of working
//!   portal indices, drained by the BFS in
//!   [`super::visibility_portal_traversal`] (Phase E4).
//!
//! Engine sizes documented but **not** asserted — our containers use
//! `Vec<…>` per `PortalHull` (vs the engine's shared
//! `c_static_aligned_stack_allocator` for points), so the bytes
//! diverge intentionally. Counts + indices are 1:1.

use crate::halo::structures::clusters::PackedClusterReference;
use crate::halo::scenario::{CLUSTER_BIT_WORDS_PER_BSP, MAX_BSPS, MAX_CLUSTERS_PER_BSP};
use crate::halo::visibility::visibility_portal_activation::PortalReference;
use crate::halo::visibility::visibility_portal_hulls::{
    PortalHull, MAXIMUM_WORKING_PORTALS,
};

// =============================================================================
// `s_working_portal` (56B)
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct SWorkingPortal {
    /// Index into [`SWorkingPortalStack::portals`] of the working
    /// portal that produced this one (parent in the BFS tree). `-1`
    /// for the seed portal.
    pub parent_working_portal_index: i16,
    pub portal_reference: PortalReference,
    pub cluster_reference: PackedClusterReference,
    /// Index into `s_visibility_region.volumes[]` once
    /// [`super::visibility_portal_traversal`]'s region builder
    /// (Phase E5) has emitted this portal's clipped volume. `-1`
    /// before emission.
    pub created_volume_index: i16,
    /// `true` once the portal has been visited / processed; engine
    /// uses this to suppress traversal back through the same portal.
    pub do_not_traverse: bool,
    pub hull: PortalHull,
    pub nearest_z: f32,
    /// Index of the working portal that "killed" this one (set when
    /// a sibling at the same cluster fully encloses this hull;
    /// engine sentinel `-1` = alive).
    pub killed_by_index: i32,
}

// =============================================================================
// `s_working_portal_stack` — append-only stack of working portals
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct SWorkingPortalStack {
    pub portals: Vec<SWorkingPortal>,
}

impl SWorkingPortalStack {
    pub fn new() -> Self {
        Self { portals: Vec::with_capacity(MAXIMUM_WORKING_PORTALS) }
    }

    pub fn clear(&mut self) {
        self.portals.clear();
    }

    /// Engine: `c_static_stack<s_working_portal,256>::push`. Returns
    /// the new slot index, or `None` if at capacity.
    pub fn push_default(&mut self) -> Option<i16> {
        if self.portals.len() >= MAXIMUM_WORKING_PORTALS {
            return None;
        }
        let i = self.portals.len() as i16;
        self.portals.push(SWorkingPortal::default());
        Some(i)
    }

    pub fn get(&self, idx: i16) -> Option<&SWorkingPortal> {
        if idx < 0 {
            return None;
        }
        self.portals.get(idx as usize)
    }

    pub fn get_mut(&mut self, idx: i16) -> Option<&mut SWorkingPortal> {
        if idx < 0 {
            return None;
        }
        self.portals.get_mut(idx as usize)
    }

    pub fn count(&self) -> usize {
        self.portals.len()
    }
}

// =============================================================================
// `s_cluster_working_portals` (9216B) — per-(bsp,cluster) → list of
// working portals targeting that cluster
//
// Engine layout (renamed for clarity):
//   first_working_portal_indices_flags  [16 BSPs][8 u32 = 256 cluster bits]
//   first_working_portal_indices        [16 BSPs][256 i16 = head per cluster]
//   next_working_portal_indices         [256 i16 = next chain per portal]
//
// `insert(cluster, wp_idx)` chains wp_idx onto cluster's list:
//   if !valid[bsp][cluster]: head = wp_idx; valid bit set
//   else: next[wp_idx] = head; head = wp_idx
// =============================================================================

#[derive(Debug, Clone)]
pub struct SClusterWorkingPortals {
    pub first_indices_flags: Box<[[u32; CLUSTER_BIT_WORDS_PER_BSP]; MAX_BSPS]>,
    pub first_indices: Box<[[i16; MAX_CLUSTERS_PER_BSP]; MAX_BSPS]>,
    pub next_indices: Box<[i16; MAXIMUM_WORKING_PORTALS]>,
}

impl SClusterWorkingPortals {
    pub fn new() -> Self {
        Self {
            first_indices_flags: Box::new([[0; CLUSTER_BIT_WORDS_PER_BSP]; MAX_BSPS]),
            first_indices: Box::new([[-1; MAX_CLUSTERS_PER_BSP]; MAX_BSPS]),
            next_indices: Box::new([-1; MAXIMUM_WORKING_PORTALS]),
        }
    }

    pub fn clear(&mut self) {
        for bsp in self.first_indices_flags.iter_mut() {
            *bsp = [0; CLUSTER_BIT_WORDS_PER_BSP];
        }
    }

    /// `insert(cluster, working_portal_index)` (Ares 0x180509A50).
    /// Push `working_portal_index` onto `cluster`'s linked list.
    pub fn insert(&mut self, cluster: PackedClusterReference, working_portal_index: i16) {
        let bsp = cluster.bsp_index;
        let c = cluster.cluster_index;
        if bsp < 0 || (bsp as usize) >= MAX_BSPS || (c as usize) >= MAX_CLUSTERS_PER_BSP {
            return;
        }
        let bsp = bsp as usize;
        let c = c as usize;
        if working_portal_index < 0
            || (working_portal_index as usize) >= MAXIMUM_WORKING_PORTALS
        {
            return;
        }
        let valid_word = self.first_indices_flags[bsp][c >> 5];
        if (valid_word >> (c & 31)) & 1 == 0 {
            // First entry for this cluster.
            self.first_indices[bsp][c] = working_portal_index;
            self.next_indices[working_portal_index as usize] = -1;
            self.first_indices_flags[bsp][c >> 5] |= 1u32 << (c & 31);
        } else {
            // Push onto front of chain.
            let prev_head = self.first_indices[bsp][c];
            self.next_indices[working_portal_index as usize] = prev_head;
            self.first_indices[bsp][c] = working_portal_index;
        }
    }

    /// `get_first(cluster)` (Ares 0x180509C30). Returns the head
    /// working_portal_index for `cluster`, or `-1` if none.
    pub fn get_first(&self, cluster: PackedClusterReference) -> i16 {
        let bsp = cluster.bsp_index;
        let c = cluster.cluster_index;
        if bsp < 0 || (bsp as usize) >= MAX_BSPS || (c as usize) >= MAX_CLUSTERS_PER_BSP {
            return -1;
        }
        let bsp = bsp as usize;
        let c = c as usize;
        let valid_word = self.first_indices_flags[bsp][c >> 5];
        if (valid_word >> (c & 31)) & 1 == 0 {
            -1
        } else {
            self.first_indices[bsp][c]
        }
    }

    /// `get_next(cluster, working_portal_index)` (Ares 0x180509D00).
    /// Returns the next working_portal_index in the chain, or `-1`
    /// at the end. Engine ignores `cluster` — chain is encoded purely
    /// in `next_indices`.
    pub fn get_next(
        &self,
        _cluster: PackedClusterReference,
        working_portal_index: i16,
    ) -> i16 {
        if working_portal_index < 0
            || (working_portal_index as usize) >= MAXIMUM_WORKING_PORTALS
        {
            return -1;
        }
        self.next_indices[working_portal_index as usize]
    }
}

impl Default for SClusterWorkingPortals {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// `s_static_index_queue<short, 256>` — circular FIFO
// =============================================================================

pub const STATIC_INDEX_QUEUE_CAPACITY: usize = 256;

/// `s_static_index_queue<short, 256>` (520B). Circular buffer of i16
/// indices used by the BFS portal walker. `read == write` means
/// empty; `(write + 1) % 256 == read` means full (one slot wasted to
/// distinguish full from empty).
#[derive(Debug, Clone)]
pub struct SStaticIndexQueue {
    pub read: i32,
    pub write: i32,
    pub indices: [i16; STATIC_INDEX_QUEUE_CAPACITY],
}

impl SStaticIndexQueue {
    pub fn new() -> Self {
        Self { read: 0, write: 0, indices: [0; STATIC_INDEX_QUEUE_CAPACITY] }
    }

    /// `empty @ 0x18050C320`. Returns `true` iff `read == write`.
    pub fn empty(&self) -> bool {
        self.read == self.write
    }

    /// `full @ 0x18050C480`. Returns `true` iff next-write would
    /// equal read (one slot wasted).
    pub fn full(&self) -> bool {
        Self::increment(self.write) == self.read
    }

    /// `increment @ 0x18050C5E0`. Wrap-around add-1.
    pub fn increment(value: i32) -> i32 {
        let next = value + 1;
        if (next as usize) == STATIC_INDEX_QUEUE_CAPACITY { 0 } else { next }
    }

    /// `enqueue @ 0x18050C350`. Append `index` to the tail. Drops
    /// silently on overflow (engine asserts).
    pub fn enqueue(&mut self, index: i16) {
        if self.full() {
            return;
        }
        self.indices[self.write as usize] = index;
        self.write = Self::increment(self.write);
    }

    /// `dequeue @ 0x18050C2A0`. Pop and return the head, or `-1` on
    /// empty.
    pub fn dequeue(&mut self) -> i16 {
        if self.empty() {
            return -1;
        }
        let idx = self.indices[self.read as usize];
        self.read = Self::increment(self.read);
        idx
    }
}

impl Default for SStaticIndexQueue {
    fn default() -> Self {
        Self::new()
    }
}
