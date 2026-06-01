//! Mirror of the region-builder phase of the projections walker
//! (Phase E5):
//!
//! - [`build_working_portal_stack_merge_working_portals`] @ 0x18050B560
//!   — coalesces working portals at the same cluster when their
//!   merged-bounds area isn't much bigger than the sum (cheap
//!   redundancy elimination before emitting volumes).
//! - [`build_working_portal_stack_add_working_portal_to_region`] @
//!   0x18050B7E0 — emits one [`VisibilityVolume`] per surviving
//!   working portal at a cluster, builds its plane equations via
//!   [`visibility_volume_build`].
//! - [`build_working_portal_stack_build_region`] @ 0x18050BC00 —
//!   walks every working portal once; emits when this portal is the
//!   head of its cluster's chain (so each cluster's chain gets
//!   processed exactly once).

use crate::halo::structures::clusters::{ClusterReference, PackedClusterReference};
use crate::halo::scenario::{LoadedScenario, MAX_BSPS, MAX_CLUSTERS_PER_BSP};
use crate::halo::visibility::visibility_projections_and_volumes::visibility_volume_build;
use crate::halo::visibility::visibility_working_portals::{
    SClusterWorkingPortals, SWorkingPortalStack,
};
use crate::halo::visibility::{
    RealRectangle2d, SVisibilityRegion, VisibilityCluster, VisibilityProjection,
};

// =============================================================================
// `clusters_to_region_clusters` lookup
//
// Engine: `short clusters_to_region_clusters[16][256]` — a 16×256
// table mapping (bsp, cluster) → region cluster slot (or -1). Lives
// on the projections walker's stack.
// =============================================================================

pub struct ClustersToRegionClusters {
    pub table: Box<[[i16; MAX_CLUSTERS_PER_BSP]; MAX_BSPS]>,
}

impl ClustersToRegionClusters {
    pub fn new() -> Self {
        Self { table: Box::new([[-1; MAX_CLUSTERS_PER_BSP]; MAX_BSPS]) }
    }

    pub fn get(&self, cluster: PackedClusterReference) -> i16 {
        let bsp = cluster.bsp_index;
        let c = cluster.cluster_index;
        if bsp < 0 || (bsp as usize) >= MAX_BSPS || (c as usize) >= MAX_CLUSTERS_PER_BSP {
            return -1;
        }
        self.table[bsp as usize][c as usize]
    }

    pub fn set(&mut self, cluster: PackedClusterReference, region_cluster_index: i16) {
        let bsp = cluster.bsp_index;
        let c = cluster.cluster_index;
        if bsp < 0 || (bsp as usize) >= MAX_BSPS || (c as usize) >= MAX_CLUSTERS_PER_BSP {
            return;
        }
        self.table[bsp as usize][c as usize] = region_cluster_index;
    }

    pub fn clear(&mut self) {
        for row in self.table.iter_mut() {
            *row = [-1; MAX_CLUSTERS_PER_BSP];
        }
    }
}

impl Default for ClustersToRegionClusters {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// `real_rectangle2d_enclose_rectangle` + `_area` helpers
// =============================================================================

#[inline]
fn rectangle2d_enclose_rectangle(bounds: &mut RealRectangle2d, other: &RealRectangle2d) {
    if other.x0 < bounds.x0 { bounds.x0 = other.x0; }
    if other.x1 > bounds.x1 { bounds.x1 = other.x1; }
    if other.y0 < bounds.y0 { bounds.y0 = other.y0; }
    if other.y1 > bounds.y1 { bounds.y1 = other.y1; }
}

#[inline]
fn rectangle2d_area(bounds: &RealRectangle2d) -> f32 {
    let w = (bounds.x1 - bounds.x0).max(0.0);
    let h = (bounds.y1 - bounds.y0).max(0.0);
    w * h
}

// =============================================================================
// `merge_working_portals @ 0x18050B560`
// =============================================================================

/// Merge sibling working portals at the same cluster when their
/// combined bounds isn't significantly larger than the sum (1.5x
/// area tolerance). The "killed" portal records `killed_by_index =
/// next` so the region builder skips it; the surviving portal absorbs
/// the merged bounds + nearer `nearest_z`.
///
/// Engine algorithm (per `0x18050B560`):
/// - For each active BSP, for each cluster `c`:
///   - For each pair (`first`, `next`) in cluster_working_portals at c:
///     - `merged = enclose(first.bounds, next.bounds)`
///     - if `area(merged) < (area(first) + area(next)) * 1.5`:
///       - `first.killed_by_index = next`
///       - `next.bounds = merged; next.nearest_z = min(...)`
pub fn build_working_portal_stack_merge_working_portals(
    cluster_working_portals: &SClusterWorkingPortals,
    working_portal_stack: &mut SWorkingPortalStack,
    scenario: &LoadedScenario,
) {
    for bsp in &scenario.active_bsps {
        let bsp_index_i8 = bsp.scenario_bsp_index as i8;
        let cluster_count = bsp.sbsp.clusters.len();
        for cluster_index in 0..cluster_count.min(256) {
            let cluster_ref = PackedClusterReference {
                bsp_index: bsp_index_i8,
                cluster_index: cluster_index as u8,
            };
            let mut first = cluster_working_portals.get_first(cluster_ref);
            while first != -1 {
                // Walk siblings AFTER `first` looking for a partner.
                let mut next = cluster_working_portals.get_next(cluster_ref, first);
                let first_bounds = match working_portal_stack.get(first) {
                    Some(wp) => wp.hull.bounds,
                    None => break,
                };
                let first_area = rectangle2d_area(&first_bounds);
                let mut chosen_next: i16 = -1;
                let mut chosen_merged = first_bounds;

                while next != -1 {
                    let next_bounds = match working_portal_stack.get(next) {
                        Some(wp) => wp.hull.bounds,
                        None => break,
                    };
                    let mut merged = next_bounds;
                    rectangle2d_enclose_rectangle(&mut merged, &first_bounds);
                    let merged_area = rectangle2d_area(&merged);
                    let next_area = rectangle2d_area(&next_bounds);
                    if merged_area < (first_area + next_area) * 1.5 {
                        chosen_next = next;
                        chosen_merged = merged;
                        break;
                    }
                    next = cluster_working_portals.get_next(cluster_ref, next);
                }

                if chosen_next != -1 {
                    let first_nearest_z = working_portal_stack
                        .get(first)
                        .map(|w| w.nearest_z)
                        .unwrap_or(f32::INFINITY);
                    if let Some(first_wp) = working_portal_stack.get_mut(first) {
                        first_wp.killed_by_index = chosen_next as i32;
                    }
                    if let Some(next_wp) = working_portal_stack.get_mut(chosen_next) {
                        let merged_nearest = first_nearest_z.min(next_wp.nearest_z);
                        next_wp.nearest_z = merged_nearest;
                        next_wp.hull.bounds = chosen_merged;
                    }
                }

                first = cluster_working_portals.get_next(cluster_ref, first);
            }
        }
    }
}

// =============================================================================
// `add_working_portal_to_region @ 0x18050B7E0`
// =============================================================================

/// Emit one cluster's working-portal chain into `region` as
/// [`VisibilityVolume`] entries. Returns `false` on volume-overflow
/// (engine logs an event); `true` on full success.
pub fn build_working_portal_stack_add_working_portal_to_region(
    cluster_working_portals: &SClusterWorkingPortals,
    working_portal_stack: &mut SWorkingPortalStack,
    clusters_to_region: &mut ClustersToRegionClusters,
    projection_index: i32,
    region: &mut SVisibilityRegion,
    mut working_portal_index: i16,
    cluster_reference: PackedClusterReference,
    scenario: &LoadedScenario,
) -> bool {
    // Resolve / allocate region cluster slot.
    let mut v11 = clusters_to_region.get(cluster_reference);
    if v11 == -1 {
        let cluster_count = region.cluster_count;
        if cluster_count >= 128 {
            return false; // engine logs "overflowed region clusters during region building"
        }
        v11 = cluster_count;
        region.cluster_count = cluster_count + 1;
        let mut new_cluster = VisibilityCluster::default();
        new_cluster.cluster_reference = cluster_reference;
        new_cluster.volume_counts = [0; 6];
        new_cluster.first_volume_indices = [-1; 6];
        region.clusters[v11 as usize] = new_cluster;
        region.cluster_remap_table[v11 as usize] = v11 as i32;
        clusters_to_region.set(cluster_reference, v11);
    }

    // Walk the cluster's chain emitting volumes for surviving portals.
    while working_portal_index != -1 {
        if region.volume_count >= 512 {
            return false; // engine logs "overflowed region volumes during region building"
        }
        let (killed, hull_bounds, portal_ref, nearest_z) =
            match working_portal_stack.get(working_portal_index) {
                Some(wp) => (
                    wp.killed_by_index,
                    wp.hull.bounds,
                    wp.portal_reference,
                    wp.nearest_z,
                ),
                None => break,
            };
        if killed == -1 {
            let volume_idx = region.volume_count as usize;
            region.volume_count += 1;

            // Build per-portal-clipped volume.
            let proj_copy = region.projections[projection_index as usize];
            let ok = visibility_volume_build(
                &proj_copy,
                projection_index as i16,
                &hull_bounds,
                &mut region.volumes[volume_idx],
            );
            if ok {
                let v18 = &mut region.volumes[volume_idx];
                v18.region_cluster_index = v11;
                v18.portal_nearest_z = nearest_z;

                // Resolve plane_index from portal_reference (sentinel
                // = -1 when portal_reference is the seed -1).
                if portal_ref.raw == u16::MAX {
                    v18.plane_index = -1;
                    v18.plane_bsp_index = -1;
                } else {
                    let bsp_idx = portal_ref.bsp_index();
                    let portal_idx = portal_ref.portal_index();
                    let plane_index = scenario
                        .active_bsps
                        .iter()
                        .find(|b| b.scenario_bsp_index as i16 == bsp_idx)
                        .and_then(|b| b.sbsp.cluster_portals.get(portal_idx as usize))
                        .map(|p| p.plane_index)
                        .unwrap_or(-1);
                    v18.plane_index = plane_index;
                    v18.plane_bsp_index = bsp_idx;
                }

                // Update cluster's volume-list pointers.
                let cluster_slot = &mut region.clusters[v11 as usize];
                if cluster_slot.first_volume_indices[projection_index as usize] == -1 {
                    cluster_slot.first_volume_indices[projection_index as usize] =
                        volume_idx as i16;
                }
                cluster_slot.volume_counts[projection_index as usize] += 1;

                // Mark the working portal with its created volume.
                if let Some(wp) = working_portal_stack.get_mut(working_portal_index) {
                    wp.created_volume_index = volume_idx as i16;
                }
            }
        }
        // Advance to the next working portal in the chain.
        working_portal_index =
            cluster_working_portals.get_next(cluster_reference, working_portal_index);
    }
    true
}

// =============================================================================
// `build_region @ 0x18050BC00` — orchestrator
// =============================================================================

/// Walk every working portal in the stack; for each that's the HEAD
/// of its cluster's chain, emit the chain into `region` via
/// [`build_working_portal_stack_add_working_portal_to_region`]. Each
/// cluster gets exactly one emission pass.
pub fn build_working_portal_stack_build_region(
    cluster_working_portals: &SClusterWorkingPortals,
    working_portal_stack: &mut SWorkingPortalStack,
    clusters_to_region: &mut ClustersToRegionClusters,
    projection_index: i32,
    region: &mut SVisibilityRegion,
    scenario: &LoadedScenario,
) {
    let count = working_portal_stack.count();
    for working_portal_index in 0..count {
        let working_portal_index = working_portal_index as i16;
        let cluster_reference = match working_portal_stack.get(working_portal_index) {
            Some(wp) => wp.cluster_reference,
            None => continue,
        };
        let head = cluster_working_portals.get_first(cluster_reference);
        if head == working_portal_index {
            let cont = build_working_portal_stack_add_working_portal_to_region(
                cluster_working_portals,
                working_portal_stack,
                clusters_to_region,
                projection_index,
                region,
                working_portal_index,
                cluster_reference,
                scenario,
            );
            if !cont {
                return;
            }
        }
    }
    let _ = ClusterReference::NONE; // silence unused-import if the adapter goes unused later
}
