//! Mirror of the BFS portal walker from
//! `Ares/source/visibility/visibility_portal_traversal.cpp`.
//!
//! Phase E4 — the heart of the projections walker. Starts from a
//! seed working portal at the camera's cluster, then iteratively
//! traverses outward through every portal in every visited cluster
//! until the BFS queue drains. Each step:
//!
//! 1. Dequeue a working portal `wp` (from cluster `C`).
//! 2. For each portal `P` in cluster `C`'s portal list:
//!    a. Skip if we've already walked through `P` (cycle detection
//!       via `search_working_portal_parents`).
//!    b. Skip if portal is inactive (door closed, etc).
//!    c. Lazy-transform `P` via the [`STransformedPortalCache`].
//!    d. Skip if portal facing is `None` (degenerate or beyond far).
//!    e. Determine adjacent cluster `C'` (the other side from `C`).
//!    f. Skip if cycle-on-cluster (`search_working_portal_parents_0`).
//!    g. Intersect `wp.hull` ∩ `transformed_portal.hull` →
//!       new hull. Skip if degenerate (< 3 vertices).
//!    h. Push new working portal `wp'` at `C'` with the clipped
//!       hull, link into [`SClusterWorkingPortals`] and the BFS
//!       queue.
//!
//! Phase E5's region builder later walks every working portal,
//! synthesizing per-portal-clipped volumes in `s_visibility_region`.

use crate::halo::structures::clusters::PackedClusterReference;
use crate::halo::scenario::LoadedScenario;
use crate::halo::visibility::visibility_portal_activation::{
    ActivePortalBitvectors, PortalReference,
};
use crate::halo::visibility::visibility_portal_hulls::{
    portal_hulls_intersect, EPortalFacing, PortalHull, MAXIMUM_POINTS_PER_PORTAL_HULL,
};
use crate::halo::visibility::visibility_transformed_portal_cache::{
    transform_portal, STransformedPortalCache,
};
use crate::halo::visibility::visibility_working_portals::{
    SClusterWorkingPortals, SStaticIndexQueue, SWorkingPortal, SWorkingPortalStack,
};
use crate::halo::visibility::VisibilityProjection;

// =============================================================================
// `setup_seed_working_portal @ 0x18050B3F0`
// =============================================================================

/// Build the BFS seed — a working portal at the camera's cluster
/// whose hull is the projection's full convex hull (the camera
/// frustum projected to 2D). Returns the new working portal index.
pub fn build_working_portal_stack_setup_seed_working_portal(
    stack: &mut SWorkingPortalStack,
    initial_cluster: PackedClusterReference,
    projection: &VisibilityProjection,
) -> Option<i16> {
    let idx = stack.push_default()?;
    let wp = stack.get_mut(idx).unwrap();
    wp.parent_working_portal_index = -1;
    wp.portal_reference = PortalReference { raw: u16::MAX }; // = -1 sentinel
    wp.created_volume_index = -1;
    wp.killed_by_index = -1;
    wp.cluster_reference = initial_cluster;
    wp.hull.bounds = projection.volume.frustum_bounds;
    wp.hull.points.clear();
    let n = projection.convex_hull_point_count.min(MAXIMUM_POINTS_PER_PORTAL_HULL as i32);
    for i in 0..(n as usize) {
        wp.hull.points.push(projection.convex_hull_points[i]);
    }
    wp.hull.point_count = n as i16;
    wp.nearest_z = 0.0;
    wp.do_not_traverse = false;
    Some(idx)
}

// =============================================================================
// `copy_working_portal @ 0x18050A7F0`
// =============================================================================

/// Duplicate an existing working portal into a new slot, with a
/// possibly different cluster_reference. Returns the new index.
pub fn copy_working_portal(
    stack: &mut SWorkingPortalStack,
    src_index: i16,
    new_cluster_reference: PackedClusterReference,
    do_not_traverse: bool,
) -> Option<i16> {
    let src = stack.get(src_index)?.clone();
    let idx = stack.push_default()?;
    let dst = stack.get_mut(idx).unwrap();
    dst.parent_working_portal_index = src.parent_working_portal_index;
    dst.portal_reference = src.portal_reference;
    dst.cluster_reference = new_cluster_reference;
    dst.created_volume_index = src.created_volume_index;
    dst.hull = src.hull;
    dst.nearest_z = src.nearest_z;
    dst.killed_by_index = src.killed_by_index;
    dst.do_not_traverse = do_not_traverse;
    Some(idx)
}

// =============================================================================
// `search_working_portal_parents @ 0x180509DB0`
// `search_working_portal_parents_0 @ 0x180509E50`
// =============================================================================

/// Walk the parent chain of `working_portal_index`; return the index
/// of the first ancestor whose `portal_reference` matches `target`,
/// or `-1` when none found. Used for cycle detection (don't traverse
/// the same portal twice in a single walk path).
pub fn search_working_portal_parents(
    stack: &SWorkingPortalStack,
    working_portal_index: i16,
    target_portal_reference: PortalReference,
) -> i16 {
    let mut idx = working_portal_index;
    while idx != -1 {
        let wp = match stack.get(idx) {
            Some(w) => w,
            None => return -1,
        };
        if wp.portal_reference == target_portal_reference {
            return idx;
        }
        idx = wp.parent_working_portal_index;
    }
    -1
}

/// Same shape as [`search_working_portal_parents`] but matches on
/// `cluster_reference` instead of `portal_reference`.
pub fn search_working_portal_parents_by_cluster(
    stack: &SWorkingPortalStack,
    working_portal_index: i16,
    target_cluster: PackedClusterReference,
) -> i16 {
    let mut idx = working_portal_index;
    while idx != -1 {
        let wp = match stack.get(idx) {
            Some(w) => w,
            None => return -1,
        };
        if wp.cluster_reference == target_cluster {
            return idx;
        }
        idx = wp.parent_working_portal_index;
    }
    -1
}

// =============================================================================
// `traverse_portals @ 0x18050A950` — the BFS body
// =============================================================================

/// Run the BFS until the queue drains. Returns `true` on success
/// (engine bool return — true means no overflow). Outputs accumulate
/// into `stack` + `cluster_working_portals`; the queue is consumed.
///
/// The BFS only handles intra-BSP portal traversal; engine's seam
/// crossing logic (Phase E6 / E7) is handled at the
/// `visibility_build_region_from_projections` outer level (it pushes
/// seam-translated seed portals into the queue before/after this
/// runs).
pub fn build_working_portal_stack_traverse_portals(
    _projection_index: i16,
    projection: &VisibilityProjection,
    scenario: &LoadedScenario,
    portal_activation: &ActivePortalBitvectors,
    stack: &mut SWorkingPortalStack,
    cluster_working_portals: &mut SClusterWorkingPortals,
    queue: &mut SStaticIndexQueue,
    transformed_portal_cache: &mut STransformedPortalCache,
) -> bool {
    let mut overflowed = false;

    while !queue.empty() {
        let wp_idx = queue.dequeue();
        if wp_idx < 0 {
            break;
        }

        // Snapshot fields we need (read-only, before mutating stack).
        let (wp_cluster, wp_do_not_traverse, wp_hull, wp_nearest_z) = {
            let wp = match stack.get(wp_idx) {
                Some(w) => w,
                None => continue,
            };
            (wp.cluster_reference, wp.do_not_traverse, wp.hull.clone(), wp.nearest_z)
        };
        if wp_do_not_traverse {
            continue;
        }

        // Resolve BSP + cluster.
        let bsp_index: i16 = wp_cluster.bsp_index as i16;
        let bsp = match scenario
            .active_bsps
            .iter()
            .find(|b| b.scenario_bsp_index as i16 == bsp_index)
        {
            Some(b) => b,
            None => continue,
        };
        let cluster = match bsp.sbsp.clusters.get(wp_cluster.cluster_index as usize) {
            Some(c) => c,
            None => continue,
        };

        // Walk this cluster's portals.
        for &portal_index_i16 in &cluster.portals {
            let portal_ref = PortalReference::new(bsp_index, portal_index_i16);

            // Cycle: don't traverse the same portal twice on this path.
            if search_working_portal_parents(stack, wp_idx, portal_ref) != -1 {
                continue;
            }

            // Door portal activation gate.
            if !portal_activation.is_active(portal_ref) {
                continue;
            }

            // Lazy-transform the portal (perspective-projected hull
            // + facing classification).
            let cache_slot = match transform_portal(
                transformed_portal_cache,
                projection,
                scenario,
                portal_ref,
            ) {
                Some(s) => s,
                None => continue,
            };
            let tp = match transformed_portal_cache.actual_transformed_portals.get(cache_slot)
            {
                Some(t) => t.clone(),
                None => continue,
            };
            if tp.facing == EPortalFacing::None {
                continue;
            }

            // Adjacent cluster = the side of the portal opposite to wp.
            let (back_idx, front_idx) = (tp.cluster_indices[0], tp.cluster_indices[1]);
            let adjacent_cluster_index = if (wp_cluster.cluster_index as i16) == back_idx {
                front_idx
            } else {
                back_idx
            };
            if adjacent_cluster_index < 0 {
                continue;
            }
            let adjacent_cluster = PackedClusterReference {
                bsp_index: bsp_index as i8,
                cluster_index: adjacent_cluster_index as u8,
            };

            // Cycle on cluster.
            if search_working_portal_parents_by_cluster(stack, wp_idx, adjacent_cluster) != -1
            {
                continue;
            }

            // Hull intersection.
            let mut new_hull = PortalHull::with_capacity(MAXIMUM_POINTS_PER_PORTAL_HULL);
            if !portal_hulls_intersect(&wp_hull, &tp.hull, &mut new_hull) {
                continue;
            }

            // Push new working portal at adjacent cluster.
            let new_idx = match stack.push_default() {
                Some(i) => i,
                None => {
                    overflowed = true;
                    break;
                }
            };
            let nz = wp_nearest_z.max(tp.nearest_z);
            {
                let new_wp: &mut SWorkingPortal = stack.get_mut(new_idx).unwrap();
                new_wp.parent_working_portal_index = wp_idx;
                new_wp.portal_reference = portal_ref;
                new_wp.cluster_reference = adjacent_cluster;
                new_wp.created_volume_index = -1;
                new_wp.do_not_traverse = false;
                new_wp.hull = new_hull;
                new_wp.nearest_z = nz;
                new_wp.killed_by_index = -1;
            }

            cluster_working_portals.insert(adjacent_cluster, new_idx);

            // Engine asserts the queue isn't full — we just drop on
            // overflow (matches `enqueue` semantics).
            if queue.full() {
                overflowed = true;
            } else {
                queue.enqueue(new_idx);
            }
        }
        if overflowed {
            break;
        }
    }

    !overflowed
}
