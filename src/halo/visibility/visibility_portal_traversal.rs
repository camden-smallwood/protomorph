//! Mirror of `Ares/source/visibility/visibility_portal_traversal.cpp`.
//!
//! Two top-level entry points:
//! - [`visibility_build_region_from_pvs`] @ 0x180508B20 (Phase D6) —
//!   simple PVS-only region (debug-render-all path).
//! - [`visibility_build_region_from_projections`] @ 0x180508520
//!   (Phase E7) — the retail camera path; recursive portal walker
//!   producing per-portal clipped volumes.

use crate::halo::structures::clusters::{ClusterReference, PackedClusterReference};
use crate::halo::scenario::{LoadedScenario, SGameClusterBitVectors};
use crate::halo::structures::structure_bsp_compute_cluster_active_pvs;
use crate::halo::visibility::visibility_portal_activation::ActivePortalBitvectors;
use crate::halo::visibility::visibility_portal_walker::{
    build_working_portal_stack_setup_seed_working_portal,
    build_working_portal_stack_traverse_portals,
};
use crate::halo::visibility::visibility_region_builder::{
    build_working_portal_stack_build_region, build_working_portal_stack_merge_working_portals,
    ClustersToRegionClusters,
};
use crate::halo::visibility::visibility_transformed_portal_cache::STransformedPortalCache;
use crate::halo::visibility::visibility_working_portals::{
    SClusterWorkingPortals, SStaticIndexQueue, SWorkingPortalStack,
};
use crate::halo::visibility::{
    visibility_volume_build, SVisibilityRegion, VisibilityCluster, VisibilityProjection,
    MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION, MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION,
    MAXIMUM_VOLUMES_PER_VISIBILITY_REGION,
};

/// `visibility_build_region_from_pvs @ 0x180508B20`. Build a
/// [`SVisibilityRegion`] using the precomputed cluster PVS.
///
/// All PVS-visible clusters across all active BSPs are added to
/// `region.clusters[]`. Each cluster gets ONE volume — the projection's
/// camera frustum — assigned via `visibility_volume_build`. This is
/// the engine's debug "render every PVS-visible cluster" path
/// (`debug_pvs_render_all`); the retail camera path uses the
/// projections walker (Phase E) which builds per-portal clipped
/// volumes instead.
///
/// Returns the number of clusters added (engine has no return value;
/// the count is convenient for callers who want to know if anything
/// matched).
pub fn visibility_build_region_from_pvs(
    scenario: &LoadedScenario,
    projection: &VisibilityProjection,
    initial_cluster_reference: ClusterReference,
    region: &mut SVisibilityRegion,
) -> usize {
    // Engine: copy `projections[0..6]` from input. Our caller passes
    // a single projection — splat it across all 6 slots so the rest
    // of the visibility code can read `region.projections[i]` for
    // `i < projection_count` (=1) without conditional access.
    for i in 0..MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION {
        region.projections[i] = *projection;
    }
    region.projection_count = 1;
    region.cluster_count = 0;
    region.volume_count = 0;

    if initial_cluster_reference.bsp_index < 0 {
        return 0;
    }

    // Compute the active PVS bitvector for the camera's cluster.
    let mut pvs = SGameClusterBitVectors::default();
    structure_bsp_compute_cluster_active_pvs(scenario, initial_cluster_reference, &mut pvs);

    let mut overflow_clusters = false;
    let mut overflow_volumes = false;

    // Iterate active BSPs; for each cluster bit set in the PVS, add
    // a (cluster, volume) entry.
    for bsp in &scenario.active_bsps {
        let bsp_idx = bsp.scenario_bsp_index;
        if bsp_idx >= 16 {
            continue;
        }
        let cluster_count = bsp.sbsp.clusters.len();
        for cluster_index in 0..cluster_count {
            // PVS bit test
            let word = pvs.bit_vectors[bsp_idx][cluster_index >> 5];
            if (word >> (cluster_index & 31)) & 1 == 0 {
                continue;
            }

            if region.cluster_count as usize >= MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION {
                overflow_clusters = true;
                continue;
            }

            let region_cluster_idx = region.cluster_count as usize;

            // Build the volume slot (same camera frustum for every cluster).
            if region.volume_count as usize >= MAXIMUM_VOLUMES_PER_VISIBILITY_REGION {
                overflow_volumes = true;
                continue;
            }
            let volume_idx = region.volume_count as usize;
            let frustum_bounds = projection.volume.frustum_bounds;
            // SAFETY: split the borrow between region.volumes (mut)
            // and region.projections (read).
            let proj_copy = region.projections[0];
            let ok = visibility_volume_build(
                &proj_copy,
                0,
                &frustum_bounds,
                &mut region.volumes[volume_idx],
            );
            if !ok {
                continue;
            }
            region.volume_count += 1;

            // Now the cluster slot.
            region.cluster_remap_table[region_cluster_idx] = region_cluster_idx as i32;
            let mut new_cluster = VisibilityCluster::default();
            new_cluster.cluster_reference = PackedClusterReference::from(ClusterReference {
                bsp_index: bsp_idx as i16,
                cluster_index: cluster_index as i16,
            });
            new_cluster.volume_counts = [0; 6];
            new_cluster.first_volume_indices = [-1; 6];
            new_cluster.volume_counts[0] = 1;
            new_cluster.first_volume_indices[0] = volume_idx as i16;
            region.clusters[region_cluster_idx] = new_cluster;
            region.cluster_count += 1;
        }
    }

    if overflow_clusters {
        // Engine logs a warning event ("overflowed region clusters
        // during visibility from pvs"). We just truncate.
    }
    if overflow_volumes {
        // Engine logs ("overflowed region volumes during visibility
        // from pvs"). We truncate.
    }

    // Engine: qsort cluster_remap_table by compare_region_cluster_indices_0.
    // Stable order helps downstream consumers; sort by (bsp, cluster).
    let count = region.cluster_count as usize;
    let mut keys: Vec<(i16, i16, i32)> = (0..count)
        .map(|i| {
            let cl = region.clusters[i].cluster_reference;
            let bsp = cl.bsp_index as i16;
            let cluster = cl.cluster_index as i16;
            (bsp, cluster, region.cluster_remap_table[i])
        })
        .collect();
    keys.sort_by_key(|(b, c, _)| (*b, *c));
    for (i, (_, _, rem)) in keys.into_iter().enumerate() {
        region.cluster_remap_table[i] = rem;
    }

    count
}

// =============================================================================
// `visiblility_find_best_initial_cluster_reference @ 0x18050BD80` (E6)
//
// Engine: looks up the active zone set, validates that
// `initial_cluster_reference.bsp_index` is in `bsp_zone_flags`. If
// the camera-cluster's BSP is INACTIVE, walks zone-set BSP flags +
// scenario seam data to find a "best" alternative cluster. Returns
// the input on success.
//
// Protomorph note: scenario_location_from_point already returns
// `Location::NONE` when the camera isn't in any active BSP, so we
// pass through the input here. Seam-translated alternatives
// (engine's primary use case) are not yet supported in protomorph.
// =============================================================================

/// `visiblility_find_best_initial_cluster_reference @ 0x18050BD80`.
/// Returns the input cluster reference unchanged unless the camera
/// is on a seam — full seam translation lands when protomorph maps
/// with multiple BSPs ship.
pub fn visibility_find_best_initial_cluster_reference(
    initial_cluster_reference: ClusterReference,
) -> ClusterReference {
    initial_cluster_reference
}

// =============================================================================
// `visibility_build_region_from_projections @ 0x180508520` (E7)
// =============================================================================

/// Engine-faithful retail camera region builder. Per projection,
/// resets all per-projection scratch (working portal stack, queue,
/// cache, cluster→portal lookup), seeds with a working portal at the
/// camera's cluster, runs the BFS walker, merges sibling working
/// portals, then emits per-portal-clipped volumes into `region`.
///
/// `single_cluster_only` skips the walker (only the seed cluster's
/// volume is emitted) — engine debug + texture-camera modes.
pub fn visibility_build_region_from_projections(
    projections: &[VisibilityProjection],
    initial_cluster_reference: ClusterReference,
    region_is_rendered_camera: bool,
    region: &mut SVisibilityRegion,
    single_cluster_only: bool,
    scenario: &LoadedScenario,
    portal_activation: &ActivePortalBitvectors,
) -> bool {
    let _ = region_is_rendered_camera; // engine uses this for debug viz; no-op here
    let projection_count = projections.len().min(MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION);

    // Copy projections into region (engine writes all 6 slots).
    for i in 0..MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION {
        region.projections[i] = if i < projection_count {
            projections[i]
        } else {
            VisibilityProjection::default()
        };
    }
    region.projection_count = projection_count as i16;
    region.cluster_count = 0;
    region.volume_count = 0;

    if initial_cluster_reference.bsp_index < 0 {
        return false;
    }
    let best_initial = visibility_find_best_initial_cluster_reference(initial_cluster_reference);
    let initial_packed: PackedClusterReference = best_initial.into();

    // Per-projection scratch, allocated once and reused across the loop.
    let mut clusters_to_region = ClustersToRegionClusters::new();
    let mut working_portal_stack = SWorkingPortalStack::new();
    let mut cluster_working_portals = SClusterWorkingPortals::new();
    let mut working_portal_index_queue = SStaticIndexQueue::new();
    let mut transformed_portal_cache = STransformedPortalCache::new();

    for projection_index in 0..projection_count {
        // Reset per-projection scratch.
        working_portal_stack.clear();
        cluster_working_portals.clear();
        working_portal_index_queue = SStaticIndexQueue::new();
        transformed_portal_cache.clear();

        // Seed the BFS at the camera's cluster.
        let projection = &region.projections[projection_index];
        let seed_idx = match build_working_portal_stack_setup_seed_working_portal(
            &mut working_portal_stack,
            initial_packed,
            projection,
        ) {
            Some(idx) => idx,
            None => continue,
        };
        cluster_working_portals.insert(initial_packed, seed_idx);
        working_portal_index_queue.enqueue(seed_idx);

        // (Engine optionally seeds with seam-translated copies +
        // audio_cluster_neighbors — Phase E6+ when multi-BSP ships.)

        // BFS unless single-cluster mode.
        if !single_cluster_only {
            let proj_copy = *projection;
            build_working_portal_stack_traverse_portals(
                projection_index as i16,
                &proj_copy,
                scenario,
                portal_activation,
                &mut working_portal_stack,
                &mut cluster_working_portals,
                &mut working_portal_index_queue,
                &mut transformed_portal_cache,
            );
        }

        // Merge redundant sibling portals at each cluster.
        build_working_portal_stack_merge_working_portals(
            &cluster_working_portals,
            &mut working_portal_stack,
            scenario,
        );

        // Emit per-cluster clipped volumes into `region`.
        build_working_portal_stack_build_region(
            &cluster_working_portals,
            &mut working_portal_stack,
            &mut clusters_to_region,
            projection_index as i32,
            region,
            scenario,
        );
    }

    // Stable sort for downstream iteration. Engine uses
    // `compare_region_cluster_indices_0` which orders by (bsp,
    // cluster); we mirror.
    let count = region.cluster_count as usize;
    let mut keys: Vec<(i16, i16, i32)> = (0..count)
        .map(|i| {
            let cl = region.clusters[i].cluster_reference;
            (cl.bsp_index as i16, cl.cluster_index as i16, region.cluster_remap_table[i])
        })
        .collect();
    keys.sort_by_key(|(b, c, _)| (*b, *c));
    for (i, (_, _, rem)) in keys.into_iter().enumerate() {
        region.cluster_remap_table[i] = rem;
    }

    true
}
