//! Phase F6 + F7 — fillers that walk `m_input.region.clusters[]` and
//! populate the global [`CVisibleItems`] storage with per-cluster +
//! per-instance + per-object visibility rows.
//!
//! Engine equivalents:
//! - [`collect_structure_elements_from_regions`] @ 0x18050FEA0 (F6)
//! - [`collect_objects_and_lights_from_regions`] @ 0x18050EA70 (F7)
//!
//! The engine versions are large because they handle every edge case
//! (LOD fade, sky expansion, dynamic light intersection, attachment
//! hierarchies, sub-mesh subpart bitvectors). This protomorph port
//! ships the structure-side minimal-correct version (cluster + BSP
//! instance emission); object/light/sky expansion ships incrementally
//! as the consumer phases (H/I) need them.

use crate::halo::structures::clusters::PackedClusterReference;
use crate::halo::scenario::LoadedScenario;
use crate::halo::visibility::visibility_collection_class::CVisibilityCollection;
use crate::halo::visibility::visibility_projections_and_volumes::visibility_region_test_box;
use crate::halo::visibility::{
    CVisibleItems, RealRectangle3d, SVisibleClusters, SVisibleInstanceList, SVisibleInstances,
    SZoneCluster,
};

// =============================================================================
// `collect_structure_elements_from_regions @ 0x18050FEA0` (F6)
// =============================================================================

/// Walk `region.clusters[]`. For each visible cluster:
///
/// 1. Emit one `SVisibleClusters` row → `m_items.clusters`.
/// 2. For each BSP instance whose AABB intersects the cluster's
///    volume(s), emit one `SVisibleInstances` row →
///    `m_items.instances` and link via `SVisibleInstanceList`.
///
/// The engine additionally populates per-mesh subpart bitvectors via
/// `generate_cluster_visible_parts` / `generate_instance_visible_parts`
/// — those land when the structure renderer (Phase H) needs sub-cluster
/// granularity.
pub fn collect_structure_elements_from_regions(
    collection: &CVisibilityCollection,
    scenario: &LoadedScenario,
    visible_items: &mut CVisibleItems,
) {
    let region = collection.get_region();
    let bsp_lookup = build_bsp_lookup(scenario);

    for region_cluster_index in 0..region.cluster_count as usize {
        let cluster = region.clusters[region_cluster_index];
        let bsp_index = cluster.cluster_reference.bsp_index;
        let cluster_index = cluster.cluster_reference.cluster_index;
        if bsp_index < 0 {
            continue;
        }

        // Resolve mesh_index for this cluster from the parsed sbsp
        // (cluster.mesh_index is the index into render_geometry.meshes).
        let bsp = match bsp_lookup
            .iter()
            .find(|b| b.scenario_bsp_index as i16 == bsp_index as i16)
        {
            Some(b) => *b,
            None => continue,
        };
        let bsp_cluster = match bsp.sbsp.clusters.get(cluster_index as usize) {
            Some(c) => c,
            None => continue,
        };

        // ---- Cluster row ----
        let row_idx = visible_items.items.clusters.add();
        if row_idx < 0 {
            break; // overflow
        }
        let mut cluster_row = SVisibleClusters::default();
        cluster_row.flags = 0;
        cluster_row.cluster = SZoneCluster {
            cluster_reference: cluster.cluster_reference,
            zone_index: -1,
        };
        cluster_row.region_cluster_index = region_cluster_index as u16;
        cluster_row.mesh_index = bsp_cluster.mesh_index as u16;
        visible_items.items.clusters[row_idx] = cluster_row;

        // ---- Instance list for this cluster ----
        let mut instance_list_first: i32 = -1;
        let mut instance_list_count: u16 = 0;

        for (instance_idx, instance) in bsp.sbsp.instanced_geometry_instances.iter().enumerate()
        {
            // AABB-vs-volume test using the instance's authored
            // world bounding sphere — engine uses tighter OBB but
            // the sphere is what we have parsed today.
            let r = instance.world_bounding_sphere_radius;
            let c = instance.world_bounding_sphere_center;
            let bounds = RealRectangle3d {
                x0: c.x - r,
                x1: c.x + r,
                y0: c.y - r,
                y1: c.y + r,
                z0: c.z - r,
                z1: c.z + r,
            };
            if !visibility_region_test_box(region, region_cluster_index as i32, &bounds, false)
            {
                continue;
            }

            let inst_row_idx = visible_items.items.instances.add();
            if inst_row_idx < 0 {
                break;
            }
            let mut inst_row = SVisibleInstances::default();
            inst_row.flags = 0;
            inst_row.structure_bsp_instance_index = instance_idx as u16;
            inst_row.structure_bsp_index = bsp_index as i16;
            inst_row.region_cluster_bitvector_start_index = region_cluster_index as i16;
            inst_row.alpha_byte = 0xFF;
            visible_items.items.instances[inst_row_idx] = inst_row;
            if instance_list_first < 0 {
                instance_list_first = inst_row_idx;
            }
            instance_list_count += 1;
        }

        if instance_list_count > 0 {
            let il_idx = visible_items.items.instance_list.add();
            if il_idx >= 0 {
                visible_items.items.instance_list[il_idx] = SVisibleInstanceList {
                    cluster: SZoneCluster {
                        cluster_reference: cluster.cluster_reference,
                        zone_index: -1,
                    },
                    instance_list_index: row_idx as u16,
                    first_visible_instance_list_instance_index: instance_list_first as u16,
                    visible_instance_list_count: instance_list_count,
                    mesh_index: bsp_cluster.mesh_index as u16,
                };
            }
        }
    }
}

/// `collect_objects_and_lights_from_regions @ 0x18050EA70` (F7).
/// Engine path walks `cluster_build_object_payload` for every visible
/// cluster, expanding into `m_items.root_objects` + `m_items.objects`
/// + per-light + sky lists. Protomorph today has minimal gameplay-
/// object lifecycle, so this is a placeholder that no-ops; lights +
/// skies populate via per-system collectors when their pipelines
/// (Phase I8 lens flares + I7 sky expansion) wire in.
pub fn collect_objects_and_lights_from_regions(
    _collection: &CVisibilityCollection,
    _scenario: &LoadedScenario,
    _visible_items: &mut CVisibleItems,
) {
    // TODO(phase-i): per-cluster object/light/sky expansion.
}

// =============================================================================
// Helpers
// =============================================================================

fn build_bsp_lookup(scenario: &LoadedScenario) -> Vec<&crate::halo::scenario::loader::LoadedBsp> {
    scenario.active_bsps.iter().collect()
}

/// Convenience wrapper: run both collectors. Used by
/// [`CVisibilityCollection::compute`] once F6+F7 are wired (Phase F8
/// wiring step).
pub fn collect_all(
    collection: &CVisibilityCollection,
    scenario: &LoadedScenario,
    visible_items: &mut CVisibleItems,
) {
    let _ = PackedClusterReference::default(); // silence unused-import if downstream removes
    collect_structure_elements_from_regions(collection, scenario, visible_items);
    collect_objects_and_lights_from_regions(collection, scenario, visible_items);
}
