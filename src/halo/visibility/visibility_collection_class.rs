//! Mirror of `c_visibility_collection` (Ares
//! `source/visibility/visibility_collection.cpp`) — Phase F1-F5.
//!
//! - [`CVisibilityCollection`] is the orchestrator that owns one
//!   per-frame [`SVisibilityInput`] (~200KB) + per-collection state.
//! - [`CVisibilityCollection::compute`] (engine `0x180391DC0`) is the
//!   single entry point used by all 4 producer view types
//!   (camera/lights/lightmap_shadows/occlusion).
//! - [`prepare_collection_for_build`] (engine `0x180392230`) packs
//!   inputs + picks the [`ECollectionShape`] dispatch.
//! - [`setup_sphere_region`] / [`setup_projections_region`] populate
//!   `m_input.region` from the chosen shape.
//!
//! F6 (`collect_structure_elements_from_regions`) and F7
//! (`collect_objects_and_lights_from_regions`) live in
//! [`super::visibility_collection_collect`].

use blam_tags::math::RealPoint3d;

use crate::halo::structures::clusters::{ClusterReference, PackedClusterReference};
use crate::halo::scenario::{scenario_location_from_point, LoadedScenario};
use crate::halo::visibility::visibility_input::{ECollectionShape, ECollectionType};
use crate::halo::visibility::visibility_portal_activation::ActivePortalBitvectors;
use crate::halo::visibility::visibility_portal_traversal::{
    visibility_build_region_from_projections, visibility_build_region_from_pvs,
};
use crate::halo::visibility::{
    visibility_volume_build, CVisibleItems, SVisibilityInput, VisibilityProjection,
    MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION,
};

// =============================================================================
// `c_visibility_collection` — Phase F1
// =============================================================================

/// `c_visibility_collection` (Ares 8B in engine — single
/// `m_input *` pointer). For protomorph we own the input box-allocated;
/// callers can hand the collection between view types and call
/// `compute` repeatedly within a frame.
pub struct CVisibilityCollection {
    pub m_input: Box<SVisibilityInput>,
}

impl CVisibilityCollection {
    pub fn new() -> Self {
        Self {
            m_input: SVisibilityInput::new_zeroed(),
        }
    }

    // =========================================================================
    // `compute @ 0x180391DC0` — Phase F3
    // =========================================================================

    /// Top-level entry point. Engine equivalent:
    ///
    /// ```c
    /// auto v16 = prepare_collection_for_build(...);
    /// region_collection_begin(this);
    /// switch (v16) {
    ///     case PROJECTIONS: setup_projections_region(this, projections, projection_count); break;
    ///     case SPHERE:      setup_sphere_region(this); break;
    ///     case PVS:         visibility_build_region_from_pvs(projections, ic_ref, &this->m_input->region); break;
    /// }
    /// collect_structure_elements_from_regions(this);
    /// collect_objects_and_lights_from_regions(this);
    /// c_render_information::add_visibility_counters(collection_type);
    /// ```
    pub fn compute(
        &mut self,
        collection_type: ECollectionType,
        flags: u32,
        projections: &[VisibilityProjection],
        sphere_center: RealPoint3d,
        sphere_radius: f32,
        intersection_marker_index: i32,
        initial_cluster_reference: ClusterReference,
        user_index: i32,
        player_window_index: i32,
        scenario: &LoadedScenario,
        portal_activation: &ActivePortalBitvectors,
        visible_items: &mut CVisibleItems,
    ) {
        let _ = visible_items; // F6/F7 will consume this once implemented
        let shape = self.prepare_collection_for_build(
            flags,
            collection_type,
            projections,
            sphere_center,
            sphere_radius,
            intersection_marker_index,
            initial_cluster_reference,
            user_index,
            player_window_index,
            scenario,
        );

        // `region_collection_begin` (engine `0x?` — empties the
        // per-frame visibility region). Our SVisibilityInput is
        // already zeroed at construction; nothing to do here.

        match shape {
            ECollectionShape::Projections => {
                self.setup_projections_region(projections, scenario, portal_activation);
            }
            ECollectionShape::Sphere => {
                self.setup_sphere_region(scenario);
            }
            ECollectionShape::Pvs => {
                if let Some(projection) = projections.first() {
                    let initial = self.m_input.cluster_reference;
                    visibility_build_region_from_pvs(
                        scenario,
                        projection,
                        ClusterReference::from(initial),
                        &mut self.m_input.region,
                    );
                }
            }
        }

        // F6 + F7 land in `visibility_collection_collect`; once they
        // ship, these two calls go here:
        //   collect_structure_elements_from_regions(self, scenario, visible_items);
        //   collect_objects_and_lights_from_regions(self, scenario, visible_items);
    }

    // =========================================================================
    // `prepare_collection_for_build @ 0x180392230` — Phase F2
    // =========================================================================

    /// Pack inputs into `m_input` + pick the dispatch shape.
    ///
    /// Camera path defaults to `Projections` (the retail walker)
    /// unless `flags & 0x40` (`debug_pvs_render_all`) is set, which
    /// forces the PVS path. Sphere/light paths pick `Sphere` when
    /// `projection_count == 0` or `sphere_radius < 3.0`.
    pub fn prepare_collection_for_build(
        &mut self,
        flags: u32,
        collection_type: ECollectionType,
        projections: &[VisibilityProjection],
        sphere_center: RealPoint3d,
        sphere_radius: f32,
        intersection_marker_index: i32,
        mut initial_cluster_reference: ClusterReference,
        user_index: i32,
        player_window_index: i32,
        scenario: &LoadedScenario,
    ) -> ECollectionShape {
        let projection_count = projections
            .len()
            .min(MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION);

        // Engine: scenario_location_from_point fallback when the
        // caller didn't supply a valid initial cluster.
        if initial_cluster_reference.bsp_index < 0 {
            let loc = scenario_location_from_point(scenario, sphere_center);
            initial_cluster_reference = loc.cluster_reference;
        }

        self.m_input.cluster_reference = PackedClusterReference::from(initial_cluster_reference);
        self.m_input.collection_type = collection_type;
        self.m_input.user_index = user_index;
        self.m_input.player_window_index = player_window_index;
        self.m_input.flags = flags as i32;
        self.m_input.projection_count = projection_count as i16;
        self.m_input.sphere_center = sphere_center;
        self.m_input.sphere_radius = sphere_radius;
        self.m_input.visible_items_marker_index = intersection_marker_index;

        // Stage projections into m_input.region.projections.
        for i in 0..MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION {
            self.m_input.region.projections[i] = if i < projection_count {
                projections[i]
            } else {
                VisibilityProjection::default()
            };
        }
        self.m_input.region.projection_count = projection_count as i16;

        // Shape dispatch (engine flag bit 0x40 = USE_PVS_VISIBILITY).
        let shape = if collection_type == ECollectionType::Camera && (flags & 0x40) != 0 {
            ECollectionShape::Pvs
        } else if projection_count == 0 || (sphere_radius < 3.0 && (flags as i32) >= 0) {
            ECollectionShape::Sphere
        } else {
            ECollectionShape::Projections
        };
        self.m_input.collection_shape = shape;
        shape
    }

    // =========================================================================
    // `setup_projections_region` — Phase F5
    // =========================================================================

    /// Wraps [`visibility_build_region_from_projections`] for the
    /// `Projections` shape (retail camera default + lightmap shadow
    /// cube faces).
    pub fn setup_projections_region(
        &mut self,
        projections: &[VisibilityProjection],
        scenario: &LoadedScenario,
        portal_activation: &ActivePortalBitvectors,
    ) {
        let initial = ClusterReference::from(self.m_input.cluster_reference);
        let single_cluster_only = (self.m_input.flags & 0x100) != 0;
        let region_is_rendered_camera =
            self.m_input.collection_type == ECollectionType::Camera;
        visibility_build_region_from_projections(
            projections,
            initial,
            region_is_rendered_camera,
            &mut self.m_input.region,
            single_cluster_only,
            scenario,
            portal_activation,
        );
    }

    // =========================================================================
    // `setup_sphere_region` — Phase F4
    // =========================================================================

    /// Build a sphere-shaped visibility region — used by the lights
    /// view (per-dynamic-light) when the light's projection isn't
    /// frustum-clip-amenable. Walks active BSPs and adds every
    /// cluster whose AABB overlaps the sphere.
    ///
    /// Engine: `c_visibility_collection::setup_sphere_region` —
    /// internally walks every cluster's bounds (no PVS shortcut for
    /// lights) and builds a single per-cluster volume from the
    /// sphere's bounds. We approximate by adding every cluster within
    /// sphere_radius of the cluster center — Phase F refinement when
    /// dynamic lights ship.
    pub fn setup_sphere_region(&mut self, scenario: &LoadedScenario) {
        // Reset region.
        self.m_input.region.cluster_count = 0;
        self.m_input.region.volume_count = 0;
        self.m_input.region.projection_count = 1;

        let center = self.m_input.sphere_center;
        let radius = self.m_input.sphere_radius;
        let radius_sq = radius * radius;

        for bsp in &scenario.active_bsps {
            let bsp_idx = bsp.scenario_bsp_index;
            for (cluster_idx, cluster) in bsp.sbsp.clusters.iter().enumerate() {
                // AABB-vs-sphere reject — distance from sphere center
                // to clamped-to-AABB point.
                let cx = center.x.clamp(cluster.bounds_x.lower, cluster.bounds_x.upper);
                let cy = center.y.clamp(cluster.bounds_y.lower, cluster.bounds_y.upper);
                let cz = center.z.clamp(cluster.bounds_z.lower, cluster.bounds_z.upper);
                let dx = center.x - cx;
                let dy = center.y - cy;
                let dz = center.z - cz;
                if dx * dx + dy * dy + dz * dz > radius_sq {
                    continue;
                }
                // Add to region. Volume is a simple frustum from the
                // sphere's bounds projected through the projection
                // (engine builds this via visibility_volume_build).
                let region = &mut self.m_input.region;
                if region.cluster_count as usize >= region.clusters.len()
                    || region.volume_count as usize >= region.volumes.len()
                {
                    break;
                }
                let region_cluster_idx = region.cluster_count as usize;
                let volume_idx = region.volume_count as usize;
                let proj = region.projections[0];
                let frustum_bounds = proj.volume.frustum_bounds;
                if visibility_volume_build(
                    &proj,
                    0,
                    &frustum_bounds,
                    &mut region.volumes[volume_idx],
                ) {
                    region.volume_count += 1;
                    let mut new_cluster =
                        crate::halo::visibility::VisibilityCluster::default();
                    new_cluster.cluster_reference = PackedClusterReference {
                        bsp_index: bsp_idx as i8,
                        cluster_index: cluster_idx as u8,
                    };
                    new_cluster.volume_counts = [0; 6];
                    new_cluster.first_volume_indices = [-1; 6];
                    new_cluster.volume_counts[0] = 1;
                    new_cluster.first_volume_indices[0] = volume_idx as i16;
                    region.clusters[region_cluster_idx] = new_cluster;
                    region.cluster_remap_table[region_cluster_idx] = region_cluster_idx as i32;
                    region.cluster_count += 1;
                }
            }
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    pub fn get_region(&self) -> &crate::halo::visibility::SVisibilityRegion {
        &self.m_input.region
    }

    pub fn get_collection_type(&self) -> ECollectionType {
        self.m_input.collection_type
    }

    pub fn get_collection_shape(&self) -> ECollectionShape {
        self.m_input.collection_shape
    }

    pub fn get_cluster_reference(&self) -> ClusterReference {
        ClusterReference::from(self.m_input.cluster_reference)
    }

    pub fn get_center(&self) -> RealPoint3d {
        self.m_input.sphere_center
    }

    pub fn get_radius(&self) -> f32 {
        self.m_input.sphere_radius
    }

    pub fn get_flags(&self) -> i32 {
        self.m_input.flags
    }
}

impl Default for CVisibilityCollection {
    fn default() -> Self {
        Self::new()
    }
}
