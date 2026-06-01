//! Subset of `Ares/source/visibility/` used by the render path.
//!
//! See [`project_visibility_port_plan_2026_05_09`] for the umbrella
//! plan. This module mirrors the engine 1:1 (engine-faithful fixed
//! storage layouts; sizes verified at compile time via
//! `const _: () = assert!(size_of::<T>() == N);`).
//!
//! Phase A (types + tag-data parsing) ships:
//! - [`visibility`] — `s_visibility_region` + `visibility_volume` +
//!   `visibility_projection` + `visibility_cluster` +
//!   `visibility_volume_intersection` + math helpers.
//! - [`visibility_collection_objects`] — `s_visible_object_hierarchy`
//!   + `s_visible_object_render_visibility` (per-object visibility
//!   rows).
//! - [`visibility_collection_structure`] — `s_zone_cluster` +
//!   `s_visible_clusters` + `s_visible_instance_list` +
//!   `s_visible_instances` (per-BSP-cluster/instance visibility rows).
//! - [`visibility_collection`] — `c_simple_list<T,K>` template +
//!   `s_visible_items` (157,112B per-frame "what to draw") +
//!   `c_visible_items` static container with marker stack.
//! - [`visibility_render_objects`] — `s_render_object_info` (per-object
//!   render data populated during visibility).
//! - [`visibility_lod_transparency`] — `s_lod_transparency` (4B alpha
//!   pack used by visibility LOD fade).
//!
//! Producer entry points (Phase G) and consumer wire-ins (Phase I) are
//! documented in MEMORY.md `project_visibility_port_plan_2026_05_09`.

pub mod visibility;
pub mod visibility_collection;
pub mod visibility_collection_class;
pub mod visibility_collection_collect;
pub mod visibility_collection_objects;
pub mod visibility_collection_structure;
pub mod visibility_input;
pub mod visibility_lod_transparency;
pub mod visibility_portal_activation;
pub mod visibility_portal_hulls;
pub mod visibility_portal_traversal;
pub mod visibility_projections_and_volumes;
pub mod visibility_portal_walker;
pub mod visibility_region_builder;
pub mod visibility_transformed_portal_cache;
pub mod visibility_working_portals;
pub mod visibility_render_objects;
pub mod visibility_test;

pub use visibility::{
    visibility_region_get_cluster, visibility_region_get_cluster_volume,
    visibility_region_get_volume, RealRectangle2d, RealRectangle3d,
    SVisibilityRegion, VisibilityCluster, VisibilityProjection, VisibilityVolume,
    VisibilityVolumeIntersection, MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION,
    MAXIMUM_INTERSECTIONS_PER_VISIBILITY_REGION, MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION,
    MAXIMUM_SECTIONS_PER_RENDERABLE_REGION, MAXIMUM_VOLUMES_PER_VISIBILITY_REGION,
    MAXIMUM_VOLUME_REFERENCES_PER_RENDERABLE_REGION,
};
pub use visibility_collection::{
    CSimpleList, CVisibleItems, CVisibleItemsMarkerIndices, SVisibleItems,
    K_MAXIMUM_ITEM_MARKERS, K_MAXIMUM_REGION_MEMORY_ITEMS, K_MAXIMUM_SKY_ITEMS,
    K_SUBPART_BITVECTOR_SIZE_IN_BITS, K_SUBPART_BITVECTOR_SIZE_IN_U32,
};
pub use visibility_collection_class::CVisibilityCollection;
pub use visibility_collection_collect::{
    collect_all, collect_objects_and_lights_from_regions,
    collect_structure_elements_from_regions,
};
pub use visibility_collection_objects::{
    visibility_object_flags, VisibleObjectHierarchy, VisibleObjectRenderVisibility,
};
pub use visibility_collection_structure::{
    SVisibleClusters, SVisibleInstanceList, SVisibleInstances, SZoneCluster,
};
pub use visibility_input::{
    visibility_collection_flags, ECollectionShape, ECollectionType, SVisibilityInput,
};
pub use visibility_test::{
    visibility_volume_build_vertex_flags, visibility_volume_test_box,
    visibility_volume_test_point, visibility_volume_test_sphere,
    visibility_volume_test_sphere_old, visibility_volume_test_triangle,
    VERTEX_OUT_ALL_BITS, VERTEX_OUT_BOTTOM_BIT, VERTEX_OUT_FAR_BIT, VERTEX_OUT_LEFT_BIT,
    VERTEX_OUT_NEAR_BIT, VERTEX_OUT_RIGHT_BIT, VERTEX_OUT_TOP_BIT, VOLUME_INSIDE,
    VOLUME_INTERSECT, VOLUME_OUTSIDE,
};
pub use visibility_portal_activation::{
    ActivePortalBitvectors, PortalReference, MAX_PORTALS_PER_BSP, PORTAL_FLAG_WORDS_PER_BSP,
};
pub use visibility_portal_hulls::{
    convex_hull2d_intersect_new, convex_polygon2d_clip_to_plane, plane3d_clip_polygon,
    polygon2d_from_bounds, portal_hulls_intersect, EPortalFacing, PortalHull, TransformedPortal,
    AVERAGE_POINTS_PER_PORTAL_HULL, MAXIMUM_POINTS_PER_PORTAL_HULL, MAXIMUM_WORKING_PORTALS,
    MAXIMUM_WORKING_PORTAL_STACK_POINTS,
};
pub use visibility_transformed_portal_cache::{
    compute_portal_facing, projection_origin_in_portal, transform_portal, STransformedPortalCache,
};
pub use visibility_portal_walker::{
    build_working_portal_stack_setup_seed_working_portal,
    build_working_portal_stack_traverse_portals, copy_working_portal,
    search_working_portal_parents, search_working_portal_parents_by_cluster,
};
pub use visibility_region_builder::{
    build_working_portal_stack_add_working_portal_to_region,
    build_working_portal_stack_build_region,
    build_working_portal_stack_merge_working_portals, ClustersToRegionClusters,
};
pub use visibility_working_portals::{
    SClusterWorkingPortals, SStaticIndexQueue, SWorkingPortal, SWorkingPortalStack, STATIC_INDEX_QUEUE_CAPACITY,
};
pub use visibility_portal_traversal::{
    visibility_build_region_from_projections, visibility_build_region_from_pvs,
    visibility_find_best_initial_cluster_reference,
};
pub use visibility_projections_and_volumes::{
    visibility_project_box, visibility_project_sphere, visibility_region_test_box,
    visibility_region_test_sphere, visibility_volume_build, visibility_volumes_intersect,
};
pub use visibility_lod_transparency::LodTransparency;
pub use visibility_render_objects::RenderObjectInfo;
