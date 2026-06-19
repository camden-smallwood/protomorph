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

pub use visibility::{
    RealRectangle2d, RealRectangle3d, SVisibilityRegion, VisibilityCluster, VisibilityProjection,
    VisibilityVolume, MAXIMUM_CLUSTERS_PER_VISIBILITY_REGION,
    MAXIMUM_PROJECTIONS_PER_VISIBILITY_REGION, MAXIMUM_VOLUMES_PER_VISIBILITY_REGION,
};
pub use visibility_collection::CVisibleItems;
pub use visibility_collection_class::CVisibilityCollection;
pub use visibility_collection_objects::{
    VisibleObjectHierarchy, VisibleObjectRenderVisibility,
};
pub use visibility_input::SVisibilityInput;
pub use visibility_portal_activation::{
    ActivePortalBitvectors, MAX_PORTALS_PER_BSP, PORTAL_FLAG_WORDS_PER_BSP,
};
pub use visibility_projections_and_volumes::visibility_volume_build;
pub use visibility_lod_transparency::LodTransparency;
pub use visibility_render_objects::RenderObjectInfo;
