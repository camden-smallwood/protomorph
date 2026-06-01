//! Structure (BSP) subsystem — sbsp tag → runtime data → dedicated draw path.
//!
//! Mirrors Ares `source/structures/` + `source/render/render_structure.cpp`.
//! BSP rendering is structurally different from per-character rendering:
//! flat per-frame draw lists (cluster_parts + instance_parts), no bone
//! hierarchy, world-coord vertices for clusters, per-instance world
//! transform for instances. See
//! `reference_h3_structure_render_pipeline.md` (auto-memory) for the
//! deep trace.

pub mod bsp3d;
pub mod bsp_gpu;
pub mod clusters;
pub mod collision_bsp;
pub mod structure_bsp_pvs;
pub mod structure_cluster_walker;

pub use bsp3d::{
    bsp3d_get_plane_from_designator, bsp3d_test_point, BSP3D_ROOT_NODE_INDEX,
    MAXIMUM_BSP3D_DEPTH,
};
pub use bsp_gpu::{BspGpu, BspMaterial, BspMesh, BspMeshPart};
pub use collision_bsp::{collision_bsp_test_vector, CollisionBspResult};
pub use structure_bsp_pvs::{
    cluster_pvs_flags, structure_bsp_compute_cluster_active_pvs,
};
pub use structure_cluster_walker::{
    sphere_intersects_cluster_portal_internal, structure_clusters_in_sphere,
    ClusterMarker,
};

// Per-frame draw lists + the `submit_visibility` body now live on
// [`crate::halo::render::structure_renderer::StructureRenderer`].
// `RenderClusterPart` + `RenderInstancePart` types moved with them.
