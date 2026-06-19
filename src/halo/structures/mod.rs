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
pub mod clusters_in_sphere;
pub mod collision_bsp;
pub mod structure_bsp_pvs;

pub use bsp3d::{
    bsp3d_get_plane_from_designator, bsp3d_test_point, BSP3D_ROOT_NODE_INDEX,
};
pub use bsp_gpu::BspGpu;
pub use structure_bsp_pvs::structure_bsp_compute_cluster_active_pvs;

// Per-frame draw lists + the `submit_visibility` body now live on
// [`crate::halo::render::structure_renderer::StructureRenderer`].
// `RenderClusterPart` + `RenderInstancePart` types moved with them.
