//! Decal subsystem — mirrors `Ares/source/effects/decals.{h,cpp}` +
//! `decal_definitions.{h,cpp}`. Pipeline:
//!
//! 1. **D1 (shipped):** load `scenario.decals[]` + `scenario.decal_palette[]`
//!    + `.decal_system` tag references into `LoadedScenario` (see
//!    `src/halo/scenario/loader.rs`).
//! 2. **D2a (shipped):** ray-cast each preplaced decal against its
//!    bound BSP's collision tree (see
//!    `src/halo/structures/collision_bsp.rs`). Produces a seed
//!    surface per decal placement.
//! 3. **D2b (this module, in progress):** turn each D2a seed into a
//!    per-decal indexed triangle mesh. Engine algorithm is a BFS
//!    over the collision surface graph driven by per-surface
//!    projection volumes (not a Sutherland-Hodgman triangle clip):
//!
//!    ```text
//!    c_decal_system::build_mesh @ 0x18039A200   sibling-chain dispatch
//!      c_decal::build_mesh @ 0x18039B820         per-decal orchestrator
//!        s_decal_mesh_builder mesh_builder           stack 51 KB work buffer
//!        s_fragment decal_fragments[16]              16 seeds max
//!        for collision in collisions:                D2a hits = seeds
//!          c_decal::build_mesh_fragment_recursive @ 0x18039C2D0
//!            ↳ BFS over collision_surface graph
//!              c_decal::build_mesh_fragment @ 0x18039CCC0   per-surface clip
//!                s_decal_projection_builder::build_projection @ 0x18039E510
//!                c_collision_surface_edge_iterator           neighbor walking
//!          c_decal::smooth_mesh_fragment @ 0x18039D220       seam normal avg
//!          c_decal::write_mesh_fragment  @ 0x18039D460       pack to rasterizer
//!    ```
//! 4. **D3-D6 (queued):** WGSL shader port + frame integration +
//!    material plumbing + polish (see
//!    `project_decals_port_plan_2026_05_10.md` memory).
//!
//! This commit lands the engine-faithful type scaffolding only;
//! algorithm bodies are split across follow-up sessions per the D2b
//! research memo. The `dead_code` allow applies until the BFS walker
//! ports land — every type below has a documented engine anchor.

#![allow(dead_code, unused_imports)]

pub mod collide;
pub mod decals;
pub mod edge_iterator;
pub mod floating_quad;
pub mod instance_raycast;
pub mod mesh_builder;
pub mod orchestrator;
pub mod projection_builder;
pub mod smoother;
pub mod types;
pub mod writer;

pub use decals::{
    CDecal, CDecalFlags, CDecalSystem, CDecalSystemFlags, CTagIndex, EPass, SClusterReference,
};

pub use types::{
    flag, CollisionSurfaceEdgeIterator, DecalFragment, DecalMeshBuilder,
    DecalMeshFragmentBuilder, DecalProjectionBuilder, Fold, FragmentBufferIntervals,
    NeighborSurface, WorkingVertex, K_MAX_NEIGHBOR_SURFACES, K_MAX_WORK_INDICES,
    K_MAX_WORK_VERTICES,
};
