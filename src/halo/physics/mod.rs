//! Engine physics subsystem — verbatim port of Halo 3 MCC `physics/*.{h,cpp}`.
//!
//! Per-file mapping (Ares `source/physics/` → this module):
//! - `collision_constants.h` → [`collision_constants`]
//! - `bsp2d.{h,cpp}` → `bsp2d` (Phase 2)
//! - `bsp3d.{h,cpp}` → `bsp3d` (Phase 3)
//! - `collision_bsp.{h,cpp}` → `collision_bsp` (Phase 4)
//! - `collisions.{h,cpp}` → `collisions` (Phase 5)
//!
//! See `/Users/camden/Source/protomorph/ENGINE_GEOMETRY_SAMPLER_PORT_PLAN.md` for the
//! full chain (this module is the foundation of `c_geometry_sampler::sample`).

pub mod bsp2d;
pub mod bsp3d;
pub mod collision_bsp;
pub mod collision_constants;
pub mod collisions;
