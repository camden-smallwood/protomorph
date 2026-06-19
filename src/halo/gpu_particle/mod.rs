//! GPU particle subsystem — the persistent `RawParticleState` grid plus
//! the compute pipelines that spawn into it (Phase 2), integrate it
//! (Phase 3), and render it (Phase 4).
//!
//! Mirrors the engine's GPU-compute particle architecture
//! (`particle_block.cpp` / `common_gpu.cpp`): the CPU emitter pulse
//! ([`crate::halo::effects::EffectStore::frame_advance`]) only produces
//! spawn requests; everything else lives on the GPU against a single
//! storage grid so per-frame CPU↔GPU traffic is just the spawn queue.
//!
//! Phase 2 scope: the grid buffer, the spawn double-buffer + scatter
//! compute (`cs_spawn`), a ring allocator for grid slots, and a debug
//! readback that verifies birth pos/vel.
//!
//! Organized to mirror `Ares/source/gpu_particle/`:
//! - [`particle_block`] — the GPU particle grid core (spawn/update/render/
//!   distortion), [`ParticleGpu`] + [`BatchMaterial`].
//! - [`common_gpu`] — shared sprite-frame baking + blend-state mapping.
//! - [`light_volume_gpu`] — the light-volume (`rmlv`) strip render.

mod common_gpu;
mod particle_block;
pub mod light_volume_gpu;

pub use common_gpu::{bake_sprite_frames, default_sprite_corner};
pub use particle_block::{BatchMaterial, ParticleGpu};
// Crate-internal re-exports — keep `crate::halo::gpu_particle::blend_state`
// and `::MAX_SPRITE_FRAMES` resolving from their canonical `common_gpu` home.
// (Consumed via the `common_gpu`/`super` paths today; the re-export keeps the
// historical `gpu_particle::*` paths valid.)
#[allow(unused_imports)]
pub(crate) use common_gpu::{blend_state, MAX_SPRITE_FRAMES};
