//! Effect / particle subsystem — protomorph's runtime side of the
//! engine `c_effect` / `c_particle_system` pipeline.
//!
//! This module is a thin root; the implementation is carved across
//! submodules that mirror Ares `source/effects/`:
//!
//! - [`effects`] (Ares `effects.cpp`) — the [`EffectStore`] orchestrator
//!   + live [`EffectInstance`]s (load_effect, spawn_instance, frame_advance,
//!   batch_descriptors).
//! - [`effect_definitions`] (Ares `effect_definitions.cpp`) — the immutable
//!   resolved tag-chain types ([`LoadedEffect`], [`LoadedEffectPart`],
//!   [`LoadedParticleSystem`]) + their loaders.
//! - [`light_volumes`] (Ares `light_volumes.cpp`) — the `ltvl` / `rmlv`
//!   light-volume types ([`LoadedLightVolume`], [`LightVolumeProfileGpu`],
//!   [`LightVolumeDraw`], …) + the loader.
//! - [`particle_emitter`] / [`particle_properties`] / [`particle_states`] /
//!   [`particle_system`] — the CPU pulse, property eval engine, packed
//!   GPU-grid state, and render/pool/batch types.
//!
//! The tag chain a scenery-attached looping effect follows:
//!
//! ```text
//! scenery (model = null_up, invisible host)
//!   └─ object/attachments[i] (type_group == b"effe") → .effect
//!        └─ events[*]/particle systems[*] → .particle (prt3)
//!             └─ actual shader? (embedded render_method) → bitmaps
//! ```
//!
//! Engine references (verified addresses in
//! `reference_particle_system_engine_deep_dive`):
//! `attachments_new @0x1807E2F60` (effe dispatch),
//! `effect_new_from_object @0x1802fb050`, `effects_update @0x1802fc080`.

pub mod effect_definitions;
pub mod effect_event_fsm;
pub mod effects;
pub mod light_volumes;
pub mod particle_emitter;
pub mod particle_properties;
pub mod particle_states;
pub mod particle_system;

// === Public surface re-exports ===
// Keep every external `crate::halo::effects::{…}` path resolving after the
// carve-out (render/, game.rs, gpu_particle/ all import through here).
// `#[allow(unused_imports)]` on the re-export groups that aren't (yet)
// reached through the flat `crate::halo::effects::` path — they're part of
// the published surface (consumers also reach some via the submodule path
// `effect_definitions::`/`light_volumes::`), mirroring the existing
// `RawParticleState` precedent below.
#[allow(unused_imports)]
pub use effect_definitions::{
    system_lod_amount, LoadedEffect, LoadedEffectPart, LoadedParticleSystem,
};
pub use effects::EffectStore;
#[allow(unused_imports)]
pub use effects::EffectInstance;
pub use light_volumes::{LightVolumeDraw, LightVolumeProfileGpu};
#[allow(unused_imports)]
pub use light_volumes::{LightVolumeRenderInfo, LoadedLightVolume};

// The packed per-particle GPU-grid state lives in `particle_states`; the
// CPU-side render/pool/batch types in `particle_system`. `pub use` keeps the
// external paths (`crate::halo::effects::{ParticleState, RawParticleState,
// RowPool, BatchDescriptor, ParticleRenderInfo, ParticleAlbedo, EmitterDraw,
// ROW_BATCH_FREE}`) resolving after the carve-out.
// `ParticleState` / `RawParticleState` are re-exported for the external
// `crate::halo::effects::…` path even though the GPU side now imports them
// via `particle_states::`.
#[allow(unused_imports)]
pub use particle_states::{ParticleState, RawParticleState};
#[allow(unused_imports)]
pub use particle_system::{
    BatchDescriptor, EmitterDraw, ParticleAlbedo, ParticleRenderInfo, RowPool, ROW_BATCH_FREE,
};

/// Persistent particle grid capacity — engine `x_overall_storage` =
/// 16 cols × 448 rows = 7168 `RawParticleState`.
pub const PARTICLE_GRID_SIZE: u32 = 7168;
/// Grid rows in the pool (engine 448-row `s_gpu_buffer`).
pub const PARTICLE_ROW_COUNT: u32 = 448;
/// Particles per row (engine `s_row`, 16-wide).
pub const PARTICLE_ROW_WIDTH: u32 = 16;
/// Max particles spawned per frame — engine `x_queued_buffer_system`
/// cap (576).
pub const MAX_SPAWN_PER_FRAME: u32 = 576;

/// Reserved effect-location direction name → world-axis basis matrix
/// (engine `effect_build_locations @0x1803005f0`). An effect can name a
/// reserved DIRECTION instead of a host marker; the engine binds it to a
/// world-axis basis whose forward (+X) is the named direction:
///   `gravity` (string_id 180) → forward = world-DOWN,
///   `up`      (string_id 187) → forward = world-UP.
/// Any other name has no reserved basis and resolves to identity (the
/// caller substitutes host markers). Used for the unattached weather
/// spawn, which is positioned at the world origin and repositioned per
/// frame.
pub fn reserved_location_basis(name: &str) -> glam::Mat4 {
    match name {
        "gravity" => glam::Mat4::from_cols(
            glam::vec4(0.0, 0.0, -1.0, 0.0), // forward = down
            glam::vec4(0.0, 1.0, 0.0, 0.0),
            glam::vec4(1.0, 0.0, 0.0, 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0), // origin
        ),
        "up" => glam::Mat4::from_cols(
            glam::vec4(0.0, 0.0, 1.0, 0.0), // forward = up
            glam::vec4(0.0, 1.0, 0.0, 0.0),
            glam::vec4(-1.0, 0.0, 0.0, 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0),
        ),
        _ => glam::Mat4::IDENTITY,
    }
}

/// Spin a set of unresolved reserved-direction base markers about the
/// host's LOCAL up axis. When an effect names a reserved direction (e.g.
/// "up") that isn't an authored marker group, the host's single null-object
/// marker stands in for it. null_up's 'marker' has up-axis −X, so an
/// emitter's relative_direction pitch leans the jet −X (sideways off the
/// ramp). The visible launch markers (man cannon 'fire', guardian crate
/// 'up') lean +Y — up the throat. A −90° spin about up turns −X→+Y (up the
/// ramp) while leaving the vertical column unchanged (a pure +Z jet like
/// holy_light is invariant under up-axis rotation). Override via
/// `PROTOMORPH_UP_YAW` (degrees); 0 disables the spin.
pub fn up_yaw_spin(base: Vec<glam::Mat4>) -> Vec<glam::Mat4> {
    let up_yaw_deg: f32 = std::env::var("PROTOMORPH_UP_YAW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-90.0);
    if up_yaw_deg != 0.0 {
        let spin = glam::Mat4::from_rotation_z(up_yaw_deg.to_radians());
        base.into_iter().map(|m| spin * m).collect()
    } else {
        base
    }
}
