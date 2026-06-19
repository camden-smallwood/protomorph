//! Effect store + live instances — the orchestrator (Ares `effects.cpp`).
//!
//! Holds every loaded effect (deduped by path) and all live instances, and
//! drives the CPU half of the engine's per-frame effect/particle pipeline:
//! `load_effect` (follow the tag chain), `spawn_instance`
//! (`effect_new_from_object`), `frame_advance` (the per-emitter pulse that
//! queues particles + builds light-volume strips), and `batch_descriptors`
//! (the per-batch render descriptors). The immutable resolved-tag types
//! live in [`super::effect_definitions`]; the light-volume types + loader in
//! [`super::light_volumes`].

use std::collections::HashMap;
use std::path::Path;

use blam_tags::effect::{EffectDefinition, EffectPartType, ParticleSystemFlags};
use blam_tags::paths::resolve_tag_path;
use blam_tags::TagFile;

use super::effect_definitions::{
    particle_system_eligible, resolve_physics, system_lod_amount, LoadedEffect, LoadedEffectPart,
    LoadedParticleSystem,
};
use super::light_volumes::{LightVolumeDraw, LightVolumeProfileGpu, LoadedLightVolume};
use super::particle_emitter::{self, EmitterRuntime};
use super::particle_properties;
use super::particle_system::{BatchDescriptor, EmitterDraw, ParticleRenderInfo, RowPool};
use super::particle_states::ParticleState;
use super::MAX_SPAWN_PER_FRAME;

/// A live placement of an effect in the scene. Phase 1 records the host
/// object so Phase 4 can sample its authoritative world matrix
/// (`object_header_data::world_matrix`, which already folds in any
/// parent-marker attachment); `origin` is the placement position kept
/// as a standalone fallback for logging and early emit tests.
#[derive(Debug, Clone)]
pub struct EffectInstance {
    /// Index into [`EffectStore::effects`].
    pub effect_index: usize,
    /// `object_index` (header) of the host object this effect rides on.
    pub host_header_index: u32,
    /// Marker the attachment binds to on the host model (empty = origin).
    pub marker: String,
    /// Host placement position — world-space fallback emit origin.
    pub origin: glam::Vec3,
    /// Per-location MARKER-LOCAL transform on the host model (engine
    /// effect location matrix = host_world × this). Indexed by the
    /// system's `location` block index. Carries the marker rotation that
    /// orients emission (e.g. null_up's "marker" → vertical spray).
    /// Identity when the marker isn't found.
    pub location_matrices: Vec<glam::Mat4>,
    /// Per-system → per-emitter CPU pulse state (`[system][emitter]`).
    pub emitter_runtimes: Vec<Vec<EmitterRuntime>>,
    /// `s_object_attachment::primary scale` — the host object-function
    /// name that gates this attachment's intensity. The engine scales the
    /// effect by `object_get_function_value(host, primary_scale)` each
    /// frame; emission is gated when it reads ~0 (e.g. the energy sword's
    /// `blade_effects`/`turning_on_effects` are 0 until the blade is
    /// drawn). Empty = always on (the `""`→1.0 short-circuit).
    pub primary_scale: String,
    /// Looping INSTANCE — engine creation-flag bit1, set by
    /// `effect_new_looping @0x1802faec0` (object effe attachments, flags
    /// |= 7) and `effect_new_weather @0x1802fbac0` (flags = 26). For such
    /// instances `effect_update_time @0x180309870` skips the whole
    /// event-duration/finish/delete machinery — the effect emits FOREVER
    /// regardless of the authored event `duration_bounds` (s3d_turf's
    /// ground splash is a `waterfall_end_small` attachment whose event
    /// says 5s; in MCC it never stops). One-shot effects (impacts etc.,
    /// not spawned yet) are created without the bit and DO honor event
    /// durations + the definition-flags-bit1 restart gate.
    pub looping: bool,
    /// Cluster-flagged WEATHER instance (engine `effect_new_weather` →
    /// `c_particle_system::m_flags & 0x100`). Anchored at the camera and
    /// uses the cluster `effect_weight` LOD path
    /// (`c_particle_location::calc_lod_amount`), NOT the per-system distance
    /// band — so its (often tiny / vestigial) `lod_in`/`lod_out` never culls
    /// the snow/rain/ash that is meant to surround the viewer. cluster weight
    /// is 1.0 in the camera's atmosphere; the per-zone atmosphere fade
    /// (weather stops indoors) is the deferred refinement.
    pub weather: bool,
    /// Destroyed instance (engine `effect_destroy` / `effect_handle_deleted_atmosphere
    /// @0x1802FDD10`). Its grid rows have been freed back to the pool; the slot
    /// is kept (so stable indices in tables like the weather per-setting table
    /// stay valid) but skipped by the pulse + render. A later
    /// [`EffectStore::spawn_instance`] reuses dead slots before growing the vec.
    pub dead: bool,
}

/// Holds every loaded effect (deduped by path) and all live instances.
/// One per [`crate::game::GameState`].
#[derive(Debug, Default)]
pub struct EffectStore {
    pub effects: Vec<LoadedEffect>,
    by_path: HashMap<String, usize>,
    /// Resolved `light_volume_system` (`ltvl`) tags, deduped by tag-path —
    /// referenced by `LightVolume` effect parts. The strip render
    /// (Track R1) looks them up by `LoadedEffectPart::type_tag_path`.
    pub light_volumes: HashMap<String, LoadedLightVolume>,
    /// Per-frame light-volume profile instances (rebuilt in `frame_advance`),
    /// consumed by the strip render. Flat; sliced by [`Self::light_volume_draws`].
    pub light_volume_profiles: Vec<LightVolumeProfileGpu>,
    /// Per-frame light-volume draws (one per live ltvl part), each a run of
    /// [`Self::light_volume_profiles`] sharing a base_map + blend.
    pub light_volume_draws: Vec<LightVolumeDraw>,
    pub instances: Vec<EffectInstance>,
    /// Particle birth states produced by the current frame's pulse —
    /// packed + scattered to the GPU grid each `render`. Refilled by
    /// [`Self::frame_advance`].
    pub pending_spawns: Vec<ParticleState>,
    /// Lockstep with [`Self::pending_spawns`]: the render-batch each
    /// spawn belongs to (diagnostics; routing is via [`Self::row_pool`]).
    pub pending_batches: Vec<u32>,
    /// Lockstep with [`Self::pending_spawns`]: the grid slot the engine
    /// row allocator assigned each birth (engine `m_gpu_address`). The GPU
    /// spawn scatters each packed state to this slot.
    pub pending_slots: Vec<u32>,
    /// The engine 448-row grid allocator — hands persistent slots to
    /// emitters and retires whole rows by lifespan.
    pub row_pool: RowPool,
    /// Monotonic batch-id allocator (one per emitter).
    next_batch_index: u32,
    /// Per-batch self-acceleration interpolant vectors in WORLD space —
    /// `[2*batch]` = starting, `[2*batch+1]` = ending. The update kernel
    /// SLERPs them by the property-11 curve and adds the result·dt to
    /// velocity (engine `vel += map_to_vector3d_range(self_accel, pre[11])
    /// · dt`). The local vectors × the emitter matrix; rotation preserves
    /// the slerp, so baking the world endpoints is faithful. E.g. the
    /// waterfall spray's local (-4,0,0) → world-down → flows down the cliff.
    pub batch_self_accel: Vec<[f32; 4]>,
    /// Per-batch per-frame state uniforms — `[batch*28 + slot].x` = value
    /// (engine `c_particle_state_list` system/location/game slots). The GPU
    /// update's curve eval reads the non-per-particle inputs (system_age=1,
    /// LOD=11, game_time=12, system seeds=8/9/25/26, location seed=16) from
    /// here; per-particle slots (age/random/emit_time) come from the packed
    /// particle. Rebuilt each frame in `frame_advance`.
    pub batch_state: Vec<[f32; 4]>,
    /// Per-batch world bounding sphere `(center, radius)` for the render
    /// pass's frustum cull + back-to-front depth sort. Engine
    /// `c_particle_emitter::submit @0x180568FA0`: cull sphere center =
    /// `get_position_world` (the emitter world origin), radius =
    /// `bounding_radius_estimate + location_radius`; the surviving
    /// emitter is sorted by that world position's depth. protomorph
    /// renders per-batch (a batch aggregates every instance of an
    /// effect's emitter), so this is the sphere enclosing the batch's
    /// emission origins, padded by the emitter bounding radius. Rebuilt
    /// each frame in `frame_advance` (tracks dynamic hosts).
    pub batch_bounds: Vec<(glam::Vec3, f32)>,
    /// Per-frame PER-EMITTER-INSTANCE draw/sort units (engine per-emitter
    /// granularity). One entry per (instance, system, emitter) with live
    /// rows; consumed by `register_particle_transparents` (one transparent
    /// element each) + `ParticleGpu::draw_emitter`. Rebuilt every
    /// `frame_advance`. See [`EmitterDraw`].
    pub emitter_draws: Vec<EmitterDraw>,
    /// Flat `(base_instance, count)` draw spans referenced by
    /// `EmitterDraw::span_start/span_count` — one per occupied grid row.
    pub emitter_draw_spans: Vec<(u32, u32)>,
    /// Wall-clock seconds since scenario load — birth-time stamp source.
    pub time: f32,
}

impl EffectStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a `.effect` tag (and all its particle systems) into the
    /// store, returning its index. Deduped by `effect_rel` so a palette
    /// shared across placements only loads once. Returns `None` if the
    /// effect tag can't be read.
    ///
    /// `effect_rel` is the tag-relative path (no extension) as it
    /// appears in the object attachment's `type_ref`.
    pub fn load_effect(
        &mut self,
        effect_rel: &str,
        tags_root: &Path,
    ) -> Option<usize> {
        if let Some(&idx) = self.by_path.get(effect_rel) {
            return Some(idx);
        }
        let effect_abs = resolve_tag_path(tags_root, effect_rel, "effect");
        let tag = match TagFile::read(&effect_abs) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[effects]   failed to read {}: {e}", effect_abs.display());
                return None;
            }
        };
        let definition = match EffectDefinition::from_tag(&tag) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[effects]   {effect_rel}: bad effect tag: {e}");
                return None;
            }
        };

        // Flatten events[*].particle_systems[*], resolving each prt3.
        let mut systems = Vec::new();
        let mut parts: Vec<LoadedEffectPart> = Vec::new();
        for (event_index, event) in definition.events.iter().enumerate() {
            for system in &event.particle_systems {
                // Engine creation-eligibility gate (see
                // [`particle_system_eligible`]). Skips systems an artist left
                // flagged "disabled for debugging" (the loose tag still
                // carries them; the runtime never creates them).
                if !particle_system_eligible(event, system) {
                    continue;
                }
                let particle_path = system.particle_tag_path.clone();
                if particle_path.is_empty() {
                    continue;
                }
                let Some((particle, bitmap_paths)) =
                    Self::load_particle(&particle_path, tags_root)
                else {
                    continue;
                };
                // Resolve the physics template chain from the first
                // emitter (all of a system's emitters share a template
                // in practice — the waterfall uses smoke.particle_physics).
                let render_info = particle
                    .shader
                    .as_ref()
                    .map(|rm| {
                        let choices = Self::load_rm_choices(rm, tags_root);
                        ParticleRenderInfo::from_shader(rm, &choices)
                    })
                    .unwrap_or_default();
                // Allocate one render batch per EMITTER and resolve each
                // emitter's own physics template — the engine streams the
                // GPU update per emitter, binding that emitter's UpdateState
                // and curve tables, not the system's.
                let mut emitter_physics = Vec::with_capacity(system.emitters.len());
                let mut emitter_batches = Vec::with_capacity(system.emitters.len());
                for emitter in &system.emitters {
                    let physics = resolve_physics(&emitter.particle_movement, tags_root, 0)
                        .unwrap_or_default();
                    let batch_index = self.next_batch_index;
                    self.next_batch_index += 1;
                    if std::env::var("PROTOMORPH_DIAG_PARTICLES").is_ok() {
                        let leaf = particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
                        let sample = |inp: f32| {
                            emitter
                                .particle_emission_rate
                                .function
                                .as_ref()
                                .map(|f| f.evaluate(inp, 0.0))
                                .unwrap_or(-1.0)
                        };
                        let ftype = emitter
                            .particle_emission_rate
                            .function
                            .as_ref()
                            .map(|f| format!("{:?} const={:?}", f.function_type(), f.as_constant()))
                            .unwrap_or_else(|| "none".into());
                        let life = emitter
                            .particle_lifespan
                            .function
                            .as_ref()
                            .map(|f| f.evaluate(0.0, 0.0))
                            .unwrap_or(-1.0);
                        eprintln!(
                            "[effects]   batch {batch_index}: {leaf} albedo={:?} blend={:?} bp={} \
                             phys(g={:.2} a={:.2}) | emit_rate[{ftype}] @0={:.1} @.5={:.1} @1={:.1} life@0={:.2}",
                            render_info.albedo, render_info.blend_mode, render_info.black_point,
                            physics.gravity_mod, physics.air_friction,
                            sample(0.0), sample(0.5), sample(1.0), life,
                        );
                    }
                    emitter_physics.push(physics);
                    emitter_batches.push(batch_index);
                }
                systems.push(LoadedParticleSystem {
                    event_index,
                    particle_path,
                    system: system.clone(),
                    particle,
                    bitmap_paths,
                    emitter_physics,
                    render_info,
                    emitter_batches,
                });
            }
            // Effect-part multiplexer (engine `event_generate_part
            // @0x180301C40`): every non-particle part — beam / contrail /
            // light_volume / decal / light / screen-effect / sound /
            // sub-effect — is decoded and carried here for the per-type
            // handlers (Tracks R/E). `prt3` particle parts are skipped (they
            // live in the dedicated `particle_systems` block, already
            // flattened above); `Empty` parts are unauthored padding.
            for part in &event.parts {
                if matches!(part.part_type, EffectPartType::Empty | EffectPartType::Particle) {
                    continue;
                }
                parts.push(LoadedEffectPart {
                    event_index,
                    part_type: part.part_type,
                    type_tag_path: part.type_tag_path.clone(),
                    location: part.location,
                    relative_offset: [
                        part.relative_offset.x,
                        part.relative_offset.y,
                        part.relative_offset.z,
                    ],
                    relative_orientation: [
                        part.relative_orientation_yaw_pitch.yaw,
                        part.relative_orientation_yaw_pitch.pitch,
                    ],
                });
                // Track R1: resolve the `ltvl` tag (deduped) so the strip
                // render has the definition + curves + shader. Other part
                // types' loaders land with their subsystems (beams/decals/…).
                if part.part_type == EffectPartType::LightVolume
                    && !part.type_tag_path.is_empty()
                    && !self.light_volumes.contains_key(&part.type_tag_path)
                {
                    if let Some(lv) = Self::load_light_volume_def(&part.type_tag_path, tags_root) {
                        if std::env::var("PROTOMORPH_DIAG_PARTS").is_ok() {
                            let dens =
                                particle_emitter::diag_eval(&lv.definition.profile_density, 0.0, 0.0);
                            let len = particle_emitter::diag_eval(&lv.definition.length, 0.0, 0.0);
                            eprintln!(
                                "[ltvl] {}: blend={:?} fog={} base='{}' len={:.2} density={:.0} brightness={:.2}",
                                part.type_tag_path, lv.render.blend_mode, lv.render.fog,
                                lv.render.base_map, len, dens, lv.definition.brightness_ratio,
                            );
                        }
                        self.light_volumes.insert(part.type_tag_path.clone(), lv);
                    }
                }
            }
        }

        // Recon: report what non-particle parts this effect carries — so we
        // know which breadth subsystems the loaded scenarios actually
        // exercise (PROTOMORPH_DIAG_PARTS=1).
        if std::env::var("PROTOMORPH_DIAG_PARTS").is_ok() && !parts.is_empty() {
            let mut counts: HashMap<String, u32> = HashMap::new();
            for p in &parts {
                *counts
                    .entry(format!("{:?}", p.part_type))
                    .or_default() += 1;
            }
            let mut summary: Vec<String> =
                counts.iter().map(|(k, v)| format!("{k}×{v}")).collect();
            summary.sort();
            eprintln!("[effect-parts] {effect_rel}: {}", summary.join(" "));
            for p in &parts {
                eprintln!(
                    "[effect-parts]    {:?} -> '{}' loc={} off={:?}",
                    p.part_type, p.type_tag_path, p.location, p.relative_offset,
                );
            }
        }

        let idx = self.effects.len();
        self.by_path.insert(effect_rel.to_string(), idx);
        self.effects.push(LoadedEffect {
            path: effect_rel.to_string(),
            definition,
            systems,
            parts,
        });
        Some(idx)
    }

    /// Spawn one live instance of `effect_index` riding on a host
    /// object. Engine equivalent: `effect_new_from_object` queued by
    /// `attachments_new` when it dispatches an `effe` attachment.
    /// `looping` = the engine creation-flag bit1 — pass `true` for
    /// attachment/weather instances (see [`EffectInstance::looping`]).
    pub fn spawn_instance(
        &mut self,
        effect_index: usize,
        host_header_index: u32,
        marker: String,
        origin: glam::Vec3,
        location_matrices: Vec<glam::Mat4>,
        primary_scale: String,
        looping: bool,
    ) -> usize {
        // Reuse a destroyed slot before growing the vec, so per-setting tables
        // (weather) and any other index references stay stable + bounded.
        let instance_index = self
            .instances
            .iter()
            .position(|i| i.dead)
            .unwrap_or(self.instances.len());
        // Build per-system/per-emitter runtime, seeded per (instance,
        // system, emitter) so emitters don't lock-step.
        let emitter_runtimes = self.effects[effect_index]
            .systems
            .iter()
            .enumerate()
            .map(|(si, sys)| {
                (0..sys.system.emitters.len())
                    .map(|ei| EmitterRuntime::new(instance_index, si, ei))
                    .collect()
            })
            .collect();
        let inst = EffectInstance {
            effect_index,
            host_header_index,
            marker,
            origin,
            location_matrices,
            emitter_runtimes,
            primary_scale,
            looping,
            weather: false,
            dead: false,
        };
        if instance_index == self.instances.len() {
            self.instances.push(inst);
        } else {
            self.instances[instance_index] = inst;
        }
        instance_index
    }

    /// Destroy a live instance — engine `effect_destroy` /
    /// `effect_handle_deleted_atmosphere @0x1802FDD10`. Frees every grid row
    /// the instance's emitters own back to the shared [`RowPool`] (else the
    /// pool leaks) and marks the slot dead (skipped by pulse + render, reusable
    /// by the next spawn). No-op on an invalid or already-dead index.
    pub fn destroy_instance(&mut self, instance_index: usize) {
        let Some(inst) = self.instances.get_mut(instance_index) else {
            return;
        };
        if inst.dead {
            return;
        }
        for sys in &mut inst.emitter_runtimes {
            for rt in sys {
                self.row_pool.release_rows(&mut rt.rows);
            }
        }
        inst.dead = true;
    }

    /// Reposition a camera-relative instance's emission to `world_pos`,
    /// keeping each location's orientation basis (forward/up axes). Used
    /// per-frame for atmosphere WEATHER effects so the emission volume
    /// follows the viewer — mirrors engine `effect_refresh_location`, which
    /// keeps weather anchored to the player's active cluster rather than the
    /// fixed world origin it was spawned at.
    pub fn set_instance_origin(&mut self, instance_index: usize, world_pos: glam::Vec3) {
        if let Some(inst) = self.instances.get_mut(instance_index) {
            // ONLY the host translation — emission resolves `host_matrix ×
            // location_matrix`, and `host_matrix` is `translation(origin)`
            // for a hostless effect. The location matrices stay basis-only at
            // the local origin (forward = gravity/up); translating them too
            // would double the offset → particles spawn at 2× the position.
            inst.origin = world_pos;
        }
    }

    /// Advance every live instance's emitters by `dt`, refilling
    /// [`Self::pending_spawns`] with this frame's world-space births.
    /// Mirrors the CPU half of `c_particle_system::frame_advance_all_gpu`
    /// (the per-emitter pulse that queues particles); the GPU scatter +
    /// integrate happen in the renderer.
    ///
    /// `camera_pos`/`fov_scale` drive the per-location LOD (engine
    /// `c_particle_location::calc_lod_amount @0x180496980`, distance path):
    /// each system computes a [0,1] LOD from its distance to the camera; a
    /// system at LOD 0 (past `lod_out`, before `lod_in`) releases its
    /// particles + frees its grid rows, and restarts when it returns.
    pub fn frame_advance(&mut self, dt: f32, camera_pos: glam::Vec3, fov_scale: f32) {
        self.time += dt;
        // Global distance-LOD knobs. `x_distance_lod_scale` is 1.0 in
        // single-player (the engine raises it in splitscreen to cull
        // harder); PROTOMORPH_NO_PARTICLE_LOD forces LOD=1 everywhere for
        // A/B comparison against the pre-LOD behaviour.
        let dist_scale = std::env::var("PROTOMORPH_PARTICLE_LOD_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        let lod_disabled = std::env::var("PROTOMORPH_NO_PARTICLE_LOD").is_ok();
        let batch_total = self.next_batch_index as usize;
        let Self {
            effects,
            instances,
            pending_spawns,
            pending_batches,
            pending_slots,
            row_pool,
            batch_self_accel,
            batch_state,
            batch_bounds,
            emitter_draws,
            emitter_draw_spans,
            light_volumes,
            light_volume_profiles,
            light_volume_draws,
            time,
            ..
        } = self;
        pending_spawns.clear();
        light_volume_profiles.clear();
        light_volume_draws.clear();
        pending_batches.clear();
        pending_slots.clear();
        emitter_draws.clear();
        emitter_draw_spans.clear();
        batch_self_accel.clear();
        batch_self_accel.resize(batch_total * 2, [0.0; 4]);
        // Per-frame state uniforms (28 slots/batch). game_time (slot 12) is a
        // 0→1 sawtooth each second (engine `game_tick_length·(tick%tick_rate)`),
        // shared by every batch; per-emitter slots filled in the loop below.
        batch_state.clear();
        batch_state.resize(batch_total * 28, [0.0; 4]);
        let game_time = *time - time.floor();
        for b in 0..batch_total {
            batch_state[b * 28 + 12][0] = game_time; // game_time
            // LOD default 1.0; active emitters overwrite slot 11 with their
            // location LOD below (a batch with no live emitter draws nothing).
            batch_state[b * 28 + 11][0] = 1.0;
        }
        // Per-batch world bounds accumulator: AABB of this batch's
        // emission origins (across every contributing instance) + the
        // max emitter bounding radius. Finalized to a (center, radius)
        // sphere after the instance loop for the render cull/sort.
        let mut bounds_acc: Vec<Option<(glam::Vec3, glam::Vec3, f32)>> = vec![None; batch_total];
        let birth_time = *time;
        // Fair per-frame spawn budget. The engine's staging buffer caps
        // total spawns at MAX_SPAWN_PER_FRAME; distributing it front-to-back
        // (the old `return` below) starved every system loaded after the
        // budget filled — the core runaway-starvation bug. Instead give each
        // emitter a max-min fair share of what's left, `ceil(remaining /
        // remaining_emitters)`, so a low-demand emitter's leftover rolls
        // forward and no emitter is starved while budget remains.
        let mut remaining_emitters: u32 = instances
            .iter()
            .filter(|inst| !inst.dead)
            .map(|inst| {
                effects[inst.effect_index]
                    .systems
                    .iter()
                    .map(|s| s.system.emitters.len() as u32)
                    .sum::<u32>()
            })
            .sum();
        let mut remaining_budget = MAX_SPAWN_PER_FRAME;
        for inst in instances.iter_mut() {
            if inst.dead {
                continue;
            }
            let effect = &effects[inst.effect_index];
            // Host (location) world matrix — orients the emission. Engine
            // `calc_matrix` builds the emitter matrix from the location
            // transform. Falls back to a translation-only matrix if the
            // host object header isn't resolved.
            let host_matrix = crate::halo::objects::object_header_data::world_matrix(
                inst.host_header_index,
            )
            .unwrap_or_else(|| glam::Mat4::from_translation(inst.origin));
            // Engine-faithful attachment gate: the host scales this effect
            // by `object_get_function_value(host, primary_scale)` each
            // frame (`effe→...object_get_function_value` in
            // `attachments_new @0x1807E2F60`). The evaluator resolves the
            // import chain — a static weapon's `turned_on`/`turning_on`
            // read 0 (so the energy sword's blade/charge systems stay
            // dark instead of igniting all their DontRenderSystem variants
            // into a white blowout); a powered teleporter's
            // `teleporter_active` reads 1. Empty name → 1.0 (the engine
            // `""`→1 short-circuit) = always on. Re-evaluated per frame so
            // it tracks dynamic state (blade drawn → effect resumes).
            let emit_scale = {
                let mut v = 0.0f32;
                let mut det = false;
                crate::halo::objects::object_get_function_value::object_get_function_value(
                    inst.host_header_index,
                    &inst.primary_scale,
                    u32::MAX,
                    *time,
                    &mut v,
                    &mut det,
                );
                v
            };
            // Looping if this is a looping INSTANCE (attachment/weather —
            // engine `effect_update_time` skips the event lifecycle
            // entirely for those, see [`EffectInstance::looping`]), or if
            // the DEFINITION restarts its events when they finish (engine
            // gate: definition flags bit1 — the restart branch in
            // `effect_update_time @0x180309870`; `loop_start_event` only
            // picks WHERE the restart begins).
            let looping = inst.looping
                || effect
                    .definition
                    .flags
                    .contains(blam_tags::effect::EffectFlags::RunEventsInParallel)
                || effect.definition.loop_start_event >= 0;
            for (si, sys) in effect.systems.iter().enumerate() {
                // Event duration drives the loop period. Use the upper bound.
                let duration = effect
                    .definition
                    .events
                    .get(sys.event_index)
                    .map(|e| e.duration_bounds.upper.max(e.duration_bounds.lower))
                    .unwrap_or(0.0);
                // Location matrix = host_world × marker_local (the marker
                // rotation orients emission). Falls back to host_world.
                let loc_idx = sys.system.location.max(0) as usize;
                let location_matrix = match inst.location_matrices.get(loc_idx) {
                    Some(m) => host_matrix * *m,
                    None => host_matrix,
                };
                // Per-location LOD (engine `c_particle_location::calc_lod_amount`,
                // distance arm): camera→location distance × fov_scale, ramped
                // through the system's lod_in/out feather band. lod==0 releases
                // every emitter below; lod∈(0,1] scales max_count + drives state
                // slot 11. (Weather/cluster-flagged systems use a PVS effect_weight
                // instead — deferred; those fall through to the distance band,
                // which is 1.0 when no band is authored.)
                let lod = if lod_disabled || inst.weather {
                    // Weather = cluster effect_weight path (≈1.0 in the
                    // camera's atmosphere), never the distance band — see
                    // [`EffectInstance::weather`].
                    1.0
                } else {
                    let dist =
                        (location_matrix.w_axis.truncate() - camera_pos).length() * fov_scale;
                    system_lod_amount(&sys.system, dist, dist_scale)
                };
                if std::env::var("PROTOMORPH_DIAG_LOD").is_ok() && *time < 0.2 {
                    diag::diag_lod(sys, location_matrix, camera_pos, lod);
                }
                if std::env::var("PROTOMORPH_DIAG_ORIENT").is_ok() && *time < 0.2 {
                    diag::diag_orient(sys, effect, inst, loc_idx, host_matrix, location_matrix);
                }
                for (ei, emitter) in sys.system.emitters.iter().enumerate() {
                    // Per-emitter particle pulse — extracted verbatim into
                    // [`super::particle_system::pulse_emitter_instance`]
                    // (behaviour-identical; the loop body's lone `continue`
                    // became a `return` there). The shared spawn-budget
                    // counters + per-batch accumulators thread through by
                    // `&mut`; `rt` is this emitter's runtime slot.
                    let rt = &mut inst.emitter_runtimes[si][ei];
                    super::particle_system::pulse_emitter_instance(
                        sys,
                        ei,
                        emitter,
                        rt,
                        location_matrix,
                        lod,
                        dt,
                        birth_time,
                        duration,
                        looping,
                        emit_scale,
                        &mut remaining_emitters,
                        &mut remaining_budget,
                        &mut bounds_acc,
                        batch_self_accel,
                        batch_state,
                        row_pool,
                        pending_spawns,
                        pending_slots,
                        pending_batches,
                        emitter_draw_spans,
                        emitter_draws,
                    );
                }
            }
            // === Track R1: light-volume strip build ===
            // Per-instance `ltvl` profile build, gated by the same host
            // attachment function (`emit_scale`) as particle emission. See
            // [`super::light_volumes::build_instance_light_volumes`] — the body
            // was extracted there verbatim (behaviour-identical).
            super::light_volumes::build_instance_light_volumes(
                effect,
                inst,
                host_matrix,
                emit_scale,
                light_volumes,
                light_volume_profiles,
                light_volume_draws,
            );
        }
        if std::env::var("PROTOMORPH_DIAG_LTVL").is_ok() && light_volume_draws.len() > 0 {
            static LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
            let sec = *time as u32;
            if LAST.swap(sec, std::sync::atomic::Ordering::Relaxed) != sec {
                eprintln!(
                    "[ltvl] t={sec}s {} strips, {} profiles | first strip: base='{}' count={} origin~({:.1},{:.1},{:.1})",
                    light_volume_draws.len(),
                    light_volume_profiles.len(),
                    light_volume_draws[0].base_map,
                    light_volume_draws[0].count,
                    light_volume_profiles[0].pos[0],
                    light_volume_profiles[0].pos[1],
                    light_volume_profiles[0].pos[2],
                );
            }
        }
        // Finalize per-batch world bounds: AABB center + (half-diagonal +
        // emitter bounding radius) so the sphere encloses every instance's
        // emission origin and the particles' authored reach. Empty batches
        // (no live instance this frame) get a zero-radius sphere → culled.
        // A negative radius marks a batch with no contributing instance
        // this frame (so the cull can skip it); a non-negative radius is a
        // real sphere. NOTE the loose-tag caveat: `bounding_radius_estimate`
        // is a tool-baked runtime field that is 0 in loose tags, so this
        // sphere only spans the emission ORIGINS, not the particles' reach.
        // The render cull is therefore opt-in (off by default) — see
        // `ParticleGpu::render`.
        batch_bounds.clear();
        batch_bounds.resize(batch_total, (glam::Vec3::ZERO, -1.0));
        for (b, acc) in bounds_acc.iter().enumerate() {
            if let Some((min, max, radius)) = acc {
                let center = (*min + *max) * 0.5;
                let spread = (*max - center).length();
                batch_bounds[b] = (center, spread + radius);
            }
        }
        // Diag: once per second, row-pool occupancy + top live consumers —
        // to catch global pool exhaustion starving late emitters.
        if std::env::var("PROTOMORPH_DIAG_POOL").is_ok() {
            diag::diag_pool(effects, instances, row_pool, pending_spawns, *time);
        }
        // Diag: tally instances-per-batch + spawns-this-frame-per-batch,
        // printed once near steady state, to isolate density (grid-cap vs
        // multi-instance collision vs under-emission).
        if std::env::var("PROTOMORPH_DIAG_DENSITY").is_ok() {
            diag::diag_density(effects, instances, pending_batches, batch_self_accel, *time, dt);
        }
    }

    /// The marker names of an effect's locations (for resolving the
    /// location matrices against the host model).
    pub fn effect_location_markers(&self, effect_index: usize) -> Vec<String> {
        self.effects
            .get(effect_index)
            .map(|e| {
                e.definition
                    .locations
                    .iter()
                    .map(|l| l.marker_name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Build the per-batch render descriptors (one per loaded particle
    /// system) — batch index → (render_info, bitmap paths). The renderer
    /// loads the bitmaps + registers grid regions/pipelines from this.
    pub fn batch_descriptors(&self) -> Vec<BatchDescriptor> {
        let mut out: Vec<BatchDescriptor> =
            vec![BatchDescriptor::default(); self.next_batch_index as usize];
        for sys in self.effects.iter().flat_map(|e| e.systems.iter()) {
            // One descriptor per emitter — its own curves + physics, sharing
            // the system's render material.
            for (ei, emitter) in sys.system.emitters.iter().enumerate() {
                let batch = sys.emitter_batches[ei];
                let Some(slot) = out.get_mut(batch as usize) else {
                    continue;
                };
                let eval_tables = particle_properties::compile_emitter(emitter, &sys.particle);
                if std::env::var("PROTOMORPH_DIAG_PARTICLES").is_ok() {
                    let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
                    let p = &eval_tables.properties;
                    let kind = |x: &particle_properties::EvalProperty| {
                        if x.a[0] != 0.0 { format!("const({:.2})", x.a[1]) }
                        else { format!("curve(fn{})", x.a[2] as u32) }
                    };
                    // Resolve the emitter_tint + particle_color RGB at
                    // position 0 (their color-index lo) to see the actual
                    // color the update multiplies in.
                    // The tint/color are GRADIENTS over particle age — show
                    // BOTH the lo (age 0) and hi (age 1) stops, else a fade-in
                    // gradient reads as a flat black at its age-0 stop (the
                    // snow "black tint" red herring).
                    let col_lo = |pp: &particle_properties::EvalProperty| {
                        eval_tables.colors.get(pp.c[0] as usize).copied().unwrap_or([0.0; 4])
                    };
                    let col_hi = |pp: &particle_properties::EvalProperty| {
                        eval_tables.colors.get(pp.c[1] as usize).copied().unwrap_or([0.0; 4])
                    };
                    let t0 = col_lo(&p[0]);
                    let t1 = col_hi(&p[0]);
                    let cc = col_lo(&p[3]);
                    eprintln!(
                        "[curves]   batch {batch} {leaf}: tint={} e_alpha={} pa={} | tint@age0=({:.2},{:.2},{:.2}) tint@age1=({:.2},{:.2},{:.2}) pcolor=({:.2},{:.2},{:.2})",
                        kind(&p[0]), kind(&p[1]), kind(&p[5]),
                        t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], cc[0], cc[1], cc[2],
                    );
                }
                // Engine adds a +0.25-turn sprite rotation when the bitmap
                // is authored vertically (appearance flag, resolved typed).
                let rotation_offset = if sys.particle.appearance_flags.contains(
                    blam_tags::particle::ParticleAppearanceFlags::BitmapAuthoredVertically,
                ) {
                    0.25
                } else {
                    0.0
                };
                // Pass the raw engine billboard_type enum (0..9); the VS
                // `billboard_basis` handles all of them (6/7 local approximate
                // to world axes).
                let billboard = std::env::var("PROTOMORPH_FORCE_BILLBOARD")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or_else(|| sys.particle.billboard_style.get() as i16 as u32);
                // Motion-blur aspect stretch: carry the prt3 scale only when
                // the `motion blur` appearance bit is set, else 0 (no stretch).
                let motion_blur_aspect_scale = if sys
                    .particle
                    .appearance_flags
                    .contains(blam_tags::particle::ParticleAppearanceFlags::MotionBlur)
                {
                    sys.particle.motion_blur_aspect_scale
                } else {
                    0.0
                };
                // Always-on near-camera fade (engine set_shader_render_state):
                // near_range = 1/near_fade_range (0 ⇒ disabled), near_cutoff =
                // override (bit 11) ? near_fade_override : near_fade_cutoff.
                // PROTOMORPH_FORCE_NEAR_FADE="range,cutoff" overrides for test.
                let (near_range, near_cutoff) = {
                    let nfr = sys.system.near_fade_range;
                    let cutoff = if sys
                        .system
                        .flags
                        .contains(ParticleSystemFlags::OverrideNearFade)
                    {
                        sys.system.near_fade_override
                    } else {
                        sys.system.near_fade_cutoff
                    };
                    let range = if nfr > 0.0 { 1.0 / nfr } else { 0.0 };
                    match std::env::var("PROTOMORPH_FORCE_NEAR_FADE").ok().and_then(|s| {
                        let mut it = s.split(',');
                        Some((it.next()?.trim().parse::<f32>().ok()?, it.next()?.trim().parse::<f32>().ok()?))
                    }) {
                        Some((r, c)) => (if r > 0.0 { 1.0 / r } else { 0.0 }, c),
                        None => (range, cutoff),
                    }
                };
                // Edge fade (appearance bit 7 `fade when viewed edge-on`):
                // edge_range = 1/radians(angle_fade_range) (0 ⇒ disabled),
                // edge_cutoff = radians(angle_fade_cutoff) — engine
                // `set_shader_render_state @0x1806a67c0` (`m_edge_fade =
                // 1/(deg2rad·angle_fade_range)`, `m_edge_cutoff =
                // deg2rad·angle_fade_cutoff`). Gated by the bit so non-edge-fade
                // particles carry 0 (disabled). Applied in the render VS as
                // `alpha *= saturate(edge_range·(billboard_angle − edge_cutoff))`.
                let edge_on = sys.particle.appearance_flags.contains(
                    blam_tags::particle::ParticleAppearanceFlags::FadeWhenViewedEdgeOn,
                );
                let (edge_range, edge_cutoff) = if edge_on
                    && std::env::var("PROTOMORPH_NO_EDGE_FADE").is_err()
                {
                    let r = sys.particle.angle_fade_range_degrees.to_radians();
                    let c = sys.particle.angle_fade_cutoff_degrees.to_radians();
                    (if r > 0.0 { 1.0 / r } else { 0.0 }, c)
                } else {
                    (0.0, 0.0)
                };
                // Diagnostic: which systems carry "fade when viewed edge-on"
                // (PROTOMORPH_DIAG_EDGE=1). Disable the whole term with
                // PROTOMORPH_NO_EDGE_FADE=1.
                if edge_on && std::env::var("PROTOMORPH_DIAG_EDGE").is_ok() {
                    let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
                    eprintln!(
                        "[edge-fade] {leaf} billboard={:?}({billboard}) range={:.1}° cutoff={:.1}° -> edge_range={edge_range:.3} edge_cutoff={edge_cutoff:.3}",
                        sys.particle.billboard_style.get(),
                        sys.particle.angle_fade_range_degrees,
                        sys.particle.angle_fade_cutoff_degrees,
                    );
                }
                if std::env::var("PROTOMORPH_DIAG_PARTICLES").is_ok()
                    && (billboard != 0 || motion_blur_aspect_scale > 0.0)
                {
                    let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
                    eprintln!(
                        "[billboard] {leaf} style={:?}({billboard}) motion_blur_aspect={motion_blur_aspect_scale:.3}",
                        sys.particle.billboard_style.get(),
                    );
                }
                use blam_tags::particle::ParticleAnimationFlags as Anim;
                use blam_tags::particle::ParticleAppearanceFlags as App;
                *slot = BatchDescriptor {
                    render_info: sys.render_info.clone(),
                    physics: sys.emitter_physics[ei],
                    eval_tables,
                    center_offset: [sys.particle.center_offset.x, sys.particle.center_offset.y],
                    rotation_offset,
                    billboard,
                    motion_blur_aspect_scale,
                    near_range,
                    near_cutoff,
                    edge_range,
                    edge_cutoff,
                    intensity_affects_alpha: sys
                        .particle
                        .appearance_flags
                        .contains(App::IntensityAffectsAlpha),
                    flip_u: sys.particle.appearance_flags.contains(App::RandomUMirror),
                    flip_v: sys.particle.appearance_flags.contains(App::RandomVMirror),
                    first_sequence_index: sys.particle.first_sequence_index,
                    anim_one_shot: sys.particle.animation_flags.contains(Anim::FrameAnimationOneShot),
                    anim_backwards: sys.particle.animation_flags.contains(Anim::CanAnimateBackwards),
                    tint_from_lightmap: sys.particle.appearance_flags.contains(App::TintFromLightmap),
                    tint_from_diffuse: sys.particle.appearance_flags.contains(App::TintFromDiffuseTexture),
                    // Engine `DontRenderSystem` (get_invisible, bit 7):
                    // created + updated but never drawn. Resolved once here
                    // (typed/name-resolved on the system) and threaded as a
                    // bool to the render side.
                    renders: !sys
                        .system
                        .flags
                        .contains(ParticleSystemFlags::DontRenderSystem),
                    valid: true,
                };
            }
        }
        out
    }

    /// Diagnostic — total particle systems across every live instance.
    pub fn live_system_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.effects[i.effect_index].systems.len())
            .sum()
    }

}

/// Gated diagnostic helpers extracted from [`EffectStore::frame_advance`].
///
/// Each `diag_*` fn reproduces the exact `eprintln!` output (and env-var
/// gate at the call site) of the inline block it replaced — pure
/// read-only over the frame-advance locals, no behavior change. Kept out
/// of the hot path so `frame_advance` reads as the real logic.
mod diag {
    use super::*;
    use crate::halo::effects::PARTICLE_ROW_COUNT;

    /// `PROTOMORPH_DIAG_LOD` — per-system distance/LOD band line. Gate
    /// (`is_ok() && *time < 0.2`) is at the call site.
    pub(super) fn diag_lod(
        sys: &LoadedParticleSystem,
        location_matrix: glam::Mat4,
        camera_pos: glam::Vec3,
        lod: f32,
    ) {
        let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
        let dist = (location_matrix.w_axis.truncate() - camera_pos).length();
        eprintln!(
            "[lod] {leaf} dist={dist:.1} lod_in={:.1} lod_out={:.1} feather_in={:.1} feather_out={:.1} always1={} -> lod={lod:.3}{}",
            sys.system.lod_in_distance,
            sys.system.lod_out_distance,
            sys.system.lod_feather_in_delta,
            sys.system.lod_feather_out_delta,
            sys.system.flags.contains(ParticleSystemFlags::LodAlways1),
            if lod <= 0.0 { " [RELEASED]" } else { "" },
        );
    }

    /// `PROTOMORPH_DIAG_ORIENT` — host/location/emitter basis dump. Gate
    /// (`is_ok() && *time < 0.2`) is at the call site.
    pub(super) fn diag_orient(
        sys: &LoadedParticleSystem,
        effect: &LoadedEffect,
        inst: &EffectInstance,
        loc_idx: usize,
        host_matrix: glam::Mat4,
        location_matrix: glam::Mat4,
    ) {
        let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
        let host_fwd = host_matrix.transform_vector3(glam::Vec3::X);
        let host_up = host_matrix.transform_vector3(glam::Vec3::Z);
        let loc_marker = effect
            .definition
            .locations
            .get(loc_idx)
            .map(|l| l.marker_name.clone())
            .unwrap_or_default();
        let is_id = inst
            .location_matrices
            .get(loc_idx)
            .map(|m| *m == glam::Mat4::IDENTITY);
        eprintln!(
            "[orient]   host_fwd(+X)=({:.2},{:.2},{:.2}) host_up(+Z)=({:.2},{:.2},{:.2}) attach_marker='{}' loc_marker='{}' loc_identity={:?}",
            host_fwd.x, host_fwd.y, host_fwd.z, host_up.x, host_up.y, host_up.z,
            inst.marker, loc_marker, is_id,
        );
        let loc_fwd = location_matrix.transform_vector3(glam::Vec3::X);
        let origin = location_matrix.w_axis;
        let emitter = sys.system.emitters.first();
        let rd = emitter
            .and_then(|e| e.relative_direction.starting_interpolant)
            .map(|d| [d.i, d.j, d.k]);
        let em_fwd = emitter
            .map(|e| {
                // Diag only needs the basis (transform_vector
                // ignores the offset translation), so age/random 0.
                particle_emitter::calc_emitter_matrix(location_matrix, e, 0.0, 0.0)
                    .transform_vector3(glam::Vec3::X)
            })
            .unwrap_or(loc_fwd);
        let ang = emitter
            .map(|e| particle_emitter::diag_emission_angle(e))
            .unwrap_or((0.0, 0.0, 0.0));
        let shape = emitter.map(|e| e.emission_shape.get());
        eprintln!(
            "[orient] {leaf} loc={} origin=({:.1},{:.1},{:.1}) loc_fwd(+X)=({:.2},{:.2},{:.2}) rel_dir={:?} EMIT_fwd=({:.2},{:.2},{:.2}) shape={:?} emit_angle[r0/.5/1]=({:.0},{:.0},{:.0})°",
            loc_idx, origin.x, origin.y, origin.z,
            loc_fwd.x, loc_fwd.y, loc_fwd.z, rd,
            em_fwd.x, em_fwd.y, em_fwd.z, shape, ang.0, ang.1, ang.2,
        );
    }

    /// `PROTOMORPH_DIAG_POOL` — once-per-second row-pool occupancy + top
    /// live consumers + per-batch gate detail. The once-per-second latch
    /// (`LAST`) is internal so the call site is just an `is_ok()` gate.
    pub(super) fn diag_pool(
        effects: &[LoadedEffect],
        instances: &[EffectInstance],
        row_pool: &RowPool,
        pending_spawns: &[ParticleState],
        time: f32,
    ) {
        static LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let sec = time as u32;
        if LAST.swap(sec, std::sync::atomic::Ordering::Relaxed) != sec {
            let mut live_per_batch = std::collections::HashMap::<u32, f32>::new();
            let mut leaf_per_batch = std::collections::HashMap::<u32, &str>::new();
            for inst in instances.iter() {
                for (si, sys) in effects[inst.effect_index].systems.iter().enumerate() {
                    for (ei, &batch) in sys.emitter_batches.iter().enumerate() {
                        *live_per_batch.entry(batch).or_default() +=
                            inst.emitter_runtimes[si][ei].live;
                        leaf_per_batch.entry(batch).or_insert_with(|| {
                            sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?")
                        });
                    }
                }
            }
            let total_live: f32 = live_per_batch.values().sum();
            let mut top: Vec<(u32, f32)> =
                live_per_batch.iter().map(|(k, v)| (*k, *v)).collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let summary: Vec<String> = top
                .iter()
                .take(8)
                .map(|(b, l)| format!("{}#{b}={l:.0}", leaf_per_batch[b]))
                .collect();
            eprintln!(
                "[pool] t={:.0}s free_rows={}/{} live={:.0} spawned={} | {}",
                time,
                row_pool.free.len(),
                PARTICLE_ROW_COUNT,
                total_live,
                pending_spawns.len(),
                summary.join(" "),
            );
            // Per-batch gate detail: which effect owns it, the emitter
            // age vs the event duration, looping, the rate eval right
            // now — to see exactly which gate stops an emitter.
            if sec == 3 || sec == 8 || sec == 18 || sec == 25 {
                let mut seen = std::collections::HashSet::<u32>::new();
                for inst in instances.iter() {
                    let effect = &effects[inst.effect_index];
                    let elooping = inst.looping
                        || effect
                            .definition
                            .flags
                            .contains(blam_tags::effect::EffectFlags::RunEventsInParallel)
                        || effect.definition.loop_start_event >= 0;
                    let eleaf = effect.path.rsplit(['\\', '/']).next().unwrap_or("?");
                    for (si, sys) in effect.systems.iter().enumerate() {
                        let dur = effect
                            .definition
                            .events
                            .get(sys.event_index)
                            .map(|e| e.duration_bounds.upper.max(e.duration_bounds.lower))
                            .unwrap_or(0.0);
                        for (ei, &batch) in sys.emitter_batches.iter().enumerate() {
                            if !seen.insert(batch) {
                                continue;
                            }
                            let rt = &inst.emitter_runtimes[si][ei];
                            let em = &sys.system.emitters[ei];
                            let rate = em
                                .particle_emission_rate
                                .function
                                .as_ref()
                                .map(|f| f.evaluate(rt.age, 0.0))
                                .unwrap_or(-1.0);
                            let life = em
                                .particle_lifespan
                                .function
                                .as_ref()
                                .map(|f| f.evaluate(rt.age, 0.5))
                                .unwrap_or(-1.0);
                            eprintln!(
                                "[gate] t={sec}s #{batch} {eleaf}/{} age={:.1} dur={dur:.1} loop={elooping} rate={rate:.1} life={life:.1} live={:.0} acc={:.2}",
                                leaf_per_batch[&batch], rt.age, rt.live, rt.accumulator,
                            );
                        }
                    }
                }
            }
        }
    }

    /// `PROTOMORPH_DIAG_DENSITY` — once-near-steady-state per-batch
    /// density/size/lifespan dump. The `time > 3.0` window + once-only
    /// latch (`DENSITY_DONE`) are internal so the call site is just an
    /// `is_ok()` gate.
    pub(super) fn diag_density(
        effects: &[LoadedEffect],
        instances: &[EffectInstance],
        pending_batches: &[u32],
        batch_self_accel: &[[f32; 4]],
        time: f32,
        dt: f32,
    ) {
        static DENSITY_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !(time > 3.0 && !DENSITY_DONE.swap(true, std::sync::atomic::Ordering::Relaxed)) {
            return;
        }
        let mut inst_per_batch = std::collections::HashMap::<u32, u32>::new();
        for inst in instances.iter() {
            for sys in effects[inst.effect_index].systems.iter() {
                for &batch in sys.emitter_batches.iter() {
                    *inst_per_batch.entry(batch).or_default() += 1;
                }
            }
        }
        let mut spawn_per_batch = std::collections::HashMap::<u32, u32>::new();
        for b in pending_batches.iter() {
            *spawn_per_batch.entry(*b).or_default() += 1;
        }
        let mut keys: Vec<u32> = inst_per_batch.keys().copied().collect();
        keys.sort();
        // Per-batch live count (Σ rt.live over instances) + the
        // emitter's evaluated max_count + a leaf name — to see what is
        // capping the waterfall population.
        let mut live_per_batch = std::collections::HashMap::<u32, f32>::new();
        let mut maxc_per_batch = std::collections::HashMap::<u32, f32>::new();
        let mut leaf_per_batch = std::collections::HashMap::<u32, String>::new();
        for inst in instances.iter() {
            for (si, sys) in effects[inst.effect_index].systems.iter().enumerate() {
                for (ei, &batch) in sys.emitter_batches.iter().enumerate() {
                    *live_per_batch.entry(batch).or_default() +=
                        inst.emitter_runtimes[si][ei].live;
                    let mc = particle_emitter::eval_max_count(
                        &sys.system.emitters[ei],
                        inst.emitter_runtimes[si][ei].age,
                    );
                    maxc_per_batch.insert(batch, mc);
                    leaf_per_batch.insert(
                        batch,
                        sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?").to_string(),
                    );
                }
            }
        }
        let mut sz_per_batch = std::collections::HashMap::<u32, (f32, f32, f32, f32, f32)>::new();
        for inst in instances.iter() {
            for sys in effects[inst.effect_index].systems.iter() {
                for (ei, &batch) in sys.emitter_batches.iter().enumerate() {
                    sz_per_batch.insert(
                        batch,
                        particle_emitter::diag_size_speed(&sys.system.emitters[ei]),
                    );
                }
            }
        }
        eprintln!("[density] dt={dt:.4} batch: live/maxc | size scale speed radius | self_accel|world|");
        for b in keys {
            let s = sz_per_batch.get(&b).copied().unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));
            let sa0 = batch_self_accel.get(b as usize * 2).copied().unwrap_or([0.0; 4]);
            let sa_mag = (sa0[0] * sa0[0] + sa0[1] * sa0[1] + sa0[2] * sa0[2]).sqrt();
            let (life_r, erad) = {
                let mut lr = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                let mut er = (0.0f32, 0.0f32);
                for inst in instances.iter() {
                    for sys in effects[inst.effect_index].systems.iter() {
                        for (ei, &bb) in sys.emitter_batches.iter().enumerate() {
                            if bb == b {
                                let em = &sys.system.emitters[ei];
                                lr = particle_emitter::diag_lifespan_range(em);
                                er = (
                                    particle_emitter::diag_eval(&em.emission_radius, 0.0, 0.0),
                                    particle_emitter::diag_eval(&em.emission_radius, 0.0, 1.0),
                                );
                            }
                        }
                    }
                }
                (lr, er)
            };
            eprintln!(
                "[density]   batch {b} {}: live={:.0} maxc={:.0} | scale={:.3} | RANGES life[{:.1}->{:.1}] size[{:.4}->{:.4}] vel[{:.2}->{:.2}] maxc[{:.0}->{:.0}] EMIT_RAD[{:.2}->{:.2}] | sa=({:.2},{:.2},{:.2})|{:.2}|",
                leaf_per_batch.get(&b).map(|s| s.as_str()).unwrap_or("?"),
                live_per_batch.get(&b).copied().unwrap_or(0.0),
                maxc_per_batch.get(&b).copied().unwrap_or(-1.0),
                s.2,
                life_r.0, life_r.1, life_r.2, life_r.3, life_r.4, life_r.5, life_r.6, life_r.7,
                erad.0, erad.1,
                sa0[0], sa0[1], sa0[2], sa_mag,
            );
        }
    }
}
