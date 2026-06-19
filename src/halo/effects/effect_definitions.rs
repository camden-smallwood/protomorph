//! Loaded effect tag-chain definitions — the resolved `.effect` /
//! `prt3` / `particle_physics` data an [`super::EffectStore`] holds
//! (Ares `effect_definitions.cpp`).
//!
//! These are the immutable, deduped-by-path results of following the tag
//! chain that drives an effect: the effect definition, every particle
//! system flattened across events (with its resolved `prt3` + render
//! state + per-emitter physics), and the decoded non-particle event parts.
//! The live-instance + simulation side lives in [`super::effects`].

use std::path::Path;

use blam_tags::effect::{
    EffectDefinition, EffectEvent, EffectEventFlags, ParticleSystemDefinition, ParticleSystemFlags,
};
use blam_tags::particle::ParticleDefinition;
use blam_tags::particle_physics::{ParticleMovementType, ParticlePhysics};
use blam_tags::paths::resolve_tag_path;
use blam_tags::render_method::RenderMethodChoices;
use blam_tags::TagFile;

use super::particle_emitter::{eval_constant, ParticlePhysicsParams};
use super::particle_system::ParticleRenderInfo;
use super::EffectStore;

/// One particle system inside a loaded effect, with its referenced
/// `prt3` definition resolved. Mirrors one
/// `events[*].particle_systems[*]` entry.
#[derive(Debug)]
pub struct LoadedParticleSystem {
    /// Index of the owning event in [`LoadedEffect::definition`]`.events`.
    pub event_index: usize,
    /// Tag-relative path of the `prt3` (for logging / dedup).
    pub particle_path: String,
    /// The effect-side particle-system block — carries the emitters,
    /// coordinate system, LOD/fade, sort bias.
    pub system: ParticleSystemDefinition,
    /// The resolved `prt3` definition — billboard style, color/alpha/
    /// intensity properties, and the embedded render_method (shader).
    pub particle: ParticleDefinition,
    /// Bitmap tag-paths referenced by the particle's shader. Collected
    /// now so Phase 4 can upload them; empty if the shader failed to
    /// resolve.
    pub bitmap_paths: Vec<String>,
    /// Per-emitter resolved smoke/template physics (gravity_mod / air /
    /// rot drag), one entry per emitter — the engine binds each emitter's
    /// own `UpdateState`. Feeds the GPU update kernel per batch.
    pub emitter_physics: Vec<ParticlePhysicsParams>,
    /// Render-method state (albedo path / blend / black_point / bitmaps).
    /// Per-system: all of a system's emitters share the prt3 shader; they
    /// differ only in curves/physics/emission.
    pub render_info: ParticleRenderInfo,
    /// Per-EMITTER global render-batch index (engine streams per emitter,
    /// not per system). Every spawn from emitter `ei` routes to
    /// `emitter_batches[ei]`'s grid region and draws with its material.
    /// Assigned in load order across all effects.
    pub emitter_batches: Vec<u32>,
}

/// One non-particle `event.parts[]` entry, resolved for the effect-part
/// multiplexer (engine `event_generate_part @0x180301C40` — the fourcc
/// switch that routes each part to its subsystem creator). protomorph
/// historically consumed ONLY `event.particle_systems`, silently dropping
/// every beam / contrail / light-volume / decal / light / sound / sub-effect
/// part. This carries the decoded part so the per-type handlers (Tracks R/E)
/// can act on it; particle (`prt3`) parts are excluded — those live in the
/// dedicated `particle_systems` block already flattened into [`LoadedEffect::
/// systems`].
#[derive(Debug, Clone)]
pub struct LoadedEffectPart {
    /// Index of the owning event in [`LoadedEffect::definition`]`.events`.
    pub event_index: usize,
    /// Decoded dispatch type (the multiplexer key).
    pub part_type: blam_tags::effect::EffectPartType,
    /// `type^` tag reference (the beam / light_volume / decal / light / …
    /// target tag), tag-relative path.
    pub type_tag_path: String,
    /// Block index into the effect's `locations[]` (-1 → default/first).
    pub location: i16,
    /// Part offset from its location origin (engine `relative_offset`).
    pub relative_offset: [f32; 3],
    /// Part orientation (yaw, pitch) relative to its location basis.
    pub relative_orientation: [f32; 2],
}

/// A fully-resolved effect tag — its definition plus every particle
/// system flattened across events.
#[derive(Debug)]
pub struct LoadedEffect {
    /// Tag-relative path of the `.effect` (dedup key).
    pub path: String,
    pub definition: EffectDefinition,
    pub systems: Vec<LoadedParticleSystem>,
    /// Non-particle event parts (beam/contrail/ltvl/decs/ligh/sefc/snd!/…),
    /// decoded by the [`LoadedEffectPart`] multiplexer. Empty for the common
    /// particle-only effect (the waterfall, weather, steam).
    pub parts: Vec<LoadedEffectPart>,
}

/// Engine-faithful particle-system **creation** eligibility, mirroring the
/// runtime gate in `c_particle_system::create` (dllcache `@0x180488510`:
/// `g_particle_create && (def->m_flags & 0x20)==0 && get_valid(def)`; Reach
/// `@0x82ece4d0` → `get_allowed_to_create` = `!get_disabled_for_debugging`).
/// Returns `false` for a system the engine would never instantiate, so the
/// caller skips it.
///
/// Honors the "disabled for debugging" toggle at BOTH levels the engine
/// does — the system definition (`ParticleSystemFlags`, runtime m_flags bit
/// 5) and the whole owning event (`EffectEventFlags` bit 0, which mutes all
/// its systems). These are tool-time switches artists leave set to silence
/// content in shipping; the runtime never creates them, but protomorph reads
/// the RAW loose tag where they're still present (e.g. guardian earth_edge's
/// `spark_ember` — the inward white embers that render in protomorph but NOT
/// in MCC/Sapien). blam-tags `Flags` is name-resolved from each tag's
/// embedded schema, so the older 2007-era bit layouts match by name.
///
/// `get_valid` (does the prt3 resolve?) and `g_particle_create` (always on)
/// are handled by the caller — it `continue`s on an empty/unloadable
/// particle path. Disposition / camera-mode / environment / splitscreen
/// gates (`effect_particle_system_allowed`) are gameplay/view-state
/// dependent and not relevant to protomorph's single static view yet.
pub(super) fn particle_system_eligible(
    event: &EffectEvent,
    system: &ParticleSystemDefinition,
) -> bool {
    !event.flags.contains(EffectEventFlags::DisabledForDebugging)
        && !system.flags.contains(ParticleSystemFlags::DisabledForDebugging)
}

/// Engine `c_particle_system_definition::calc_lod_amount @0x180567f30` — the
/// distance arm of the per-location LOD. Maps a camera distance (already
/// scaled by the camera's `field_of_view_scale`) to a [0,1] amount with a
/// near cutoff (`lod_in`), a feather-in ramp up to full, a full-LOD band, a
/// feather-out ramp, and a far cutoff (`lod_out`):
///
/// ```text
///  amount  0 |__       _________        __| 0
///            in↑      /         \      ↓out
///                in+feather   out-feather
/// ```
///
/// `LodAlways1` (m_flags&4) short-circuits to 1.0; `LodSameInSplitscreen`
/// (m_flags&8) skips the global `x_distance_lod_scale` (1.0 in single
/// player). Returns 1.0 when no LOD band is authored (`lod_out <= lod_in`,
/// the stripped/loose-tag default) so an un-authored system is never
/// silently culled.
pub fn system_lod_amount(sys: &ParticleSystemDefinition, distance: f32, dist_scale: f32) -> f32 {
    if sys.flags.contains(ParticleSystemFlags::LodAlways1) {
        return 1.0;
    }
    let lod_in = sys.lod_in_distance;
    let lod_out = sys.lod_out_distance;
    // No usable band (default 0/0, or inverted) → engine systems always
    // carry one; protomorph's loose tags may not, so don't cull.
    if lod_out <= lod_in {
        return 1.0;
    }
    let mut d = distance;
    if !sys.flags.contains(ParticleSystemFlags::LodSameInSplitscreen) {
        d *= dist_scale;
    }
    if d <= lod_in || d >= lod_out {
        return 0.0;
    }
    if d < lod_in + sys.lod_feather_in_delta {
        return (d - lod_in) * sys.inverse_lod_feather_in;
    }
    if d <= lod_out - sys.lod_feather_out_delta {
        return 1.0;
    }
    (lod_out - d) * sys.inverse_lod_feather_out
}

/// Walk a `particle_physics` template chain to its physics controller
/// and read the three constants (gravity_mod / air / rot drag). Returns
/// `None` if no physics controller is found in the chain.
pub(super) fn resolve_physics(
    pm: &ParticlePhysics,
    tags_root: &Path,
    depth: u32,
) -> Option<ParticlePhysicsParams> {
    if depth > 8 {
        return None;
    }
    // A physics controller at this authoring level wins.
    if let Some(ctrl) = pm
        .movements
        .iter()
        .find(|c| matches!(c.controller_type.get(), ParticleMovementType::Physics))
    {
        // Engine `set_shader_update_state @0x1806A65E0` initializes air/rot
        // friction to FLOAT_1_0 (1.0) and only overwrites them when the
        // movement def actually authors the property; gravity defaults to 0.
        // Since reaching here means a Physics controller (movement def)
        // exists, an ABSENT air/rot parameter defaults to 1.0 — NOT 0. (Our
        // old `unwrap_or(0.0)` left frictionless any emitter whose physics
        // omits drag.)
        let get = |id: i32, default: f32| {
            ctrl.parameters
                .iter()
                .find(|p| p.parameter_id == id)
                .map(|p| eval_constant(&p.property))
                .unwrap_or(default)
        };
        return Some(ParticlePhysicsParams {
            gravity_mod: get(0, 0.0),
            air_friction: get(1, 1.0),
            rot_friction: get(2, 1.0),
        });
    }
    // Otherwise inherit from the template.
    let template = pm.template.as_ref()?;
    let abs = resolve_tag_path(tags_root, template, "particle_physics");
    let tag = TagFile::read(&abs).ok()?;
    let next = ParticlePhysics::from_tag(&tag).ok()?;
    resolve_physics(&next, tags_root, depth + 1)
}

impl EffectStore {
    /// Load a render_method's rmdf and resolve its category choices BY
    /// NAME (the canonical blam-tags layer). Particles all share
    /// `shaders/particle.render_method_definition`. Returns empty choices
    /// (→ `from_shader` falls back to per-category defaults) if the rmdf
    /// can't be read.
    pub(super) fn load_rm_choices(
        rm: &blam_tags::render_method::RenderMethod,
        tags_root: &Path,
    ) -> RenderMethodChoices {
        let rmdf_abs =
            resolve_tag_path(tags_root, &rm.definition_path, "render_method_definition");
        match TagFile::read(&rmdf_abs)
            .ok()
            .and_then(|t| blam_tags::render_method::RenderMethodDefinition::from_tag(&t).ok())
        {
            Some(rmdf) => RenderMethodChoices::resolve(rm, &rmdf),
            None => {
                eprintln!(
                    "[particle] failed to load rmdf '{}' — render categories default",
                    rm.definition_path
                );
                RenderMethodChoices::default()
            }
        }
    }

    /// Read a `prt3` tag and pull the bitmap paths out of its embedded
    /// shader. Returns `None` if the tag can't be read/parsed.
    pub(super) fn load_particle(
        particle_rel: &str,
        tags_root: &Path,
    ) -> Option<(ParticleDefinition, Vec<String>)> {
        let abs = resolve_tag_path(tags_root, particle_rel, "particle");
        let tag = match TagFile::read(&abs) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[effects]   failed to read particle {}: {e}", abs.display());
                return None;
            }
        };
        let particle = match ParticleDefinition::from_tag(&tag) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[effects]   {particle_rel}: bad particle tag: {e}");
                return None;
            }
        };
        let bitmap_paths = particle
            .shader
            .as_ref()
            .map(|rm| {
                rm.parameters
                    .iter()
                    .filter(|p| !p.bitmap_path.is_empty())
                    .map(|p| p.bitmap_path.clone())
                    .collect()
            })
            .unwrap_or_default();
        Some((particle, bitmap_paths))
    }
}
