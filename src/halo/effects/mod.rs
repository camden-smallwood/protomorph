//! Effect / particle subsystem — protomorph's runtime side of the
//! engine `c_effect` / `c_particle_system` pipeline.
//!
//! **Phase 1 (this file's current scope): attachment loading + effect
//! instances.** We follow the tag chain that drives a scenery-attached
//! looping effect:
//!
//! ```text
//! scenery (model = null_up, invisible host)
//!   └─ object/attachments[i] (type_group == b"effe") → .effect
//!        └─ events[*]/particle systems[*] → .particle (prt3)
//!             └─ actual shader? (embedded render_method) → bitmaps
//! ```
//!
//! The riverworld waterfall (`waterfall_base/mid/top.scenery`) is the
//! baseline: each scenery's effe attachment fires one looping event of
//! 2–3 particle systems (`rolling_mist`, `water_spray`, `mist`).
//!
//! Phase 1 only *resolves and stores* this data + creates one
//! [`EffectInstance`] per placed host object. Simulation (the
//! `RawParticleState` GPU grid + spawn/update compute) lands in Phase 2,
//! billboard render in Phase 4. The store is intentionally render-API
//! agnostic so the GPU backend can attach behind it.
//!
//! Engine references (verified addresses in
//! `reference_particle_system_engine_deep_dive`):
//! `attachments_new @0x1807E2F60` (effe dispatch),
//! `effect_new_from_object @0x1802fb050`, `effects_update @0x1802fc080`.

use std::collections::HashMap;
use std::path::Path;

use blam_tags::effect::{
    EffectEvent, EffectEventFlags, EffectDefinition, EffectPartType, ParticleSystemDefinition,
    ParticleSystemFlags,
};
use blam_tags::particle::ParticleDefinition;
use blam_tags::particle_physics::{ParticleMovementType, ParticlePhysics};
use blam_tags::paths::resolve_tag_path;
use blam_tags::render_method::{AlphaBlendMode, RenderMethodChoices};
use blam_tags::TagFile;
use std::str::FromStr;

pub mod particle_curves;
pub mod particle_emitter;

use particle_emitter::{
    eval_constant, pulse_emitter, EmitterRuntime, ParticlePhysicsParams, ParticleState,
};

/// Persistent particle grid capacity — engine `x_overall_storage` =
/// 16 cols × 448 rows = 7168 `RawParticleState`.
pub const PARTICLE_GRID_SIZE: u32 = 7168;
/// Grid rows in the pool (engine 448-row `s_gpu_buffer`).
pub const PARTICLE_ROW_COUNT: u32 = 448;
/// Particles per row (engine `s_row`, 16-wide).
pub const PARTICLE_ROW_WIDTH: u32 = 16;
/// `row_batch` sentinel — a row that owns no live emitter (free / dead).
/// The update kernel skips any slot whose row reads this.
pub const ROW_BATCH_FREE: u32 = 0xFFFF_FFFF;
/// Max particles spawned per frame — engine `x_queued_buffer_system`
/// cap (576).
pub const MAX_SPAWN_PER_FRAME: u32 = 576;

/// The engine grid allocator (`particle_block.{h,cpp}`): a global pool of
/// 448 rows, each 16 particles wide, handed out on demand to emitters.
/// A particle keeps its slot for life (no overwrite); a whole row retires
/// at once when its lifespan countdown expires. This replaces the old
/// fixed-even-region ring, whose `cursor % region_len` overwrote live
/// particles once a region filled.
#[derive(Debug)]
pub struct RowPool {
    /// Per-row state, indexed by row 0..448.
    rows: Vec<RowState>,
    /// Free row indices (LIFO).
    free: Vec<u16>,
    /// Per-row owning render batch (or [`ROW_BATCH_FREE`]). Uploaded to
    /// the GPU each frame so the update kernel routes each slot's
    /// physics/curves by `row_batch[slot / 16]`.
    batch: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct RowState {
    /// Particles placed in this row so far (0..16). Slots fill
    /// right-to-left: `slot = 16·(row+1) − used`.
    used: u8,
    /// Retirement countdown (seconds). Set to the longest-lived member's
    /// lifespan on each insert (engine `m_lifespan = max`); the row frees
    /// when this goes negative — i.e. once its youngest member has died.
    lifespan: f32,
}

impl Default for RowPool {
    fn default() -> Self {
        let n = PARTICLE_ROW_COUNT as usize;
        Self {
            rows: vec![RowState { used: 0, lifespan: 0.0 }; n],
            // Pop ascending (0,1,2,…) for readable layouts.
            free: (0..PARTICLE_ROW_COUNT as u16).rev().collect(),
            batch: vec![ROW_BATCH_FREE; n],
        }
    }
}

impl RowPool {
    /// Decrement and free this emitter's expired rows (engine
    /// `frame_advance`: `row.lifespan -= dt; if <0 destroy`). Returns
    /// each freed row to the pool and marks its batch slot free.
    fn retire(&mut self, emitter_rows: &mut Vec<u16>, dt: f32) {
        let mut w = 0;
        for i in 0..emitter_rows.len() {
            let r = emitter_rows[i];
            let st = &mut self.rows[r as usize];
            st.lifespan -= dt;
            if st.lifespan < 0.0 {
                st.used = 0;
                self.free.push(r);
                self.batch[r as usize] = ROW_BATCH_FREE;
            } else {
                emitter_rows[w] = r;
                w += 1;
            }
        }
        emitter_rows.truncate(w);
    }

    /// Allocate a grid slot for one new particle (engine `allocate_particle`
    /// → `s_row`): fill the head row right-to-left, grabbing a fresh row
    /// from the pool when it's full. Returns the flat grid slot, or `None`
    /// when the pool is exhausted (≥7168 live particles → drop the spawn).
    fn alloc(&mut self, emitter_rows: &mut Vec<u16>, lifespan: f32, batch: u32) -> Option<u32> {
        let head = emitter_rows
            .last()
            .copied()
            .filter(|&r| self.rows[r as usize].used < PARTICLE_ROW_WIDTH as u8);
        let row = match head {
            Some(r) => r,
            None => {
                let r = self.free.pop()?;
                self.rows[r as usize] = RowState { used: 0, lifespan: 0.0 };
                self.batch[r as usize] = batch;
                emitter_rows.push(r);
                r
            }
        };
        let st = &mut self.rows[row as usize];
        st.used += 1;
        st.lifespan = st.lifespan.max(lifespan);
        // slot = 16·(row+1) − used (fills right-to-left within the row).
        Some(PARTICLE_ROW_WIDTH * (row as u32 + 1) - st.used as u32)
    }

    /// Force-free every row this emitter owns and clear its list — the
    /// CPU mirror of `c_particle_emitter_gpu::clear` (called from
    /// `release_particles @0x180569370` when a location's LOD hits 0). The
    /// rows return to the shared pool immediately so a nearer effect can
    /// claim them, and the now-empty emitter restarts fresh on return.
    /// Unlike [`Self::retire`] this ignores per-row lifespan — it drops
    /// every live particle the emitter has.
    fn release_rows(&mut self, emitter_rows: &mut Vec<u16>) {
        for &r in emitter_rows.iter() {
            let st = &mut self.rows[r as usize];
            st.used = 0;
            st.lifespan = 0.0;
            self.free.push(r);
            self.batch[r as usize] = ROW_BATCH_FREE;
        }
        emitter_rows.clear();
    }

    /// The per-row batch table (one `u32` per row) for GPU upload.
    pub fn batch_table(&self) -> &[u32] {
        &self.batch
    }

    /// The emitter's true live-particle count = Σ row `used` over its rows
    /// (engine emitter_gpu live count at +0x14). Includes particles that
    /// have died but whose row hasn't retired yet — matching the engine,
    /// whose count decrements at row-retirement granularity.
    fn live_count(&self, emitter_rows: &[u16]) -> f32 {
        emitter_rows
            .iter()
            .map(|&r| self.rows[r as usize].used as f32)
            .sum()
    }

    /// Append one `(base_instance, count)` draw span per occupied row, for
    /// drawing exactly this emitter's particles (engine
    /// `c_particle_emitter_gpu::render`: per row, `used` instances starting
    /// at slot `16*(row+1)-used`). Returns the number of spans appended.
    fn append_spans(&self, emitter_rows: &[u16], out: &mut Vec<(u32, u32)>) -> u32 {
        let mut n = 0;
        for &r in emitter_rows {
            let used = self.rows[r as usize].used as u32;
            if used == 0 {
                continue;
            }
            let base = PARTICLE_ROW_WIDTH * (r as u32 + 1) - used;
            out.push((base, used));
            n += 1;
        }
        n
    }
}

/// Particle `albedo` category option, resolved BY NAME from the
/// render-method category (never by raw option index — see
/// [`ParticleRenderInfo::from_shader`]). The discriminant is the GPU
/// contract with `particle_render.wgsl::sample_diffuse`; it is held in
/// `shaders/particle.render_method_definition` albedo-category order so
/// the shader's index checks stay valid, but it is *derived* from the
/// option name, not the authored position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ParticleAlbedo {
    #[default]
    DiffuseOnly = 0,
    DiffusePlusBillboardAlpha = 1,
    Palettized = 2,
    PalettizedPlusBillboardAlpha = 3,
    DiffusePlusSpriteAlpha = 4,
    PalettizedPlusSpriteAlpha = 5,
}

impl ParticleAlbedo {
    fn from_name(name: &str) -> Self {
        match name {
            "diffuse_only" => Self::DiffuseOnly,
            "diffuse_plus_billboard_alpha" => Self::DiffusePlusBillboardAlpha,
            "palettized" => Self::Palettized,
            "palettized_plus_billboard_alpha" => Self::PalettizedPlusBillboardAlpha,
            "diffuse_plus_sprite_alpha" => Self::DiffusePlusSpriteAlpha,
            "palettized_plus_sprite_alpha" => Self::PalettizedPlusSpriteAlpha,
            "" => Self::DiffuseOnly, // category absent → first option
            other => {
                eprintln!(
                    "[particle] unknown albedo option '{other}' — defaulting to diffuse_only"
                );
                Self::DiffuseOnly
            }
        }
    }

    /// True when the albedo path needs a palette gradient lookup.
    pub fn is_palettized(self) -> bool {
        matches!(
            self,
            Self::Palettized | Self::PalettizedPlusBillboardAlpha | Self::PalettizedPlusSpriteAlpha
        )
    }
}

/// The render-method state a particle system needs to shade correctly,
/// resolved from the embedded `c_render_method` (category options +
/// bitmap parameters) BY NAME via blam-tags [`RenderMethodChoices`].
/// Drives the billboard pass's sampling/blend.
///
/// All category fields are name-resolved — the authored option *index*
/// is per-rmdf and unstable (e.g. `pre_multiplied_alpha` is option 5 in
/// `shader.rmdf` but option 10 in `particle.rmdf`, and `blend_mode` is a
/// different category number per rmdf), so positional decode is wrong by
/// construction. Names are the only stable key.
#[derive(Debug, Clone, Default)]
pub struct ParticleRenderInfo {
    /// `albedo` category option (name-resolved).
    pub albedo: ParticleAlbedo,
    /// `blend_mode` category option, resolved by name to the runtime
    /// `e_alpha_blend_mode`. Drives both the wgpu blend state and the
    /// shader's non-linear output conversion.
    pub blend_mode: AlphaBlendMode,
    /// `black_point` option != off → apply `remap_alpha`.
    pub black_point: bool,
    /// `depth_fade` category (5) != off → apply the soft depth fade.
    /// Engine gates the soft fade on this option; the riverworld
    /// water_spray has it OFF, so applying it unconditionally wrongly
    /// thins the spray near the cliff/pool. (Particle rmdf category order,
    /// resolved by name: 0 albedo, 1 blend_mode, 2 specialized_rendering,
    /// 3 lighting, 4 render_targets, 5 depth_fade, 6 black_point, 7 fog,
    /// 8 frame_blend, 9 self_illumination.)
    pub depth_fade: bool,
    /// `specialized_rendering` category (2) != normal → a distortion
    /// variant. Distortion writes a screen-space warp (not implemented),
    /// so distortion particles must be skipped, not drawn as color — else
    /// their warp/normal bitmap renders as rainbow garbage (man cannon
    /// `distortion_shimmer`).
    pub distortion: bool,
    /// `frame_blend` category (8) != off → cross-fade consecutive
    /// sprite-sheet frames (engine lerps `sample_diffuse` of frame N and
    /// N+1 by `frac(variant)`).
    pub frame_blend: bool,
    /// `fog` category (7) != off → per-vertex analytic atmosphere fog
    /// (engine `particle_render_hlsl.hlsl:673` `IF_CATEGORY_OPTION(fog,
    /// on)`: `compute_scattering` → `m_color ×= extinction; m_color_add
    /// += inscatter·v_exposure`). This is what blends fogged particles
    /// (s3d_turf `waterdrops_falling`, weather snow) into the scene —
    /// without it they punch through fog banks as crisp unfogged sprites.
    pub fog: bool,
    /// `base_map` parameter bitmap (rgb, or palette index for
    /// palettized).
    pub base_map: String,
    /// `alpha_map` parameter bitmap (the billboard-UV alpha source).
    pub alpha_map: String,
    /// `palette` gradient bitmap (palettized albedo paths only).
    pub palette: Option<String>,
}

impl ParticleRenderInfo {
    /// Resolve from a particle's embedded render_method + its rmdf
    /// category choices (name-resolved via blam-tags
    /// [`RenderMethodChoices`]). Bitmap parameters come straight off the
    /// rmsh's per-instance parameter list.
    fn from_shader(
        shader: &blam_tags::render_method::RenderMethod,
        choices: &RenderMethodChoices,
    ) -> Self {
        let bitmap = |name: &str| {
            shader
                .parameters
                .iter()
                .find(|p| p.parameter_name == name && !p.bitmap_path.is_empty())
                .map(|p| p.bitmap_path.clone())
        };
        // `blend_mode` → runtime enum BY NAME (order-/drift-proof).
        let blend_mode = match choices.get("blend_mode") {
            Some(name) => AlphaBlendMode::from_str(name).unwrap_or_else(|_| {
                eprintln!("[particle] unknown blend_mode option '{name}' — defaulting to opaque");
                AlphaBlendMode::Opaque
            }),
            None => AlphaBlendMode::Opaque, // no blend_mode category (rare)
        };
        Self {
            albedo: ParticleAlbedo::from_name(choices.get_or("albedo", "")),
            blend_mode,
            black_point: choices.get_or("black_point", "off") != "off",
            depth_fade: choices.get_or("depth_fade", "off") != "off",
            // `specialized_rendering` != none → a distortion variant.
            distortion: choices.get_or("specialized_rendering", "none") != "none",
            frame_blend: choices.get_or("frame_blend", "off") != "off",
            fog: choices.get_or("fog", "off") != "off",
            base_map: bitmap("base_map").unwrap_or_default(),
            alpha_map: bitmap("alpha_map").unwrap_or_default(),
            palette: bitmap("palette"),
        }
    }

    /// True when the albedo path needs a palette gradient lookup
    /// (`palettized*`).
    pub fn is_palettized(&self) -> bool {
        self.albedo.is_palettized()
    }
}

/// A render-batch description handed to the GPU particle subsystem —
/// one per EMITTER (engine streams per emitter), indexed by batch index.
#[derive(Debug, Clone)]
pub struct BatchDescriptor {
    pub render_info: ParticleRenderInfo,
    /// This system's resolved physics — routed per-region by the update
    /// kernel so each emitter integrates with its own gravity/drag.
    pub physics: ParticlePhysicsParams,
    /// Compiled GPU curve-evaluation tables for this emitter — the update
    /// kernel re-evaluates color/size/alpha/etc. from these each frame.
    pub eval_tables: particle_curves::EmitterEvalTables,
    /// prt3 `center_offset` — shifts the baked sprite registration point
    /// (engine `postprocess_frame_animation`: reg.x+=co.x, reg.y-=co.y).
    pub center_offset: [f32; 2],
    /// Extra in-plane rotation (turns) the render adds to every sprite —
    /// `0.25` when the prt3 is authored "bitmap authored vertically",
    /// resolved CPU-side from the typed appearance flag.
    pub rotation_offset: f32,
    /// Billboard basis selector for the render — the raw engine
    /// `particle_billboard_type_enum` value (0..9) from the typed
    /// `ParticleBillboardStyle`, driving `billboard_basis` in the VS:
    /// 0 screen-facing, 1 camera-facing, 2 screen-parallel (velocity
    /// streaks), 3 screen-perpendicular, 4 screen-vertical, 5 screen-
    /// horizontal, 6 local-vertical, 7 local-horizontal, 8 world, 9
    /// velocity-horizontal. (6/7 approximate to world axes in the VS —
    /// the per-emitter local basis isn't threaded to the render yet.)
    pub billboard: u32,
    /// Motion-blur aspect-stretch scale (engine prt3 `motion_blur_aspect
    /// _scale`, used as `aspect += |rel_vel|·scale/(30·size)` when the
    /// `motion blur` appearance bit is set — particle_render_hlsl.hlsl:588).
    /// 0.0 when the bit is off (or the scale is 0): no stretch. Stretches
    /// fast particles along their motion (sparks/rain/debris streak).
    pub motion_blur_aspect_scale: f32,
    /// Always-on near-camera fade (engine set_shader_render_state
    /// @0x1806a67c0). `near_range = 1/near_fade_range` (0 = disabled);
    /// `near_cutoff = near_fade_cutoff` (or `near_fade_override` when the
    /// system's bit-11 `override near fade` flag is set). The render does
    /// `alpha *= saturate(near_range·(depth−near_cutoff))` so particles
    /// dissolve as the camera gets close instead of filling the lens.
    pub near_range: f32,
    pub near_cutoff: f32,
    /// Edge fade (appearance bit 7 `fade when viewed edge-on`, engine
    /// `set_shader_render_state @0x1806a67c0` + `particle_render.hlsl:714`).
    /// `edge_range = 1/radians(angle_fade_range)` (0 = disabled — only set
    /// when the bit is on); `edge_cutoff = radians(angle_fade_cutoff)`. The VS
    /// fades alpha as the billboard turns edge-on to the view.
    pub edge_range: f32,
    pub edge_cutoff: f32,
    /// Appearance `intensity affects alpha` (bit 6) — engine
    /// `particle_render_hlsl.hlsl:701-703` multiplies the output alpha by
    /// `m_intensity`. Resolved by-name (drift-proof) from the typed
    /// appearance flag.
    pub intensity_affects_alpha: bool,
    /// Appearance `random u mirror` / `random v mirror` (bits 0/1) — engine
    /// `particle_render_hlsl.hlsl:623-630` randomly flips each sprite's u/v
    /// per particle (a coin off `m_random`). By-name resolved.
    pub flip_u: bool,
    pub flip_v: bool,
    /// prt3 `first sequence index` — selects which of the base bitmap's
    /// sprite-sheet sequences this particle animates through. The frame
    /// UV rects are baked from that sequence at batch registration.
    pub first_sequence_index: i16,
    /// Sprite-sheet animation flags (engine `_frame_animation_one_shot_bit`
    /// / `_can_animate_backwards_bit`) — drive `compute_variant`.
    pub anim_one_shot: bool,
    pub anim_backwards: bool,
    /// Appearance `tint from lightmap` (bit3) / `tint from diffuse` (bit4)
    /// — gate the engine `initialize_particle` lightprobe tint of
    /// `m_initial_color` (resolved at register time from the scene
    /// lightprobe DC). Both off → white.
    pub tint_from_lightmap: bool,
    pub tint_from_diffuse: bool,
    /// Engine `DontRenderSystem` flag (`c_particle_system_definition::
    /// get_invisible`, bit 7) INVERTED: `false` for a system the engine
    /// creates + frame-advances but never draws (it exists only to drive
    /// event timing / spawn sub-effects / play sound). Such a batch still
    /// spawns + updates on the GPU; it's just skipped at render (no
    /// transparent element, no draw). Default `true`.
    pub renders: bool,
    /// `false` for batch slots that never got a real system (shouldn't
    /// happen, but keeps indexing total).
    pub valid: bool,
}

impl Default for BatchDescriptor {
    fn default() -> Self {
        Self {
            render_info: ParticleRenderInfo::default(),
            physics: ParticlePhysicsParams::default(),
            eval_tables: particle_curves::EmitterEvalTables::default(),
            center_offset: [0.0, 0.0],
            rotation_offset: 0.0,
            billboard: 0,
            motion_blur_aspect_scale: 0.0,
            near_range: 0.0,
            near_cutoff: 0.0,
            edge_range: 0.0,
            edge_cutoff: 0.0,
            intensity_affects_alpha: false,
            flip_u: false,
            flip_v: false,
            first_sequence_index: 0,
            anim_one_shot: false,
            anim_backwards: false,
            tint_from_lightmap: false,
            tint_from_diffuse: false,
            renders: true,
            valid: false,
        }
    }
}

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

/// Render-method state a light volume (`rmlv`) needs to shade — resolved
/// by name from the embedded `c_render_method_shader_light_volume`. Far
/// simpler than the particle path: light volumes only have blend_mode, fog,
/// and a single `diffuse_only` albedo `base_map` (engine `light_volume_fx
/// .hlsl` PS samples just `base_map`).
#[derive(Debug, Clone, Default)]
pub struct LightVolumeRenderInfo {
    pub blend_mode: AlphaBlendMode,
    pub fog: bool,
    pub base_map: String,
}

/// A resolved `light_volume_system` (`ltvl`) tag — the first definition's
/// shader + the 8 authored profile curves. The strip render
/// (Track R1 increment 2) consumes this: it stacks `profile_density`
/// camera-facing-but-roll-locked quads along `origin + direction·(offset +
/// i·profile_distance)`, each profile's thickness/color/alpha/intensity
/// evaluated from the curves at its percentile (engine `light_volume_fx
/// .hlsl`). blam-tags decodes the whole 380B definition.
#[derive(Debug, Clone)]
pub struct LoadedLightVolume {
    /// The first `light_volumes[]` definition (the common single-def case).
    pub definition: blam_tags::light_volume_system::LightVolumeDefinition,
    pub render: LightVolumeRenderInfo,
    /// `base_map` bitmap tag-paths referenced by the rmlv shader.
    pub bitmap_paths: Vec<String>,
}

/// One light-volume profile (a cross-section quad) as the GPU strip render
/// consumes it. Per-profile fields (position/color/intensity/thickness/alpha)
/// are evaluated CPU-side from the curves at the profile's percentile; the
/// per-volume fields (direction / profile_length / profile_distance /
/// num_profiles / brightness_ratio) are denormalized into every profile so a
/// whole strip draws as one instanced call. Mirrors `s_profile_state` +
/// `g_all_state` from `light_volume_fx.hlsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightVolumeProfileGpu {
    /// xyz = world position (origin + dir·(offset + i·profile_distance)); w unused.
    pub pos: [f32; 4],
    /// rgb = profile color, a = intensity.
    pub color_intensity: [f32; 4],
    /// xyz = shaft direction (normalized), w = profile thickness.
    pub dir_thickness: [f32; 4],
    /// profile_length, profile_distance, num_profiles, brightness_ratio.
    pub params: [f32; 4],
    /// x = profile alpha; yzw pad.
    pub alpha: [f32; 4],
}

/// A contiguous run of [`LightVolumeProfileGpu`] sharing a base_map + blend —
/// one draw of the strip pipeline (4 verts × `count` profile instances).
#[derive(Debug, Clone)]
pub struct LightVolumeDraw {
    /// `base_map` bitmap path (the draw's texture; also the group key).
    pub base_map: String,
    pub blend_mode: AlphaBlendMode,
    pub fog: bool,
    /// `[first .. first+count)` into [`EffectStore::light_volume_profiles`].
    pub first: u32,
    pub count: u32,
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
}

/// Holds every loaded effect (deduped by path) and all live instances.
/// One per [`crate::game::GameState`].
/// One transparent-sort + draw unit = a single emitter of a single live
/// instance, mirroring the engine's PER-EMITTER `c_particle_emitter::submit
/// @0x180568FA0` (one `add_element` per emitter at `get_position_world`) +
/// `c_particle_emitter_gpu::render @0x1806A5140` (draws only THAT emitter's
/// rows). Replaces the old per-BATCH registration whose single averaged
/// centroid mis-sorted multi-instance systems (11 s3d_turf steam vents share
/// one batch → one global-average depth → the fence/drips wouldn't blend).
/// Rebuilt every frame in [`EffectStore::frame_advance`].
#[derive(Debug, Clone)]
pub struct EmitterDraw {
    /// World emission origin (engine `get_position_world` = emitter
    /// `m_matrix.origin`) — the sort point.
    pub sort_pos: [f32; 3],
    /// Render batch (material/pipeline/eval) this emitter draws with.
    pub batch: u16,
    /// `add_element` offset = `sort_bias * 0.05` (engine `v27`).
    pub offset: f32,
    /// Emitter bounding radius (for the opt-in per-emitter frustum cull).
    pub radius: f32,
    /// Authored transparent sort layer (the particle shader's `sort_layer`,
    /// engine `v32->m_shader.m_sort_layer`), as a [`TransparentSortLayer`]
    /// discriminant.
    pub sort_layer: u8,
    /// `[span_start..span_start+span_count]` slice into
    /// [`EffectStore::emitter_draw_spans`] — one `(base_instance, count)`
    /// per occupied grid row (engine per-row draw: `count` instances from
    /// slot `16*(row+1)-used`).
    pub span_start: u32,
    pub span_count: u32,
}

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
fn particle_system_eligible(event: &EffectEvent, system: &ParticleSystemDefinition) -> bool {
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
fn system_lod_amount(sys: &ParticleSystemDefinition, distance: f32, dist_scale: f32) -> f32 {
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

    /// Load a render_method's rmdf and resolve its category choices BY
    /// NAME (the canonical blam-tags layer). Particles all share
    /// `shaders/particle.render_method_definition`. Returns empty choices
    /// (→ `from_shader` falls back to per-category defaults) if the rmdf
    /// can't be read.
    fn load_rm_choices(
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
    fn load_particle(
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

    /// Read a `light_volume_system` (`ltvl`) tag and resolve its first
    /// definition + render state (the `rmlv` shader's blend_mode / fog /
    /// base_map, name-resolved). Returns `None` if the tag can't be
    /// read/parsed. Mirrors [`Self::load_particle`].
    fn load_light_volume_def(ltvl_rel: &str, tags_root: &Path) -> Option<LoadedLightVolume> {
        let abs = resolve_tag_path(tags_root, ltvl_rel, "light_volume_system");
        let tag = match TagFile::read(&abs) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[effects]   failed to read ltvl {}: {e}", abs.display());
                return None;
            }
        };
        let system = match blam_tags::light_volume_system::LightVolumeSystem::from_tag(&tag) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[effects]   {ltvl_rel}: bad light_volume_system: {e}");
                return None;
            }
        };
        // The common case is a single definition; take the first.
        let definition = system.definitions.into_iter().next()?;
        let render = definition
            .shader
            .as_ref()
            .map(|rm| {
                let choices = Self::load_rm_choices(rm, tags_root);
                // Light shafts are additive by default (the engine's typical
                // rmlv blend) when the category is absent/unknown.
                let blend_mode = match choices.get("blend_mode") {
                    Some(name) => AlphaBlendMode::from_str(name).unwrap_or(AlphaBlendMode::Additive),
                    None => AlphaBlendMode::Additive,
                };
                let base_map = rm
                    .parameters
                    .iter()
                    .find(|p| p.parameter_name == "base_map" && !p.bitmap_path.is_empty())
                    .map(|p| p.bitmap_path.clone())
                    .unwrap_or_default();
                LightVolumeRenderInfo {
                    blend_mode,
                    fog: choices.get_or("fog", "off") != "off",
                    base_map,
                }
            })
            .unwrap_or_default();
        let bitmap_paths = definition
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
        Some(LoadedLightVolume { definition, render, bitmap_paths })
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
    ) {
        let instance_index = self.instances.len();
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
        self.instances.push(EffectInstance {
            effect_index,
            host_header_index,
            marker,
            origin,
            location_matrices,
            emitter_runtimes,
            primary_scale,
            looping,
            weather: false,
        });
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
                if std::env::var("PROTOMORPH_DIAG_ORIENT").is_ok() && *time < 0.2 {
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
                for (ei, emitter) in sys.system.emitters.iter().enumerate() {
                    // Max-min fair share of the remaining budget. Every
                    // emitter consumes a slot of `remaining_emitters` even
                    // when its share is 0, so the divisor shrinks and unused
                    // budget concentrates on the emitters that want it.
                    let fair_share = if remaining_emitters > 0 {
                        remaining_budget.div_ceil(remaining_emitters)
                    } else {
                        0
                    };
                    remaining_emitters = remaining_emitters.saturating_sub(1);
                    let batch = sys.emitter_batches[ei];
                    // Self-acceleration interpolants → world (× this
                    // emitter's matrix). The update slerps the two endpoints
                    // by the property-11 curve and adds ·dt to velocity
                    // (engine vector-property path). Rotation preserves the
                    // slerp, so baking world endpoints is faithful. Per
                    // emitter — its own batch slot.
                    // Only the basis is used here (transform_vector ignores
                    // the offset translation), so age/random 0 is fine.
                    let em = particle_emitter::calc_emitter_matrix(location_matrix, emitter, 0.0, 0.0);
                    let to_world = |v: Option<blam_tags::math::RealVector3d>| {
                        let v = v.unwrap_or(blam_tags::math::RealVector3d { i: 0.0, j: 0.0, k: 0.0 });
                        let w = em.transform_vector3(glam::Vec3::new(v.i, v.j, v.k));
                        [w.x, w.y, w.z, 0.0]
                    };
                    let b = batch as usize;
                    // Accumulate this emitter's world origin + bounding
                    // radius into the batch's render bounds (engine
                    // submit() cull sphere = emitter world pos + def
                    // radius). Override beats estimate when authored.
                    if b < bounds_acc.len() {
                        let origin = location_matrix.w_axis.truncate();
                        let radius = if emitter.bounding_radius_override > 0.0 {
                            emitter.bounding_radius_override
                        } else {
                            emitter.bounding_radius_estimate
                        };
                        let acc = bounds_acc[b]
                            .get_or_insert((origin, origin, 0.0));
                        acc.0 = acc.0.min(origin);
                        acc.1 = acc.1.max(origin);
                        acc.2 = acc.2.max(radius);
                    }
                    if b * 2 + 1 < batch_self_accel.len() {
                        batch_self_accel[b * 2] =
                            to_world(emitter.particle_self_acceleration.starting_interpolant);
                        batch_self_accel[b * 2 + 1] =
                            to_world(emitter.particle_self_acceleration.ending_interpolant);
                    }
                    let rt = &mut inst.emitter_runtimes[si][ei];
                    // Engine `c_particle_location::pulse @0x180496450`: when the
                    // location LOD is 0 (too far past lod_out, or nearer than
                    // lod_in), the emitter is RELEASED — `release_particles
                    // @0x180569370` destroys its CPU particles and
                    // `c_particle_emitter_gpu::clear`s the GPU rows. Free the
                    // rows back to the shared pool (a nearer effect reclaims
                    // them) and reset the runtime so the emitter restarts fresh
                    // (re-fires starting_count) when the location returns to
                    // LOD>0. age keeps running, matching the engine.
                    if lod <= 0.0 {
                        row_pool.release_rows(&mut rt.rows);
                        rt.accumulator = 0.0;
                        rt.live = 0.0;
                        rt.started = false;
                        continue;
                    }
                    // Retire this emitter's expired rows, then recompute its live
                    // count — both at ROW granularity, matching the engine's
                    // CHUNKY retirement (bible §03 / canonical correction #2):
                    // `frame_advance` decrements each row's `lifespan` by dt and
                    // frees the whole 16-wide row when it goes negative; the
                    // emitter live count (engine emitter_gpu +0x14) is Σ `used`
                    // over the surviving rows, so it steps down 16-at-a-time, not
                    // per-particle. (A per-particle death-time model — `retain(d >
                    // now)` — was tried and REVERTED as divergent: the engine has
                    // no per-particle alive count.)
                    row_pool.retire(&mut rt.rows, dt);
                    rt.live = row_pool.live_count(&rt.rows);
                    // Per-frame SYSTEM-level state for this batch (engine
                    // `update_states_internal`): slot 1 = system age in raw
                    // seconds (the eval saturates it → a first-second ramp),
                    // slots 8/9/25/26 = system random correlations, slot 16 =
                    // location seed. The correlations are a stable per-batch
                    // hash so curves keyed on them don't flicker frame-to-frame.
                    let bs = b * 28;
                    if bs + 26 < batch_state.len() {
                        batch_state[bs + 1][0] = rt.age;
                        let hash = |k: u32| -> f32 {
                            let mut x = (batch.wrapping_mul(0x9E37_79B9))
                                .wrapping_add(k.wrapping_mul(0x85EB_CA6B));
                            x ^= x >> 16;
                            x = x.wrapping_mul(0x7FEB_352D);
                            x ^= x >> 15;
                            (x & 0x00FF_FFFF) as f32 / 16_777_216.0
                        };
                        batch_state[bs + 8][0] = hash(0);
                        batch_state[bs + 9][0] = hash(1);
                        batch_state[bs + 25][0] = hash(2);
                        batch_state[bs + 26][0] = hash(3);
                        batch_state[bs + 16][0] = hash(4);
                        batch_state[bs + 11][0] = lod; // location LOD (engine slot 11)
                        // Exported-function slots 13/14 = effect_scale_a /
                        // effect_scale_b (engine `update_states_internal
                        // @0x180487020` → `c_particle_system::get_exported_function
                        // @0x180489660` → `m_effect_scale_a`/`m_effect_scale_b`).
                        // These are the effect-level scale multipliers parts opt
                        // into via the `SCALE_*` masks; their neutral default is
                        // 1.0 (a scale identity). Leaving them 0 zeroed any
                        // property keyed on slot 13/14 (input → eval at x=0;
                        // modifier-mul → ×0). The per-frame exported-function
                        // override (effect exported-function block → host object
                        // functions) is not yet wired — 1.0 is the engine default
                        // for effects spawned at unit scale (object attachments /
                        // weather).
                        batch_state[bs + 13][0] = 1.0;
                        batch_state[bs + 14][0] = 1.0;
                    }
                    let start = pending_spawns.len();
                    let random_rotation = sys.particle.appearance_flags.contains(
                        blam_tags::particle::ParticleAppearanceFlags::RandomStartingRotation,
                    );
                    // Gate emission by the host attachment function. Row
                    // retirement + age already advanced above, so a gated
                    // effect ages out cleanly and resumes the frame the
                    // function rises (e.g. the blade is drawn). `<= ~0`
                    // (the resolver also returns 0 on a failed lookup) →
                    // emit nothing this frame.
                    let n = if emit_scale > 1e-4 {
                        pulse_emitter(
                            emitter,
                            rt,
                            location_matrix,
                            dt,
                            birth_time,
                            duration,
                            looping,
                            random_rotation,
                            lod,
                            fair_share,
                            // Inherit effect velocity (engine system flag
                            // m_flags&0x40 → `vel += effect_get_velocity`):
                            // particles born from a MOVING emitter carry its
                            // motion. The emitter's frame-to-frame origin delta
                            // is the velocity source (see pulse_emitter).
                            sys.system
                                .flags
                                .contains(ParticleSystemFlags::InheritEffectVelocity),
                            pending_spawns,
                        )
                    } else {
                        0
                    };
                    // Allocate a persistent grid slot per new birth from the
                    // emitter's rows. The pool exhausts only at 7168 live
                    // particles; drop the unplaceable tail (engine caps too).
                    let mut placed = 0u32;
                    for k in 0..n as usize {
                        // Row retires on the particle's REMAINING life
                        // `(1-age)/invlife`, not the full lifespan — so a
                        // pre-warmed particle born partway through its life
                        // (age>0) frees its row slot when it actually dies. This
                        // is what spreads weather row-retirement across the
                        // lifespan (vs. one synchronized wave) → continuous
                        // re-emission. Normal particles (age≈0) → full life.
                        let life = 1.0 / pending_spawns[start + k].inverse_lifespan.max(1e-6);
                        match row_pool.alloc(&mut rt.rows, life, batch) {
                            Some(slot) => {
                                pending_slots.push(slot);
                                pending_batches.push(batch);
                                placed += 1;
                            }
                            None => break,
                        }
                    }
                    if (placed as usize) < n as usize {
                        pending_spawns.truncate(start + placed as usize);
                    }
                    rt.live = rt.live - (n - placed) as f32; // un-count dropped
                    remaining_budget = remaining_budget.saturating_sub(placed);

                    // PER-EMITTER transparent sort+draw unit (engine
                    // `c_particle_emitter::submit` registers one `add_element`
                    // PER EMITTER at `get_position_world`; `render_callback`
                    // draws only that emitter's rows). Built only when the
                    // emitter has occupied rows. `em.w_axis` = the emitter
                    // matrix origin (engine `m_matrix.origin`). The render side
                    // filters non-color (distortion / DontRenderSystem)
                    // batches via `ParticleGpu::is_color_batch`.
                    let span_start = emitter_draw_spans.len() as u32;
                    let span_count = row_pool.append_spans(&rt.rows, emitter_draw_spans);
                    if span_count > 0 {
                        let pos = em.w_axis.truncate();
                        let radius = if emitter.bounding_radius_override > 0.0 {
                            emitter.bounding_radius_override
                        } else {
                            emitter.bounding_radius_estimate
                        };
                        // Particle shader's authored sort layer (engine
                        // `v32->m_shader.m_sort_layer`). GlobalSortLayer and
                        // TransparentSortLayer share discriminants; map
                        // Invalid→Normal (engine `from_global`).
                        let sort_layer = sys.particle.shader.as_ref().map_or(2u8, |rm| {
                            use blam_tags::render_method::GlobalSortLayer as G;
                            match rm.sort_layer.get() {
                                G::PrePass => 1,
                                G::PostPass => 3,
                                _ => 2,
                            }
                        });
                        emitter_draws.push(EmitterDraw {
                            sort_pos: [pos.x, pos.y, pos.z],
                            batch: batch as u16,
                            offset: sys.system.sort_bias as f32 * 0.05,
                            radius,
                            sort_layer,
                            span_start,
                            span_count,
                        });
                    }
                }
            }
            // === Track R1: light-volume strip build ===
            // For each `ltvl` part on this effect, evaluate the strip
            // profiles into the per-frame light-volume buffers. Light volumes
            // are stateless (engine re-evaluates every profile each frame), so
            // we build them CPU-side here, gated by the same host attachment
            // function (`emit_scale`) as particle emission. Profiles stack
            // along `origin + dir·(offset + i·profile_distance)`; per-profile
            // color/thickness/alpha/intensity come from the 8 curves at the
            // profile percentile (engine light_volume_fx.hlsl + the GPU
            // re-eval). Each strip is one [`LightVolumeDraw`].
            if emit_scale > 1e-4 {
                for part in &effect.parts {
                    if part.part_type != EffectPartType::LightVolume {
                        continue;
                    }
                    let Some(lv) = light_volumes.get(&part.type_tag_path) else {
                        continue;
                    };
                    let loc_idx = part.location.max(0) as usize;
                    let location_matrix = match inst.location_matrices.get(loc_idx) {
                        Some(m) => host_matrix * *m,
                        None => host_matrix,
                    };
                    let origin =
                        location_matrix.transform_point3(glam::Vec3::from(part.relative_offset));
                    let direction = location_matrix
                        .transform_vector3(glam::Vec3::X)
                        .normalize_or_zero();
                    if direction.length_squared() < 1e-6 {
                        continue;
                    }
                    let def = &lv.definition;
                    let eval = |p: &blam_tags::effects_properties::EditableProperty, input: f32| {
                        particle_emitter::diag_eval(p, input, 0.0)
                    };
                    // Diagnostic: PROTOMORPH_LV_FORCE=<scale> forces
                    // brightness_ratio=1 (visible head-on) and ×scale thickness
                    // — to confirm the strip render rasterizes, independent of
                    // the (often tiny / side-only) authored values.
                    let force = std::env::var("PROTOMORPH_LV_FORCE")
                        .ok()
                        .and_then(|s| s.parse::<f32>().ok());
                    // Profile count — engine `c_light_volume_gpu::render
                    // @0x1806D29A0`: `count = (int)(m_length · m_profile_density)`
                    // (TRUNCATED, not rounded; no min-1 floor), rendered only if
                    // `> 0`. The old `round(profile_density).clamp(1,128)` ignored
                    // length entirely → wrong count + spurious profiles for
                    // zero-length / sub-1 volumes. Cap at the GPU strip buffer
                    // max (8 rows × 16 = 128).
                    let length = eval(&def.length, 0.0);
                    let profile_density = eval(&def.profile_density, 0.0);
                    let count_i = (length * profile_density) as i32;
                    if count_i <= 0 {
                        continue;
                    }
                    let num = (count_i as u32).min(128);
                    let offset = eval(&def.offset, 0.0);
                    let profile_length = eval(&def.profile_length, 0.0);
                    let profile_distance = if num > 1 {
                        length / (num as f32 - 1.0)
                    } else {
                        0.0
                    };
                    let first = light_volume_profiles.len() as u32;
                    for i in 0..num {
                        let pct = if num > 1 {
                            i as f32 / (num as f32 - 1.0)
                        } else {
                            0.0
                        };
                        let pos = origin + direction * (offset + i as f32 * profile_distance);
                        let thickness = eval(&def.profile_thickness, pct) * force.unwrap_or(1.0);
                        let alpha = eval(&def.profile_alpha, pct);
                        let intensity = eval(&def.profile_intensity, pct);
                        let color = def
                            .profile_color
                            .function
                            .as_ref()
                            .map(|f| {
                                let c = f.evaluate_color(pct, 0.0);
                                [c.red, c.green, c.blue]
                            })
                            .unwrap_or([1.0, 1.0, 1.0]);
                        light_volume_profiles.push(LightVolumeProfileGpu {
                            pos: [pos.x, pos.y, pos.z, 0.0],
                            color_intensity: [color[0], color[1], color[2], intensity],
                            dir_thickness: [direction.x, direction.y, direction.z, thickness],
                            params: [
                                profile_length,
                                profile_distance,
                                num as f32,
                                if force.is_some() { 1.0 } else { def.brightness_ratio },
                            ],
                            alpha: [alpha, 0.0, 0.0, 0.0],
                        });
                    }
                    light_volume_draws.push(LightVolumeDraw {
                        base_map: lv.render.base_map.clone(),
                        blend_mode: lv.render.blend_mode,
                        fog: lv.render.fog,
                        first,
                        count: num,
                    });
                }
            }
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
            static LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let sec = *time as u32;
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
                    *time,
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
        // Diag: tally instances-per-batch + spawns-this-frame-per-batch,
        // printed once near steady state, to isolate density (grid-cap vs
        // multi-instance collision vs under-emission).
        static DENSITY_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("PROTOMORPH_DIAG_DENSITY").is_ok()
            && *time > 3.0
            && !DENSITY_DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
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
                let eval_tables = particle_curves::compile_emitter(emitter, &sys.particle);
                if std::env::var("PROTOMORPH_DIAG_PARTICLES").is_ok() {
                    let leaf = sys.particle_path.rsplit(['\\', '/']).next().unwrap_or("?");
                    let p = &eval_tables.properties;
                    let kind = |x: &particle_curves::EvalProperty| {
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
                    let col_lo = |pp: &particle_curves::EvalProperty| {
                        eval_tables.colors.get(pp.c[0] as usize).copied().unwrap_or([0.0; 4])
                    };
                    let col_hi = |pp: &particle_curves::EvalProperty| {
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
                let (edge_range, edge_cutoff) = if sys.particle.appearance_flags.contains(
                    blam_tags::particle::ParticleAppearanceFlags::FadeWhenViewedEdgeOn,
                ) {
                    let r = sys.particle.angle_fade_range_degrees.to_radians();
                    let c = sys.particle.angle_fade_cutoff_degrees.to_radians();
                    (if r > 0.0 { 1.0 / r } else { 0.0 }, c)
                } else {
                    (0.0, 0.0)
                };
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

    pub fn batch_count(&self) -> u32 {
        self.next_batch_index
    }

    /// Diagnostic — total particle systems across every live instance.
    pub fn live_system_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.effects[i.effect_index].systems.len())
            .sum()
    }

}

/// Walk a `particle_physics` template chain to its physics controller
/// and read the three constants (gravity_mod / air / rot drag). Returns
/// `None` if no physics controller is found in the chain.
fn resolve_physics(
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
