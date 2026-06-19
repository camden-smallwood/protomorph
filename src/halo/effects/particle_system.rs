//! CPU-side particle render/pool/batch plumbing for the effect subsystem.
//!
//! Holds the row-pool grid allocator, the particle albedo/render-info
//! resolution, the per-emitter render `BatchDescriptor`, and the per-frame
//! `EmitterDraw` sort/draw unit.
//!
//! NOTE: [`RowPool`]'s strict engine home is `gpu_particle/particle_block`
//! (engine `s_row` / `s_overall_storage`), but it is kept here CPU-side to
//! avoid a module cycle between `effects` and `gpu_particle` (the GPU side
//! depends on `effects`, not the reverse).

use std::str::FromStr;

use blam_tags::effect::ParticleSystemFlags;
use blam_tags::render_method::{AlphaBlendMode, RenderMethodChoices};

use super::effect_definitions::LoadedParticleSystem;
use super::particle_emitter::{self, pulse_emitter, EmitterRuntime, ParticlePhysicsParams};
use super::particle_properties::EmitterEvalTables;
use super::particle_states::ParticleState;
use super::{PARTICLE_ROW_COUNT, PARTICLE_ROW_WIDTH};

/// `row_batch` sentinel — a row that owns no live emitter (free / dead).
/// The update kernel skips any slot whose row reads this.
pub const ROW_BATCH_FREE: u32 = 0xFFFF_FFFF;

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
    pub(crate) free: Vec<u16>,
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
    pub(crate) fn retire(&mut self, emitter_rows: &mut Vec<u16>, dt: f32) {
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
    pub(crate) fn alloc(&mut self, emitter_rows: &mut Vec<u16>, lifespan: f32, batch: u32) -> Option<u32> {
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
    pub(crate) fn release_rows(&mut self, emitter_rows: &mut Vec<u16>) {
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
    pub(crate) fn live_count(&self, emitter_rows: &[u16]) -> f32 {
        emitter_rows
            .iter()
            .map(|&r| self.rows[r as usize].used as f32)
            .sum()
    }

    /// Append one `(base_instance, count)` draw span per occupied row, for
    /// drawing exactly this emitter's particles (engine
    /// `c_particle_emitter_gpu::render`: per row, `used` instances starting
    /// at slot `16*(row+1)-used`). Returns the number of spans appended.
    pub(crate) fn append_spans(&self, emitter_rows: &[u16], out: &mut Vec<(u32, u32)>) -> u32 {
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
    pub(crate) fn from_shader(
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
    pub eval_tables: EmitterEvalTables,
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
            eval_tables: EmitterEvalTables::default(),
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

/// One transparent-sort + draw unit = a single emitter of a single live
/// instance, mirroring the engine's PER-EMITTER `c_particle_emitter::submit
/// @0x180568FA0` (one `add_element` per emitter at `get_position_world`) +
/// `c_particle_emitter_gpu::render @0x1806A5140` (draws only THAT emitter's
/// rows). Replaces the old per-BATCH registration whose single averaged
/// centroid mis-sorted multi-instance systems (11 s3d_turf steam vents share
/// one batch → one global-average depth → the fence/drips wouldn't blend).
/// Rebuilt every frame in [`super::EffectStore::frame_advance`].
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
    /// engine `v32->m_shader.m_sort_layer`), as a `TransparentSortLayer`
    /// discriminant.
    pub sort_layer: u8,
    /// `[span_start..span_start+span_count]` slice into
    /// [`super::EffectStore::emitter_draw_spans`] — one `(base_instance, count)`
    /// per occupied grid row (engine per-row draw: `count` instances from
    /// slot `16*(row+1)-used`).
    pub span_start: u32,
    pub span_count: u32,
}

/// The per-emitter particle-pulse body of `EffectStore::frame_advance`,
/// extracted verbatim (behaviour-identical) for one emitter `ei` of system
/// `sys` belonging to a single live instance.
///
/// All parameters are exactly the per-frame state the original inline loop
/// body read or mutated: the shared spawn-budget counters
/// (`remaining_emitters` / `remaining_budget`), the per-batch accumulators
/// (`bounds_acc` / `batch_self_accel` / `batch_state`), the row pool, this
/// emitter's runtime (`rt`), and the pending-spawn / draw-span output vectors.
///
/// The single early `continue` in the original loop body becomes a `return`
/// (it was the last statement of the loop iteration, so this is exact).
#[allow(clippy::too_many_arguments)]
pub(super) fn pulse_emitter_instance(
    sys: &LoadedParticleSystem,
    ei: usize,
    emitter: &blam_tags::effect::ParticleSystemEmitter,
    rt: &mut EmitterRuntime,
    location_matrix: glam::Mat4,
    lod: f32,
    dt: f32,
    birth_time: f32,
    duration: f32,
    looping: bool,
    emit_scale: f32,
    remaining_emitters: &mut u32,
    remaining_budget: &mut u32,
    bounds_acc: &mut [Option<(glam::Vec3, glam::Vec3, f32)>],
    batch_self_accel: &mut [[f32; 4]],
    batch_state: &mut [[f32; 4]],
    row_pool: &mut RowPool,
    pending_spawns: &mut Vec<ParticleState>,
    pending_slots: &mut Vec<u32>,
    pending_batches: &mut Vec<u32>,
    emitter_draw_spans: &mut Vec<(u32, u32)>,
    emitter_draws: &mut Vec<EmitterDraw>,
) {
    // Max-min fair share of the remaining budget. Every
    // emitter consumes a slot of `remaining_emitters` even
    // when its share is 0, so the divisor shrinks and unused
    // budget concentrates on the emitters that want it.
    let fair_share = if *remaining_emitters > 0 {
        remaining_budget.div_ceil(*remaining_emitters)
    } else {
        0
    };
    *remaining_emitters = remaining_emitters.saturating_sub(1);
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
        return;
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
    *remaining_budget = remaining_budget.saturating_sub(placed);

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
