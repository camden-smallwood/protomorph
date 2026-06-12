//! CPU emitter pulse — produces particle spawn requests, the only part
//! of the particle pipeline the engine runs on the CPU. Spawn/physics/
//! aging/render all run on the GPU against the persistent grid.
//!
//! Faithful to `c_particle_emitter_definition::initialize_particle
//! @0x1805673f0` (the 11 emission shapes, +X forward basis) and
//! `c_particle_system::real_internal_random_range @0x18048a1f0` (the LCG
//! RNG). Emission math runs in emitter-LOCAL space (+X = forward, YZ =
//! the perpendicular plane); Phase 2 applies only the host translation
//! to reach world space — the full location/relative_direction
//! orientation matrix lands in Phase 4.

use blam_tags::effect::{EmissionShape, ParticleSystemEmitter};
use blam_tags::effects_properties::EditableProperty;
use glam::Vec3;

/// Engine particle LCG (`real_internal_random_range`): advance the seed
/// with `1664525·s + 1013904223`, take the high 16 bits, scale by
/// `1/65535`. 8 such seeds per particle drive its lifetime randomness.
#[derive(Debug, Clone, Copy)]
pub struct ParticleRng {
    pub seed: u32,
}

impl ParticleRng {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Advance and return the raw 32-bit state.
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }

    /// Engine `real_internal_random_range(lo, hi)` — exact bit-for-bit.
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let v = self.next_u32();
        let hiword = (v >> 16) as f32; // HIWORD(v) ∈ [0, 65535]
        hiword * 0.000015259022 * (hi - lo) + lo
    }

    /// One unorm8 random byte (for the particle's stored seed bytes).
    #[inline]
    fn next_byte(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }
}

/// The logical per-particle birth state — the engine `s_particle_state`.
/// The CPU builds this fully (engine `spawn_particle` /
/// `initialize_particle`); [`Self::pack`] produces the 80-byte
/// [`RawParticleState`] the GPU grid stores (engine `pack_particle_state`,
/// `particle_pack_fx.hlsl`). The GPU spawn is a pure scatter-copy of the
/// packed state — it does no initialization.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleState {
    pub position: [f32; 3],
    pub size: f32,
    pub velocity: [f32; 3],
    pub aspect: f32,
    /// 8 random correlations in [0,1] (state slots 4..7 / 21..24).
    pub random: [f32; 4],
    pub random2: [f32; 4],
    pub physical_rotation: f32,
    pub manual_rotation: f32,
    pub animated_frame: f32,
    pub manual_frame: f32,
    pub birth_time: f32,
    pub inverse_lifespan: f32,
    pub age: f32,
    pub intensity: f32,
    pub rotational_velocity: f32,
    pub frame_velocity: f32,
    pub black_point: f32,
    pub palette_v: f32,
    pub axis: [f32; 3],
    /// `m_color` (per-frame-evaluated; birth value).
    pub color: [f32; 4],
    /// `m_initial_color` rgb + the HDR exponent in `.w` (engine
    /// `exp2(initial_color.w)`).
    pub initial_color: [f32; 4],
}

/// The packed 80-byte GPU grid state (engine DX11 `s_raw_particle_state`,
/// 20×u32). Field word ranges: pos[0..4] vel[4..6] rnd[6..8] rnd2[8..10]
/// rot[10..12] time[12..14] anm[14] anm2[15] axis[16] col[17] col2[18]
/// pad[19].
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawParticleState {
    pub words: [u32; 20],
}

const _: () = assert!(std::mem::size_of::<RawParticleState>() == 80);

#[inline]
fn pack_half2(x: f32, y: f32) -> u32 {
    (half::f16::from_f32(x).to_bits() as u32) | ((half::f16::from_f32(y).to_bits() as u32) << 16)
}
#[inline]
fn pack_ushort2n(x: f32, y: f32) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0) as u32;
    q(x) | (q(y) << 16)
}
#[inline]
fn pack_argb8(c: [f32; 4]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u32;
    q(c[2]) | (q(c[1]) << 8) | (q(c[0]) << 16) | (q(c[3]) << 24)
}
#[inline]
fn pack_dec3n(v: [f32; 3]) -> u32 {
    let q = |x: f32| (((x.clamp(-1.0, 1.0) * 511.0) as i32) & 0x3ff) as u32;
    q(v[0]) | (q(v[1]) << 10) | (q(v[2]) << 20)
}

impl ParticleState {
    /// Pack into the GPU grid format — exact port of
    /// `pack_particle_state` + `packed_vector.fx`.
    pub fn pack(&self) -> RawParticleState {
        let mut w = [0u32; 20];
        w[0] = self.position[0].to_bits();
        w[1] = self.position[1].to_bits();
        w[2] = self.position[2].to_bits();
        // Engine spawn-frame "hack" (particle_update_hlsl.hlsl:156-157):
        // store m_size NEGATIVE so the first GPU update frame runs with
        // dt=0 (no integration, no age advance). The particle is born
        // mid-frame and the CPU already folded its sub-frame offset into
        // `age`; integrating it a full dt the same frame would double-
        // advance it. `cs_update` rewrites size positive from the curves
        // (mirroring HLSL line 201), so every later frame integrates
        // normally. `size` is always ≥ 0.0001, so `-size` is reliably < 0.
        w[3] = (-self.size).to_bits();
        w[4] = pack_half2(self.velocity[0], self.velocity[1]);
        w[5] = pack_half2(self.velocity[2], self.aspect);
        w[6] = pack_ushort2n(self.random[0], self.random[1]);
        w[7] = pack_ushort2n(self.random[2], self.random[3]);
        w[8] = pack_ushort2n(self.random2[0], self.random2[1]);
        w[9] = pack_ushort2n(self.random2[2], self.random2[3]);
        w[10] = pack_ushort2n(self.physical_rotation, self.manual_rotation);
        w[11] = pack_ushort2n(self.animated_frame, self.manual_frame);
        w[12] = pack_half2(self.birth_time, self.inverse_lifespan);
        w[13] = pack_half2(self.age, self.intensity);
        w[14] = pack_half2(self.rotational_velocity, self.frame_velocity);
        w[15] = pack_ushort2n(self.black_point, self.palette_v);
        w[16] = pack_dec3n(self.axis);
        w[17] = pack_argb8(self.color);
        w[18] = pack_argb8(self.initial_color);
        w[19] = 0;
        RawParticleState { words: w }
    }
}

/// Per-emitter runtime state carried across frames by an effect
/// instance. One per emitter per particle-system per instance.
#[derive(Debug, Clone)]
pub struct EmitterRuntime {
    /// Fractional particle accumulator — `acc += rate·dt`, spawn
    /// `floor(acc)` each frame, keep the remainder.
    pub accumulator: f32,
    /// Seconds the emitter has been alive (the `system age` state input).
    pub age: f32,
    /// Stable per-system random in [0,1] — the `system random` state slot
    /// (engine fills it from the system's `m_random_seed`). Drives ranged
    /// SYSTEM-level properties like `max_count`/`emission_rate` (e.g.
    /// snowflake_heavy's `max_count`, ranged on system_random). Shared across
    /// a system's emitters (computed from instance+system, not emitter).
    pub system_random: f32,
    pub rng: ParticleRng,
    /// `false` until the first pulse — gates the `starting_count` burst.
    /// For a LOOPING instance it is re-cleared each event-restart period so
    /// the burst re-fires (engine `effect_restart_all_events`).
    pub started: bool,
    /// Age (seconds) at which this emitter's event next RESTARTS — for a
    /// looping instance the event `duration` is the restart PERIOD (engine
    /// `effect_update_time` re-runs `event_create_particle_systems` each
    /// cycle). 0 = uninitialised (set to `duration` on first pulse). Burst
    /// effects (sparks: `starting_count` > 0, `rate` = 0) re-fire each
    /// period; constant-rate effects (steam/snow) are unaffected.
    pub next_restart: f32,
    /// Live-particle count feeding the `max_count` headroom clamp — the sum of
    /// the emitter's row `used` counts (engine `c_particle_emitter_gpu::
    /// m_particle_count` at +0x14: ++ per spawn, -= row.used per row
    /// retirement). Set by the caller after row retirement.
    pub live: f32,
    /// Grid rows this emitter owns (engine `s_row` list per
    /// `c_particle_emitter_gpu`). Each row holds ≤16 particles; the last
    /// entry is the partially-filled head. Rows retire (free) once their
    /// lifespan countdown expires. Indices into [`RowPool::rows`].
    pub rows: Vec<u16>,
    /// The emitter's world origin on the PREVIOUS pulse — engine
    /// `spawn_particle @0x180569A80` `previous_position`. The world-coordinate
    /// spawn lerps `previous_position → current origin` by each particle's
    /// sub-frame `particle_t`, so a MOVING emitter streams particles along its
    /// motion path instead of clumping them at the current origin each frame.
    /// Also the basis for inherit-velocity `(cur − prev)/dt`. `None` until the
    /// first pulse (→ no interp / zero inherited velocity on frame 0).
    /// Stationary emitter ⇒ prev == cur ⇒ identical to the old single-origin
    /// transform.
    pub prev_origin: Option<glam::Vec3>,
}

impl EmitterRuntime {
    /// Seed the RNG deterministically from an instance + emitter index so
    /// distinct emitters don't lock-step (the engine seeds from the
    /// effect's `m_random_seed`; we mirror the spirit, not the value).
    pub fn new(instance_index: usize, system_index: usize, emitter_index: usize) -> Self {
        let seed = 0x9E3779B9u32
            .wrapping_mul(instance_index as u32 + 1)
            .wrapping_add(0x85EBCA6Bu32.wrapping_mul(system_index as u32 + 1))
            .wrapping_add(0xC2B2AE35u32.wrapping_mul(emitter_index as u32 + 1));
        // `system_random` is per (instance, system) — shared by the system's
        // emitters (engine: one `m_random_seed` per c_particle_system).
        let sys_seed = 0x9E3779B9u32
            .wrapping_mul(instance_index as u32 + 1)
            .wrapping_add(0x85EBCA6Bu32.wrapping_mul(system_index as u32 + 1));
        let mut x = sys_seed ^ (sys_seed >> 16);
        x = x.wrapping_mul(0x7FEB_352D);
        x ^= x >> 15;
        let system_random = (x & 0x00FF_FFFF) as f32 / 16_777_216.0;
        Self {
            accumulator: 0.0,
            age: 0.0,
            system_random,
            rng: ParticleRng::new(seed | 1),
            started: false,
            next_restart: 0.0,
            live: 0.0,
            rows: Vec::new(),
            prev_origin: None,
        }
    }
}

/// `g_physics_constants->gravity` — the engine particle gravity
/// magnitude (world-units/s²) the physics controller scales by each
/// emitter's `gravity_modifier`.
pub const PARTICLE_GRAVITY: f32 = 4.1712594;

/// Resolved per-emitter physics — the three constants the GPU update
/// kernel needs (engine `update_particles_physics @0x1805793B0`).
/// `gravity_mod` scales [`PARTICLE_GRAVITY`] (negative = buoyant, like
/// smoke); the frictions are per-second drag coefficients.
#[derive(Debug, Clone, Copy)]
pub struct ParticlePhysicsParams {
    pub gravity_mod: f32,
    pub air_friction: f32,
    pub rot_friction: f32,
}

impl Default for ParticlePhysicsParams {
    fn default() -> Self {
        // No template resolved → inert (no gravity, no drag).
        Self { gravity_mod: 0.0, air_friction: 0.0, rot_friction: 0.0 }
    }
}

/// Evaluate a property at the particle-birth state (input/range 0) — the
/// constant-ish value a controller parameter resolves to. Used for the
/// physics-template constants that feed the GPU update uniform.
pub fn eval_constant(p: &EditableProperty) -> f32 {
    eval_prop(p, 0.0, 0.0)
}

/// Evaluate an emitter's `particle_max_count` at the given system age
/// (diagnostics).
pub fn eval_max_count(emitter: &ParticleSystemEmitter, sys_age: f32) -> f32 {
    eval_prop(&emitter.particle_max_count, sys_age, 0.0)
}

/// Diagnostics: evaluate any property at (input, range). For probing the
/// prt3 aspect_ratio / particle_scale ranges from the batch diag.
pub fn diag_eval(p: &EditableProperty, input: f32, range: f32) -> f32 {
    eval_prop(p, input, range)
}

/// Diagnostics: key emitter properties evaluated at range 0 vs range 1 (the
/// `particle random` axis): (life0,life1, size0,size1, vel0,vel1, maxc0,maxc1).
/// r0 != r1 ⇒ the property is genuinely ranged; the GPU path previously
/// collapsed ranged constants to r0 (the low bound).
pub fn diag_lifespan_range(emitter: &ParticleSystemEmitter) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let at = |p: &EditableProperty, r: f32| eval_prop(p, 0.0, r);
    (
        at(&emitter.particle_lifespan, 0.0), at(&emitter.particle_lifespan, 1.0),
        at(&emitter.particle_size, 0.0), at(&emitter.particle_size, 1.0),
        at(&emitter.particle_initial_velocity, 0.0), at(&emitter.particle_initial_velocity, 1.0),
        at(&emitter.particle_max_count, 0.0), at(&emitter.particle_max_count, 1.0),
    )
}

/// Diagnostics: (emitter_size@birth, emitter_size@death, particle_scale,
/// initial_speed, emission_radius) for one emitter.
pub fn diag_size_speed(emitter: &ParticleSystemEmitter) -> (f32, f32, f32, f32, f32) {
    (
        eval_prop(&emitter.particle_size, 0.0, 0.0),
        eval_prop(&emitter.particle_size, 1.0, 0.0),
        eval_prop(&emitter.particle_scale, 0.0, 0.0),
        eval_prop(&emitter.particle_initial_velocity, 0.0, 0.0),
        eval_prop(&emitter.emission_radius, 0.0, 0.0),
    )
}

/// Diagnostics: the emission angle (cone half-angle, degrees) evaluated
/// across the particle-random range `[0,1]` — the velocity spread.
pub fn diag_emission_angle(emitter: &ParticleSystemEmitter) -> (f32, f32, f32) {
    (
        eval_prop(&emitter.emission_angle, 0.0, 0.0),
        eval_prop(&emitter.emission_angle, 0.0, 0.5),
        eval_prop(&emitter.emission_angle, 0.0, 1.0),
    )
}

/// Evaluate an [`EditableProperty`] curve. `input`/`range` are the
/// state-list values its `input_index`/`range_input_index` select; for
/// constant curves they're ignored. Loose tags leave `constant_value`
/// (a `!` runtime field) at 0, so the authored `function` is the real
/// source — we only fall back to `constant_value` when there's no curve.
#[inline]
fn eval_prop(p: &EditableProperty, input: f32, range: f32) -> f32 {
    match &p.function {
        Some(f) => f.evaluate(input, range),
        None => p.constant_value,
    }
}

/// Number of particle state-list slots (engine `c_particle_state_list`,
/// 27 inputs; we round up). Indexed by a property's `input`/`range`/
/// `modifier` variable enum: 0 particle_age, 1 system_age, 2 particle_random,
/// 3 system_random, 11 location_lod, 12 game_time, …
pub const STATE_SLOTS: usize = 28;

/// Faithful port of the engine's SINGLE property evaluator
/// `c_editable_property_base::evaluate @0x180567fa0` — the one evaluator the
/// engine uses for every emitter property (and mirrors on the GPU for the
/// visual props). It reads the function INPUT from `state[input_index]` and
/// the RANGE from `state[range_input_index]` (indices into the state list),
/// evaluates, then applies the output modifier (1 = add, 2 = multiply) with
/// `state[modifier_input_index]`.
///
/// This replaces the old `eval_prop(prop, sys_age, 0)` stub that hardcoded
/// the input to `system_age`, the range to `0`, and DROPPED the modifier —
/// which collapsed ranged SYSTEM-level properties to their low bound (e.g.
/// snowflake_heavy's `max_count`, ranged on system_random, evaluated to 0 so
/// the visible heavy snow never spawned) and ignored the `× location_lod`
/// modifier authored on `max_count`.
fn eval_property(p: &EditableProperty, state: &[f32]) -> f32 {
    let slot = |i: u8| state.get(i as usize).copied().unwrap_or(0.0);
    let Some(f) = p.function.as_ref() else {
        return p.constant_value;
    };
    // Constant functions ignore the input (engine: `input = 0` for constants).
    let input = if f.is_constant() { 0.0 } else { slot(p.input_index) };
    let range = if f.header().flags.is_ranged() {
        slot(p.range_input_index)
    } else {
        0.0
    };
    let mut out = f.evaluate(input, range);
    match p.output_modifier_type {
        1 => out += slot(p.output_modifier_input_index), // add
        2 => out *= slot(p.output_modifier_input_index), // multiply
        _ => {}
    }
    out
}

/// Color variant of [`eval_property`] for the emitter tint gradient.
fn eval_color_property(p: &EditableProperty, state: &[f32]) -> [f32; 3] {
    let slot = |i: u8| state.get(i as usize).copied().unwrap_or(0.0);
    match p.function.as_ref() {
        Some(f) => {
            let input = if f.is_constant() { 0.0 } else { slot(p.input_index) };
            let range = if f.header().flags.is_ranged() {
                slot(p.range_input_index)
            } else {
                0.0
            };
            let c = f.evaluate_color(input, range);
            [c.red, c.green, c.blue]
        }
        None => [1.0, 1.0, 1.0],
    }
}

/// Evaluate a color property (the emitter tint). Returns white when the
/// property carries no color graph.
#[inline]
fn eval_color_prop(p: &EditableProperty, input: f32, range: f32) -> [f32; 3] {
    match &p.function {
        Some(f) => {
            let c = f.evaluate_color(input, range);
            [c.red, c.green, c.blue]
        }
        None => [1.0, 1.0, 1.0],
    }
}

/// Advance one emitter for `dt` seconds against a host world transform,
/// pushing world-space [`SpawnRequest`]s onto `out`. Returns the number
/// spawned. `birth_time` stamps each particle for the update/render
/// shaders' aging.
///
/// `max_new` caps spawns this frame (grid headroom). Engine analogue:
/// `c_particle_emitter::queue_particle` driven by the per-emitter
/// accumulator in `frame_advance`.
/// Build the emitter world matrix from the location (host_world ×
/// marker_local) matrix — engine `calc_matrix @0x180569500`. Pre-rotates
/// the location's basis by `relative_direction` (yaw=atan2(dir.y,dir.x),
/// pitch=asin(dir.z)) and offsets the origin by the rotated
/// `translational_offset`. relative_direction = +X (identity) → no tilt.
pub(crate) fn calc_emitter_matrix(
    location: glam::Mat4,
    emitter: &ParticleSystemEmitter,
    sys_age: f32,
    sys_random: f32,
) -> glam::Mat4 {
    // Engine `calc_matrix` builds the emitter transform on an ORTHONORMAL
    // rotation basis with a SEPARATE scalar scale (real_matrix4x3.scale).
    // For an object-attached effect that scalar is 1 — the host object's
    // RENDER scale does NOT reach particle emission. protomorph's
    // host_world matrix bakes the render scale into its basis vectors, so a
    // scaled host (the guardian 130lb_earth hologram is placed at 4×) would
    // multiply the emission radius (2.75 → 11) AND the translational offset,
    // flinging the spawn shell out to ~11 units — a radius-11 ember shell
    // that envelops the nearby camera and reads as "white dots streaming in
    // from outside the room toward the hologram." Sapien keeps them tight
    // at the hologram. Normalize the basis to unit length (keep rotation +
    // translation, drop the host scale) to match the engine. Unit-scale
    // hosts (waterfall, man cannon, etc.) are unaffected (no-op).
    let t = location.w_axis;
    let mut m = glam::Mat4::from_cols(
        location.x_axis.truncate().normalize_or_zero().extend(0.0),
        location.y_axis.truncate().normalize_or_zero().extend(0.0),
        location.z_axis.truncate().normalize_or_zero().extend(0.0),
        t,
    );
    // relative_direction → rotation about the location basis.
    if let Some(d) = emitter.relative_direction.starting_interpolant {
        let dir = glam::Vec3::new(d.i, d.j, d.k);
        if dir.length_squared() > 1e-9 {
            let n = dir.normalize();
            let yaw = n.y.atan2(n.x);
            let pitch = n.z.clamp(-1.0, 1.0).asin();
            if yaw.abs() > 1e-4 || pitch.abs() > 1e-4 {
                // Engine `matrix3x3_from_angles(yaw, pitch, 0)` @0x1802c76e0
                // is ROW-major (used as v×M); glam is column-major (M×v), so
                // we build its TRANSPOSE — glam columns = engine rows:
                //   col0 (forward) = the normalized direction `n`
                //   col1 = (-sin yaw, cos yaw, 0)
                //   col2 = (-cos yaw sin pitch, -sin yaw sin pitch, cos pitch)
                // Built explicitly because glam::from_euler's Ry(pitch) has
                // the OPPOSITE sign to the engine's pitch (engine forward.z =
                // +sin(pitch)), which inverted every vertically-aimed emitter
                // (man cannon, powerups) — aiming/offsetting them the wrong
                // way. Horizontal (pitch≈0) emitters were unaffected.
                let (sy, cy) = yaw.sin_cos();
                let (sp, cp) = pitch.sin_cos();
                let rot = glam::Mat3::from_cols(
                    n,
                    glam::Vec3::new(-sy, cy, 0.0),
                    glam::Vec3::new(-cy * sp, -sy * sp, cp),
                );
                // Engine `matrix3x3_multiply(m_rot, rotate, m_rot)` = rotate·
                // m_rot (rotate applied first), i.e. glam `location · rotateᵀ`.
                let t = m.w_axis;
                m = m * glam::Mat4::from_mat3(rot);
                m.w_axis = t;
            }
        }
    }
    // translational_offset → EVALUATE the property, then rotate by the
    // emitter basis + add to the origin. Engine `calc_matrix` calls
    // `evaluate_emitter_offset @0x18056A5D0` →
    // `c_editable_property_type<...,real_point3d>::evaluate @0x18056A450`:
    //   offset = (1-t)·starting_interpolant + t·ending_interpolant
    // where `t` = the property's scalar eval against system_age/random.
    // Using the raw `starting_interpolant` left animated offsets stuck at
    // their BIRTH value — the powerup core's plasma_noise/ring author
    // translational_offset starting=(0.15,0,0), ending=0 keyed on
    // system_age, so the bubble sat 0.15 off-centre instead of animating to
    // the centred ending as system_age saturates (~1s). Constant offsets
    // (starting==ending) are unaffected.
    let start = emitter
        .translational_offset
        .starting_interpolant
        .map(|o| glam::Vec3::new(o.i, o.j, o.k))
        .unwrap_or(glam::Vec3::ZERO);
    let end = emitter
        .translational_offset
        .ending_interpolant
        .map(|o| glam::Vec3::new(o.i, o.j, o.k))
        .unwrap_or(glam::Vec3::ZERO);
    if start != glam::Vec3::ZERO || end != glam::Vec3::ZERO {
        let t = eval_prop(&emitter.translational_offset, sys_age, sys_random).clamp(0.0, 1.0);
        let local = start.lerp(end, t);
        let off = m.transform_vector3(local);
        m.w_axis += glam::Vec4::new(off.x, off.y, off.z, 0.0);
    }
    m
}

pub fn pulse_emitter(
    emitter: &ParticleSystemEmitter,
    rt: &mut EmitterRuntime,
    location_matrix: glam::Mat4,
    dt: f32,
    birth_time: f32,
    duration: f32,
    looping: bool,
    random_rotation: bool,
    lod: f32,
    max_new: u32,
    inherit_velocity: bool,
    out: &mut Vec<ParticleState>,
) -> u32 {
    rt.age += dt;
    // `rt.live` is the emitter's true live-particle count (sum of its row
    // `used` counts), set by the caller after row retirement — the engine
    // reads the same from the emitter_gpu record at +0x14 for the clamp.
    // Engine `c_particle_emitter::pulse @0x180568690`: keep only the
    // fractional accumulator at frame start, then add this frame's
    // emission.
    // Event lifecycle by `duration` (engine `effect_update_time
    // @0x180309870`):
    //  - non-looping: the event runs once and stops at `duration`.
    //  - looping: `duration` is the RESTART PERIOD — the event re-runs each
    //    cycle (`effect_restart_all_events`), which RE-FIRES `starting_count`.
    //    Constant-rate effects (steam/snow) are unchanged (their burst is 0
    //    and the rate is continuous); burst effects (sparks: starting_count
    //    > 0, rate = 0) re-emit each period instead of firing once and
    //    vanishing.
    if duration > 0.0 {
        if looping {
            if rt.next_restart <= 0.0 {
                rt.next_restart = duration; // first period
            }
            if rt.age >= rt.next_restart {
                rt.started = false; // re-fire starting_count below
                rt.next_restart += duration;
            }
        } else if rt.age >= duration {
            return 0;
        }
    }
    let sys_age = rt.age;

    // Emitter matrix + world origin for this frame (engine `this->m_matrix`).
    // Computed ONCE here (sys_age constant per frame, sys_random=0) instead of
    // per particle, and used as the spawn frame for the whole pulse.
    let emitter_matrix = calc_emitter_matrix(location_matrix, emitter, sys_age, 0.0);
    let cur_origin = emitter_matrix.transform_point3(Vec3::ZERO);
    // `previous_position` (engine `spawn_particle`): last pulse's origin, or
    // the current one on the first pulse (→ no interp). Update it EVERY pulse
    // (even no-spawn frames) so a moving emitter's motion is always tracked.
    let prev_origin = rt.prev_origin.unwrap_or(cur_origin);
    rt.prev_origin = Some(cur_origin);
    // Inherited effect velocity (engine `effect_get_velocity`, applied when the
    // system's `inherit effect velocity` flag — m_flags&0x40 — is set): the
    // emitter's frame-to-frame motion. Zero for a stationary emitter.
    let inherited_vel = if inherit_velocity && dt > 1e-6 {
        (cur_origin - prev_origin) / dt
    } else {
        Vec3::ZERO
    };

    // Engine `c_editable_property_base::evaluate` reads each property's
    // input/range/modifier from the 27-slot state list. Build the SYSTEM-level
    // slots here (the per-particle slots — particle_age/particle_random — are
    // set in the spawn loop below). slot 1 system_age, slot 3 system_random,
    // slot 11 location_lod.
    let mut state = [0.0f32; STATE_SLOTS];
    state[1] = sys_age;
    state[3] = rt.system_random;
    state[11] = lod;

    // First pulse: the emitter's starting_count burst (engine adds it once on
    // the first emit). Then the per-frame rate. Both evaluated through the
    // faithful property evaluator (state-list input/range + output modifier),
    // NOT the old stub — so a ranged/modified rate evaluates against the real
    // system_random / location_lod instead of 0.
    if !rt.started {
        rt.started = true;
        rt.accumulator += eval_property(&emitter.particle_starting_count, &state).max(0.0);
    }
    let rate = eval_property(&emitter.particle_emission_rate, &state).max(0.0);
    rt.accumulator += rate * dt;
    // max_count caps the accumulator by the live-particle headroom — engine
    // `if (max>0) acc = min(acc, max_count - live)`. The `× location_lod`
    // thinning is the property's authored OUTPUT MODIFIER (× state[11]),
    // applied inside `eval_property` — NOT a blanket `× lod` on every emitter
    // (the engine only thins those that author the modifier; the rest are
    // gated by the lod==0 release in the caller). This is also what was
    // collapsing snowflake_heavy's ranged-on-system_random max_count to 0.
    let max_count = eval_property(&emitter.particle_max_count, &state);
    if max_count > 0.0 {
        let headroom = (max_count - rt.live).max(0.0);
        if rt.accumulator > headroom {
            rt.accumulator = headroom;
        }
    }
    let acc = rt.accumulator;
    let mut n = acc.floor();
    if n < 1.0 {
        return 0;
    }
    rt.accumulator -= n;
    if n > max_new as f32 {
        n = max_new as f32;
    }
    let count = n as u32;
    // Sub-frame spread: each particle born `particle_t = i/acc` into the
    // frame (engine `t += 1/acc`), used to offset its initial age.
    let t_step = 1.0 / acc;

    for i in 0..count {
        // Sub-frame birth time: spread the frame's particles in age so
        // they emit as a stream, not a single clump (engine particle_t).
        let particle_t = i as f32 * t_step;
        // Per-particle property evaluation (engine evaluates these
        // inside initialize_particle for every particle, so randomness
        // in the curves varies per particle).
        // Engine `c_particle_emitter::spawn_particle @0x180569A80`: the 8
        // correlated per-particle seeds (`m_random_seed[8]`) are drawn FIRST,
        // `update_states_internal @0x180487020` publishes them to state slots
        // 4-7 / 21-24 (`seed * 1/65535`), and ONLY THEN does `initialize_
        // particle` evaluate the spawn-time properties — so a property ranged
        // on one of those slots reads its real seed at birth. (Previously the
        // 8 were drawn AFTER the spawn-time eval and never placed in the eval
        // state → any spawn property ranged on slots 4-7/21-24 saw 0.)
        // slot 2 = `_particle_random_seed` — the engine's separate local-RNG
        // fresh random (drawn before the 8 here to match the engine order).
        let p_rand = rt.rng.range(0.0, 1.0);
        let mut random = [0.0f32; 4];
        let mut random2 = [0.0f32; 4];
        for r in random.iter_mut() {
            *r = rt.rng.range(0.0, 1.0);
        }
        for r in random2.iter_mut() {
            *r = rt.rng.range(0.0, 1.0);
        }
        state[0] = 0.0; // particle_age ≈ 0 at birth
        state[2] = p_rand; // _particle_random_seed
        state[4] = random[0];
        state[5] = random[1];
        state[6] = random[2];
        state[7] = random[3];
        state[21] = random2[0];
        state[22] = random2[1];
        state[23] = random2[2];
        state[24] = random2[3];
        let lifespan = eval_property(&emitter.particle_lifespan, &state).max(0.01);
        let emission_radius = eval_property(&emitter.emission_radius, &state);
        let emission_angle = eval_property(&emitter.emission_angle, &state);
        let speed_raw = eval_property(&emitter.particle_initial_velocity, &state);
        let speed = if speed_raw == 0.0 { 0.001 } else { speed_raw };
        let size = eval_property(&emitter.particle_size, &state).max(0.0001);
        let rotation_rate = eval_property(&emitter.particle_initial_rotation_rate, &state);
        // Alpha black-point — the soft-edge fade input the PS feeds into
        // `remap_alpha`. Authored on the emitter's particle_alpha_black_point.
        let black_point =
            eval_property(&emitter.particle_alpha_black_point, &state).clamp(0.0, 1.0);
        // Emitter tint (the particle's authored color). engine
        // initialize_particle seeds m_color from this × the particle's
        // color property; we apply the tint (the per-age gradient lands
        // with animated color).
        let tint = eval_color_property(&emitter.particle_tint, &state);
        let alpha = eval_property(&emitter.particle_alpha, &state).clamp(0.0, 1.0);

        let (local_pos, local_vel) = emit_shape(
            emitter.emission_shape.get(),
            emission_radius,
            emission_angle,
            &mut rt.rng,
        );

        // normalize3d(velocity) then ×speed — exactly the engine tail of
        // initialize_particle. (Most shapes already produce a unit
        // direction, but disc/line/etc. don't, and the engine normalizes
        // unconditionally.)
        let dir = if local_vel.length_squared() > 1e-12 {
            local_vel.normalize()
        } else {
            Vec3::X
        };
        let vel = dir * speed;

        // Orient emission by the emitter matrix = calc_matrix(location). The
        // location matrix (host_world × marker_local) carries the marker
        // rotation (e.g. null_up's "marker" → vertical), so a `line` emitter
        // firing along +X sprays the authored direction. Engine
        // `spawn_particle @0x180569A80` (WORLD coordinate system): the spawn
        // ORIGIN is interpolated `previous_position → current origin` by this
        // particle's sub-frame `particle_t`, then the rotated local offset is
        // added — `pos = lerp(prev,cur,t) + R·local`. For a stationary emitter
        // `lerp(prev,cur,t) == cur` and `cur + R·local == transform_point3`, so
        // this is identical to the old single-origin transform; a moving
        // emitter streams particles along its path instead of clumping.
        let spawn_origin = prev_origin.lerp(cur_origin, particle_t);
        let world_pos = spawn_origin + emitter_matrix.transform_vector3(local_pos);
        // velocity = R·local_vel, plus the inherited effect velocity when the
        // system opts in (engine `if (m_flags & 0x40) vel += effect_velocity`).
        let world_vel = emitter_matrix.transform_vector3(vel) + inherited_vel;

        // (`random` / `random2` — the 8 correlated seeds, engine
        // `m_random_seed[8]` stored ushort4N — were drawn up-front BEFORE the
        // spawn-time property eval and published to state slots 4-7/21-24.)
        out.push(ParticleState {
            position: world_pos.to_array(),
            size,
            velocity: world_vel.to_array(),
            aspect: 1.0,
            random,
            random2,
            // Starting rotation phase — RANDOM only when the prt3 has the
            // `random starting rotation` appearance flag (engine
            // `c_particle_definition::initialize_particle`: physical_rotation
            // = 0, randomized only if appearance bit set). Velocity-aligned
            // sprites (man cannon antigrav_light, no flag) must NOT be spun,
            // or their streaks cross into X patterns instead of running along
            // the jet. The engine only draws the RNG inside that gate.
            physical_rotation: if random_rotation { rt.rng.range(0.0, 1.0) } else { 0.0 },
            manual_rotation: 0.0,
            animated_frame: 0.0,
            manual_frame: 0.0,
            birth_time,
            inverse_lifespan: 1.0 / lifespan,
            // Sub-frame age offset (engine: age += particle_t·dt·invlife).
            age: particle_t * dt / lifespan,
            intensity: 1.0,
            rotational_velocity: rotation_rate,
            frame_velocity: 0.0,
            black_point,
            palette_v: 0.0,
            axis: [1.0, 0.0, 0.0],
            // Birth color = emitter tint × alpha. Per-frame property
            // re-evaluation (P1/P2) will animate this on the GPU; until
            // then it's a static baked value. initial_color.w is the HDR
            // exponent (0 → exp2(0)=1).
            color: [tint[0], tint[1], tint[2], alpha],
            initial_color: [1.0, 1.0, 1.0, 0.0],
        });
    }
    count
}

/// The 11 emission shapes from `initialize_particle @0x1805673f0`,
/// computed in emitter-LOCAL space (+X = forward). Returns
/// `(position, velocity)` — velocity is later normalized and scaled by
/// speed. `r` = emission radius, `angle_deg` = emission angle (degrees).
fn emit_shape(
    shape: EmissionShape,
    r: f32,
    angle_deg: f32,
    rng: &mut ParticleRng,
) -> (Vec3, Vec3) {
    const TWO_PI: f32 = std::f32::consts::TAU;
    let deg2rad = std::f32::consts::PI / 180.0;
    match shape {
        // case 0 — sprayer: cone of half-angle `theta` around +X.
        EmissionShape::Sprayer => {
            let a = rng.range(0.0, TWO_PI);
            let theta = rng.range(0.0, angle_deg * deg2rad);
            let st = theta.sin();
            (
                Vec3::ZERO,
                Vec3::new(theta.cos(), a.sin() * st, a.cos() * st),
            )
        }
        // case 1 — disc: ring of radius R in the YZ plane, velocity
        // radially outward in that plane.
        EmissionShape::Disc => {
            let a = rng.range(0.0, TWO_PI);
            let (sa, ca) = (a.sin(), a.cos());
            (Vec3::new(0.0, sa * r, ca * r), Vec3::new(0.0, sa, ca))
        }
        // case 2 — globe: random unit dir, position pushed out by R.
        EmissionShape::Globe => {
            let d = random_direction3d(rng);
            (d * r, d)
        }
        // case 3 — implode: random unit dir position, velocity inward.
        EmissionShape::Implode => {
            let d = random_direction3d(rng);
            (d * r, -d)
        }
        // case 4 — tube: ring@R, velocity = cone tilted by ring angle.
        EmissionShape::Tube => {
            let a = rng.range(0.0, TWO_PI);
            let theta = angle_deg * deg2rad; // fixed, not random
            let (sa, ca) = (a.sin(), a.cos());
            let st = theta.sin();
            (
                Vec3::new(0.0, sa * r, ca * r),
                Vec3::new(theta.cos(), st * sa, st * ca),
            )
        }
        // case 5 — halo: cone velocity, position on a random-dir sphere@R.
        EmissionShape::Halo => {
            let a = rng.range(0.0, TWO_PI);
            let theta = rng.range(0.0, angle_deg * deg2rad);
            let st = theta.sin();
            let d = random_direction3d(rng);
            (
                d * r,
                Vec3::new(theta.cos(), a.sin() * st, a.cos() * st),
            )
        }
        // case 8 — debris: ring@R, velocity straight +Z.
        EmissionShape::Debris => {
            let a = rng.range(0.0, TWO_PI);
            (
                Vec3::new(0.0, a.sin() * r, a.cos() * r),
                Vec3::new(0.0, 0.0, 1.0),
            )
        }
        // case 9 — line: position along Y in [-R, R], velocity cone.
        EmissionShape::Line => {
            let a = rng.range(0.0, TWO_PI);
            let theta = rng.range(0.0, angle_deg * deg2rad);
            let st = theta.sin();
            let y = rng.range(-r, r);
            (
                Vec3::new(0.0, y, 0.0),
                Vec3::new(theta.cos(), a.sin() * st, a.cos() * st),
            )
        }
        // case 6/7 impact rect + case 10 breakable need effect-impact /
        // breakable-surface context we don't carry yet — emit at origin
        // forward (+X) as a safe placeholder.
        EmissionShape::ImpactContour
        | EmissionShape::ImpactArea
        | EmissionShape::BreakableSurface => (Vec3::ZERO, Vec3::X),
    }
}

/// `c_particle_system::internal_random_direction3d` — a uniform unit
/// vector. The engine reads a 1026-entry precomputed table; we generate
/// an equivalent uniform-on-sphere direction from two LCG draws (same
/// distribution, different stream). Used by globe/implode/halo.
fn random_direction3d(rng: &mut ParticleRng) -> Vec3 {
    let z = rng.range(-1.0, 1.0);
    let phi = rng.range(0.0, std::f32::consts::TAU);
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(z, r * phi.cos(), r * phi.sin())
}

#[cfg(test)]
mod prev_origin_tests {
    use glam::{Mat4, Quat, Vec3};

    /// The world-coordinate spawn formula `lerp(prev,cur,t) + R·local` must be
    /// BIT-IDENTICAL to the old `emitter_matrix.transform_point3(local)` for a
    /// stationary emitter (prev == cur) at every sub-frame `particle_t`. This is
    /// the backward-compat invariant the previous_position change rests on — a
    /// regression here would shift every particle's spawn position.
    #[test]
    fn world_spawn_stationary_equals_transform_point() {
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4),
            Vec3::new(3.0, -4.0, 5.0),
        );
        let cur = m.transform_point3(Vec3::ZERO); // = translation column
        for &local in &[
            Vec3::new(0.3, -0.2, 0.5),
            Vec3::ZERO,
            Vec3::new(-1.0, 2.0, -3.0),
        ] {
            for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let spawn_origin = cur.lerp(cur, t); // stationary: prev == cur
                let got = spawn_origin + m.transform_vector3(local);
                let want = m.transform_point3(local);
                assert!((got - want).length() < 1e-5, "local={local} t={t}: {got} vs {want}");
            }
        }
    }

    /// A MOVING emitter streams: the particle born at frame start (t=0) spawns
    /// at the previous origin, the one at frame end (t=1) at the current origin
    /// (each plus the same rotated local offset), and inherit-velocity is the
    /// origin delta over dt.
    #[test]
    fn world_spawn_moving_streams_and_inherits() {
        let r = Mat4::from_rotation_z(0.3);
        let prev = Vec3::new(0.0, 0.0, 0.0);
        let cur = Vec3::new(6.0, 3.0, -1.5);
        let local = Vec3::new(0.4, -0.1, 0.2);
        let off = r.transform_vector3(local);
        let at0 = prev.lerp(cur, 0.0) + off;
        let at1 = prev.lerp(cur, 1.0) + off;
        assert!((at0 - (prev + off)).length() < 1e-5);
        assert!((at1 - (cur + off)).length() < 1e-5);
        // midpoint particle lands halfway along the motion path.
        let mid = prev.lerp(cur, 0.5) + off;
        assert!((mid - ((prev + cur) * 0.5 + off)).length() < 1e-5);
        // inherit-velocity = (cur − prev)/dt.
        let dt = 0.5f32;
        let inherited = (cur - prev) / dt;
        assert!((inherited - Vec3::new(12.0, 6.0, -3.0)).length() < 1e-5);
    }
}
