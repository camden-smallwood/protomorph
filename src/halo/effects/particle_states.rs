//! Packed per-particle GPU-grid state (engine `particle_states.cpp`).
//!
//! [`ParticleState`] is the logical per-particle birth state the CPU builds
//! (engine `s_particle_state`); [`ParticleState::pack`] produces the 80-byte
//! [`RawParticleState`] the GPU grid stores (engine `s_raw_particle_state`,
//! `particle_pack_fx.hlsl`). The GPU spawn is a pure scatter-copy of the
//! packed state — it does no initialization.

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
