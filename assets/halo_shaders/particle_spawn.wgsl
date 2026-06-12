// particle_spawn.wgsl — scatter copy. The engine's GPU spawn does NO
// initialization: the CPU builds the full 80-byte RawParticleState into
// a staging buffer + an address buffer, and this kernel copies each
// staged state into the live grid at its decoded address.
//
// Engine `particle_spawn_hlsl.hlsl::default_cs`:
//   out = (addr & 0xffff) + ((addr >> 16) * 16);   // 16-wide grid
//   cs_particle_state_buffer[out] = cs_particle_state_spawn_buffer[i];
//
// RawParticleState = 20 u32 (80 bytes) — exact DX11 s_raw_particle_state.

struct RawParticleState {
    words: array<u32, 20>,
};

@group(0) @binding(0) var<storage, read_write> grid: array<RawParticleState>;
@group(0) @binding(1) var<storage, read> staging: array<RawParticleState>;
@group(0) @binding(2) var<storage, read> addresses: array<u32>;
// params.x = spawn count this frame.
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn cs_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.x) {
        return;
    }
    let addr = addresses[i];
    let out = (addr & 0xffffu) + ((addr >> 16u) * 16u);
    grid[out] = staging[i];
}
