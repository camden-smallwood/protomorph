// particle_update.wgsl — per-frame integrate + property re-evaluation.
// Prepended by particle_eval.wgsl (the curve-evaluation engine + its
// bindings 3-6: g_properties / g_functions / g_colors / g_state_uniform).
//
// Engine `particle_update_hlsl.hlsl::update_particle_state`: age, then if
// alive, integrate (pos uses OLD vel; gravity -Z; saturated frictions;
// frac rotations) and RE-EVALUATE the visual properties from the per-
// emitter curve tables. The visible props are particle-age-keyed, so
// they animate over each particle's life.
//
// RawParticleState word layout (20 u32) — see particle_spawn.wgsl.

struct RawParticleState { words: array<u32, 20>, };

@group(0) @binding(0) var<storage, read_write> grid: array<RawParticleState>;
struct UpdateParams {
    dt: f32,
    region_len: u32,
    batch_count: u32,
    _pad: u32,
};
@group(0) @binding(1) var<uniform> params: UpdateParams;
// Per-batch physics: (gravity·grav_mod, air_friction, rot_friction, 0).
@group(0) @binding(2) var<storage, read> physics_table: array<vec4<f32>>;
// Per-row owning batch (engine row allocation): row = slot / 16. A row
// reading 0xffffffff is free/dead — skip it. Replaces the old fixed-region
// `batch = slot / region_len` routing.
@group(0) @binding(8) var<storage, read> row_batch: array<u32>;

fn pack_argb8(c: vec4<f32>) -> u32 {
    let q = clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0;
    return u32(q.z) | (u32(q.y) << 8u) | (u32(q.x) << 16u) | (u32(q.w) << 24u);
}

@compute @workgroup_size(64)
fn cs_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&grid)) {
        return;
    }
    let batch = row_batch[idx / 16u];
    if (batch == 0xffffffffu) {
        return; // free / dead row
    }
    let phys = physics_table[batch];

    var s = grid[idx];

    // time: [12]=(birth, invlife) [13]=(age, intensity)
    let t0 = unpack2x16float(s.words[12]);
    let birth = t0.x;
    let invlife = t0.y;
    if (invlife <= 0.0) {
        return; // empty / dead slot
    }
    let t1 = unpack2x16float(s.words[13]);
    var age = t1.x;

    // Engine spawn-frame "hack" (particle_update_hlsl.hlsl:156-157): a
    // freshly-spawned particle is stored with NEGATIVE m_size, so its first
    // update runs dt=0 — no integration AND no age advance (the CPU already
    // set its sub-frame birth age). `size` is rewritten positive below
    // (word[3]), so every subsequent frame integrates with the real dt.
    let spawn_frame = bitcast<f32>(s.words[3]) < 0.0;
    let dt = select(params.dt, 0.0, spawn_frame);
    age = age + invlife * dt;
    if (age >= 1.0) {
        s.words[12] = pack2x16float(vec2<f32>(birth, 0.0)); // free slot
        s.words[13] = pack2x16float(vec2<f32>(1.0, 0.0));
        grid[idx] = s;
        return;
    }

    // --- unpack the rest ---
    var pos = vec3<f32>(
        bitcast<f32>(s.words[0]), bitcast<f32>(s.words[1]), bitcast<f32>(s.words[2]),
    );
    let v0 = unpack2x16float(s.words[4]);
    let v1 = unpack2x16float(s.words[5]);
    var vel = vec3<f32>(v0.x, v0.y, v1.x);
    let anm = unpack2x16float(s.words[14]);
    var rot_vel = anm.x;
    let frame_vel = anm.y;
    let rotw = unpack2x16unorm(s.words[10]);
    var phys_rot = rotw.x;
    let anim_frame = unpack2x16unorm(s.words[11]);

    // --- build the eval state (per-particle inputs) ---
    var st: EvalState;
    st.age = age;
    st.birth_time = birth;
    st.random = vec4<f32>(unpack2x16unorm(s.words[6]), unpack2x16unorm(s.words[7]));
    st.random2 = vec4<f32>(unpack2x16unorm(s.words[8]), unpack2x16unorm(s.words[9]));

    // --- preevaluate the properties used for the visible state ---
    let pre_tint = particle_evaluate(st, batch, P_EMITTER_TINT);
    let pre_e_alpha = particle_evaluate(st, batch, P_EMITTER_ALPHA);
    let pre_e_size = particle_evaluate(st, batch, P_EMITTER_SIZE);
    let pre_scale = particle_evaluate(st, batch, P_PARTICLE_SCALE);
    let pre_rotation = particle_evaluate(st, batch, P_PARTICLE_ROTATION);
    let pre_black = particle_evaluate(st, batch, P_PARTICLE_BLACK_POINT);
    let pre_self_accel_t = particle_evaluate(st, batch, P_PARTICLE_SELF_ACCEL);
    // prt3-level properties (now curve-wired).
    let pre_p_color = particle_evaluate(st, batch, P_PARTICLE_COLOR);
    let pre_p_alpha = particle_evaluate(st, batch, P_PARTICLE_ALPHA);
    let pre_intensity = particle_evaluate(st, batch, P_PARTICLE_INTENSITY);
    let pre_aspect = particle_evaluate(st, batch, P_PARTICLE_ASPECT);
    let pre_palette = particle_evaluate(st, batch, P_PARTICLE_PALETTE);
    let pre_frame = particle_evaluate(st, batch, P_PARTICLE_FRAME);

    // --- integrate (engine update_particle_state order) ---
    pos = pos + vel * dt;                    // OLD velocity
    vel = vel + particle_self_accel(batch, pre_self_accel_t) * dt; // self-accel (slerp'd vector)
    vel.z = vel.z - dt * phys.x;             // gravity · grav_mod
    vel = vel * (1.0 - clamp(phys.y * dt, 0.0, 1.0));
    rot_vel = rot_vel * (1.0 - clamp(phys.z * dt, 0.0, 1.0));
    phys_rot = fract(phys_rot + rot_vel * dt);
    let manual_rot = fract(pre_rotation);
    let new_anim_frame = fract(anim_frame.x + frame_vel * dt);

    // --- recompute the visible state from the curves (engine
    // update_particle_state): emitter × particle for color/alpha. ---
    let rgb = particle_map_to_color(batch, P_EMITTER_TINT, pre_tint)
        * particle_map_to_color(batch, P_PARTICLE_COLOR, pre_p_color);
    let alpha = pre_e_alpha * pre_p_alpha;
    let size = pre_e_size * pre_scale;
    let intensity = pre_intensity;
    let black_point = saturate(pre_black) * 0.9999995;
    let palette_v = saturate(pre_palette) * 0.9999995;
    let aspect = pre_aspect;
    let manual_frame = fract(pre_frame);

    // --- write back ---
    s.words[0] = bitcast<u32>(pos.x);
    s.words[1] = bitcast<u32>(pos.y);
    s.words[2] = bitcast<u32>(pos.z);
    s.words[3] = bitcast<u32>(size);
    s.words[4] = pack2x16float(vec2<f32>(vel.x, vel.y));
    s.words[5] = pack2x16float(vec2<f32>(vel.z, aspect));
    s.words[10] = pack2x16unorm(vec2<f32>(phys_rot, manual_rot));
    s.words[11] = pack2x16unorm(vec2<f32>(new_anim_frame, manual_frame));
    s.words[13] = pack2x16float(vec2<f32>(age, intensity));
    s.words[14] = pack2x16float(vec2<f32>(rot_vel, frame_vel));
    s.words[15] = pack2x16unorm(vec2<f32>(black_point, palette_v));
    s.words[17] = pack_argb8(vec4<f32>(rgb, alpha));
    grid[idx] = s;
}
