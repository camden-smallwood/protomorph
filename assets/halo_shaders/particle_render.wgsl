// particle_render.wgsl — billboard expand + shade the persistent grid,
// one draw per render-batch (a particle system's material).
//
// Faithful to particle_render_hlsl.hlsl:
//   - VS color  (line 656): m_color = state.color · initial_color ·
//     intensity · exp2(initial_color.w), then ·= v_exposure.
//   - albedo `sample_diffuse` (line 810): per-option rgb/alpha sourcing,
//     incl. palettized index→palette[index, palette_v].
//   - black_point `remap_alpha` (line 768) smooth edge fade.
//   - blended *= m_color ; multiply mode does lerp(1,rgb,a).
//
// group0 = frame-shared (camera, grid, exposure); group1 = per-batch
// material (base/alpha/palette + sampler + the albedo/blend/black_point
// selectors).

struct RawParticleState { words: array<u32, 20>, };

struct Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};
// frame params: v.x = view exposure, v.y = size scale, v.z = near_clip,
// v.w = depth_fade_range (world units). v2.x = distortion strength
// (UV-fraction scale for the displacement write; only the distortion pass
// uses it).
struct FrameParams { v: vec4<f32>, v2: vec4<f32>, };
// per-batch material selectors: x = albedo option, y = black_point on,
// z = blend_mode option. `corner` = the baked sprite corner
// (xy = bottom-left, zw = extent) the engine bakes in
// c_particle_definition::postprocess_frame_animation — billboard vertex =
// shift·corner.zw + corner.xy, then ·particle_size. For a square,
// centered, no-sequence bitmap this is (-1,-1,2,2) → a [-1,1] quad
// (2·size world extent), NOT a unit quad.
// extra.x = rotation_offset (turns added to every sprite).
// Must match gpu_particle::MAX_SPRITE_FRAMES.
const MAX_SPRITE_FRAMES = 64u;
struct MatParams {
    v: vec4<u32>,
    corner: vec4<f32>,
    extra: vec4<f32>,
    // Per-emitter initial color (engine `m_initial_color`): rgb tint +
    // HDR exponent in .w. White `(1,1,1,0)` unless the appearance flags
    // request lightprobe tinting (bit3 `tint from lightmap` → the
    // emitter's lightprobe DC color; bit4 `tint from diffuse`). The VS
    // folds `rgb · exp2(.w)` into the particle color (engine
    // `particle_render_hlsl.hlsl:656`). protomorph carries the HDR value
    // directly in `rgb` (float uniform), so `.w` is 0.
    initial_color: vec4<f32>,
    // frames_meta: x = frame count (≤1 → plain bitmap, full quad),
    // y = anim flags (bit0 one_shot, bit1 can_animate_backwards),
    // z = frame_blend on, w = fog on.
    frames_meta: vec4<u32>,
    // fade: x = near_range (1/near_fade_range, 0 = disabled), y = near_cutoff
    // (engine always-on near-camera fade, set_shader_render_state
    // @0x1806a67c0); z = intensity-affects-alpha (appearance bit 6, by-name
    // resolved: 1 → `m_color.w *= m_intensity`); w = random-UV-mirror flags
    // (appearance bits 0/1, by-name: bit0 = flip-u, bit1 = flip-v).
    fade: vec4<f32>,
    // edge: x = edge_range (1/radians(angle_fade_range), 0 = disabled),
    // y = edge_cutoff (radians(angle_fade_cutoff)). Appearance bit 7 "fade
    // when viewed edge-on" (set_shader_render_state @0x1806a67c0 /
    // particle_render.hlsl:714): alpha *= saturate(edge_range·(billboard_angle
    // − edge_cutoff)). z/w unused.
    edge: vec4<f32>,
    // Per-frame sprite-sheet UV rects in engine frame_texcoord form
    // (left, top, right-left, bottom-top).
    frame_uv: array<vec4<f32>, MAX_SPRITE_FRAMES>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> grid: array<RawParticleState>;
@group(0) @binding(2) var<uniform> frame: FrameParams;
// Scene (opaque+water) depth — sampled for the soft depth fade. The
// engine particle pass uses NO hardware depth test; the fade both
// occludes (behind geometry → 0) and softens near-surface intersection.
@group(0) @binding(3) var scene_depth: texture_depth_2d;

// Frame-default atmosphere setting (the cluster-weighted accumulator
// entry of the renderer's shared atmosphere table). Struct + accessor
// mirror `engine_bindings.wgsl` 1:1 — `compute_scattering` (appended
// from `atmosphere_fx.wgsl` at module assembly) reads it through
// `make_engine_atmosphere()`. Used by the `fog` category's per-vertex
// analytic fog (engine particle_render_hlsl.hlsl:673).
struct EngineAtmosphereRaw {
    slot0_sun_dir_dist_bias: vec4<f32>,        // (sun_dir.xyz, distance_bias)
    slot1_sun_int_norm_thickness: vec4<f32>,   // (sun_int / (β_m+β_p), max_fog_thickness OR -1)
    slot2_beta_m_log2e_g1: vec4<f32>,          // (β_m × log2(e), g+1)
    slot3_beta_p_log2e_refh: vec4<f32>,        // (β_p × log2(e), reference_height)
    slot4_beta_m_angular_mieh: vec4<f32>,      // (β_m_angular, mie_height)
    slot5_beta_p_angular_rayh: vec4<f32>,      // ((1-g²)×β_p_angular, rayleigh_height)
    slot6_g2: vec4<f32>,                       // (2g, _, _, _)
    slot7_sun_disc: vec4<f32>,
    slot8_sun_glow: vec4<f32>,
    slot9_sun_tint_horizon: vec4<f32>,
    slot10_horizon_pad: vec4<f32>,
}
@group(0) @binding(4) var<uniform> engine_atmosphere_raw: EngineAtmosphereRaw;

struct EngineAtmosphere {
    sun_direction: vec3<f32>,
    distance_bias: f32,
    sun_intensity_normalized: vec3<f32>,
    max_fog_thickness: f32,
    beta_m_log2e: vec3<f32>,
    mie_g_plus_one: f32,
    beta_p_log2e: vec3<f32>,
    reference_height: f32,
    beta_m_angular: vec3<f32>,
    mie_height_scale: f32,
    beta_p_angular: vec3<f32>,
    rayleigh_height_scale: f32,
    mie_g_times_two: f32,
}
fn make_engine_atmosphere() -> EngineAtmosphere {
    let r = engine_atmosphere_raw;
    var out: EngineAtmosphere;
    out.sun_direction = r.slot0_sun_dir_dist_bias.xyz;
    out.distance_bias = r.slot0_sun_dir_dist_bias.w;
    out.sun_intensity_normalized = r.slot1_sun_int_norm_thickness.xyz;
    out.max_fog_thickness = r.slot1_sun_int_norm_thickness.w;
    out.beta_m_log2e = r.slot2_beta_m_log2e_g1.xyz;
    out.mie_g_plus_one = r.slot2_beta_m_log2e_g1.w;
    out.beta_p_log2e = r.slot3_beta_p_log2e_refh.xyz;
    out.reference_height = r.slot3_beta_p_log2e_refh.w;
    out.beta_m_angular = r.slot4_beta_m_angular_mieh.xyz;
    out.mie_height_scale = r.slot4_beta_m_angular_mieh.w;
    out.beta_p_angular = r.slot5_beta_p_angular_rayh.xyz;
    out.rayleigh_height_scale = r.slot5_beta_p_angular_rayh.w;
    out.mie_g_times_two = r.slot6_g2.x;
    return out;
}

// ---- inlined from atmosphere_fx.wgsl (compute_scattering + consts;
// the full file carries assembly-time placeholders so it can't be
// concatenated raw) ----
const k_log2_e: f32 = 1.4426950;

// DIAGNOSTIC: when true, force every opaque entry-point's atmosphere
// term to a no-op (extinction=1, inscatter=0). Used to bisect the
// "everything washed-out white" symptom — if bypass restores
// saturated lit color across distance, the problem is in the
// atmosphere term (cbuffer values or formula); if it doesn't, the
// problem is upstream (exposure / sqrt / lit).
const ATMOSPHERE_BYPASS: bool = false;

fn compute_scattering(
    view_point: vec3<f32>,
    world_scene_point: vec3<f32>,
    extinction: ptr<function, vec3<f32>>,
    inscatter: ptr<function, vec3<f32>>,
) {
    if (ATMOSPHERE_BYPASS) {
        *extinction = vec3<f32>(1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }

    let atm = make_engine_atmosphere();

    // Halo uses slot1.w (max_fog_thickness) as the disable sentinel —
    // negative when atmosphere is off (atmosphere_fx.hlsl line 21+37).
    if (atm.max_fog_thickness < 0.0) {
        *extinction = vec3<f32>(1.0, 1.0, 1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }

    var view_vector = view_point - world_scene_point;
    var dist = length(view_vector);
    if (dist < 1e-6) {
        *extinction = vec3<f32>(1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }
    view_vector = view_vector / dist;
    let c_theta = -dot(view_vector, atm.sun_direction);

    // HLSL line 49-50: dist = max(dist + DIST_BIAS, 0); dist = min(dist, MAX_FOG_THICKNESS).
    // Riverworld's haze_level setting has Distance Bias = -15, meaning
    // close-range fog is suppressed for the first 15 world units.
    dist = max(dist + atm.distance_bias, 0.0);
    dist = min(dist, atm.max_fog_thickness);

    var view_height = max(view_point.z - atm.reference_height, 0.0);
    var scene_height = max(world_scene_point.z - atm.reference_height, 0.0);
    let diff = view_height - scene_height;

    view_height = view_height * k_log2_e;
    scene_height = scene_height * k_log2_e;

    let mie_h = max(atm.mie_height_scale, 1e-4);
    let ray_h = max(atm.rayleigh_height_scale, 1e-4);

    if (diff * diff > 0.001) {
        let dp = -(exp2(-view_height / mie_h) - exp2(-scene_height / mie_h)) * dist * mie_h / diff;
        let dm = -(exp2(-view_height / ray_h) - exp2(-scene_height / ray_h)) * dist * ray_h / diff;
        *extinction = exp2(-(atm.beta_m_log2e * dm + atm.beta_p_log2e * dp));
    } else {
        let dp = exp2(-view_height / mie_h) * dist;
        let dm = exp2(-view_height / ray_h) * dist;
        *extinction = exp2(-(atm.beta_m_log2e * dm + atm.beta_p_log2e * dp));
    }

    // Rayleigh phase: RAYLEIGH_THETA_PREFIX × (1 + cos²θ)
    let beta_m_theta = atm.beta_m_angular * (1.0 + c_theta * c_theta);

    // Mie HG phase: MIE_THETA_PREFIX_HGC × (1+g²-2g·cosθ)^-1.5
    // (β_p_angular is already pre-multiplied by (1-g²) in set_constant.)
    let heyey_term = atm.mie_g_plus_one - atm.mie_g_times_two * c_theta;
    let heyey_term_one_pt_five = pow(max(heyey_term, 1e-4), -1.5);
    let beta_p_theta = atm.beta_p_angular * heyey_term_one_pt_five;

    // SUN_INTENSITY_OVER_TR_PLUS_TM × (β_m_θ + β_p_θ) × (1 - extinction)
    *inscatter = atm.sun_intensity_normalized
              * (beta_m_theta + beta_p_theta)
              * (vec3<f32>(1.0) - *extinction);
}

@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var alpha_tex: texture_2d<f32>;
@group(1) @binding(2) var palette_tex: texture_2d<f32>;
@group(1) @binding(3) var samp: sampler;
@group(1) @binding(4) var<uniform> mat: MatParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    /// Billboard UV (the [0,1] quad) — used for billboard-alpha albedo
    /// variants and the distortion dudv lookup.
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) black_point: f32,
    @location(3) palette_v: f32,
    /// View-space distance (engine `depth = dot(Camera_Forward, pos-Camera_Position)`).
    @location(4) view_z: f32,
    /// Billboard u/v axes projected into view space (xy = screen-plane
    /// directions of increasing u / v). The distortion pass orients the
    /// bitmap displacement by these (engine `m_tangent` / `m_binormal`,
    /// view space — particle_render_hlsl.hlsl:612-615).
    @location(5) view_tan: vec2<f32>,
    @location(6) view_bin: vec2<f32>,
    /// Sprite-sheet frame UVs (remapped from `uv` into the frame's sub-rect).
    /// For plain bitmaps these equal `uv` (full quad). `sprite_uv1` + the
    /// blend feed the frame_blend cross-fade.
    @location(7) sprite_uv0: vec2<f32>,
    @location(8) sprite_uv1: vec2<f32>,
    @location(9) frame_blend: f32,
    /// Engine `m_color_add` — currently just the `fog` category's
    /// analytic inscatter (`compute_scattering` × v_exposure, VS line
    /// 678). Zero when the category is off. The PS adds it after the
    /// texture multiply in every non-multiply blend mode (engine
    /// particle_render_hlsl.hlsl:1006).
    @location(10) color_add: vec3<f32>,
};

// Engine compute_variant (particle_render_hlsl.hlsl:355) — maps the
// normalized (animated + manual) frame input to a frame position in
// [0, count). one_shot stops at the last frame (no wrap); backwards reverses.
fn compute_variant(input: f32, count: u32, one_shot: bool, backwards: bool) -> f32 {
    if (count == 1u) { return 0.0; }
    var c = f32(count);
    if (one_shot) { c = c - 1.0; }
    var variant = c * input;
    if (backwards) { variant = c - variant; }
    return variant;
}

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_particle(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    var out: VsOut;
    let s = grid[ii];

    // time: [12]=(birth, invlife) [13]=(age, intensity)
    let t0 = unpack2x16float(s.words[12]);
    let invlife = t0.y;
    let t1 = unpack2x16float(s.words[13]);
    let age = t1.x;
    let intensity = t1.y;
    if (invlife <= 0.0 || age >= 1.0) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0); // offscreen → culled
        return out;
    }

    let pos = vec3<f32>(
        bitcast<f32>(s.words[0]),
        bitcast<f32>(s.words[1]),
        bitcast<f32>(s.words[2]),
    );
    // Engine billboard: vertex_pos = shift·m_corner.zw + m_corner.xy, then
    // ·m_size (particle_render_hlsl.hlsl:563,597). For the default square
    // sprite m_corner = (-1,-1,2,2), so vertex_pos ∈ [-1,1] → 2·m_size
    // world extent. aspect scales x.
    let size = bitcast<f32>(s.words[3]) * frame.v.y;
    // World velocity (words[4]=vx,vy; words[5].x=vz) — drives velocity-
    // aligned billboards (types 2/3/9) and the motion-blur stretch.
    let vw = unpack2x16float(s.words[4]);
    let vel = vec3<f32>(vw.x, vw.y, unpack2x16float(s.words[5]).x);
    // Sprite aspect (x stretch). Motion blur (engine
    // particle_render_hlsl.hlsl:588): when the `motion blur` appearance bit is
    // set (mat.extra.z = prt3 motion_blur_aspect_scale, 0 = off) add
    // |rel_vel|·scale/(30·size) to x so fast particles streak along their
    // motion. Static free-cam → relative velocity ≈ the particle's world
    // velocity (the observer velocity/rotation terms are ~0).
    var aspect = unpack2x16float(s.words[5]).y;
    if (mat.extra.z > 0.0 && size > 1.0e-5) {
        aspect += length(vel) * mat.extra.z / (30.0 * size);
    }
    // col (argb8) at word[17]; unpack4x8unorm gives (r,g,b,a) from our
    // pack order? Our pack_argb8 stores b|g<<8|r<<16|a<<24, so byte0=b,
    // byte1=g, byte2=r, byte3=a. unpack4x8unorm returns x=byte0..w=byte3
    // = (b,g,r,a); swizzle to rgba.
    let raw_col = unpack4x8unorm(s.words[17]);
    var color = vec4<f32>(raw_col.z, raw_col.y, raw_col.x, raw_col.w);
    // anm2 word[15] = (black_point, palette_v) ushort2N
    let bp = unpack2x16unorm(s.words[15]);

    let shift = CORNERS[vi] * 0.5 + 0.5;            // [0,1] quad
    let vp = shift * mat.corner.zw + mat.corner.xy;  // sprite-space position
    // Per-particle in-plane rotation (engine vertex_orientation =
    // sincos(2π·(physical_rotation + manual_rotation)),
    // particle_render_hlsl.hlsl:567). rot at word[10] ushort2N
    // (x = physical, y = manual). `mat.rotation_offset` adds the
    // bitmap-authored-vertically +0.25 turn (resolved CPU-side from the
    // typed appearance flag). Without this every sprite drew axis-aligned —
    // a static grid instead of a churning cloud.
    let rotw = unpack2x16unorm(s.words[10]);
    let rotation = rotw.x + rotw.y + mat.extra.x;
    let ang = rotation * 6.2831855;
    let cs = cos(ang);
    let sn = sin(ang);
    // In-plane position (aspect on x), then rotate.
    let pp = vec2<f32>(vp.x * aspect, vp.y);
    let rp = vec2<f32>(cs * pp.x - sn * pp.y, sn * pp.x + cs * pp.y);
    let center_view = camera.view * vec4<f32>(pos, 1.0);

    // Billboard basis in WORLD space (engine `billboard_basis`,
    // particle_render_hlsl.hlsl:220) — the in-plane (u, v) world axes the
    // quad expands along. Camera right/up/position come from the view
    // matrix (rows of its rotation = camera axes in world).
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let vt = vec3<f32>(camera.view[3][0], camera.view[3][1], camera.view[3][2]);
    let cam_pos = -vec3<f32>(
        dot(camera.view[0].xyz, vt),
        dot(camera.view[1].xyz, vt),
        dot(camera.view[2].xyz, vt),
    );
    // Billboard basis (engine billboard_basis, particle_render_hlsl.hlsl:220)
    // selected by mat.v.w = particle_billboard_type_enum (0..9). We need only
    // the in-plane u(b0)/v(b1); basis[2] (toward camera, for backface) is
    // implicit. Types 6/7 (local) approximate to world axes — the per-emitter
    // local basis isn't threaded to the render yet.
    let bb = mat.v.w;
    let to_cam = pos - cam_pos;
    var b0 = cam_right; // 0 screen-facing: u = -Camera_Left, v = Camera_Up
    var b1 = cam_up;
    if (bb == 1u) {
        // camera-facing: a stable up reference that doesn't roll with the cam.
        let eye = normalize(to_cam);
        let perp = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(eye.z) < 0.99);
        b0 = normalize(cross(eye, perp));
        b1 = normalize(cross(b0, eye));
    } else if (bb == 2u) {
        // screen-parallel: u along velocity, v toward camera → motion streak
        // (man cannon antigrav_light).
        if (length(vel) > 1.0e-4) {
            b0 = normalize(vel);
            let cv = cross(b0, to_cam);
            if (length(cv) > 1.0e-5) { b1 = normalize(cv); }
        }
    } else if (bb == 3u) {
        // screen-perpendicular: the plane perpendicular to velocity.
        if (length(vel) > 1.0e-4) {
            let motion = normalize(vel);
            let perp = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(motion.z) < 0.99);
            b0 = normalize(cross(motion, perp));
            b1 = cross(motion, b0);
        }
    } else if (bb == 4u || bb == 6u) {
        // screen-vertical (4) / local-vertical (6 → world-up fallback): stand
        // upright on world +Z, rotate about it to face the camera.
        b0 = vec3<f32>(0.0, 0.0, 1.0);
        let cv = cross(b0, to_cam);
        if (length(cv) > 1.0e-5) { b1 = normalize(cv); }
    } else if (bb == 5u || bb == 7u || bb == 8u) {
        // screen-horizontal (5) / local-horizontal (7 fallback) / world (8):
        // the world XY plane.
        b0 = vec3<f32>(1.0, 0.0, 0.0);
        b1 = vec3<f32>(0.0, 1.0, 0.0);
    } else if (bb == 9u) {
        // velocity-horizontal: u along velocity, v in the horizontal plane.
        if (length(vel) > 1.0e-4) {
            b0 = normalize(vel);
            let perp = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(b0.x) < 0.99);
            b1 = normalize(cross(perp, b0));
        }
    }
    let world_pos = pos + (rp.x * b0 + rp.y * b1) * size;
    out.clip = camera.projection * (camera.view * vec4<f32>(world_pos, 1.0));
    // Random per-particle UV mirror (appearance bits 0/1, engine
    // particle_render_hlsl.hlsl:623-630): flip u/v when the by-name flag is
    // set AND a per-particle coin lands (`int(256·m_random)&1`). Decorrelates
    // repeated sprites so a cloud doesn't read as a tiled grid. Applied to the
    // TEXCOORD only (`uv`), never the quad geometry (`shift`/`vp`). fade.w
    // packs the two flags: bit0 = flip-u, bit1 = flip-v. m_random.xy =
    // word[6] (the first two ushort2N correlations).
    let flip_flags = u32(mat.fade.w);
    let rnd_uv = unpack2x16unorm(s.words[6]);
    var uv = shift;
    if ((flip_flags & 1u) != 0u && (u32(256.0 * rnd_uv.x) & 1u) != 0u) { uv.x = 1.0 - uv.x; }
    if ((flip_flags & 2u) != 0u && (u32(256.0 * rnd_uv.y) & 1u) != 0u) { uv.y = 1.0 - uv.y; }
    out.uv = uv; // billboard UV (alpha-billboard variants + distortion)

    // Sprite-sheet frame selection (engine particle_render_hlsl.hlsl:635).
    // Plain bitmaps (frame_count ≤ 1) keep the full [0,1] quad → no change.
    var sprite_uv0 = uv;
    var sprite_uv1 = uv;
    var fblend = 0.0;
    let fcount = mat.frames_meta.x;
    if (fcount > 1u) {
        let one_shot = (mat.frames_meta.y & 1u) != 0u;
        // backwards is per-particle (engine: flag AND a random bit of random.z).
        let rnd_z = unpack2x16unorm(s.words[7]).x;
        let backwards = ((mat.frames_meta.y & 2u) != 0u)
            && ((u32(256.0 * rnd_z) & 1u) != 0u);
        // anim_frame + manual_frame from word[11] (ushort2N).
        let fr = unpack2x16unorm(s.words[11]);
        let variant = compute_variant(fr.x + fr.y, fcount, one_shot, backwards);
        let f0 = u32(floor(variant)) % fcount;
        let r0 = mat.frame_uv[f0];
        sprite_uv0 = uv * r0.zw + r0.xy; // engine frame_texcoord (scroll 0, scale 1)
        if (mat.frames_meta.z != 0u) {
            let f1 = (u32(floor(variant)) + 1u) % fcount;
            let r1 = mat.frame_uv[f1];
            sprite_uv1 = uv * r1.zw + r1.xy;
            fblend = fract(variant);
        }
    }
    out.sprite_uv0 = sprite_uv0;
    out.sprite_uv1 = sprite_uv1;
    out.frame_blend = fblend;

    out.view_z = -center_view.z; // RH view: in-front is -z → positive distance
    // u/v billboard axes in view space, SCALED by the particle's world
    // half-extent (size·corner) — engine particle_render_hlsl.hlsl:612
    // `vertex_basis *= particle_scale * g_sprite.m_corner.z/.w`. Carrying the
    // size is essential: the distortion PS divides by depth, so basis·1/depth
    // gives the billboard's true *screen-space* size. Using unit axes (the
    // first cut) made the warp magnitude independent of size/distance — a
    // uniform smear. .y is negated to convert view-space (y up) to UV (y down).
    let vt0 = (camera.view * vec4<f32>(b0, 0.0)).xy;
    let vt1 = (camera.view * vec4<f32>(b1, 0.0)).xy;
    out.view_tan = vec2<f32>(vt0.x, -vt0.y) * size * mat.corner.z;
    out.view_bin = vec2<f32>(vt1.x, -vt1.y) * size * mat.corner.w;

    // VS color (particle_render_hlsl.hlsl:656-668):
    //   m_color = color · initial_color.rgb · exp2(initial_color.w) ·
    //             intensity, then ·= exposure, branched by blend mode:
    //     multiply              → ×1 (no exposure)
    //     BLEND_MODE_SELF_ILLUM → ×V_ILLUM_EXPOSURE  (additive=1, add_src
    //                             _times_srcalpha=9 — the engine macro at
    //                             particle_render_hlsl.hlsl:78)
    //     else (alpha_blend, …) → ×v_exposure.x  (render_exposure)
    // V_ILLUM_EXPOSURE = m_illum_render_scale · render_exposure. The engine
    // pulls self-illum/additive effects back by a SEPARATE fixed self-illum
    // exposure (`setup_targets_static_lighting(render_exposure,
    // illum_render_scale, …)` → `setup_render_target_globals_with_exposure
    // @0x180670AD0`: alt_exposure.g = illum_scale·view_exposure). In a dark
    // scene the auto-exposure climbs (~2.5) but self-illum stays fixed, so
    // illum_render_scale << 1 — without it additive embers (guardian
    // earth_edge spark_ember) blow out to bright white dots instead of
    // staying invisible like the real game. frame.v.x = render_exposure,
    // frame.v2.x = V_ILLUM_EXPOSURE.
    var exposure = frame.v.x;
    if (mat.v.z == 2u) {
        exposure = 1.0; // multiply: no exposure
    } else if (mat.v.z == 1u || mat.v.z == 9u) {
        exposure = frame.v2.x; // additive / add_src_times_srcalpha → V_ILLUM
    }
    let base_rgb = color.rgb * mat.initial_color.rgb * exp2(mat.initial_color.w);
    // Engine always-on near-camera fade (particle_render_hlsl.hlsl:705-709,
    // set_shader_render_state @0x1806a67c0): alpha *= saturate(near_range·
    // (depth − near_cutoff)) so particles dissolve as the camera nears them
    // instead of filling the lens. view_z = view-forward depth = engine
    // `depth`; near_range 0 = disabled. Always applied (protomorph is never
    // the first-person view the engine exempts).
    var out_alpha = color.a;
    // intensity affects alpha (appearance bit 6, engine
    // particle_render_hlsl.hlsl:701-703): `m_color.w *= m_intensity`.
    // Resolved BY NAME on the CPU (`mat.fade.z = 1`); applied BEFORE the
    // near-camera fade, matching the engine's order.
    if (mat.fade.z != 0.0) {
        out_alpha = out_alpha * intensity;
    }
    if (mat.fade.x > 0.0) {
        out_alpha = out_alpha * clamp(mat.fade.x * (out.view_z - mat.fade.y), 0.0, 1.0);
    }
    // Edge fade (appearance bit 7 "fade when viewed edge-on", engine
    // particle_render.hlsl:714): fade as the billboard turns edge-on to the
    // view (independent of camera roll). The billboard normal = cross of the
    // in-plane basis; billboard_angle = π/2 − acos(|view·normal|) (π/2 face-on,
    // 0 edge-on); alpha *= saturate(edge_range·(billboard_angle − edge_cutoff)).
    // A no-op for camera-facing billboards (normal ∥ view); it bites for the
    // oriented particle types. edge_range = 0 ⇒ disabled (bit clear).
    if (mat.edge.x > 0.0) {
        let cam_to_vtx = normalize(world_pos - cam_pos);
        let nrm = normalize(cross(b0, b1));
        let billboard_angle = 1.5707964 - acos(clamp(abs(dot(cam_to_vtx, nrm)), 0.0, 1.0));
        out_alpha = out_alpha * clamp(mat.edge.x * (billboard_angle - mat.edge.y), 0.0, 1.0);
    }
    out.color = vec4<f32>(base_rgb * intensity * exposure, out_alpha);
    // `fog` category on (frames_meta.w) — engine particle_render_hlsl
    // .hlsl:673 `IF_CATEGORY_OPTION(fog, on)`: per-vertex analytic
    // atmosphere fog. `m_color ×= extinction` and the inscatter
    // (×v_exposure.x, the render exposure — NOT the blend-branched
    // `exposure` above) is added by the PS as m_color_add. Without this
    // fogged particles (s3d_turf drips, weather snow) punch through fog
    // banks as crisp unfogged sprites.
    out.color_add = vec3<f32>(0.0);
    if (mat.frames_meta.w != 0u) {
        var fog_extinction: vec3<f32>;
        var fog_inscatter: vec3<f32>;
        compute_scattering(cam_pos, world_pos, &fog_extinction, &fog_inscatter);
        out.color = vec4<f32>(out.color.rgb * fog_extinction, out.color.a);
        out.color_add = fog_inscatter * frame.v.x;
    }
    out.black_point = bp.x;
    out.palette_v = bp.y;
    return out;
}

// remap_alpha (particle_render_hlsl.hlsl:768) — smooth black-point fade.
fn remap_alpha(black_point: f32, alpha: f32) -> f32 {
    let mid = (black_point + 1.0) * 0.5;
    return mid * saturate((alpha - black_point) / (mid - black_point))
        + saturate(alpha - mid);
}

// sample_diffuse (particle_render_hlsl.hlsl:810). Base/palette index is read
// at the SPRITE UV (the animated frame's sub-rect); the alpha map is read at
// the SPRITE UV for *_sprite_alpha variants (4, 5) and at the BILLBOARD UV
// for *_billboard_alpha variants (1, 3). For plain bitmaps sprite==billboard.
fn sample_diffuse(sprite_uv: vec2<f32>, billboard_uv: vec2<f32>, palette_v: f32, albedo: u32) -> vec4<f32> {
    let base = textureSample(base_tex, samp, sprite_uv);
    if (albedo == 2u || albedo == 3u || albedo == 5u) {
        let pal = textureSample(palette_tex, samp, vec2<f32>(base.x, palette_v));
        var a = pal.a;
        if (albedo == 3u) { a = textureSample(alpha_tex, samp, billboard_uv).a; }
        else if (albedo == 5u) { a = textureSample(alpha_tex, samp, sprite_uv).a; }
        return vec4<f32>(pal.rgb, a);
    } else if (albedo == 1u) {
        return vec4<f32>(base.rgb, textureSample(alpha_tex, samp, billboard_uv).a);
    } else if (albedo == 4u) {
        return vec4<f32>(base.rgb, textureSample(alpha_tex, samp, sprite_uv).a);
    }
    return base;
}

// Two color targets matching the shared transparents rpass: RT0 =
// lighting_base (HDR composite), RT1 = hdr_dark
// (_surface_screenshot_composite_cubemap). PC DARK_MULT=1.0 → RT1 == RT0.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) dark: vec4<f32>,
}

@fragment
fn fs_particle(in: VsOut) -> FsOut {
    let albedo = mat.v.x;

    // sample_diffuse at frame 0; cross-fade to frame 1 when frame_blend is on
    // (engine particle_render_hlsl.hlsl:922 lerps the full sample by frac(frame)).
    var sampled = sample_diffuse(in.sprite_uv0, in.uv, in.palette_v, albedo);
    if (mat.frames_meta.z != 0u && mat.frames_meta.x > 1u) {
        let s1 = sample_diffuse(in.sprite_uv1, in.uv, in.palette_v, albedo);
        sampled = mix(sampled, s1, in.frame_blend);
    }

    var alpha = sampled.a;
    if (mat.v.y != 0u) {
        alpha = remap_alpha(in.black_point, alpha);
    }
    var rgb = sampled.rgb * in.color.rgb;
    alpha = alpha * in.color.a;

    // Depth occlusion. The particle pass has no hardware depth test (we sample
    // the depth target in-shader, so it can't also be a depth attachment), so
    // occlusion is done here. The engine ALWAYS occludes particles by a
    // hardware depth-test (geometry in front → killed) and ADDITIONALLY applies
    // the soft `compute_depth_fade` only when the shader's depth_fade category
    // is on. Mirror that: always hard-cull behind geometry; soft-fade the
    // intersection only when the category (mat.extra.y) is set. Earlier this
    // whole block was category-gated, so water_spray (depth_fade OFF) had NO
    // occlusion and bled through rock walls. Reverse-Z: scene_view_z = near/depth.
    {
        let px = vec2<i32>(in.clip.xy);
        let scene_d = textureLoad(scene_depth, px, 0);
        // Compare the FRAGMENT's own depth (`in.clip.z`, interpolated across the
        // quad), not the billboard CENTER (`in.view_z`, constant per quad). The
        // center test was all-or-nothing: a large billboard whose center fell
        // behind a surface was killed entirely — most of an additive lift column
        // vanished behind the platform lip. Per-fragment matches the engine's
        // hardware depth test. Reverse-Z: bigger depth = closer.
        let frag_d = in.clip.z;
        // frame.v2.y != 0 (PROTOMORPH_NO_PARTICLE_DEPTH) disables ALL particle
        // depth interaction (soft fade AND hard occlusion) — a diagnostic to
        // isolate whether a vanishing layer is occluded by scene_depth
        // (opaque + water).
        if (frame.v2.y == 0.0) {
            if (mat.extra.y != 0.0) {
                // soft fade (depth_fade category on): linearize both (near/depth),
                // fade the intersection over `range` world units.
                var scene_z = 1.0e30; // sky / far → no fade
                if (scene_d > 0.0) { scene_z = frame.v.z / scene_d; }
                let frag_z = frame.v.z / max(frag_d, 1.0e-6);
                let range = max(frame.v.w, 1.0e-4);
                alpha = alpha * saturate((scene_z - frag_z) / range);
            } else if (scene_d > 0.0 && frag_d < scene_d) {
                // hard occlusion only (category off): fragment behind opaque
                // geometry (smaller reverse-Z depth) is killed; no thinning in front.
                alpha = 0.0;
            }
        }
    }

    // Non-linear blend output conventions (the linear blend state is set
    // per-batch on the pipeline; these match its expectations).
    // `mat.v.z` is the RUNTIME `e_alpha_blend_mode` value (name-resolved
    // on the CPU; matches `blend_state()` / `AlphaBlendMode`), NOT the
    // authored category index: 2 = multiply, 5 = pre_multiplied_alpha,
    // 7 = multiply_add.
    let blend_mode = mat.v.z;
    if (blend_mode == 2u) {
        // multiply: lerp(1, rgb, a) — engine line 1002.
        rgb = mix(vec3<f32>(1.0), rgb, alpha);
    } else if (blend_mode == 5u || blend_mode == 7u) {
        // pre_multiplied_alpha / multiply_add: premultiply rgb by alpha.
        rgb = rgb * alpha;
    }
    if (blend_mode != 2u) {
        // engine line 1006: `blended.xyz += IN.m_color_add` (fog
        // inscatter; self_illum constant_color would land here too) for
        // every non-multiply mode, after the texture multiply.
        rgb = rgb + in.color_add;
    }

    if (alpha <= 0.0 && blend_mode != 2u) {
        discard;
    }
    let c = vec4<f32>(rgb, alpha);
    return FsOut(c, c);
}

// Distortion particles (specialized_rendering category != none, e.g. the
// man cannon `distortion_shimmer`). Instead of color, they write a
// screen-space displacement delta into an accumulation buffer (cleared to
// 0.5, additive blend). A later full-screen pass warps the scene by
// `(sample - 0.5)`. Engine `compute_normalized_distortion`
// (particle_render_hlsl.hlsl:153): displacement = bitmap · alpha ·
// depth_fade, oriented by the view-space billboard basis, ÷ depth.
@fragment
fn fs_distortion(in: VsOut) -> @location(0) vec4<f32> {
    // The distortion bitmap is a DXN/BC5 dudv map, uploaded as Bc5RgSnorm —
    // the hardware decodes R/G to SIGNED [-1,1] (displacement already centered
    // at 0). Use it raw, exactly like the engine's `blended.xy` (which samples
    // the same signed format). Subtracting 0.5 here was the bug: it injected a
    // constant -0.5 skew, warping everything one direction.
    let base = textureSample(base_tex, samp, in.uv);
    // Engine compute_normalized_distortion (particle_render_hlsl.hlsl:884)
    // flips the v-component of the sampled displacement before orienting it
    // by the billboard basis: `blended.y = -blended.y`.
    let local = vec2<f32>(base.x, -base.y);

    var alpha = in.color.a;
    // Soft depth fade so the shimmer fades into occluding geometry, matching
    // the color particles (gated on the same category bit).
    if (mat.extra.y != 0.0) {
        let px = vec2<i32>(in.clip.xy);
        let scene_d = textureLoad(scene_depth, px, 0);
        var scene_z = 1.0e30;
        if (scene_d > 0.0) {
            scene_z = frame.v.z / scene_d;
        }
        let range = max(frame.v.w, 1.0e-4);
        alpha = alpha * saturate((scene_z - in.view_z) / range);
    }

    // Orient the local displacement by the view-space billboard basis, scale
    // by strength · alpha, and divide by depth (far → less warp).
    let basis = mat2x2<f32>(in.view_tan, in.view_bin);
    let disp = local * frame.v2.x * alpha;
    let frame_disp = (basis * disp) / max(in.view_z, 0.1);
    return vec4<f32>(frame_disp, 0.0, 0.0);
}
