// Port of `patchy_fog.hlsl:33-226` (default_vs + default_ps).
//
// 8 screen-space fog sheets, evenly spaced from `distance_to_first_sheet`
// along the camera forward axis at intervals of `distance_between_sheets`.
// Each sheet samples a tiling noise bitmap, fades by depth + height, and
// contributes to optical-depth integration. The integrated extinction
// modulates engine-style Mie inscatter (HG phase) using the active
// atmosphere setting. Output is `(inscatter × exposure, extinction)` —
// an additive HDR tint with extinction in the alpha for blend.
//
// Engine cbuffer slots (`patchy_fog_registers_h.hlsl`):
//   0x3F0000  inverse_z_transform        (per-frame)
//   0x3F0001  texcoord_basis             (cos, -sin, sin, cos rotation)
//   0x3F0002  attenuation_data           (full_intensity_height,
//                                         attenuation_rate,
//                                         depth_fade_factor, _)
//   0x3F0003  eye_position
//   0x3F0004  window_pixel_bounds        (UV scale + offset)
//   0x400000  sheet_fade_factors0        (sheets 0-3, density × pop-fix)
//   0x400001  sheet_fade_factors1        (sheets 4-7)
//   0x400002  sheet_depths0              (4 view-space depths)
//   0x400003  sheet_depths1
//   0x400004..0x40000B  tex_coord_transform0..7  (per-sheet UV center
//                                                 + half-extent)
//
// Atmosphere constants (from `c_atmosphere_fog_interface::set_constant
// @ 0x1803AE530`, fed by the active atmosphere setting):
//   atmosphere_constant_0.xyz = SUN_DIR
//   atmosphere_constant_1.xyz = SUN_INTENSITY_OVER_TM
//   atmosphere_constant_2.w   = HG_CONSTANT_PLUS_ONE
//   atmosphere_constant_3.xyz = TOTAL_MIE_LOG2E
//   atmosphere_constant_5.xyz = MIE_THETA_PREFIX_HGC
//   atmosphere_constant_extra.x = HG_CONSTANT_TIMES_TWO

struct PatchyFogParams {
    /// Per-frame view-space-depth reconstruction `(m33, m43, m34, m44)`.
    /// Engine: rows 2 + 3 of `inverse(view × projection)` packed as
    /// `(.row[2].zw, .row[3].zw)` so we can do `vsd = depth * xy + zw`,
    /// then divide x by -y.
    inverse_z_transform: vec4<f32>,
    /// `(cos(roll), -sin(roll), sin(roll), cos(roll))` — accumulated
    /// camera roll for the noise-UV basis.
    texcoord_basis: vec4<f32>,
    /// `(full_intensity_height, attenuation_rate, depth_fade_factor, _)`.
    attenuation_data: vec4<f32>,
    /// `(eye_world.xyz, 1.0)`.
    eye_position: vec4<f32>,
    /// `((0.5(top+bottom)+0.5)/h, (0.5(left+right)+0.5)/w,
    ///   0.5(bottom-top)/h, -0.5(right-left)/w)` — engine
    /// `c_patchy_fog::set_rasterizer_state` wraps the screen UVs.
    window_pixel_bounds: vec4<f32>,

    /// 4 sheets of `(density × pop-fix-fade)` weights per sheet group.
    sheet_fade_factors0: vec4<f32>,
    sheet_fade_factors1: vec4<f32>,
    /// 4 view-space depths per sheet group (sheets 0-3 / 4-7).
    sheet_depths0: vec4<f32>,
    sheet_depths1: vec4<f32>,
    /// Per-sheet noise UV transform: `(center.x, center.y, half_u, half_v)`.
    tex_coord_transform0: vec4<f32>,
    tex_coord_transform1: vec4<f32>,
    tex_coord_transform2: vec4<f32>,
    tex_coord_transform3: vec4<f32>,
    tex_coord_transform4: vec4<f32>,
    tex_coord_transform5: vec4<f32>,
    tex_coord_transform6: vec4<f32>,
    tex_coord_transform7: vec4<f32>,
}

struct PatchyFogAtmosphere {
    /// `atmosphere_constant_0.xyz` — world-space sun direction.
    sun_dir: vec4<f32>,
    /// `atmosphere_constant_1.xyz` — `SUN_INTENSITY / TOTAL_MIE`.
    sun_intensity_over_tm: vec4<f32>,
    /// `.w` = `1 + g²` for the Henyey-Greenstein phase.
    constant_2: vec4<f32>,
    /// `.xyz` = `TOTAL_MIE × log2(e)`.
    total_mie_log2e: vec4<f32>,
    _4: vec4<f32>,
    /// `.xyz` = `MIE_THETA_PREFIX × (1 - g²)`.
    mie_theta_prefix_hgc: vec4<f32>,
    /// `.x` = `2g` for HG phase.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: PatchyFogParams;
@group(0) @binding(1) var<uniform> atm: PatchyFogAtmosphere;
@group(0) @binding(2) var tex_noise: texture_2d<f32>;
@group(0) @binding(3) var smp_noise: sampler;
@group(0) @binding(4) var tex_scene_depth: texture_depth_2d;
@group(0) @binding(5) var smp_scene_depth: sampler;
/// `inverse(view × projection)`. Clip-space corners are passed through
/// to `out.clip` directly; this matrix reconstructs the world-space
/// corner needed for view-direction + height-fade math. Engine builds
/// world-space corners on CPU each frame via `inv_VP × clip-corner` —
/// we do the equivalent in the VS.
@group(0) @binding(6) var<uniform> inverse_view_projection: mat4x4<f32>;
@group(0) @binding(7) var<uniform> g_exposure: vec4<f32>;

struct VsIn {
    @location(0) position: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    /// `.xy` = engine sheet texcoord ∈ {-1,1}² with corners
    /// {-1,1},{-1,-1},{1,-1},{1,1}; `.z` = clip-space `w` (used for
    /// near-surface fade).
    @location(0) texcoord: vec3<f32>,
    @location(1) world_space: vec4<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // Vertex buffer carries clip-space corners (±1, ±1, 1, 1). Pass
    // straight through — the quad already covers the full screen.
    out.clip = in.position;
    // World-space corner via `inverse(VP) × clip-corner`. Used by the
    // PS for height fading + view-direction reconstruction.
    let ws = inverse_view_projection * in.position;
    out.world_space = vec4<f32>(ws.xyz / max(ws.w, 1e-6), 1.0);
    out.texcoord = vec3<f32>(in.texcoord.x, in.texcoord.y, in.position.w);
    return out;
}

// Engine `accum_pixel` (render_target_fx.hlsl:8-18). `patchy_fog.hlsl:225`
// uses `convert_to_render_target(...)` → writes BOTH RT0 (`_surface_screenshot
// _composite_depth` = our `lighting_base`) AND RT1 (`_surface_screenshot
// _composite_cubemap` = our `hdr_dark`, the bloom-extract feed source). On
// MCC PC `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0` so RT0 content == RT1
// content. Without RT1 writes, auto-exposure's bloom-feed sample misses the
// fog contribution and over-corrects upward.
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> AccumPixel {
    // Engine: window-clipped UV. `texcoord ∈ [-1, 1]` from VS; multiplied
    // by half-extents + biased by center → screen UV in [0, 1].
    let screen_normalized_uv = vec2<f32>(
        u.window_pixel_bounds.x + in.texcoord.x * u.window_pixel_bounds.z,
        u.window_pixel_bounds.y + in.texcoord.y * u.window_pixel_bounds.w,
    );
    let scene_depth = textureSample(tex_scene_depth, smp_scene_depth, screen_normalized_uv);

    // View-space depth from depth-buffer value via the inverse-Z
    // transform. Engine HLSL line 102-105.
    var view_space_scene_depth = vec2<f32>(
        u.inverse_z_transform.x * scene_depth + u.inverse_z_transform.z,
        u.inverse_z_transform.y * scene_depth + u.inverse_z_transform.w,
    );
    view_space_scene_depth.x = view_space_scene_depth.x / max(-view_space_scene_depth.y, 1e-6);

    // Engine: positive when sheet is in FRONT of the scene point.
    let view_space_depth_diff0 = vec4<f32>(view_space_scene_depth.x) - u.sheet_depths0;
    let view_space_depth_diff1 = vec4<f32>(view_space_scene_depth.x) - u.sheet_depths1;

    // Depth-fade — 0 behind scene, fades to 1 well in front.
    var fade_factor0: vec4<f32> = clamp(
        vec4<f32>(1.0) - exp(-view_space_depth_diff0 * u.attenuation_data.z),
        vec4<f32>(0.0), vec4<f32>(1.0),
    );
    var fade_factor1: vec4<f32> = clamp(
        vec4<f32>(1.0) - exp(-view_space_depth_diff1 * u.attenuation_data.z),
        vec4<f32>(0.0), vec4<f32>(1.0),
    );

    // Per-sheet density × pop-prevention.
    fade_factor0 = fade_factor0 * u.sheet_fade_factors0;
    fade_factor1 = fade_factor1 * u.sheet_fade_factors1;

    // Sheet-noise-UV basis. `texcoord ∈ [-1, 1]` directly.
    let screen_normalized_biased = in.texcoord.xy;

    // 8 sheets × noise sample. Each sheet's `tex_coord_transform`
    // packs (center_uv, half_extent_uv); `texcoord_basis` rotates the
    // screen-aligned axes by accumulated camera roll. Unrolled to
    // avoid any dynamic indexing concerns.
    let basis_u = u.texcoord_basis.xy;
    let basis_v = u.texcoord_basis.zw;
    var noise_values0: vec4<f32>;
    var noise_values1: vec4<f32>;
    let uv0 = u.tex_coord_transform0.xy + screen_normalized_biased.x * u.tex_coord_transform0.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform0.w * basis_v;
    noise_values0.x = textureSample(tex_noise, smp_noise, uv0).x;
    let uv1 = u.tex_coord_transform1.xy + screen_normalized_biased.x * u.tex_coord_transform1.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform1.w * basis_v;
    noise_values0.y = textureSample(tex_noise, smp_noise, uv1).x;
    let uv2 = u.tex_coord_transform2.xy + screen_normalized_biased.x * u.tex_coord_transform2.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform2.w * basis_v;
    noise_values0.z = textureSample(tex_noise, smp_noise, uv2).x;
    let uv3 = u.tex_coord_transform3.xy + screen_normalized_biased.x * u.tex_coord_transform3.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform3.w * basis_v;
    noise_values0.w = textureSample(tex_noise, smp_noise, uv3).x;
    let uv4 = u.tex_coord_transform4.xy + screen_normalized_biased.x * u.tex_coord_transform4.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform4.w * basis_v;
    noise_values1.x = textureSample(tex_noise, smp_noise, uv4).x;
    let uv5 = u.tex_coord_transform5.xy + screen_normalized_biased.x * u.tex_coord_transform5.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform5.w * basis_v;
    noise_values1.y = textureSample(tex_noise, smp_noise, uv5).x;
    let uv6 = u.tex_coord_transform6.xy + screen_normalized_biased.x * u.tex_coord_transform6.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform6.w * basis_v;
    noise_values1.z = textureSample(tex_noise, smp_noise, uv6).x;
    let uv7 = u.tex_coord_transform7.xy + screen_normalized_biased.x * u.tex_coord_transform7.z * basis_u + screen_normalized_biased.y * u.tex_coord_transform7.w * basis_v;
    noise_values1.w = textureSample(tex_noise, smp_noise, uv7).x;
    // Square the noise so it's always positive and biased toward dense
    // (per HLSL line 186-187).
    noise_values0 = noise_values0 * noise_values0;
    noise_values1 = noise_values1 * noise_values1;

    // Height fading — fog density decays exponentially above the
    // configured "Full intensity height" world-z.
    let view_vector = normalize(in.world_space.xyz - u.eye_position.xyz);
    let height0 = vec4<f32>(view_vector.z) * u.sheet_depths0 + vec4<f32>(u.eye_position.z);
    let height_fade0 = min(
        vec4<f32>(1.0),
        exp(vec4<f32>(u.attenuation_data.y)
            * (vec4<f32>(u.attenuation_data.x) - height0)),
    );
    fade_factor0 = fade_factor0 * height_fade0;
    let height1 = vec4<f32>(view_vector.z) * u.sheet_depths1 + vec4<f32>(u.eye_position.z);
    let height_fade1 = min(
        vec4<f32>(1.0),
        exp(vec4<f32>(u.attenuation_data.y)
            * (vec4<f32>(u.attenuation_data.x) - height1)),
    );
    fade_factor1 = fade_factor1 * height_fade1;

    // Optical-depth line integral.
    let optical_depth = dot(fade_factor0, noise_values0)
                      + dot(fade_factor1, noise_values1);

    // View-dependent Mie scattering (engine HLSL line 208-214).
    let c_theta = dot(view_vector, atm.sun_dir.xyz);
    let extinction = 1.0 - exp2(-atm.total_mie_log2e.x * optical_depth);
    let heyey_term = atm.constant_2.w - atm.extra.x * c_theta;
    let heyey_term_neg_1_5 = pow(max(heyey_term, 1e-6), -1.5);
    let beta_p_theta = atm.mie_theta_prefix_hgc.xyz * heyey_term_neg_1_5;
    var inscatter = atm.sun_intensity_over_tm.xyz * beta_p_theta * extinction;

    // Near-surface fade (engine HLSL line 217-223).
    let fog_depth = in.texcoord.z;
    let depth_diff = view_space_scene_depth.x - fog_depth;
    let fog_fade = smoothstep(0.03, 0.3, depth_diff);
    inscatter = inscatter * fog_fade;

    // Output mirrors `convert_to_render_target(float4(inscatter ×
    // g_exposure.r, extinction), false, true)`. inscatter is already
    // pre-multiplied by extinction at line 236; the pipeline blend
    // (`_alpha_blend_multiply_add` = ONE / INV_SRC_ALPHA / ADD on both
    // channels) consumes src.a=extinction to attenuate the existing
    // scene by `(1 - extinction)` and adds inscatter on top.
    //
    // ── DIAGNOSTIC — `g_exposure.w` repurposed as debug mode (0 = off).
    // Each mode rewrites the output to a flat color × alpha=1 so the
    // multiply-add blend lights it onto lighting_base directly.
    let dbg = g_exposure.w;
    if (dbg > 0.5 && dbg < 1.5) {
        // Mode 1: per-sheet noise sample (sheet 0 only). If this is flat,
        // tex_coord_transform UVs / repeat_rate / basis are degenerate.
        let n = noise_values0.x; // already squared
        let v = vec4<f32>(vec3<f32>(n) * 5.0, 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 1.5 && dbg < 2.5) {
        // Mode 2: variation across all 8 sheets (sum of sheet noises).
        let n = noise_values0.x + noise_values0.y + noise_values0.z + noise_values0.w
              + noise_values1.x + noise_values1.y + noise_values1.z + noise_values1.w;
        let v = vec4<f32>(vec3<f32>(n / 8.0) * 5.0, 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 2.5 && dbg < 3.5) {
        // Mode 3: optical_depth — pre-extinction. If high everywhere,
        // density × beta_m saturates. Should look noisy.
        let v = vec4<f32>(vec3<f32>(optical_depth * 0.01), 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 3.5 && dbg < 4.5) {
        // Mode 4: extinction (post-Mie). If ≈1 everywhere, we're saturated
        // — the spatial detail from noise has been crushed by exp2.
        let v = vec4<f32>(vec3<f32>(extinction) * 5.0, 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 4.5 && dbg < 5.5) {
        // Mode 5: view-space scene depth as luminance (scaled). Verifies
        // inverse_z_transform reconstruction.
        let v = vec4<f32>(vec3<f32>(view_space_scene_depth.x * 0.05), 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 5.5 && dbg < 6.5) {
        // Mode 6: height fade for sheet 0. If 0 everywhere, height fade
        // kills fog (likely sign issue or bad full_intensity_height).
        let v = vec4<f32>(vec3<f32>(height_fade0.x) * 2.0, 1.0);
        return AccumPixel(v, v);
    }
    if (dbg > 6.5 && dbg < 7.5) {
        // Mode 7: view_vector visualized as RGB. R=view_vector.x+.5,
        // G=view_vector.y+.5, B=view_vector.z+.5. Should radially
        // gradient from screen center (eye-aligned) outward. If colors
        // are static / don't vary across screen → world_space
        // reconstruction is degenerate. If z-channel (blue) is uniformly
        // bright/dark → camera-axis mismatch.
        let v_dbg = vec4<f32>(view_vector * 0.5 + 0.5, 1.0);
        return AccumPixel(v_dbg, v_dbg);
    }
    if (dbg > 7.5 && dbg < 8.5) {
        // Mode 8: raw height at sheet 0 (= view_vector.z × sheet_depth
        // + eye.z). Visualized as luminance / 30 so heights up to 30
        // wu map to full brightness. Should show eye-level fog horizon
        // mid-screen, gradient toward top (looking up = higher height)
        // and bottom (looking down = lower height). If horizon shifts
        // when CAMERA YAWS, world_up / world_space reconstruction is
        // confused.
        let v_dbg = vec4<f32>(vec3<f32>(height0.x / 30.0), 1.0);
        return AccumPixel(v_dbg, v_dbg);
    }
    if (dbg > 8.5 && dbg < 9.5) {
        // Mode 9: HG cos(theta) = dot(view_vector, sun_dir). Should
        // be brightest where camera looks AT sun (white at sun
        // direction), dark at opposite. If gradient doesn't shift
        // correctly when camera turns toward sun → sun_dir basis
        // mismatch.
        let v_dbg = vec4<f32>(vec3<f32>(c_theta * 0.5 + 0.5), 1.0);
        return AccumPixel(v_dbg, v_dbg);
    }
    if (dbg > 9.5 && dbg < 10.5) {
        // Mode 10: UV of sheet 0 (after texcoord_basis rotation + per-
        // sheet center + half-extent scale). R=uv0.x×0.1, G=uv0.y×0.1
        // (clamped to [0,1] by sampler wrapping anyway). Should show
        // a smooth gradient across screen. If it tiles/wraps strangely
        // → tex_coord_transform scale is off. If it stays constant
        // across camera roll → texcoord_basis isn't being applied.
        let v_dbg = vec4<f32>(fract(uv0 * 0.1), 0.5, 1.0);
        return AccumPixel(v_dbg, v_dbg);
    }
    if (dbg > 10.5 && dbg < 11.5) {
        // Mode 11: inscatter (pre-extinction-bake) as RGB. Tests sun_dir
        // chromaticity + beta_p coefficients + HG phase. If colors look
        // monochrome white → chromaticity normalize is wrong. If
        // saturated brightness without spatial variation → fade chain
        // is broken upstream of this.
        let inscatter_raw = atm.sun_intensity_over_tm.xyz * beta_p_theta;
        let v_dbg = vec4<f32>(inscatter_raw, 1.0);
        return AccumPixel(v_dbg, v_dbg);
    }
    let v = vec4<f32>(inscatter * g_exposure.r, extinction);
    return AccumPixel(v, v);
}
