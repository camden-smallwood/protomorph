// `final_composite_base_hlsl.hlsl::default_ps` (lines 95-132) — verbatim port.
// Engine source: `source/rasterizer/hlsl/final_composite_base.hlsl`.
// Dispatched by `c_screen_postprocess::copy_accumulation_target @
// 0x1806B71E0` (the final pass before swapchain present).
//
// Hierarchy: `final_composite.hlsl` (`//@generate screen` entry) wraps
// `final_composite_base.hlsl` (the actual body); variants are
// `final_composite_dof.hlsl`, `final_composite_debug.hlsl`, and
// `final_composite_zoom.hlsl`. protomorph collapses base + DoF into one
// shader (the DoF branch is conditional on `dof_focus.z > 0`); the
// debug + zoom variants are not yet ported.
//
// Body steps:
//   combined = surface_sampler  (HDR composite)
//   bloom    = bloom_sampler    (postprocess output)
//   blend    = combined + bloom                              // CALC_BLEND default
//   blend    = mul(float4(blend, 1), hue_saturation_matrix)  // hue/sat (4×4)
//   luminance = dot(blend, (0.333, 0.333, 0.333))            // contrast
//   blend    *= pow(luminance, contrast.x) / luminance       // contrast
//   clamped  = min(blend, tone_curve_constants.xxx)          // tone curve
//   result   = ((clamped*w + z) * clamped + y) * clamped     // cubic
//   ... 3D LUT color grading (h:121-127) ...
//   result.a = sqrt(dot(result.rgb, NTSC_weights))
//
// E1 — Depth of field. When `dof.max_blur > 0`, lerp the sharp HDR
// with the half-res blurred copy by per-pixel depth distance from
// focus. Engine equivalent: `final_composite_dof_hlsl.hlsl` (a
// separate shader 33 selected at pipeline-create time), with cbuffers
// `depth_constants` (0x1E0000) and `depth_constants2` (0x1E0001). DoF
// stays inert (no visible change) when `max_blur=0`.
//
// Tone-curve defaults from HLSL comments
// (final_composite_base_hlsl.hlsl:115-118):
//   x = 1.4938015821857215  (saturation clamp)
//   y = 1.0041494251232542  (linear term)
//   z = 0                   (quadratic term)
//   w = -0.15               (cubic term — compresses brights)

struct CompositeParams {
    /// `ps_postprocess_hue_saturation_matrix` — column-major 4×4
    /// applied as `mul(float4(blend, 1), matrix)`. Identity for
    /// the default no-color-shift case.
    hue_saturation_matrix: mat4x4<f32>,
    /// `(contrast.x, _, _, _)` — luminance exponent. 1.0 = no change.
    contrast: vec4<f32>,
    /// `(cg_blend_factor.x, _, _, _)` — lerp weight between the two
    /// 3D LUTs. 0 = use cg0 only; 1 = use cg1 only.
    cg_blend_factor: vec4<f32>,
    /// DoF — `(focus_depth_ndc, half_width_ndc, max_blur, _)`.
    /// `focus_depth_ndc` is the depth-buffer value (0..1) at the
    /// focus distance; `half_width_ndc` is the in-focus band's half-
    /// width in the same depth-buffer units. Pixels within the band
    /// get full sharp; pixels outside fade to blurry over a soft
    /// transition. `max_blur ≤ 0` disables (sharp passthrough).
    dof_focus: vec4<f32>,
    /// Engine `tone_curve_constants` cbuffer (slot 0x1D0001).
    /// `(.x = max_clamp, .y = linear, .z = quadratic, .w = cubic)`
    /// per `final_composite_registers_fx.hlsl` declaration order.
    /// Computed per-frame by `compute_tone_curve_constants` in
    /// `final_composite_pass.rs` from `m_tone_curve` /
    /// `m_tone_curve_white_point`. Engine default
    /// (WHITE=1.0, m_tone_curve=true): `(1.0, 1.5000, 0.0, -0.5000)`.
    tone_curve_constants: vec4<f32>,
    /// Engine `intensity` cbuffer (slot 0x1D0000) = `(natural, bloom,
    /// bling, persist)`. The shader-18 (`copy_target.hlsl:50-53`) weighted
    /// combine of the separate scene / bloom / bling / persist buffers.
    /// protomorph: natural=1.0; bloom=1.0 (bloom_intensity is baked into
    /// the bloom pyramid upstream in `postprocess_bloom_buffer`, so it's
    /// not re-applied here); bling=`m_bling_intensity` (0 when inactive);
    /// persist=0 (persist buffer not ported).
    intensity: vec4<f32>,
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var s_bloom: sampler;
@group(0) @binding(4) var<uniform> u: CompositeParams;
/// `color_grading0` — primary 3D LUT, 16³ RGBA. Identity = no change.
@group(0) @binding(5) var t_cg0: texture_3d<f32>;
@group(0) @binding(6) var s_cg0: sampler;
/// `color_grading1` — secondary 3D LUT for crossfade. Identity by
/// default; cg_blend_factor.x crossfades cg0→cg1.
@group(0) @binding(7) var t_cg1: texture_3d<f32>;
@group(0) @binding(8) var s_cg1: sampler;
/// E1 DoF — half-res blurred copy of the HDR composite. Engine:
/// `_surface_aux_small2` after the 11-tap binomial gaussian blur.
@group(0) @binding(9) var t_dof_blur: texture_2d<f32>;
@group(0) @binding(10) var s_dof_blur: sampler;
/// E1 DoF — scene depth (depth-2d non-comparison). Used to compute
/// per-pixel sharp/blurry weight from focus distance.
@group(0) @binding(11) var t_scene_depth: texture_depth_2d;
@group(0) @binding(12) var s_scene_depth: sampler;
/// Bling/star buffer — engine `copy_target.hlsl` sampler 3
/// (`g_star_buffer` from `bling_generate`). Sampled SEPARATELY from
/// bloom and weighted by `intensity.z` in the same composite pass, per
/// the engine's single-pass combine (no pre-fuse). Holds 0 when bling is
/// inactive, and `intensity.z` is 0 then too, so the term vanishes.
@group(0) @binding(13) var t_bling: texture_2d<f32>;
@group(0) @binding(14) var s_bling: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const NTSC_WEIGHTS = vec3<f32>(0.299, 0.587, 0.114);

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Fullscreen triangle (no vertex buffer needed)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
    );
    var out: VsOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let sharp = textureSample(t_src, s_src, in.uv).rgb;

    // E1 — DoF blend. `max_blur ≤ 0` → sharp passthrough (default,
    // when no scenario has authored DoF). Otherwise blend toward the
    // half-res blurred copy by per-pixel depth distance from focus.
    var combined = sharp;
    if (u.dof_focus.z > 0.0) {
        let blurry = textureSample(t_dof_blur, s_dof_blur, in.uv).rgb;
        let depth = textureSample(t_scene_depth, s_scene_depth, in.uv);
        let focus = u.dof_focus.x;
        let half_width = max(u.dof_focus.y, 1.0e-4);
        let dist = abs(depth - focus);
        let outside = max(dist - half_width, 0.0) / half_width;
        let blur_factor = clamp(outside, 0.0, 1.0) * u.dof_focus.z;
        combined = mix(sharp, blurry, blur_factor);
    }

    let bloom = textureSample(t_bloom, s_bloom, in.uv).rgb;

    // Bling/star term. When bling is inactive `bling_generate` is skipped,
    // so the star buffer (`aux_star`) is NEVER cleared/written this frame
    // and holds uninitialized / stale content — which is NaN or Inf on
    // Metal. `intensity.z` is 0 then, but `0 * NaN = NaN` (IEEE), which
    // would poison the entire composite → opaque black boxes wherever the
    // garbage tiles land. Guard on the weight so the term contributes
    // EXACTLY 0 when off, and sanitize the sample so an active-but-Inf
    // spike (bright HDR source) can't blow the composite to NaN either.
    var bling_term = vec3<f32>(0.0);
    if (u.intensity.z != 0.0) {
        let bling = textureSample(t_bling, s_bling, in.uv).rgb;
        // Sanitize the spike sample to a finite range. Metal's min/max
        // (and thus clamp) are NaN-suppressing — they return the finite
        // operand — so `clamp` maps NaN→0 and Inf→65504 reliably, without
        // depending on a `v != v` NaN test (which Metal's default fast-math
        // assumes-no-NaN may optimize away). 65504 = max half-float.
        let safe_bling = clamp(bling, vec3<f32>(0.0), vec3<f32>(65504.0));
        bling_term = u.intensity.z * safe_bling;
    }

    // Engine `copy_target.hlsl:50-53` (shader 18 = copy_accumulation_target):
    // single-pass weighted combine of the SEPARATE scene / bloom / bling
    // buffers (no pre-fuse). `blend = natural*combined + bloom_int*bloom +
    // bling_int*bling (+ persist*persist)`. bloom_intensity is baked into the
    // pyramid upstream so intensity.y=1; persist not ported.
    var blend: vec3<f32> = u.intensity.x * combined
                         + u.intensity.y * bloom
                         + bling_term;

    // Hue/saturation 4×4 (final_composite_base_hlsl.hlsl:103).
    blend = (u.hue_saturation_matrix * vec4<f32>(blend, 1.0)).rgb;

    // Contrast (final_composite_base_hlsl.hlsl:106-112). Halo guards
    // `if (luminance > 0)` to avoid pow(0)/0; mirror that.
    let luminance = dot(blend, vec3<f32>(0.333, 0.333, 0.333));
    if (luminance > 0.0) {
        blend = blend * (pow(luminance, u.contrast.x) / luminance);
    }

    // PROTOMORPH SCENE-RADIANCE COMPAT LAYER — inline sqrt before tone
    // curve. NOT engine-faithful per the verified runtime path
    // (`c_rasterizer::setup_render_target_globals_with_exposure
    // @ 0x180670AD0` writes `gamma2_flags={0,0}` so engine RT0+RT1 are
    // both LINEAR on PC, `convert_to_render_target.hlsl`'s `safe_sqrt`
    // branches never fire). Engine tone curve reads LINEAR and the
    // cubic `(WHITE, k*1.0041, 0, -0.15*k³)` is near-identity for
    // shadows/midtones (`tc(0.25)=0.249, tc(0.5)=0.483`), only
    // compressing highlights >1.0 toward `1.0` at WHITE clamp.
    //
    // Why we keep the sqrt anyway: protomorph's raw lit scene radiance
    // is ~2× dimmer than engine MCC produces for the same scene
    // (verified via `[diag] actual_brightness` — m_exposure converges
    // toward the cfxs clamp ceiling because pre-exposure raw avg lands
    // near 0.12 vs engine's expected ~0.25). Feeding our dim values to
    // the near-identity engine tone curve produces a visibly dark
    // output. The sqrt boosts midtones (0.5 linear → sqrt → 0.707 →
    // tone curve → 0.687, vs without-sqrt 0.5 → 0.483 — ~42% brighter
    // midtones) and approximates engine's visual output.
    //
    // **Proper fix:** find and close the ~2× scene-radiance gap in the
    // lighting pipeline (SH/lightmap decode, sun/sky contribution,
    // material albedo path — see the riverworld scene_avg=0.12 raw
    // measurement). When raw scene values hit ~0.25 average, the
    // engine tone curve renders correctly without compensation and we
    // can drop this sqrt. Confirmed via dllcache decompile 2026-05-20.
    let clamped = min(sqrt(max(blend, vec3<f32>(0.0))),
                      vec3<f32>(u.tone_curve_constants.x));
    var result = ((clamped * u.tone_curve_constants.w + vec3<f32>(u.tone_curve_constants.z))
                  * clamped
                  + vec3<f32>(u.tone_curve_constants.y))
                 * clamped;

    // 3D LUT color grading (h:121-127). Halo's r_size = 1/16 so the
    // input is biased into [0.03125, 0.96875] to avoid the LUT's edge
    // texels (which would be sampled with bilinear filter wrapping
    // outside the intended range).
    let r_size = 1.0 / 16.0;
    let cg_scale = vec3<f32>(1.0 - r_size);
    let cg_offset = vec3<f32>(0.5 * r_size);
    let cg_uvw = result * cg_scale + cg_offset;
    let cg0 = textureSample(t_cg0, s_cg0, cg_uvw).rgb;
    let cg1 = textureSample(t_cg1, s_cg1, cg_uvw).rgb;
    result = mix(cg0, cg1, u.cg_blend_factor.x);

    // alpha from luminance (h:129) — used for downstream sRGB blits.
    let alpha = dot(result, NTSC_WEIGHTS);
    return vec4<f32>(result, alpha);
}
