// Port of Halo 3 `shadow_apply_hlsl.hlsl::default_ps` (9-tap PCF stamp
// with per-pixel slope bias + cosine/depth falloff).
//
// Engine flow (after `setup_targets_static_lighting_alpha_blend(1, 1)`):
//   1. Each pixel reconstructs its world position from scene depth +
//      `inverse(view × projection)` (engine: `CAMERA_TO_*` cbuffer
//      constants `0x2E0000..2`).
//   2. Transforms world position into light-space via the per-caster
//      `world_to_shadow` matrix.
//   3. Samples normal_buffer for receiver normal; computes
//      `cosine = dot(n, light_dir)` and a slope-based depth bias
//      `bias = (sqrt(1-cos²)/cos + 0.2) × SHADOW_PIXELSIZE` —
//      engine `shadow_apply_hlsl.hlsl:189-198` and matching
//      `terrain_fx.hlsl:1476-1485`.
//   4. PCF-samples `_surface_shadow_1` to get an `unshadowed_percentage`
//      (engine kernel: 9-tap manual, `shadow_apply_hlsl.hlsl:74-110`).
//   5. Modulates by cosine_falloff × (1 - shadow_falloff⁴) — engine
//      `shadow_apply_hlsl.hlsl:154-163`.
//   6. Blits `vec4<f32>(0, 0, 0, 1.0 - unshadowed_percentage × shadow_darkness)`
//      to the LDR target with alpha-blend (`SrcAlpha, OneMinusSrcAlpha`).
//
// Single-caster invocation: this pass gets called ONCE per shadow caster
// after that caster's `_surface_shadow_1` was populated by
// `shadow_generate.wgsl`. Multiple casters serialize through this same
// pass against the same target.
//
// Three PCF kernel variants are available; pick at apply time by the
// `quality_mode` uniform field:
//   0 = default 9-tap 3×3            — `shadow_apply_hlsl.hlsl:74-110`
//   1 = fancy 16-tap 5×5 predicated  — `shadow_apply_fancy_hlsl.hlsl:13-89`
//   2 = faster 4-tap 2×2 bilinear    — `shadow_apply_faster_hlsl.hlsl:13-41`
// Engine compiles these as separate shader binaries via #define;
// we ship all three in one pipeline with a runtime branch (cheap on
// modern GPUs since the kernel choice is uniform across the draw).

struct ShadowApplyUniforms {
    /// World-to-light-clip matrix. Same matrix that
    /// `shadow_generate.wgsl` used to write the depth target.
    world_to_shadow: mat4x4<f32>,
    /// Inverse of (camera.view * camera.projection). Used to
    /// reconstruct world position from screen-space (uv, depth).
    inverse_view_projection: mat4x4<f32>,
    /// Engine `SHADOW_PIXELSIZE = sqrt(2) / k_shadow_resolution`,
    /// in `0x2E0003.x`. Drives PCF tap spacing AND the slope-bias
    /// `half_pixel_size` term.
    shadow_pixel_size: vec4<f32>,
    /// Caster opacity / fade (engine: `k_ps_constant_shadow_alpha.r`).
    /// Scaled by `cosine_falloff × (1 - shadow_falloff⁴)` per pixel
    /// before being applied to the alpha-blend stamp.
    caster_opacity: vec4<f32>,
    /// `(sub_res / k_shadow_resolution, sub_res / k_shadow_resolution,
    /// 0, 0)` — per-caster viewport sub-rect coverage. The generate
    /// pass writes only the bottom-left `sub_res × sub_res` corner of
    /// `_surface_shadow_1`; this scale collapses the world_to_shadow's
    /// UV [0..1] output into that sub-rect. Engine equivalent: per-
    /// object resolution clamp inside `c_lightmap_shadows_view::render`
    /// → `clamp(diameter_pixels × shadow_quality_lod, 16..512)`.
    shadow_uv_scale: vec4<f32>,
    /// World-space light direction (toward the light). Engine equivalent:
    /// `SHADOW_DIRECTION_WORLDSPACE = p_lighting_constant_9.xyz`. Used
    /// for cosine falloff + slope bias.
    light_dir_world: vec4<f32>,
    /// `(quality_mode, _, _, _)` — kernel selector. 0=default 3×3, 1=fancy
    /// 5×5, 2=faster 2×2. Cast to u32 by truncation in fs_main.
    pcf_quality: vec4<f32>,
    /// `(eye_world.xyz, 1.0)` — camera world position. Used as the
    /// `view_point` for `compute_scattering`, so the shadow stamp tints
    /// toward the SAME atmospheric inscatter the forward pass applied at
    /// this pixel's depth (engine `shadow_apply_hlsl.hlsl:211` inscatter).
    camera_world: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: ShadowApplyUniforms;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var depth_sampler: sampler;
@group(0) @binding(3) var shadow_depth: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var receiver_normal: texture_2d<f32>;
@group(0) @binding(6) var normal_sampler: sampler;

// ─── Atmosphere fog inscatter (engine `shadow_apply_hlsl.hlsl:210-217`) ───
// The engine emits `inscatter * g_exposure.rrr` as the shadow stamp color,
// NOT black: with the alpha-blend `Src*(1-darken) + Dst*darken`, this
// preserves the fog inscatter on shadowed pixels (the shadow darkens only
// the `lit*extinction` term). Engine packs inscatter as a depth-linear
// near/far fit of `c_atmosphere_fog_interface::compute_scattering` (dllcache
// `submit_visibility_and_render @ 0x1806BB7A0` → `compute_scattering @
// 0x1803AF020`); since we already reconstruct world position below, we call
// `compute_scattering(camera, world)` directly — exact match to the forward
// pass instead of a linear approximation, so shadowed fog == unshadowed fog.
//
// `EngineAtmosphereRaw` + `EngineExposure` layouts mirror
// `engine_bindings.wgsl` (same `GpuAtmosphereData` / `GpuEngineExposure`
// source buffers bound by the renderer).
const k_log2_e: f32 = 1.4426950;

struct EngineAtmosphereRaw {
    slot0_sun_dir_dist_bias: vec4<f32>,
    slot1_sun_int_norm_thickness: vec4<f32>,
    slot2_beta_m_log2e_g1: vec4<f32>,
    slot3_beta_p_log2e_refh: vec4<f32>,
    slot4_beta_m_angular_mieh: vec4<f32>,
    slot5_beta_p_angular_rayh: vec4<f32>,
    slot6_g2: vec4<f32>,
    slot7_sun_disc: vec4<f32>,
    slot8_sun_glow: vec4<f32>,
    slot9_sun_tint_horizon: vec4<f32>,
    slot10_horizon_pad: vec4<f32>,
}
@group(0) @binding(7) var<uniform> engine_atmosphere_raw: EngineAtmosphereRaw;

struct EngineExposure {
    g_exposure: vec4<f32>,
    g_alt_exposure: vec4<f32>,
}
@group(0) @binding(8) var<uniform> engine_exposure: EngineExposure;

// Verbatim port of `atmosphere_fx.hlsl::compute_scattering` (same as
// `atmosphere_fx.wgsl`, used by every opaque forward entry point). Returns
// the in-scatter radiance at `world_scene_point` viewed from `view_point`.
fn shadow_inscatter(view_point: vec3<f32>, world_scene_point: vec3<f32>) -> vec3<f32> {
    let r = engine_atmosphere_raw;
    let sun_direction = r.slot0_sun_dir_dist_bias.xyz;
    let distance_bias = r.slot0_sun_dir_dist_bias.w;
    let sun_intensity_normalized = r.slot1_sun_int_norm_thickness.xyz;
    let max_fog_thickness = r.slot1_sun_int_norm_thickness.w;
    let beta_m_log2e = r.slot2_beta_m_log2e_g1.xyz;
    let mie_g_plus_one = r.slot2_beta_m_log2e_g1.w;
    let beta_p_log2e = r.slot3_beta_p_log2e_refh.xyz;
    let reference_height = r.slot3_beta_p_log2e_refh.w;
    let beta_m_angular = r.slot4_beta_m_angular_mieh.xyz;
    let mie_height_scale = r.slot4_beta_m_angular_mieh.w;
    let beta_p_angular = r.slot5_beta_p_angular_rayh.xyz;
    let rayleigh_height_scale = r.slot5_beta_p_angular_rayh.w;
    let mie_g_times_two = r.slot6_g2.x;

    // Atmosphere disabled (slot1.w negative) → no fog.
    if (max_fog_thickness < 0.0) {
        return vec3<f32>(0.0);
    }

    var view_vector = view_point - world_scene_point;
    var dist = length(view_vector);
    if (dist < 1e-6) {
        return vec3<f32>(0.0);
    }
    view_vector = view_vector / dist;
    let c_theta = -dot(view_vector, sun_direction);

    dist = max(dist + distance_bias, 0.0);
    dist = min(dist, max_fog_thickness);

    var view_height = max(view_point.z - reference_height, 0.0);
    var scene_height = max(world_scene_point.z - reference_height, 0.0);
    let diff = view_height - scene_height;

    view_height = view_height * k_log2_e;
    scene_height = scene_height * k_log2_e;

    let mie_h = max(mie_height_scale, 1e-4);
    let ray_h = max(rayleigh_height_scale, 1e-4);

    var extinction: vec3<f32>;
    if (diff * diff > 0.001) {
        let dp = -(exp2(-view_height / mie_h) - exp2(-scene_height / mie_h)) * dist * mie_h / diff;
        let dm = -(exp2(-view_height / ray_h) - exp2(-scene_height / ray_h)) * dist * ray_h / diff;
        extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    } else {
        let dp = exp2(-view_height / mie_h) * dist;
        let dm = exp2(-view_height / ray_h) * dist;
        extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    }

    let beta_m_theta = beta_m_angular * (1.0 + c_theta * c_theta);
    let heyey_term = mie_g_plus_one - mie_g_times_two * c_theta;
    let heyey_term_one_pt_five = pow(max(heyey_term, 1e-4), -1.5);
    let beta_p_theta = beta_p_angular * heyey_term_one_pt_five;

    return sun_intensity_normalized
        * (beta_m_theta + beta_p_theta)
        * (vec3<f32>(1.0) - extinction);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Standard fullscreen triangle (matches our other postprocess passes).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
    );
    var out: VsOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

// 9-tap manual PCF, per engine `shadow_apply_hlsl.hlsl:74-110`. Each
// tap is a single hardware-comparison sample (returns 1.0 if frag
// depth ≤ stored, 0.0 otherwise). 9 taps in a 3×3 grid centered on
// `light_uv`, summed and divided by 9. `compare_ref` already includes
// the slope-based depth bias subtracted (so deeper-than-actual occluders
// don't false-positive shadow this pixel).
fn pcf_9tap(light_uv: vec2<f32>, compare_ref: f32) -> f32 {
    // Engine `sample_percentage_closer_PCF_3x3_block` (PC path,
    // shadow_apply_hlsl.hlsl:74-110): nine POINT comparison taps on a
    // 1-texel grid, weighted BILINEARLY by the fragment's sub-texel
    // position so the effective footprint is ~2×2 (sharp) rather than a
    // flat 3×3 box. `shadow_sampler` is Nearest, so each tap is a single
    // point comparison (1 = unshadowed); the smoothing is the weighting.
    // `shadow_pixel_size.x` is √2/res; ×0.7071 → one texel.
    let texel = u.shadow_pixel_size.x * 0.70710678;
    let blend = fract(light_uv / texel);          // engine blend.xy
    let inv = vec2<f32>(1.0) - blend;             // engine blend.zw
    let t = texel;
    let c =
          inv.x   * inv.y   * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(-t, -t), compare_ref)
        +           inv.y   * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(0.0, -t), compare_ref)
        + blend.x * inv.y   * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>( t, -t), compare_ref)
        + inv.x             * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(-t, 0.0), compare_ref)
        +                     textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(0.0, 0.0), compare_ref)
        + blend.x           * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>( t, 0.0), compare_ref)
        + inv.x   * blend.y * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(-t,  t), compare_ref)
        +           blend.y * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>(0.0,  t), compare_ref)
        + blend.x * blend.y * textureSampleCompare(shadow_depth, shadow_sampler, light_uv + vec2<f32>( t,  t), compare_ref);
    return c * 0.25;
}

// 16-tap fancy 5×5 predicated PCF, per engine
// `shadow_apply_fancy_hlsl.hlsl:13-89`. Taps at half-integer offsets
// {-1.5, -0.5, +0.5, +1.5} × pixel. Uses sub-texel `frac` blend so the
// kernel adapts to where the fragment falls within a shadow texel —
// edge taps weighted by one fractional component, corner taps by two,
// center taps full weight. Total weight sums to 9.0 → divide.
fn pcf_5x5_fancy(light_uv: vec2<f32>, compare_ref: f32) -> f32 {
    let step = u.shadow_pixel_size.x;
    let frac_pos = fract(light_uv / step + vec2<f32>(0.5));
    let blend_xy = frac_pos;
    let blend_zw = vec2<f32>(1.0) - blend_xy;

    let offsets = array<f32, 4>(-1.5, -0.5, 0.5, 1.5);
    var sum: f32 = 0.0;

    for (var iy: i32 = 0; iy < 4; iy = iy + 1) {
        for (var ix: i32 = 0; ix < 4; ix = ix + 1) {
            let ox = offsets[ix] * step;
            let oy = offsets[iy] * step;
            let s = textureSampleCompare(
                shadow_depth, shadow_sampler, light_uv + vec2<f32>(ox, oy), compare_ref,
            );
            // Per HLSL: x-edge taps (ix=0 left, ix=3 right) take blend.z / blend.x;
            // x-center taps (ix=1, ix=2) take 1.0. Same for y.
            var wx: f32;
            if (ix == 0) { wx = blend_zw.x; }
            else if (ix == 3) { wx = blend_xy.x; }
            else { wx = 1.0; }
            var wy: f32;
            if (iy == 0) { wy = blend_zw.y; }
            else if (iy == 3) { wy = blend_xy.y; }
            else { wy = 1.0; }
            sum = sum + wx * wy * s;
        }
    }
    return sum * (1.0 / 9.0);
}

// 4-tap faster 2×2 bilinear PCF, per engine
// `shadow_apply_faster_hlsl.hlsl:13-41`. Taps at half-integer offsets
// {-0.5, +0.5}. Each tap weighted by product of `frac` blend components
// so total weight sums to 1.0 — no divide needed.
fn pcf_2x2_faster(light_uv: vec2<f32>, compare_ref: f32) -> f32 {
    let step = u.shadow_pixel_size.x;
    let frac_pos = fract(light_uv / step + vec2<f32>(0.5));
    let blend_xy = frac_pos;
    let blend_zw = vec2<f32>(1.0) - blend_xy;

    let s00 = textureSampleCompare(
        shadow_depth, shadow_sampler, light_uv + vec2<f32>(-0.5 * step, -0.5 * step), compare_ref,
    );
    let s10 = textureSampleCompare(
        shadow_depth, shadow_sampler, light_uv + vec2<f32>( 0.5 * step, -0.5 * step), compare_ref,
    );
    let s01 = textureSampleCompare(
        shadow_depth, shadow_sampler, light_uv + vec2<f32>(-0.5 * step,  0.5 * step), compare_ref,
    );
    let s11 = textureSampleCompare(
        shadow_depth, shadow_sampler, light_uv + vec2<f32>( 0.5 * step,  0.5 * step), compare_ref,
    );

    return blend_zw.x * blend_zw.y * s00
         + blend_xy.x * blend_zw.y * s10
         + blend_zw.x * blend_xy.y * s01
         + blend_xy.x * blend_xy.y * s11;
}

// Dual-target output matching the engine's convert_to_render_target
// pattern. RT0 = lighting_base (LDR), RT1 = hdr_dark (bloom-extract /
// auto-exposure feed). On MCC PC `DARK_COLOR_MULTIPLIER = g_exposure.g
// = 1.0` so both targets receive the same value. Without writing RT1
// the bloom-extract sample sees the un-shadowed scene → auto-exposure
// undershoots by ~0.6 stops on shadow-heavy scenes.
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> AccumPixel {
    // ── DEBUG: pcf_quality.y as flat-stamp test mode. When > 0.5, every
    // fragment stamps `(0, 0, 0, debug_alpha)` regardless of depth/PCF
    // math. If the screen visibly darkens with this on but not normally,
    // the math (depth reconstruction, light_uv, PCF) is the bug — not
    // the pipeline plumbing.
    let dbg_alpha = u.pcf_quality.y;
    if (dbg_alpha > 0.001) {
        let v = vec4<f32>(0.0, 0.0, 0.0, dbg_alpha);
        return AccumPixel(v, v);
    }

    // 1. Sample scene depth at this pixel. wgpu textureSample on a
    //    `texture_depth_2d` returns linear depth in [0, 1].
    let depth = textureSample(scene_depth, depth_sampler, in.uv);

    // 2. Reconstruct world position from clip-space (uv, depth) via the
    //    inverse view-projection matrix. uv -> NDC: x' = uv.x*2-1,
    //    y' = 1-uv.y*2 (wgpu y points down in textures, up in clip).
    let ndc = vec3<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, depth);
    let world_h = u.inverse_view_projection * vec4<f32>(ndc, 1.0);
    let world = world_h.xyz / world_h.w;

    // 3. Transform world position into light-clip space.
    let light_clip = u.world_to_shadow * vec4<f32>(world, 1.0);
    if (light_clip.w <= 0.0) {
        // Frag is behind the light's projection plane — leave unshadowed.
        let v = vec4<f32>(0.0);
        return AccumPixel(v, v);
    }
    let light_ndc = light_clip.xyz / light_clip.w;

    // Cull pixels outside the shadow frustum (no stamp written there).
    if (any(light_ndc.xy < vec2<f32>(-1.0)) || any(light_ndc.xy > vec2<f32>(1.0))
        || light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        let v = vec4<f32>(0.0);
        return AccumPixel(v, v);
    }
    // ── DEBUG: pcf_quality.w ≥ 0.5 stamps 0.4 alpha for every fragment
    // INSIDE the caster's frustum, bypassing depth comparison + PCF.
    // If shadows appear (as solid darker rectangles under casters), the
    // matrix is fine and the bug is in PCF / depth comparison. If still
    // no darkening, the matrix doesn't land receivers in the frustum.
    let frustum_only = u.pcf_quality.w;
    if (frustum_only > 0.5) {
        let v = vec4<f32>(0.0, 0.0, 0.0, 0.4);
        return AccumPixel(v, v);
    }
    let light_uv_full = vec2<f32>(light_ndc.x * 0.5 + 0.5, 0.5 - light_ndc.y * 0.5);
    // Generate wrote into a sub-rect at the top-left of the target
    // (wgpu viewport origin is top-left). Scale UV [0..1] into
    // [0..uv_scale] so we sample the populated region.
    let light_uv = light_uv_full * u.shadow_uv_scale.xy;

    // 4. Receiver normal + slope-based depth bias. Engine
    //    `shadow_apply_hlsl.hlsl:189-198`. Normal MRT stores
    //    `(n * 0.5 + 0.5, 1.0)` in Rgba8Unorm.
    let normal_packed = textureSample(receiver_normal, normal_sampler, in.uv).xyz;
    let normal_world = normalize(normal_packed * 2.0 - 1.0);
    let cosine_raw = dot(normal_world, u.light_dir_world.xyz);
    // engine: max(cosine, 0.24253562503633297351890646211612) — limits
    // max slope to 4.0 and prevents divide-by-zero. Per
    // `shadow_apply_hlsl.hlsl:190`.
    let cosine_for_slope = max(cosine_raw, 0.24253562503633297);
    let slope = sqrt(max(0.0, 1.0 - cosine_for_slope * cosine_for_slope))
        / cosine_for_slope + 0.2;
    let depth_bias = slope * u.shadow_pixel_size.x;
    // textureSampleCompare semantics: returns 1.0 if compare_ref ≤
    // stored_depth. Subtract bias so we count near occluders as
    // unshadowing — matches engine `step(max_depth, sampled.r)` after
    // `max_depth = frag_z - bias`.
    let compare_ref = light_ndc.z - depth_bias;

    // 5. PCF — quality-mode-driven kernel selector (engine-equivalent of
    //    compiling `shadow_apply.hlsl` with #defined SAMPLE_PERCENTAGE_CLOSER).
    let quality = u32(u.pcf_quality.x);
    var unshadowed: f32;
    if (quality == 1u) {
        unshadowed = pcf_5x5_fancy(light_uv, compare_ref);
    } else if (quality == 2u) {
        unshadowed = pcf_2x2_faster(light_uv, compare_ref);
    } else {
        unshadowed = pcf_9tap(light_uv, compare_ref);
    }

    // 6. Cosine falloff + depth-z falloff (engine `shadow_apply_hlsl.hlsl:
    //    154-163`): shadow_darkness = alpha × (1 - saturate(2z-1)⁴) ×
    //    cosine. The z-falloff fades the shadow toward the FAR end of the
    //    volume — this is the engine's mechanism that keeps a caster's
    //    shadow from stamping at full strength onto geometry far behind the
    //    intended receiver (e.g. a weapon on a ledge stamping onto the wall
    //    below): that geometry sits near the volume's back plane (high z) →
    //    faded out. (Removing this is what let those phantom second shadows
    //    appear; it also mildly fades a tall caster's far footprint, which
    //    is the engine's behaviour too.)
    let shadow_falloff_a = saturate(light_ndc.z * 2.0 - 1.0);
    let shadow_falloff = shadow_falloff_a * shadow_falloff_a;
    let no_cosine = u.pcf_quality.z;
    let cosine_falloff = select(saturate(cosine_raw), 1.0, no_cosine > 0.5);
    let shadow_darkness =
        u.caster_opacity.x * (1.0 - shadow_falloff * shadow_falloff) * cosine_falloff;
    // Engine `shadow_apply_hlsl.hlsl:204-205`:
    //   darken = saturate(1 - shadow_darkness + percentage_closer*shadow_darkness)
    //   darken *= darken            // <-- the SQUARING (was missing here)
    // `darken` is the light-KEEP factor (1 = fully lit, 0 = fully shadowed);
    // `unshadowed` == engine `percentage_closer`. The square roughly doubles
    // shadow strength and sharpens the umbra/penumbra contrast — without it
    // the stamp was linear (`(1-unshadowed)*shadow_darkness`) and read washed
    // out: umbra only 85% dark vs the engine's ~97.75%. With SrcAlpha/
    // OneMinusSrcAlpha blend and rgb=0, stamp_alpha = 1-darken reproduces the
    // engine's `dst*darken + shadow_color*(1-darken)` (shadow_color=black for
    // now; engine uses atmosphere inscatter·exposure — deferred).
    var darken = saturate(1.0 - shadow_darkness + unshadowed * shadow_darkness);
    darken = darken * darken;
    let stamp_alpha = 1.0 - darken;

    // Output: alpha-blend (SrcAlpha, OneMinusSrcAlpha). Engine
    // `shadow_apply_hlsl.hlsl:217` emits `inscatter * g_exposure.rrr` as the
    // stamp color (NOT black). With this blend the result is
    //   inscatter*exposure*(1-darken) + Dst*darken
    //   = exposure * (lit*darken*extinction + inscatter)
    // i.e. the shadow darkens only the lit*extinction term and the
    // atmospheric inscatter is preserved — so shadows in fog keep the fog
    // color instead of crushing to pure black. `inscatter` is recomputed
    // per-pixel here (camera → reconstructed world) so it matches exactly
    // what the forward opaque pass added at this pixel. Same value to both
    // RT0 and RT1 since DARK_COLOR_MULTIPLIER = 1.0 on PC.
    let inscatter = shadow_inscatter(u.camera_world.xyz, world);
    // Sanitize the stamp color. The frustum culls above use comparisons on
    // `light_ndc`, which are all FALSE when `world` is non-finite (NaN/Inf
    // compares false) — so sky / far-plane pixels, whose depth reconstructs
    // to a degenerate (w≈0) `world` and thus an Inf position, slip through to
    // here. `shadow_inscatter` then returns NaN (Inf/Inf in the view-vector
    // normalize) or −Inf (negative optical depth → extinction overflow), and
    // the alpha blend does `Src.rgb * SrcAlpha` → `NaN * alpha = NaN`, which
    // poisons the whole stamp quad to opaque black (the regression boxes over
    // sky). `clamp` is built on fmin/fmax, which are NaN-suppressing on Metal
    // (return the finite operand), so this maps NaN→0 and −Inf→0 — exactly the
    // harmless `rgb=0` the stamp used before inscatter tinting — while leaving
    // real receiver pixels' finite inscatter untouched. 65504 = max half-float.
    let stamp_rgb = clamp(
        inscatter * engine_exposure.g_exposure.x,
        vec3<f32>(0.0),
        vec3<f32>(65504.0),
    );
    let v = vec4<f32>(stamp_rgb, stamp_alpha);
    return AccumPixel(v, v);
}
