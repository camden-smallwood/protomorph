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
}

@group(0) @binding(0) var<uniform> u: ShadowApplyUniforms;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var depth_sampler: sampler;
@group(0) @binding(3) var shadow_depth: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var receiver_normal: texture_2d<f32>;
@group(0) @binding(6) var normal_sampler: sampler;

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
    let step = u.shadow_pixel_size.x;
    var sum: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let off = vec2<f32>(f32(dx), f32(dy)) * step;
            sum = sum + textureSampleCompare(
                shadow_depth, shadow_sampler, light_uv + off, compare_ref,
            );
        }
    }
    return sum * (1.0 / 9.0);
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

    // 6. Cosine falloff + depth-z falloff (engine
    //    `shadow_apply_hlsl.hlsl:154-163`):
    //      shadow_falloff  = saturate(z*2 - 1)²        (only bottom half)
    //      cosine_falloff  = saturate(cosine)
    //      shadow_darkness = caster_opacity * (1 - falloff⁴) * cosine_falloff
    let shadow_falloff_a = saturate(light_ndc.z * 2.0 - 1.0);
    let shadow_falloff = shadow_falloff_a * shadow_falloff_a;
    // ── DEBUG: pcf_quality.z ≥ 0.5 forces cosine_falloff = 1.0. Lets us
    // tell whether the normal-MRT path is killing stamps. If shadows
    // appear with no_cosine but not normally, the bug is that receiver
    // pixels have cleared normals (or the normal decode is wrong).
    let no_cosine = u.pcf_quality.z;
    let cosine_falloff = select(saturate(cosine_raw), 1.0, no_cosine > 0.5);
    let shadow_darkness =
        u.caster_opacity.x * (1.0 - shadow_falloff * shadow_falloff) * cosine_falloff;
    let stamp_alpha = (1.0 - unshadowed) * shadow_darkness;

    // Output: alpha-blend (SrcAlpha, OneMinusSrcAlpha) with rgb=0 darkens
    // toward black by `stamp_alpha`. Engine emits `inscatter * exposure`
    // here instead of black; we use straight black until atmosphere-aware
    // shadow-color modulation lands. Same value to both RT0 and RT1
    // since DARK_COLOR_MULTIPLIER = 1.0 on PC.
    let v = vec4<f32>(0.0, 0.0, 0.0, stamp_alpha);
    return AccumPixel(v, v);
}
