// Port of self_illumination_halogram_fx.hlsl:130-159 — scope_blur.
//
// NOT parallax — used for weapon scope HUD overlays. Samples the
// self_illum_map at 4 diagonal offsets, averages, then computes a
// 2-channel weighted blend between two cbuffer colors:
//   color = avg.r * self_illum_color
//         + (1 - avg.r) * avg.g * self_illum_heat_color
// `self_illum_heat_color` is unique to this variant (HLSL line 130).
//
// HLSL constants (PC path, line 141):
//   texStep = float2(0.001736 / 2.0, 0.003125 / 2.0)
//          ≈ float2(0.000868, 0.001563)

// HLSL: float2(0.001736/2.0, 0.003125/2.0) = (0.000868, 0.0015625).
const SCOPE_BLUR_TEX_STEP: vec2<f32> = vec2<f32>(0.000868, 0.0015625);

fn calc_self_illumination_scope_blur_ps(
    texcoord_in: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_albedo = albedo;
    let _u_view = view_dir;

    let texcoord = transform_texcoord(texcoord_in, material.self_illum_map_xform);

    // 4 diagonal taps — HLSL lines 142-145.
    let color_0 = textureSample(self_illum_map, self_illum_map_sampler,
        vec2<f32>(texcoord.x + SCOPE_BLUR_TEX_STEP.x, texcoord.y + SCOPE_BLUR_TEX_STEP.y),
    );
    let color_1 = textureSample(self_illum_map, self_illum_map_sampler,
        vec2<f32>(texcoord.x - SCOPE_BLUR_TEX_STEP.x, texcoord.y + SCOPE_BLUR_TEX_STEP.y),
    );
    let color_2 = textureSample(self_illum_map, self_illum_map_sampler,
        vec2<f32>(texcoord.x - SCOPE_BLUR_TEX_STEP.x, texcoord.y - SCOPE_BLUR_TEX_STEP.y),
    );
    let color_3 = textureSample(self_illum_map, self_illum_map_sampler,
        vec2<f32>(texcoord.x + SCOPE_BLUR_TEX_STEP.x, texcoord.y - SCOPE_BLUR_TEX_STEP.y),
    );

    let average = (color_0 + color_1 + color_2 + color_3).xy * 0.25;

    let color = average.r * material.self_illum_color.rgb
        + (1.0 - average.r) * average.g * material.self_illum_heat_color.rgb;

    return color * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_scope_blur_ps(texcoord, albedo, view_dir);
}
