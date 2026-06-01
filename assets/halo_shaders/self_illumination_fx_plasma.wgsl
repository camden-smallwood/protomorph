// Port of `self_illumination_fx.hlsl` — `plasma` variant
// (`calc_self_illumination_plasma_ps`, line 52-72).
//
// HLSL:
//   float3 calc_self_illumination_plasma_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float alpha   = sample2D(alpha_mask_map, transform_texcoord(texcoord, alpha_mask_map_xform)).a;
//       float noise_a = sample2D(noise_map_a,    transform_texcoord(texcoord, noise_map_a_xform)).r;
//       float noise_b = sample2D(noise_map_b,    transform_texcoord(texcoord, noise_map_b_xform)).r;
//
//       float diff        = 1.0f - abs(noise_a - noise_b);
//       float medium_diff = pow(diff, thinness_medium);
//       float sharp_diff  = pow(diff, thinness_sharp);
//       float wide_diff   = pow(diff, thinness_wide);
//
//       wide_diff   -= medium_diff;
//       medium_diff -= sharp_diff;
//
//       float3 color = color_medium.rgb * color_medium.a * medium_diff
//                    + color_sharp.rgb  * color_sharp.a  * sharp_diff
//                    + color_wide.rgb   * color_wide.a   * wide_diff;
//
//       return color * alpha * self_illum_intensity;
//   }

fn calc_self_illumination_plasma_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_albedo = albedo;

    let alpha = textureSample(alpha_mask_map, alpha_mask_map_sampler,
        transform_texcoord(texcoord, material.alpha_mask_map_xform),
    ).a;
    let noise_a = textureSample(noise_map_a, noise_map_a_sampler,
        transform_texcoord(texcoord, material.noise_map_a_xform),
    ).r;
    let noise_b = textureSample(noise_map_b, noise_map_b_sampler,
        transform_texcoord(texcoord, material.noise_map_b_xform),
    ).r;

    let diff = 1.0 - abs(noise_a - noise_b);
    var medium_diff = pow(diff, material.thinness_medium.x);
    var sharp_diff  = pow(diff, material.thinness_sharp.x);
    var wide_diff   = pow(diff, material.thinness_wide.x);

    wide_diff   = wide_diff - medium_diff;
    medium_diff = medium_diff - sharp_diff;

    let color =
          material.color_medium.rgb * material.color_medium.a * medium_diff
        + material.color_sharp.rgb  * material.color_sharp.a  * sharp_diff
        + material.color_wide.rgb   * material.color_wide.a   * wide_diff;

    return color * alpha * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_plasma_ps(texcoord, albedo, view_dir);
}
