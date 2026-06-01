// Port of `self_illumination_fx.hlsl` — `times_diffuse` variant
// (line 145). HLSL function: `calc_self_illumination_times_diffuse_ps`.
//
// HLSL:
//   float3 calc_self_illumination_times_diffuse_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float3 self_illum_texture_sample = sample(self_illum_map, xform(uv)).rgb;
//       float albedo_blend = max(self_illum_texture_sample.g * 10.0 - 9.0, 0.0);
//       float3 albedo_part = albedo_blend + (1 - albedo_blend) * albedo;
//       float3 mix_illum_color = primary_change_color_blend * primary_change_color
//                              + (1 - primary_change_color_blend) * self_illum_color.xyz;
//       float3 self_illum = albedo_part * mix_illum_color * self_illum_intensity * self_illum_texture_sample;
//       return self_illum;
//   }

fn calc_self_illumination_times_diffuse_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let xformed = transform_texcoord(texcoord, material.self_illum_map_xform);
    let self_illum_sample = textureSample(self_illum_map, self_illum_map_sampler, xformed).rgb;

    let albedo_blend = max(self_illum_sample.g * 10.0 - 9.0, 0.0);
    let albedo_part = vec3<f32>(albedo_blend) + (1.0 - albedo_blend) * (*albedo);

    let p_blend = material.primary_change_color_blend.x;
    let mix_color = p_blend * material.primary_change_color.xyz
                  + (1.0 - p_blend) * material.self_illum_color.xyz;

    return albedo_part * mix_color * material.self_illum_intensity.x * self_illum_sample;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_times_diffuse_ps(texcoord, albedo, view_dir);
}
