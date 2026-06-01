// Port of `self_illumination_fx.hlsl` — `simple` variant (line 16).
//
// HLSL:
//   float3 calc_self_illumination_simple_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float4 result = sample(self_illum_map, transform_texcoord(texcoord, self_illum_map_xform))
//                     * self_illum_color;
//       result.rgb *= self_illum_intensity;
//       return result.rgb;
//   }

fn calc_self_illumination_simple_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_albedo = albedo;
    let result = textureSample(self_illum_map, self_illum_map_sampler, transform_texcoord(texcoord, material.self_illum_map_xform))
               * material.self_illum_color;
    return result.rgb * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_simple_ps(texcoord, albedo, view_dir);
}
