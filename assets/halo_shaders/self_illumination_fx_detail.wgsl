// Port of `self_illumination_fx.hlsl` — `detail` variant (line 109).
// (rmdf option name: `illum_detail`.)
//
// HLSL:
//   float3 calc_self_illumination_detail_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float4 self_illum        = sample(self_illum_map,        xform(texcoord, self_illum_map_xform));
//       float4 self_illum_detail = sample(self_illum_detail_map, xform(texcoord, self_illum_detail_map_xform));
//       float4 result = self_illum * (self_illum_detail * DETAIL_MULTIPLIER) * self_illum_color;
//       result.rgb *= self_illum_intensity;
//       return result.rgb;
//   }

fn calc_self_illumination_detail_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_albedo = albedo;
    let xformed = transform_texcoord(texcoord, material.self_illum_map_xform);
    let self_illum = textureSample(self_illum_map, self_illum_map_sampler, xformed);
    let xformed_detail = transform_texcoord(texcoord, material.self_illum_detail_map_xform);
    let self_illum_detail = textureSample(self_illum_detail_map, self_illum_detail_map_sampler, xformed_detail);
    let result = self_illum * (self_illum_detail * DETAIL_MULTIPLIER) * material.self_illum_color;
    return result.rgb * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_detail_ps(texcoord, albedo, view_dir);
}
