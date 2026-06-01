// Port of `self_illumination_fx.hlsl` — `three_channel` variant
// (line 78).
//
// HLSL:
//   float3 calc_self_illumination_three_channel_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float4 self_illum = sample(self_illum_map, transform_texcoord(texcoord, self_illum_map_xform));
//       self_illum.rgb = self_illum.r * channel_a.a * channel_a.rgb
//                      + self_illum.g * channel_b.a * channel_b.rgb
//                      + self_illum.b * channel_c.a * channel_c.rgb;
//       return self_illum.rgb * self_illum_intensity;
//   }

fn calc_self_illumination_three_channel_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_albedo = albedo;
    let xformed = transform_texcoord(texcoord, material.self_illum_map_xform);
    let self_illum = textureSample(self_illum_map, self_illum_map_sampler, xformed);
    let result =
          self_illum.r * material.channel_a.a * material.channel_a.rgb
        + self_illum.g * material.channel_b.a * material.channel_b.rgb
        + self_illum.b * material.channel_c.a * material.channel_c.rgb;
    return result * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_three_channel_ps(texcoord, albedo, view_dir);
}
