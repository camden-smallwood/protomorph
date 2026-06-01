// Port of `albedo_fx.hlsl` — `two_detail` variant (line 208).
//
// HLSL:
//   void calc_albedo_two_detail_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base    = sample(base_map,    xform(uv, base_map_xform));
//       float4 detail  = sample(detail_map,  xform(uv, detail_map_xform));
//       float4 detail2 = sample(detail_map2, xform(uv, detail_map2_xform));
//       albedo.xyz = base.xyz * (DETAIL_MULTIPLIER²) * detail.xyz * detail2.xyz;
//       albedo.w   = base.w * detail.w * detail2.w;
//       apply_pc_albedo_modifier(albedo, normal);
//   }

fn calc_albedo_two_detail_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let detail2 = textureSample(detail_map2, detail_map2_sampler, transform_texcoord(texcoord, material.detail_map2_xform));
    let dm2 = DETAIL_MULTIPLIER * DETAIL_MULTIPLIER;
    (*albedo).r = base.r * dm2 * detail.r * detail2.r;
    (*albedo).g = base.g * dm2 * detail.g * detail2.g;
    (*albedo).b = base.b * dm2 * detail.b * detail2.b;
    (*albedo).a = base.a * detail.a * detail2.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_two_detail_ps(texcoord, albedo, normal, misc);
}
