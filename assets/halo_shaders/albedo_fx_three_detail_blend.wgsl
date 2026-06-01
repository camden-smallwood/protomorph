// Port of `albedo_fx.hlsl` — `three_detail_blend` variant (line 114).
//
// HLSL:
//   void calc_albedo_three_detail_blend_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base    = sample(base_map,     xform(uv, base_map_xform));
//       float4 detail1 = sample(detail_map,   xform(uv, detail_map_xform));
//       float4 detail2 = sample(detail_map2,  xform(uv, detail_map2_xform));
//       float4 detail3 = sample(detail_map3,  xform(uv, detail_map3_xform));
//
//       float blend1 = saturate(2.0 * base.w);
//       float blend2 = saturate(2.0 * base.w - 1.0);
//
//       float4 first_blend  = (1 - blend1) * detail1     + blend1 * detail2;
//       float4 second_blend = (1 - blend2) * first_blend + blend2 * detail3;
//
//       albedo.rgb = DETAIL_MULTIPLIER * base.rgb * second_blend.rgb;
//       albedo.a   = second_blend.a;
//   }

fn calc_albedo_three_detail_blend_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;
    let base    = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail1 = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let detail2 = textureSample(detail_map2, detail_map2_sampler, transform_texcoord(texcoord, material.detail_map2_xform));
    let detail3 = textureSample(detail_map3, detail_map3_sampler, transform_texcoord(texcoord, material.detail_map3_xform));

    let blend1 = clamp(2.0 * base.a, 0.0, 1.0);
    let blend2 = clamp(2.0 * base.a - 1.0, 0.0, 1.0);

    let first_blend  = (1.0 - blend1) * detail1     + blend1 * detail2;
    let second_blend = (1.0 - blend2) * first_blend + blend2 * detail3;

    (*albedo).r = DETAIL_MULTIPLIER * base.r * second_blend.r;
    (*albedo).g = DETAIL_MULTIPLIER * base.g * second_blend.g;
    (*albedo).b = DETAIL_MULTIPLIER * base.b * second_blend.b;
    (*albedo).a = second_blend.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_three_detail_blend_ps(texcoord, albedo, normal, misc);
}
