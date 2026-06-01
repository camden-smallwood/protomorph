// Port of `albedo_fx.hlsl` — `detail_blend` variant (line ~94).
//
// HLSL:
//   void calc_albedo_detail_blend_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base    = sample(base_map,    xform(uv, base_map_xform));
//       float4 detail  = sample(detail_map,  xform(uv, detail_map_xform));
//       float4 detail2 = sample(detail_map2, xform(uv, detail_map2_xform));
//       albedo.xyz = (1 - base.w) * detail.xyz + base.w * detail2.xyz;
//       albedo.xyz = DETAIL_MULTIPLIER * base.xyz * albedo.xyz;
//       albedo.w   = (1 - base.w) * detail.w + base.w * detail2.w;
//       apply_pc_albedo_modifier(albedo, normal);
//   }

fn calc_albedo_detail_blend_ps(
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
    let blended = (1.0 - base.a) * detail.rgb + base.a * detail2.rgb;
    let scaled = DETAIL_MULTIPLIER * base.rgb * blended;
    (*albedo).r = scaled.r;
    (*albedo).g = scaled.g;
    (*albedo).b = scaled.b;
    (*albedo).a = (1.0 - base.a) * detail.a + base.a * detail2.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_detail_blend_ps(texcoord, albedo, normal, misc);
}
