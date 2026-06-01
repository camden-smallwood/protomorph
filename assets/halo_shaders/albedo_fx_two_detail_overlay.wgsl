// Port of `albedo_fx.hlsl` — `two_detail_overlay` variant (~line 195).
//
// HLSL:
//   void calc_albedo_two_detail_overlay_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base           = sample(base_map,           xform(uv, base_map_xform));
//       float4 detail         = sample(detail_map,         xform(uv, detail_map_xform));
//       float4 detail2        = sample(detail_map2,        xform(uv, detail_map2_xform));
//       float4 detail_overlay = sample(detail_map_overlay, xform(uv, detail_map_overlay_xform));
//       float4 detail_blend = (1 - base.w) * detail + base.w * detail2;
//       albedo.xyz = base.xyz * (DETAIL_MULTIPLIER²) * detail_blend.xyz * detail_overlay.xyz;
//       albedo.w   = detail_blend.w * detail_overlay.w;
//       apply_pc_albedo_modifier(albedo, normal);
//   }

fn calc_albedo_two_detail_overlay_ps(
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
    let detail_overlay = textureSample(detail_map_overlay, detail_map_overlay_sampler, transform_texcoord(texcoord, material.detail_map_overlay_xform));

    let detail_blend = (1.0 - base.a) * detail + base.a * detail2;
    let dm2 = DETAIL_MULTIPLIER * DETAIL_MULTIPLIER;
    (*albedo).r = base.r * dm2 * detail_blend.r * detail_overlay.r;
    (*albedo).g = base.g * dm2 * detail_blend.g * detail_overlay.g;
    (*albedo).b = base.b * dm2 * detail_blend.b * detail_overlay.b;
    (*albedo).a = detail_blend.a * detail_overlay.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_two_detail_overlay_ps(texcoord, albedo, normal, misc);
}
