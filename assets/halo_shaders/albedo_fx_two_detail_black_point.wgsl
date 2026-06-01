// Port of `albedo_fx.hlsl` — `two_detail_black_point` variant
// (line ~250).
//
// HLSL:
//   void calc_albedo_two_detail_black_point_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base    = sample(base_map,    xform(uv, base_map_xform));
//       float4 detail  = sample(detail_map,  xform(uv, detail_map_xform));
//       float4 detail2 = sample(detail_map2, xform(uv, detail_map2_xform));
//       albedo.xyz = base.xyz * (DETAIL_MULTIPLIER²) * detail.xyz * detail2.xyz;
//       albedo.w   = apply_black_point(base.w, detail.w * detail2.w);
//       apply_pc_albedo_modifier(albedo, normal);
//   }
//
// `apply_black_point` from utilities_fx.hlsl:123 — soft remap of `alpha`
// to a black-point pivot at `bp = base.w`:
//   mid_point = (bp + 1) / 2
//   return mid_point * saturate((alpha - bp) / (mid_point - bp))
//        + saturate(alpha - mid_point);

fn apply_black_point(black_point: f32, alpha: f32) -> f32 {
    let mid_point = (black_point + 1.0) * 0.5;
    return mid_point * saturate((alpha - black_point) / (mid_point - black_point))
         + saturate(alpha - mid_point);
}

fn calc_albedo_two_detail_black_point_ps(
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
    (*albedo).a = apply_black_point(base.a, detail.a * detail2.a);
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_two_detail_black_point_ps(texcoord, albedo, normal, misc);
}
