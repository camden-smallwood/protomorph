// Port of `albedo_fx.hlsl` — `two_change_color` variant (line 144).
//
// HLSL:
//   void calc_albedo_two_change_color_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base   = sample(base_map,         xform(uv, base_map_xform));
//       float4 detail = sample(detail_map,       xform(uv, detail_map_xform));
//       float4 cc     = sample(change_color_map, xform(uv, change_color_map_xform));
//       float3 cc_tint = ((1-cc.x) + cc.x * primary_change_color)
//                      * ((1-cc.y) + cc.y * secondary_change_color);
//       albedo.rgb = DETAIL_MULTIPLIER * base.rgb * detail.rgb * cc_tint;
//       albedo.a   = base.a * detail.a;
//       apply_pc_albedo_modifier(albedo, normal);
//   }

fn calc_albedo_two_change_color_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let cc = textureSample(change_color_map, change_color_map_sampler, transform_texcoord(texcoord, material.change_color_map_xform));

    let cc_tint = (vec3<f32>(1.0) - vec3<f32>(cc.r) + vec3<f32>(cc.r) * material.primary_change_color.xyz)
                * (vec3<f32>(1.0) - vec3<f32>(cc.g) + vec3<f32>(cc.g) * material.secondary_change_color.xyz);

    (*albedo).r = DETAIL_MULTIPLIER * base.r * detail.r * cc_tint.r;
    (*albedo).g = DETAIL_MULTIPLIER * base.g * detail.g * cc_tint.g;
    (*albedo).b = DETAIL_MULTIPLIER * base.b * detail.b * cc_tint.b;
    (*albedo).a = base.a * detail.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_two_change_color_ps(texcoord, albedo, normal, misc);
}
