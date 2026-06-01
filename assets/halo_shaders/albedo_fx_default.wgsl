// Port of `albedo_fx.hlsl` — `default` variant (line 76).
//
// HLSL:
//   void calc_albedo_default_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base   = sampleBiasGlobal2D(base_map,   xform(uv, base_map_xform));
//       float4 detail = sampleBiasGlobal2D(detail_map, xform(uv, detail_map_xform));
//       albedo.rgb = base.rgb × (detail.rgb × DETAIL_MULTIPLIER) × albedo_color.rgb;
//       albedo.w   = base.w × detail.w × albedo_color.w;
//       apply_pc_albedo_modifier(albedo, normal);    // no-op (GPU sRGB-decodes via view)
//   }

fn calc_albedo_default_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * material.albedo_color.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * material.albedo_color.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * material.albedo_color.b;
    (*albedo).a = base.a * detail.a * material.albedo_color.a;
}

// HLSL `CALC_ALBEDO(albedo_type)` macro alias for this variant.
fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_default_ps(texcoord, albedo, normal, misc);
}
