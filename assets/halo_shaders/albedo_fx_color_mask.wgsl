// Port of `albedo_fx.hlsl` — `color_mask` variant (line 229).
//
// Three-channel color mask drives a 3-stage tint multiplier. Each
// channel of `color_mask_map.rgb` selects between (1.0) and the
// corresponding albedo_color slot, gamma-divided by `neutral_gray`.
//
// HLSL:
//   void calc_albedo_color_mask_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       float4 base       = sample(base_map,       xform(uv, base_map_xform));
//       float4 detail     = sample(detail_map,     xform(uv, detail_map_xform));
//       float4 color_mask = sample(color_mask_map, xform(uv, color_mask_map_xform));
//       float4 tint =
//           ((1 - cm.x) + cm.x * albedo_color.xyzw  / float4(neutral_gray.xyz, 1)) *
//           ((1 - cm.y) + cm.y * albedo_color2.xyzw / float4(neutral_gray.xyz, 1)) *
//           ((1 - cm.z) + cm.z * albedo_color3.xyzw / float4(neutral_gray.xyz, 1));
//       albedo.rgb = base.rgb * (detail.rgb * DETAIL_MULTIPLIER) * tint.rgb;
//       albedo.w   = base.w * detail.w * tint.w;
//       apply_pc_albedo_modifier(albedo, normal);  // (no-op on PC path)
//   }

fn calc_albedo_color_mask_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;

    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let cm = textureSample(color_mask_map, color_mask_map_sampler, transform_texcoord(texcoord, material.color_mask_map_xform));

    let neutral_rgba = vec4<f32>(material.neutral_gray.xyz, 1.0);
    let stage0 = (vec4<f32>(1.0) - vec4<f32>(cm.r)) + vec4<f32>(cm.r) * (material.albedo_color  / neutral_rgba);
    let stage1 = (vec4<f32>(1.0) - vec4<f32>(cm.g)) + vec4<f32>(cm.g) * (material.albedo_color2 / neutral_rgba);
    let stage2 = (vec4<f32>(1.0) - vec4<f32>(cm.b)) + vec4<f32>(cm.b) * (material.albedo_color3 / neutral_rgba);
    let tint = stage0 * stage1 * stage2;

    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * tint.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * tint.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * tint.b;
    (*albedo).a = base.a * detail.a * tint.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_color_mask_ps(texcoord, albedo, normal, misc);
}
