// Port of `albedo_custom_fx.hlsl::calc_albedo_waterfall_ps` (line 20-45).
//
// Four-texture frothy-water blend used by waterfall transparents. The
// engine's older commented-out path was a layered alpha blend; the
// shipped formula is a multiplicative froth color × base mask, with
// a weighted alpha combining the layer alphas + base alpha + bias.
//
// HLSL:
//   float4 base_mask = sample2D(waterfall_base_mask, ...);
//   float4 layer0    = sample2D(waterfall_layer0,    ...);
//   float4 layer1    = sample2D(waterfall_layer1,    ...);
//   float4 layer2    = sample2D(waterfall_layer2,    ...);
//   float4 frothy_color = layer0 * layer1 * layer2;
//   float  frothy_transparency = layer0.a + layer1.a + layer2.a;
//   albedo.rgb = frothy_color.rgb * base_mask.rgb;
//   albedo.a   = clamp(transparency_frothy_weight * frothy_transparency
//                    + transparency_base_weight  * base_mask.a
//                    + transparency_bias, 0, 1);
//   apply_pc_albedo_modifier(albedo, normal);   // GPU sRGB-decode via view
//
// Rmt2 packs scalar PARAMs as vec4 — read `.x` for the three transparency
// weights.

fn calc_albedo_waterfall_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_normal = normal;
    let _u_misc = misc;
    let base_mask = textureSample(waterfall_base_mask, waterfall_base_mask_sampler,
        transform_texcoord(texcoord, material.waterfall_base_mask_xform),
    );
    let layer0 = textureSample(waterfall_layer0, waterfall_layer0_sampler,
        transform_texcoord(texcoord, material.waterfall_layer0_xform),
    );
    let layer1 = textureSample(waterfall_layer1, waterfall_layer1_sampler,
        transform_texcoord(texcoord, material.waterfall_layer1_xform),
    );
    let layer2 = textureSample(waterfall_layer2, waterfall_layer2_sampler,
        transform_texcoord(texcoord, material.waterfall_layer2_xform),
    );

    let frothy_color = layer0 * layer1 * layer2;
    let frothy_transparency = layer0.a + layer1.a + layer2.a;

    (*albedo).r = frothy_color.r * base_mask.r;
    (*albedo).g = frothy_color.g * base_mask.g;
    (*albedo).b = frothy_color.b * base_mask.b;
    (*albedo).a = clamp(
        material.transparency_frothy_weight.x * frothy_transparency
            + material.transparency_base_weight.x * base_mask.a
            + material.transparency_bias.x,
        0.0,
        1.0,
    );
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_waterfall_ps(texcoord, albedo, normal, misc);
}
