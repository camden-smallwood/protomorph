// Port of `alpha_test_fx.hlsl` — `on` variant (line 16).
//
// HLSL:
//   void calc_alpha_test_on_ps(in float2 texcoord, out float output_alpha)
//   {
//       float alpha = sample(alpha_test_map, transform_texcoord(texcoord, alpha_test_map_xform)).a;
//       output_alpha = alpha;
//       // clip(alpha-0.5) — handled by umbrella's `if (alpha < 0.5) { discard; }`.
//   }

fn calc_alpha_test_on_ps(
    texcoord: vec2<f32>,
    output_alpha: ptr<function, f32>,
) {
    let xformed = transform_texcoord(texcoord, material.alpha_test_map_xform);
    *output_alpha = textureSample(alpha_test_map, alpha_test_map_sampler, xformed).a;
}

fn calc_alpha_test(texcoord: vec2<f32>, output_alpha: ptr<function, f32>) {
    calc_alpha_test_on_ps(texcoord, output_alpha);
}
