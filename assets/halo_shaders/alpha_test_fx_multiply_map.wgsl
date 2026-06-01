// Port of `alpha_test_custom_fx.hlsl::calc_alpha_test_multiply_map_ps`.
// Used by shader_custom (`rmcs`) materials whose rmdf
// `alpha_test = multiply_map` option fires — e.g. deadlock_net,
// deadlock_netb (chain-link fence + canopy materials).
//
// HLSL (verbatim, alpha_test_custom_fx.hlsl):
//   void calc_alpha_test_multiply_map_ps(
//       in float2 texcoord,
//       out float output_alpha)
//   {
//       float4 alpha_test_layer = sample2D(alpha_test_map,
//           transform_texcoord(texcoord, alpha_test_map_xform));
//       float4 multiply_layer   = sample2D(multiply_map,
//           transform_texcoord(texcoord, multiply_map_xform));
//       float alpha = alpha_test_layer.a * multiply_layer.a;
//       output_alpha = alpha;
//       clip(alpha - 0.5f);
//   }
//
// The `clip(alpha - 0.5)` is handled by the umbrella's
// `if (output_alpha < 0.5) { discard; }` check — same pattern as
// `alpha_test_fx_on.wgsl`.

fn calc_alpha_test_multiply_map_ps(
    texcoord: vec2<f32>,
    output_alpha: ptr<function, f32>,
) {
    let xformed_a = transform_texcoord(texcoord, material.alpha_test_map_xform);
    let xformed_m = transform_texcoord(texcoord, material.multiply_map_xform);
    let a = textureSample(alpha_test_map, alpha_test_map_sampler, xformed_a).a;
    let m = textureSample(multiply_map, multiply_map_sampler, xformed_m).a;
    *output_alpha = a * m;
}

fn calc_alpha_test(texcoord: vec2<f32>, output_alpha: ptr<function, f32>) {
    calc_alpha_test_multiply_map_ps(texcoord, output_alpha);
}
