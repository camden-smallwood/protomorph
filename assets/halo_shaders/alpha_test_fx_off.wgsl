// Port of `alpha_test_fx.hlsl` — `off` variant (line 6).
//
// HLSL:
//   void calc_alpha_test_off_ps(in float2 texcoord, out float output_alpha)
//   { output_alpha = 1.0; }

fn calc_alpha_test_off_ps(
    texcoord: vec2<f32>,
    output_alpha: ptr<function, f32>,
) {
    let _u_uv = texcoord;
    *output_alpha = 1.0;
}

fn calc_alpha_test(texcoord: vec2<f32>, output_alpha: ptr<function, f32>) {
    calc_alpha_test_off_ps(texcoord, output_alpha);
}
