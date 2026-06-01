// Port of `specular_mask_fx.hlsl` — `no_specular_mask` variant (line 1).
//
// HLSL:
//   void calc_specular_mask_no_specular_mask_ps(
//       in float2 texcoord, in float in_specular_mask, out float specular_mask)
//   { specular_mask = 1.0; }

fn calc_specular_mask_no_specular_mask_ps(
    texcoord: vec2<f32>,
    in_specular_mask: f32,
    specular_mask: ptr<function, f32>,
) {
    let _u_uv = texcoord;
    let _u_in = in_specular_mask;
    *specular_mask = 1.0;
}

fn calc_specular_mask(texcoord: vec2<f32>, in_specular_mask: f32, specular_mask: ptr<function, f32>) {
    calc_specular_mask_no_specular_mask_ps(texcoord, in_specular_mask, specular_mask);
}
