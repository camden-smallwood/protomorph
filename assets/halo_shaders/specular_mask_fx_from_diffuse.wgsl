// Port of `specular_mask_fx.hlsl` — `from_diffuse` variant (line 9).
//
// HLSL:
//   void calc_specular_mask_from_diffuse_ps(
//       in float2 texcoord, in float in_specular_mask, out float specular_mask)
//   { specular_mask = in_specular_mask; }

fn calc_specular_mask_from_diffuse_ps(
    texcoord: vec2<f32>,
    in_specular_mask: f32,
    specular_mask: ptr<function, f32>,
) {
    let _u_uv = texcoord;
    *specular_mask = in_specular_mask;
}

fn calc_specular_mask(texcoord: vec2<f32>, in_specular_mask: f32, specular_mask: ptr<function, f32>) {
    calc_specular_mask_from_diffuse_ps(texcoord, in_specular_mask, specular_mask);
}
