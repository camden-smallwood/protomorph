// Port of `specular_mask_fx.hlsl` — `specular_mask_from_color_texture`
// variant (line 29-35). HLSL function is `calc_specular_mask_color_texture_ps`:
//
//   void calc_specular_mask_color_texture_ps(
//       in float2 texcoord, in float in_specular_mask, out float specular_mask)
//   {
//       specular_mask = sample2D(specular_mask_texture,
//           texcoord*specular_mask_texture_xform.xy + specular_mask_texture_xform.zw).a;
//   }
//
// Algorithmically identical to `_texture_ps`: sample the
// `specular_mask_texture` at the xformed UV, take alpha. The
// variant exists to mark rmop authoring intent — the texture is a
// full color map with a useful alpha (vs a dedicated mask). The
// shader code is the same.

fn calc_specular_mask_color_texture_ps(
    texcoord: vec2<f32>,
    in_specular_mask: f32,
    specular_mask: ptr<function, f32>,
) {
    let _u_in = in_specular_mask;
    let xformed = transform_texcoord(texcoord, material.specular_mask_texture_xform);
    *specular_mask = textureSample(specular_mask_texture, specular_mask_texture_sampler, xformed).a;
}

fn calc_specular_mask(texcoord: vec2<f32>, in_specular_mask: f32, specular_mask: ptr<function, f32>) {
    calc_specular_mask_color_texture_ps(texcoord, in_specular_mask, specular_mask);
}
