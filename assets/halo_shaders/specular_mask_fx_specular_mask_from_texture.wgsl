// Port of `specular_mask_fx.hlsl` — `specular_mask_from_texture`
// variant (line 20). HLSL function is `calc_specular_mask_texture_ps`.
//
// HLSL:
//   void calc_specular_mask_texture_ps(
//       in float2 texcoord, in float in_specular_mask, out float specular_mask)
//   {
//       float4 m = sample(specular_mask_texture,
//                         texcoord*specular_mask_texture_xform.xy + specular_mask_texture_xform.zw);
//       specular_mask = m.a;
//   }

fn calc_specular_mask_texture_ps(
    texcoord: vec2<f32>,
    in_specular_mask: f32,
    specular_mask: ptr<function, f32>,
) {
    let _u_in = in_specular_mask;
    let xformed = transform_texcoord(texcoord, material.specular_mask_texture_xform);
    *specular_mask = textureSample(specular_mask_texture, specular_mask_texture_sampler, xformed).a;
}

fn calc_specular_mask(texcoord: vec2<f32>, in_specular_mask: f32, specular_mask: ptr<function, f32>) {
    calc_specular_mask_texture_ps(texcoord, in_specular_mask, specular_mask);
}
