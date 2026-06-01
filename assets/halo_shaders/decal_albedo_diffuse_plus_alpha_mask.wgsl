// Engine `decal_fx.hlsl::sample_diffuse` — albedo=diffuse_plus_alpha_mask
// branch (line 196-199):
//   return float4(
//       sampleBiasGlobal2D(base_map, texcoord).xyz,
//       sampleBiasGlobal2D(alpha_map, texcoord_tile).w);
//
// Variant of `diffuse_plus_alpha`: alpha is read from the un-tiled
// (`texcoord_tile`) coord so a single alpha tile masks the whole
// sprite atlas even when the decal repeats. RGB still comes from the
// post-sprite `texcoord`.

fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    let _u_palette = palette_v;
    let rgb = textureSample(base_map, base_map_sampler, texcoord).xyz;
    let alpha = textureSample(alpha_map, alpha_map_sampler, texcoord_tile).w;
    return vec4<f32>(rgb, alpha);
}
