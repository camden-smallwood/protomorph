// Engine `decal_fx.hlsl::sample_diffuse` — albedo=diffuse_plus_alpha
// branch (decal_fx.hlsl:191-194):
//   return float4(
//       sampleBiasGlobal2D(base_map, texcoord).xyz,
//       sampleBiasGlobal2D(alpha_map, texcoord).w);
//
// Distinct from `diffuse_plus_alpha_mask` — here the alpha is read
// at the post-sprite `texcoord` (matching the RGB lookup), so the
// alpha tiles + animates with the sprite. The `_mask` variant samples
// alpha at `texcoord_tile` so a single alpha tile gates the entire
// atlas.

fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    let _u_tile = texcoord_tile;
    let _u_palette = palette_v;
    let rgb = textureSample(base_map, base_map_sampler, texcoord).xyz;
    let alpha = textureSample(alpha_map, alpha_map_sampler, texcoord).w;
    return vec4<f32>(rgb, alpha);
}
