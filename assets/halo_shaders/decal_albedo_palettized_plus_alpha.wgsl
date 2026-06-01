// Engine `decal_fx.hlsl::sample_diffuse` — albedo=palettized_plus_alpha
// branch (decal_fx.hlsl:210-215):
//   float index = sampleBiasGlobal2D(base_map, texcoord).x;
//   float alpha = sampleBiasGlobal2D(alpha_map, texcoord).w;
//   return float4(sample2D(palette, float2(index, palette_v)).xyz, alpha);
//
// Same dependent palette fetch as `palettized`, but the alpha comes
// from the separate `alpha_map` (the original texture) instead of the
// palette's W channel. Distinct from `palettized_plus_alpha_mask`,
// which samples alpha at `texcoord_tile`.

fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    let _u_tile = texcoord_tile;
    let index = textureSample(base_map, base_map_sampler, texcoord).x;
    let alpha = textureSample(alpha_map, alpha_map_sampler, texcoord).w;
    let rgb = textureSample(palette, palette_sampler, vec2<f32>(index, palette_v)).xyz;
    return vec4<f32>(rgb, alpha);
}
