// Engine `decal_fx.hlsl::sample_diffuse` — albedo=palettized branch
// (line 203-207):
//   float index = sampleBiasGlobal2D(base_map, texcoord).x;
//   return sample2D(palette, float2(index, palette_v));
//
// Dependent texture fetch: base_map.r drives the U coordinate into the
// `palette` texture; `palette_v` (from the entry's PS call) is the V.
// The palette can be any size — engine comment warns about filtering
// artifacts (palette either smooth or unfiltered). We follow the
// engine's `sample2D` which has filtering on; the rmop authors
// downsample/tweak palettes appropriately.

fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    let _u_tile = texcoord_tile;
    let index = textureSample(base_map, base_map_sampler, texcoord).x;
    return textureSample(palette, palette_sampler, vec2<f32>(index, palette_v));
}
