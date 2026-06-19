// Engine `decal_fx.hlsl::sample_diffuse` — albedo=diffuse_only branch
// (line 184-188):
//   return sampleBiasGlobal2D(base_map, texcoord);
//
// Engine convention: PS samples base_map at the post-sprite texcoord
// (the .zw of m_texcoord). The `texcoord_tile` arg matters only for
// *_mask variants that sample alpha at the un-tiled texcoord.
//
// ### Why forced LOD 0 (an ARCHITECTURAL deviation, not a math gap) ###
//
// The engine samples `sampleBiasGlobal2D(base_map, texcoord)` (auto-LOD +
// a global LOD bias). protomorph forces LOD 0 instead. This is NOT a
// shader-faithfulness divergence — it's a workaround for a protomorph
// decal-MESH artifact:
//
// BFS-mesh decal triangles emit vertices at BSP polygon corners that can
// lie FAR outside the [0,1] projection footprint (UV ranges of [-3.9, 2.3]
// observed in riverworld). UV derivatives are CONSTANT per triangle (linear
// interpolation), so even the in-range fragments (the rest are clipped at
// the top of `fs_main`) see the inflated `span/screen` gradient → the
// sampler picks a near-1×1 mip → FLAT BROWN/OLIVE decals instead of the
// authored detail. Floating-quad decals span exactly [0,1] (clean corner
// verts) so they don't hit this.
//
// NOTE for future work: synthesizing derivatives from `texcoord_tile`
// (`m_texcoord.xy`) does NOT help — it's the same `in.texcoord` as the
// sampled `m_texcoord.zw` up to an affine transform, so it carries the
// SAME inflated per-triangle derivative. The real fixes are upstream:
// clamp/cap UVs when generating BFS decal meshes, distinguish BFS vs
// floating-quad at draw time, or add a MaxLOD-clamped sampler (matching the
// engine's global bias). LOD 0 is the safe, visually-correct stopgap that
// works for both decal kinds (clean decals lose only distant minification).
fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    return textureSampleLevel(base_map, base_map_sampler, texcoord, 0.0);
}
