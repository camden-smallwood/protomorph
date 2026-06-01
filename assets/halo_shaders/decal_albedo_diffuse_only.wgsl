// Engine `decal_fx.hlsl::sample_diffuse` — albedo=diffuse_only branch
// (line 184-188):
//   return sampleBiasGlobal2D(base_map, texcoord);
//
// Engine convention: PS samples base_map at the post-sprite texcoord
// (the .zw of m_texcoord). The `texcoord_tile` arg matters only for
// *_mask variants that sample alpha at the un-tiled texcoord.
//
// ### Why `textureSampleGrad` with synthesized derivatives ###
//
// BFS-mesh decal triangles emit vertices at BSP polygon corners that
// can lie FAR outside the [0,1] projection footprint (UV ranges of
// [-3.9, 2.3] observed in riverworld). The PS clip at the top of
// `entry_decal.wgsl::fs_main` discards out-of-range UVs, but the
// texture sampler computes LOD from screen-space UV derivatives
// (du/dx, dv/dy) BEFORE the discard — and those derivatives are
// inflated by the wide UV span across the rasterized triangle. The
// sampler then picks a very high mip (close to 1x1 = the bitmap's
// average color), producing FLAT BROWN/OLIVE BFS-mesh decals instead
// of the authored rock-detail art. Floating-quad decals don't hit
// this because their UVs span exactly [0,1] (clean corner verts) →
// reasonable derivatives.
//
// Engine `sampleBiasGlobal2D` (PC) likely sidesteps this via a
// MaxLOD-clamped sampler or a strong negative LOD bias; we don't have
// that infrastructure yet. As a near-engine-faithful workaround we
// synthesize derivatives from the texcoord_tile (= un-sprited UV in
// [0,1] for in-range fragments) so the sampler sees a derivative that
// reflects ACTUAL pixel-to-bitmap-texel coverage rather than the
// inflated BFS triangle span. fwidth() on the in-range coord gives a
// per-fragment derivative that's the same as what a clean
// floating-quad fragment would see.
fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    // TEST: force LOD 0 (full-resolution mip) — confirms whether the
    // BFS flat-brown output is caused by automatic-LOD picking a tiny
    // mip due to inflated screen-space derivatives. If this restores
    // detailed rock art on the BFS meshes, swap to a derivative-fix
    // strategy that preserves distance LOD for floating-quad decals.
    return textureSampleLevel(base_map, base_map_sampler, texcoord, 0.0);
}
