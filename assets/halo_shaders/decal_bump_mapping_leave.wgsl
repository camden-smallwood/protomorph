// Engine `decal_fx.hlsl::sample_bump` — bump_mapping=leave branch
// (line 312-315):
//   calc_bumpmap_off_ps(texcoord, unused, tangent_frame, bump_normal);
// `calc_bumpmap_off_ps` returns the geometric tangent-frame Z (or
// zero for IS_FLAT_VERTEX). Since bump_mapping=leave implies
// IS_FLAT_VERTEX (the rmdf entry-point's `category dependencies`
// gates on bump_mapping), there's no TBN to return — engine yields
// `bump_normal = 0`.
//
// IS_FLAT_VERTEX (line 64): bump_mapping=leave is the flat-vertex
// gate; VS strips the TBN interpolators when this is set.
//
// Signature mirrors the bumped variants (5 args) so `entry_decal::fs_main`
// can call uniformly regardless of which fragment got concatenated.

fn sample_bump(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, tangent: vec3<f32>, binormal: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let _tile = texcoord_tile;
    let _uv = texcoord;
    let _t = tangent;
    let _b = binormal;
    let _n = normal;
    return vec3<f32>(0.0, 0.0, 0.0);
}

const IS_FLAT_VERTEX: bool = true;
