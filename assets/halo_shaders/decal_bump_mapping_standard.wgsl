// Engine `decal_fx.hlsl::sample_bump` — bump_mapping=standard branch
// (line 317-320):
//   calc_bumpmap_default_ps(texcoord, unused, tangent_frame, bump_normal);
// `calc_bumpmap_default_ps` (bump_mapping_fx.hlsl:63) samples
// `bump_map` at `transform_texcoord(texcoord, bump_map_xform)`, then
// transforms tangent-space bump by `tangent_frame` to get the
// world-space normal.

fn sample_bump(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, tangent: vec3<f32>, binormal: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let uv = transform_texcoord(texcoord, material.bump_map_xform);
    let bump_ts = textureSample(bump_map, bump_map_sampler, uv).xy;
    let bump_z = sqrt(max(0.0, 1.0 - dot(bump_ts, bump_ts)));
    let n_ts = vec3<f32>(bump_ts, bump_z);
    return normalize(n_ts.x * tangent + n_ts.y * binormal + n_ts.z * normal);
}

const IS_FLAT_VERTEX: bool = false;
