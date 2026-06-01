// Engine `decal_fx.hlsl::sample_bump` — bump_mapping=standard_mask
// branch (decal_fx.hlsl:322-325):
//   calc_bumpmap_default_ps(texcoord_tile, unused, tangent_frame, bump_normal);
//
// Distinct from `standard` — samples `bump_map` at `texcoord_tile`
// (the un-tiled, single-sprite coord) so the bump tile masks the
// whole decal even when the diffuse repeats. `calc_bumpmap_default_ps`
// (`bump_mapping_fx.hlsl:63`) samples through
// `transform_texcoord(texcoord_tile, bump_map_xform)`, reads two
// channels (BC5 tangent-space xy), reconstructs z, then rotates into
// world space via the per-vertex tangent frame.

fn sample_bump(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, tangent: vec3<f32>, binormal: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let _u_tex = texcoord;
    let uv = transform_texcoord(texcoord_tile, material.bump_map_xform);
    let bump_ts = textureSample(bump_map, bump_map_sampler, uv).xy;
    let bump_z = sqrt(max(0.0, 1.0 - dot(bump_ts, bump_ts)));
    let n_ts = vec3<f32>(bump_ts, bump_z);
    return normalize(n_ts.x * tangent + n_ts.y * binormal + n_ts.z * normal);
}

const IS_FLAT_VERTEX: bool = false;
