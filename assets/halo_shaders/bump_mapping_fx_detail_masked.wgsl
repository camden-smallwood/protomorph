// Port of `bump_mapping_fx.hlsl::calc_bumpmap_detail_masked_ps`
// (line 107-123). Standard detail bump plus a per-texel mask
// controlling how much detail bump is mixed in (optionally inverted
// via `invert_mask` rmop bool).
//
// HLSL:
//   bump   = sample_bumpmap(bump_map, bump_map_sampler, xform(uv, bump_map_xform));
//   detail = sample_bumpmap(bump_detail_map, bump_detail_map_sampler, xform(uv, bump_detail_map_xform));
//   mask   = sample2D(bump_detail_mask_map,  xform(uv, bump_detail_mask_map_xform)).r;
//   mask   = invert_mask ? 1.0 - mask : mask;
//   bump.xy += detail.xy * mask * bump_detail_coefficient;
//   bump_normal = normalize(mul(bump, tangent_frame));
//
// `invert_mask` resolves to a build-time bool — protomorph mirrors
// other rmop-bool patterns by reading the resolved value through the
// material cbuffer (`material.invert_mask.x != 0`). When the rmop
// chain doesn't declare it the param defaults to 0 (no invert).

fn calc_bumpmap_detail_masked_ps(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    let _u_view = fragment_to_camera_world;
    var bump = sample_bumpmap(bump_map, bump_map_sampler, transform_texcoord(texcoord, material.bump_map_xform));
    let detail = sample_bumpmap(bump_detail_map, bump_detail_map_sampler, transform_texcoord(texcoord, material.bump_detail_map_xform));
    var mask = textureSample(bump_detail_mask_map, bump_detail_mask_map_sampler, transform_texcoord(texcoord, material.bump_detail_mask_map_xform)).r;
    if (material.invert_mask.x != 0.0) {
        mask = 1.0 - mask;
    }
    let masked = detail.xy * mask * material.bump_detail_coefficient.x;
    bump = vec3<f32>(bump.x + masked.x, bump.y + masked.y, bump.z);
    bump = normalize(bump);
    *bump_normal = normalize(tangent_frame_t * bump.x + tangent_frame_b * bump.y + tangent_frame_n * bump.z);
}

fn calc_bumpmap(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    calc_bumpmap_detail_masked_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
