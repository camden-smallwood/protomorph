// Port of `bump_mapping_fx.hlsl::calc_bumpmap_detail_plus_detail_masked_ps`
// (line 127-144). Standard bump + two detail layers: an unconditional
// `detail` plus a masked `detail_masked` controlled by the mask map's R
// channel. No `invert_mask` arm in this variant (compare to
// `detail_masked`).
//
// HLSL:
//   bump          = sample_bumpmap(bump_map, bump_map_sampler, xform(uv, bump_map_xform));
//   detail        = sample_bumpmap(bump_detail_map, bump_detail_map_sampler, xform(uv, bump_detail_map_xform));
//   detail_masked = sample_bumpmap(bump_detail_masked_map, bump_detail_masked_map_sampler, xform(uv, bump_detail_masked_map_xform));
//   mask          = sample2D(bump_detail_mask_map,          xform(uv, bump_detail_mask_map_xform)).r;
//   bump.xy += detail.xy        * bump_detail_coefficient;
//   bump.xy += detail_masked.xy * mask * bump_detail_masked_coefficient;
//   bump = normalize(bump);
//   bump_normal = normalize(mul(bump, tangent_frame));

fn calc_bumpmap_detail_plus_detail_masked_ps(
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
    let detail_masked = sample_bumpmap(bump_detail_masked_map, bump_detail_masked_map_sampler, transform_texcoord(texcoord, material.bump_detail_masked_map_xform));
    let mask = textureSample(bump_detail_mask_map, bump_detail_mask_map_sampler, transform_texcoord(texcoord, material.bump_detail_mask_map_xform)).r;

    let d1 = detail.xy * material.bump_detail_coefficient.x;
    let d2 = detail_masked.xy * mask * material.bump_detail_masked_coefficient.x;
    bump = vec3<f32>(bump.x + d1.x + d2.x, bump.y + d1.y + d2.y, bump.z);
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
    calc_bumpmap_detail_plus_detail_masked_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
