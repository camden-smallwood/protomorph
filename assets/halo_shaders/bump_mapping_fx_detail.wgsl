// Port of `bump_mapping_fx.hlsl` — `detail` variant (line 70).
//
// HLSL:
//   void calc_bumpmap_detail_ps(
//       in float2 texcoord, in float3 fragment_to_camera_world,
//       in float3x3 tangent_frame, out float3 bump_normal)
//   {
//       float3 bump   = sample_bumpmap(bump_map, bump_map_sampler, ...);
//       float3 detail = sample_bumpmap(bump_detail_map, bump_detail_map_sampler, ...);
//       bump.xy += detail.xy * bump_detail_coefficient;
//       bump = normalize(bump);
//       bump_normal = normalize( mul(bump, tangent_frame) );
//   }

fn calc_bumpmap_detail_ps(
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
    bump = vec3<f32>(
        bump.x + detail.x * material.bump_detail_coefficient.x,
        bump.y + detail.y * material.bump_detail_coefficient.x,
        bump.z,
    );
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
    calc_bumpmap_detail_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
