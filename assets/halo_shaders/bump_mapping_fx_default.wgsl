// Port of `bump_mapping_fx.hlsl` — `default` variant (line 57).
//
// HLSL:
//   void calc_bumpmap_default_ps(
//       in float2 texcoord, in float3 fragment_to_camera_world,
//       in float3x3 tangent_frame, out float3 bump_normal)
//   {
//       float3 bump = sample_bumpmap(bump_map, bump_map_sampler, transform_texcoord(texcoord, bump_map_xform));
//       bump_normal = normalize( mul(bump, tangent_frame) );
//   }

fn calc_bumpmap_default_ps(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    let _u_view = fragment_to_camera_world;
    let bump = sample_bumpmap(bump_map, bump_map_sampler, transform_texcoord(texcoord, material.bump_map_xform));
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
    calc_bumpmap_default_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
