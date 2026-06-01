// Port of `bump_mapping_fx.hlsl` — `detail_unorm` variant.
//
// HLSL:
//   void calc_bumpmap_detail_unorm_ps(
//       in float2 texcoord, in float3 fragment_to_camera_world,
//       in float3x3 tangent_frame, out float3 bump_normal)
//   {
//       float3 bump = sample_bumpmap(bump_map, bump_map_sampler, transform_texcoord(texcoord, bump_map_xform));
//       bump = bump * 2.0 - 1.0;
//       float3 detail = sample_bumpmap(bump_detail_map, bump_detail_map_sampler, transform_texcoord(texcoord, bump_detail_map_xform));
//       bump.xy += detail.xy * bump_detail_coefficient;
//       bump = normalize(bump);
//       bump_normal = normalize(mul(bump, tangent_frame));
//   }
//
// Differs from `detail` in that `bump` gets `*2-1` applied first
// (the HLSL `_unorm` suffix indicates the bump_map is sampled as
// UNORM and decoded inline rather than via SNORM bitm view).

fn calc_bumpmap_detail_unorm_ps(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    let _u_view = fragment_to_camera_world;
    // Engine HLSL `_unorm` variant does `bump = bump*2-1` because it
    // assumes the bump_map is bound as UNORM (DX9 path). wgpu binds
    // every MCC normal map as `Bc5RgSnorm` (per `feedback_mcc_bc5_is_snorm
    // .md`), so `sample_bumpmap` already returns signed [-1,1] data.
    // Re-applying `*2-1` would shift the bump to [-3, 1]. Skip the
    // decode; the SNORM data is the engine post-decode value.
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
    calc_bumpmap_detail_unorm_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
