// Port of `bump_mapping_fx.hlsl` — `off` variant.
//
// HLSL:
//   void calc_bumpmap_off_ps(
//       in float2 texcoord, in float3 fragment_to_camera_world,
//       in float3x3 tangent_frame, out float3 bump_normal)
//   { bump_normal = tangent_frame[2]; }
//
// `mat3x3<f32>` doesn't pass cleanly across function boundaries here,
// so the tangent_frame rows (T, B, N) come in as three vec3 args.
// This matches naga's calling-convention quirks; HLSL's `mul(v, M)`
// becomes `T·v.x + B·v.y + N·v.z` at the call sites that need it.

fn calc_bumpmap_off_ps(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    let _u_uv = texcoord;
    let _u_view = fragment_to_camera_world;
    let _u_t = tangent_frame_t;
    let _u_b = tangent_frame_b;
    *bump_normal = tangent_frame_n;
}

fn calc_bumpmap(
    texcoord: vec2<f32>,
    fragment_to_camera_world: vec3<f32>,
    tangent_frame_t: vec3<f32>,
    tangent_frame_b: vec3<f32>,
    tangent_frame_n: vec3<f32>,
    bump_normal: ptr<function, vec3<f32>>,
) {
    calc_bumpmap_off_ps(texcoord, fragment_to_camera_world, tangent_frame_t, tangent_frame_b, tangent_frame_n, bump_normal);
}
