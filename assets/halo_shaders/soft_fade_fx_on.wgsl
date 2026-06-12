// soft_fade category — `on` option (apply_soft_fade_on).
// Port of common_fx.hlsl:69 apply_soft_fade_on.
//
//   val = 1
//   if use_soft_fresnel: val *= pow(calc_fresnel_dp(wnorm,wview), soft_fresnel_power)
//   if use_soft_z:       val *= get_softness(z_to_w(depth_map), linearDepth, soft_z_range)
//   alpha_blend ? albedo.w *= val : albedo.rgb *= val
//
// `calc_fresnel_dp` = SmoothStep(saturate(abs(dot(n,v))))  (common_fx.hlsl:50).
//
// The `use_soft_fresnel`/`use_soft_z` PARAM(bool)s are resolved per
// material at WGSL-assembly time (like `no_dynamic_lights`), so each
// branch below is emitted only when its bool is set — when neither is
// set, `apply_soft_fade` is a no-op (val stays 1).
//
// The `use_soft_z` depth-fade branch IS now emitted (see
// `pick_soft_fade` in render_methods/mod.rs): it point-loads the scene
// depth at the fragment's pixel from `scene_depth_tex` (camera_bgl
// binding 15, real only in the read-only-depth transparent pass) and
// fades by `saturate((scene_dist - frag_dist) * soft_z_range)`. Both
// distances come from `sf_linear_depth` (camera-projection
// reconstruction). `frag_device_z` = clip_position.z, `vpos` =
// clip_position.xy.
fn sf_calc_fresnel_dp(wnorm: vec3<f32>, wview: vec3<f32>) -> f32 {
    var ndotv = saturate(abs(dot(wnorm, wview)));
    ndotv = ndotv * ndotv * (3.0 - 2.0 * ndotv); // SmoothStep (common_fx.hlsl:45)
    return ndotv;
}

// Linear view-space distance (positive, increasing with depth) from a
// device-depth value. The engine's `z_to_w` (common_fx.hlsl:23) bakes
// fixed projection constants (zn=10240, zf=0.0078125) because its depth
// buffer was rendered with that projection; protomorph uses a reverse-Z
// buffer with the live camera projection, so we reconstruct from
// `camera.projection` instead. Derivation (perspective, clip.w = -view_z,
// input w = 1):
//   device_z = (P[2][2]*vz + P[3][2]) / (-vz)
//   → dist = -vz = P[3][2] / (device_z + P[2][2])
// Both the scene sample and the fragment go through this same monotonic
// function, so `get_softness` (the saturated difference) behaves exactly
// as the engine's regardless of the absolute constants.
fn sf_linear_depth(device_z: f32) -> f32 {
    let p = camera.projection;
    return p[3][2] / (device_z + p[2][2]);
}

fn apply_soft_fade(
    albedo: ptr<function, vec4<f32>>,
    wnorm: vec3<f32>,
    wview: vec3<f32>,
    frag_device_z: f32,
    vpos: vec2<f32>,
) {
    var val = 1.0;
__SOFT_FADE_FRESNEL__
__SOFT_FADE_SOFTZ__
__SOFT_FADE_APPLY__
}
