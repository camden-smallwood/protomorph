// Overlay category — `none` option (overlays_fx.hlsl:5-8). No-op so the
// entry point can always call `calc_overlay_ps` (mirrors the HLSL
// APPLY_OVERLAYS macro, which is unconditionally present).
fn calc_overlay_ps(color: vec3<f32>, texcoord: vec2<f32>) -> vec3<f32> {
    return color;
}
