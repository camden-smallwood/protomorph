// Edge-fade category — `none` option (overlays_fx.hlsl:62-65). No-op so
// the entry point can always call `calc_edge_fade_ps`.
fn calc_edge_fade_ps(color: vec3<f32>, view_dot_normal: f32) -> vec3<f32> {
    return color;
}
