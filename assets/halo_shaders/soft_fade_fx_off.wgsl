// soft_fade category — default `off` option (apply_soft_fade_off).
// Port of common_fx.hlsl:64 — a no-op. Always emitted so the entry
// point's unconditional `apply_soft_fade(...)` call resolves.
fn apply_soft_fade(
    albedo: ptr<function, vec4<f32>>,
    wnorm: vec3<f32>,
    wview: vec3<f32>,
    frag_device_z: f32,
    vpos: vec2<f32>,
) {
}
