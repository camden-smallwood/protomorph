// Edge-fade category — `simple` option. Port of overlays_fx.hlsl:71-75:
//
//   float fade_alpha = pow(abs(view_dot_normal), edge_fade_power);
//   return color * lerp(edge_fade_edge_tint, edge_fade_center_tint, fade_alpha);
//
// `view_dot_normal` is dot(view_dir, surface_normal): ~1 face-on (center),
// ~0 at the silhouette (edge). So fade_alpha→1 at the center tint and →0
// at the edge tint. Softens the sphere's silhouette (the hologram reads
// less solid at grazing angles). `mix(a,b,t)` == HLSL `lerp(a,b,t)`.
fn calc_edge_fade_ps(color: vec3<f32>, view_dot_normal: f32) -> vec3<f32> {
    let fade_alpha = pow(abs(view_dot_normal), material.edge_fade_power.x);
    let tint = mix(
        material.edge_fade_edge_tint.rgb,
        material.edge_fade_center_tint.rgb,
        fade_alpha,
    );
    return color * tint;
}
