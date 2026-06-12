// Overlay category — `multiply` option.
// Port of overlays_fx.hlsl:37-43 (calc_overlay_multiply_ps):
//
//   overlay       = sample2D(overlay_map, xform)
//   overlay_color = overlay.rgb * overlay_tint.rgb * overlay_intensity
//   return color * overlay_color
//
// Modulates the lit color by a tinted/scaled overlay map.
fn calc_overlay_ps(color: vec3<f32>, texcoord: vec2<f32>) -> vec3<f32> {
    let overlay = textureSample(
        overlay_map, overlay_map_sampler,
        transform_texcoord(texcoord, material.overlay_map_xform),
    ).rgb;
    let overlay_color = overlay * material.overlay_tint.rgb * material.overlay_intensity.x;

    return color * overlay_color;
}
