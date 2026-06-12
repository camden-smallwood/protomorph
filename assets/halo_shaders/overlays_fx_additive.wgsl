// Overlay category — `additive` option.
// Port of overlays_fx.hlsl:16-19 (calc_overlay_additive_ps):
//
//   return color + (overlay_tint.rgb * overlay_intensity)
//                  * sample2D(overlay_map, xform).rgb
//
// A single additive overlay layer tinted/scaled by overlay_tint *
// overlay_intensity.
fn calc_overlay_ps(color: vec3<f32>, texcoord: vec2<f32>) -> vec3<f32> {
    let overlay = textureSample(
        overlay_map, overlay_map_sampler,
        transform_texcoord(texcoord, material.overlay_map_xform),
    ).rgb;

    return color + (material.overlay_tint.rgb * material.overlay_intensity.x) * overlay;
}
