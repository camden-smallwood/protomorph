// Overlay category — `additive_detail` option.
// Port of overlays_fx.hlsl:26-34 (calc_overlay_additive_detail_ps):
//
//   overlay        = sample2D(overlay_map, xform)
//   overlay_detail = sample2D(overlay_detail_map, xform)
//   overlay_color  = overlay.rgb * overlay_detail.rgb * DETAIL_MULTIPLIER
//                    * overlay_tint.rgb * overlay_intensity
//   return color + overlay_color
//
// `DETAIL_MULTIPLIER = 4.59479` (albedo_fx.hlsl:1).
const OVERLAY_DETAIL_MULTIPLIER: f32 = 4.594790;

fn calc_overlay_ps(color: vec3<f32>, texcoord: vec2<f32>) -> vec3<f32> {
    let overlay = textureSample(
        overlay_map, overlay_map_sampler,
        transform_texcoord(texcoord, material.overlay_map_xform),
    ).rgb;
    let detail = textureSample(
        overlay_detail_map, overlay_detail_map_sampler,
        transform_texcoord(texcoord, material.overlay_detail_map_xform),
    ).rgb;
    let overlay_color = overlay * detail * OVERLAY_DETAIL_MULTIPLIER
        * material.overlay_tint.rgb * material.overlay_intensity.x;

    return color + overlay_color;
}
