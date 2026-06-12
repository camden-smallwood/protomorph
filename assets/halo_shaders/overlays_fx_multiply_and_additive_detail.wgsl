// Overlay category — `multiply_and_additive_detail` option.
// Port of overlays_fx.hlsl:48-54 (multiply_and_additive_detail) which
// chains :26-34 (additive_detail):
//
//   calc_overlay_multiply_and_additive_detail_ps(color, texcoord):
//     overlay_color = sample(overlay_multiply_map, ...).rgb
//     return calc_overlay_additive_detail_ps(color * overlay_color, texcoord)
//
//   calc_overlay_additive_detail_ps(color, texcoord):
//     overlay        = sample(overlay_map, ...)
//     overlay_detail = sample(overlay_detail_map, ...)
//     overlay_color  = overlay.rgb * overlay_detail.rgb * DETAIL_MULTIPLIER
//                      * overlay_tint.rgb * overlay_intensity
//     return color + overlay_color
//
// i.e. multiply the lit color by the multiply map, then ADD the
// (overlay × detail) layer — the guardian hologram's cell-like inner
// surface. `DETAIL_MULTIPLIER = 4.59479` (albedo_fx.hlsl:1).
const OVERLAY_DETAIL_MULTIPLIER: f32 = 4.594790;

fn calc_overlay_ps(color: vec3<f32>, texcoord: vec2<f32>) -> vec3<f32> {
    let multiply = textureSample(
        overlay_multiply_map, overlay_multiply_map_sampler,
        transform_texcoord(texcoord, material.overlay_multiply_map_xform),
    ).rgb;
    let base = color * multiply;

    let overlay = textureSample(
        overlay_map, overlay_map_sampler,
        transform_texcoord(texcoord, material.overlay_map_xform),
    ).rgb;
    let detail = textureSample(
        overlay_detail_map, overlay_detail_map_sampler,
        transform_texcoord(texcoord, material.overlay_detail_map_xform),
    ).rgb;
    let add = overlay * detail * OVERLAY_DETAIL_MULTIPLIER
        * material.overlay_tint.rgb * material.overlay_intensity.x;

    return base + add;
}
