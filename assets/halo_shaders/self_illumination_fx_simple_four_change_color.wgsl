// ⚠️  RECONSTRUCTED — no HLSL source ships in either H3 MCC or Reach MCC.
//
// `calc_self_illumination_simple_four_change_color_ps` is referenced
// by rmt2 option tables (s3d_powerhouse and other 343i-ported MP maps
// select this option for some self-illuminated materials) but the
// matching pixel shader function is absent from:
//   - `/Users/camden/Halo/halo3_mcc_hlsl_extracted/self_illumination_fx.hlsl`
//     (stops at `calc_self_illumination_palette_ps`; no four-channel variant)
//   - `tags/shaders/self_illumination_fx.hlsl_include` (cached HLSL fragment)
//   - Reach MCC `tags/shaders/templated/self_illumination.hlsl_include`
//     (Reach adds `window_room`, `blend_box`, `change_color_detail` —
//     no `simple_four_change_color`)
//
// **TODO**: revisit when newer-game tags (Halo Infinite / Halo 4 / Halo 5)
// become available — they likely still ship the source for this variant
// since 343i would have authored it then. The 343i porting of MCC maps
// is known-incomplete for shader options of this kind.
//
// Algorithmic reconstruction extends `three_channel_ps` to 4 channels:
//   sample = textureSample(self_illum_map, xformed_tc)         // rgba
//   color  = sample.r * primary_change_color.rgb
//          + sample.g * secondary_change_color.rgb
//          + sample.b * tertiary_change_color.rgb
//          + sample.a * quaternary_change_color.rgb
//   return color * self_illum_intensity
//
// The "simple_" prefix in the option name suggests the math is otherwise
// `simple_ps`-shaped (no parallax, no plasma noise, no falloff curves);
// the "four_change_color" suffix mirrors the engine's
// k_object_change_color_count = 4 convention (Ares object_constants.h:19).
// `albedo` is `inout` in HLSL but never written to by the `simple_*`
// family — keep as a no-op out-ref.

fn calc_self_illumination_simple_four_change_color_ps(
    texcoord_in: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_albedo = albedo;
    let _u_view = view_dir;

    let xformed = transform_texcoord(texcoord_in, material.self_illum_map_xform);
    let sample = textureSample(self_illum_map, self_illum_map_sampler, xformed);

    let mixed = sample.r * material.primary_change_color.rgb
              + sample.g * material.secondary_change_color.rgb
              + sample.b * material.tertiary_change_color.rgb
              + sample.a * material.quaternary_change_color.rgb;

    return mixed * material.self_illum_intensity.x;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_simple_four_change_color_ps(texcoord, albedo, view_dir);
}
