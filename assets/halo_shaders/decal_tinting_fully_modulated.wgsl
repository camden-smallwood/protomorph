// Engine `decal_fx.hlsl::tint_and_modulate` — tinting=fully_modulated
// branch. With fully_modulated:
//   tint_color_internal       = tint_color   (PARAM, rmd cbuffer)
//   intensity_internal        = intensity    (PARAM, rmd cbuffer)
//   modulation_factor_internal= 1.0
//
// Common math (HLSL line 358-360):
//   recip_sqrt_3 = 1 / sqrt(3)
//   Y            = recip_sqrt_3 * length(diffuse.xyz)
//   diffuse.xyz *= lerp(tint_color.xyz, 1.0, Y) * intensity
//
// Then the post_lighting / blend-mode block (HLSL line 364-380) runs
// identically to the `none` branch — exposure scale on post-SL decals,
// skipped for multiply/double_multiply blends.
//
// Build-time substitutions (assembled per-variant by render_methods):
//   __DECAL_POST_LIGHTING__       — `true` iff render_pass=post_lighting
//   __DECAL_MULTIPLICATIVE_BLEND__ — `true` iff blend_mode∈{multiply,double_multiply}
// `BLEND_MODE_SELF_ILLUM` comes from the picked `decal_blend_mode_*.wgsl`.

fn tint_and_modulate(diffuse: vec4<f32>) -> vec4<f32> {
    const DECAL_POST_LIGHTING: bool = __DECAL_POST_LIGHTING__;
    const DECAL_MULTIPLICATIVE_BLEND: bool = __DECAL_MULTIPLICATIVE_BLEND__;

    // HLSL line 358: `recip_sqrt_3 * length(diffuse.xyz)`.
    let recip_sqrt_3: f32 = 0.57735026;
    let y: f32 = recip_sqrt_3 * length(diffuse.xyz);

    // HLSL line 359-360 for fully_modulated:
    //   modulation_factor_internal = 1.0 → lerp(tint, 1, Y).
    // rmt2 packs scalar PARAMs as vec4 for cbuffer alignment — read .x
    // for `intensity`. `tint_color` is RGBA so .xyz is correct.
    let tint_rgb = mix(material.tint_color.xyz, vec3<f32>(1.0), y);
    var tinted = vec4<f32>(diffuse.xyz * tint_rgb * material.intensity.x, diffuse.w);

    // HLSL line 364-380: render_pass=post_lighting exposure block.
    if (DECAL_POST_LIGHTING && !DECAL_MULTIPLICATIVE_BLEND) {
        if (BLEND_MODE_SELF_ILLUM) {
            // HLSL line 374: `*= ILLUM_EXPOSURE` (= g_alt_exposure.g).
            tinted = vec4<f32>(tinted.xyz * engine_exposure.g_alt_exposure.g, tinted.w);
        } else {
            // HLSL line 378: `*= g_exposure.x`.
            tinted = vec4<f32>(tinted.xyz * engine_exposure.g_exposure.x, tinted.w);
        }
    }
    return tinted;
}
