// Engine `decal_fx.hlsl::tint_and_modulate` — tinting=unmodulated
// branch (decal_fx.hlsl:347-349). With unmodulated:
//   tint_color_internal       = tint_color    (PARAM, rmd cbuffer)
//   intensity_internal        = intensity     (PARAM, rmd cbuffer)
//   modulation_factor_internal= 0.0           (unchanged from default)
//
// Common math (HLSL line 358-360):
//   Y           = recip_sqrt_3 * length(diffuse.xyz)
//   diffuse.xyz*= lerp(tint_color.xyz, 1.0, 0 * Y) * intensity
//               = tint_color.xyz * intensity
//
// The luminance-driven lerp folds away because the modulation factor
// is zero — no need to compute `Y`. Then the post_lighting /
// blend-mode exposure block (HLSL line 364-380) runs identically to
// the `none`/`fully_modulated` branches.
//
// Build-time substitutions (assembled per-variant by render_methods):
//   __DECAL_POST_LIGHTING__       — `true` iff render_pass=post_lighting
//   __DECAL_MULTIPLICATIVE_BLEND__ — `true` iff blend_mode∈{multiply,double_multiply}
// `BLEND_MODE_SELF_ILLUM` comes from the picked `decal_blend_mode_*.wgsl`.

fn tint_and_modulate(diffuse: vec4<f32>) -> vec4<f32> {
    const DECAL_POST_LIGHTING: bool = __DECAL_POST_LIGHTING__;
    const DECAL_MULTIPLICATIVE_BLEND: bool = __DECAL_MULTIPLICATIVE_BLEND__;

    // HLSL line 359-360 collapses to: tint_color.xyz * intensity.
    var tinted = vec4<f32>(
        diffuse.xyz * material.tint_color.xyz * material.intensity.x,
        diffuse.w,
    );

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
