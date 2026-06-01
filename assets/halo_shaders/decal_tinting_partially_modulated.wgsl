// Engine `decal_fx.hlsl::tint_and_modulate` — tinting=partially_modulated
// branch. With partially_modulated:
//   tint_color_internal       = tint_color        (PARAM, rmd cbuffer)
//   intensity_internal        = intensity         (PARAM, rmd cbuffer)
//   modulation_factor_internal= modulation_factor (PARAM, rmd cbuffer)
//
// Common math (HLSL line 358-360):
//   recip_sqrt_3 = 1 / sqrt(3)
//   Y            = recip_sqrt_3 * length(diffuse.xyz)
//   diffuse.xyz *= lerp(tint_color.xyz, 1.0, modulation_factor * Y) * intensity
//
// Differs from `fully_modulated` only in that `modulation_factor`
// comes from a PARAM instead of being baked to 1.0 — the lerp slope
// is author-controlled, allowing partial tint regardless of pixel
// brightness.
//
// Then the post_lighting / blend-mode block (HLSL line 364-380) runs
// identically to the other tinting branches.
//
// Build-time substitutions (assembled per-variant by render_methods):
//   __DECAL_POST_LIGHTING__       — `true` iff render_pass=post_lighting
//   __DECAL_MULTIPLICATIVE_BLEND__ — `true` iff blend_mode∈{multiply,double_multiply}

fn tint_and_modulate(diffuse: vec4<f32>) -> vec4<f32> {
    const DECAL_POST_LIGHTING: bool = __DECAL_POST_LIGHTING__;
    const DECAL_MULTIPLICATIVE_BLEND: bool = __DECAL_MULTIPLICATIVE_BLEND__;

    let recip_sqrt_3: f32 = 0.57735026;
    let y: f32 = recip_sqrt_3 * length(diffuse.xyz);

    // rmt2 packs scalar PARAMs as vec4 — `.x` for modulation_factor + intensity.
    let mod_factor = material.modulation_factor.x;
    let tint_rgb = mix(material.tint_color.xyz, vec3<f32>(1.0), mod_factor * y);
    var tinted = vec4<f32>(diffuse.xyz * tint_rgb * material.intensity.x, diffuse.w);

    if (DECAL_POST_LIGHTING && !DECAL_MULTIPLICATIVE_BLEND) {
        if (BLEND_MODE_SELF_ILLUM) {
            tinted = vec4<f32>(tinted.xyz * engine_exposure.g_alt_exposure.g, tinted.w);
        } else {
            tinted = vec4<f32>(tinted.xyz * engine_exposure.g_exposure.x, tinted.w);
        }
    }
    return tinted;
}
