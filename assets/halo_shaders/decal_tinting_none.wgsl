// Engine `decal_fx.hlsl::tint_and_modulate` — tinting=none branch
// (line 340-342). With tinting=none, the tint × intensity body is a
// no-op. But the `render_pass=post_lighting` block (HLSL line 364-380)
// runs regardless of the tinting branch — post-SL decals get an
// exposure scale (skipped for multiply / double_multiply blend modes).
// Build-time substitutions (assembled per-variant):
//   __DECAL_POST_LIGHTING__       — `true` iff render_pass=post_lighting
//   __DECAL_MULTIPLICATIVE_BLEND__ — `true` iff blend_mode∈{multiply,double_multiply}
// `BLEND_MODE_SELF_ILLUM` is defined by the picked `decal_blend_mode_*.wgsl`
// fragment (assembled before this file).

fn tint_and_modulate(diffuse: vec4<f32>) -> vec4<f32> {
    const DECAL_POST_LIGHTING: bool = __DECAL_POST_LIGHTING__;
    const DECAL_MULTIPLICATIVE_BLEND: bool = __DECAL_MULTIPLICATIVE_BLEND__;

    if (DECAL_POST_LIGHTING && !DECAL_MULTIPLICATIVE_BLEND) {
        if (BLEND_MODE_SELF_ILLUM) {
            // HLSL line 374: `diffuse.xyz *= ILLUM_EXPOSURE` (=g_alt_exposure.g).
            return vec4<f32>(diffuse.xyz * engine_exposure.g_alt_exposure.g, diffuse.w);
        } else {
            // HLSL line 378: `diffuse.xyz *= g_exposure.x`.
            return vec4<f32>(diffuse.xyz * engine_exposure.g_exposure.x, diffuse.w);
        }
    }
    return diffuse;
}
