// Engine `decal_fx.hlsl::fade_out` — blend_mode=pre_multiplied_alpha
// branch (line 384-421). Behavior for blend_mode=pre_multiplied_alpha:
//   - NOT additive/multiply/double_multiply → skip the .xyz scale-by-fade.
//   - BLEND_MODE_USES_SRC_ALPHA = true: the macro (HLSL line 141-148)
//     excludes only opaque/additive/multiply/double_multiply/maximum/
//     multiply_add, so pre_multiplied_alpha → true (D3 fix: was wrongly
//     `false`). So the alpha clause `color.w *= fade` ALWAYS fires, BEFORE
//     the premultiply block — the premultiply reads the FADED alpha (D2).
//   - pre_multiplied_alpha block (line 405-420):
//       if albedo NOT vector_alpha / vector_alpha_drop_shadow:
//           color.xyz *= color.w;       // premultiply by the faded alpha
//       color.xyz *= fade.x;            // always
//
// vector_alpha already premultiplies inside `sample_diffuse`
// (`return float4(color * vector_alpha, vector_alpha)`); if it's
// composed with pre_multiplied_alpha we'd double-premultiply and
// blacken the decal. Detect via the `__DECAL_ALBEDO_IS_VECTOR_ALPHA__`
// build-time substitution emitted by the dispatcher (defaults to
// `false` for non-vector albedos).
//
// `BLEND_MODE_SELF_ILLUM` for pre_multiplied_alpha = false (only
// additive / add_src_times_srcalpha enable self-illum per HLSL line 67).
//
// Pipeline blend state: src=One, dst=OneMinusSrcAlpha (the canonical
// "premultiplied" blend per `reference_halo_blend_modes.md`).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    const ALBEDO_IS_VECTOR_ALPHA: bool = __DECAL_ALBEDO_IS_VECTOR_ALPHA__;
    // HLSL line 399-402 alpha clause — fires before the premultiply block;
    // USES_SRC_ALPHA=true → always. The premultiply below reads this faded w.
    var a = color.w;
    if (!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA) {
        a = a * decal_constants.fade;
    }
    var rgb = color.xyz;
    if (!ALBEDO_IS_VECTOR_ALPHA) {
        // HLSL line 417: premultiply RGB by the (faded) source alpha.
        rgb = rgb * a;
    }
    // HLSL line 419: fade scale always applied.
    rgb = rgb * decal_constants.fade;
    return vec4<f32>(rgb, a);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = true;
