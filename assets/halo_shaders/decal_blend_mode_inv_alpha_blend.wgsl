// Engine `decal_fx.hlsl::fade_out` — blend_mode=inv_alpha_blend branch
// (line 384-421). Behavior matches alpha_blend in fade_out:
//   - NOT additive/multiply/double_multiply → no .xyz scale by fade.
//   - NOT pre_multiplied_alpha → skip the premultiply block.
//   - BLEND_MODE_USES_SRC_ALPHA = true: the macro (HLSL line 141-148)
//     EXCLUDES only opaque/additive/multiply/double_multiply/maximum/
//     multiply_add, so inv_alpha_blend → true (D3 fix: this const was
//     wrongly `false`). With it true the alpha clause always fires →
//     `color.w *= fade` (the prior behavior), so this is a label-only fix.
//
// Pipeline blend state: src=OneMinusSrcAlpha, dst=SrcAlpha (the
// "inverse alpha" variant per `reference_halo_blend_modes.md`). Engine
// uses this for decals that should bias toward the destination color
// (e.g., shadow stains that darken less the brighter the underlying).
//
// `BLEND_MODE_SELF_ILLUM` for inv_alpha_blend = false (HLSL line 67
// only flags additive / add_src_times_srcalpha).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    var a = color.w;
    if (!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA) {
        a = a * decal_constants.fade;
    }
    return vec4<f32>(color.xyz, a);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = true;
