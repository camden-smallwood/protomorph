// Engine `decal_fx.hlsl::fade_out` — blend_mode=additive branch
// (line 386-389):
//   color.xyz *= fade;
//
// BLEND_MODE_USES_SRC_ALPHA = false for additive (line 141-148 — additive
// is in the excluded set) → no unconditional alpha scale here. The HLSL
// alpha guard at line 400-403 only fires when `!IS_FLAT_VERTEX ||
// !specular_leave`; the common bump_mapping=leave + specular=leave path
// skips it, and additive blending (`SRC=ONE, DST=ONE`) ignores the alpha
// channel anyway. We mirror the existing multiply / double_multiply
// ports and leave color.w untouched.
//
// BLEND_MODE_SELF_ILLUM = true (line 67). Only consumed by
// `tint_and_modulate` under `render_pass=post_lighting` (line 364-380);
// the pre_lighting path the current `decal_tinting_none` port emits
// already skips that branch.
//
// Pipeline blend state: src=One, dst=One (Halo `_alpha_blend_additive`,
// wired in `render_methods/pipeline_cache.rs::pick_blend_for_mode`).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.xyz * decal_constants.fade, color.w);
}

const BLEND_MODE_SELF_ILLUM: bool = true;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
