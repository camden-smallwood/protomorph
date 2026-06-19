// Engine `decal_fx.hlsl::fade_out` — blend_mode=additive branch
// (line 386-389):
//   color.xyz *= fade;
//
// BLEND_MODE_USES_SRC_ALPHA = false for additive (line 141-148 — additive
// is in the excluded set). The HLSL alpha clause (line 399-402) still fades
// `color.w` when `!IS_FLAT_VERTEX || !specular_leave || USES_SRC_ALPHA`;
// decals have no specular category (always `leave`), so the guard is
// `(!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA)`. For a BUMPED additive
// decal that fades RT1.w (the bump-normal alpha) with the decal.
//
// BLEND_MODE_SELF_ILLUM = true (line 67). Only consumed by
// `tint_and_modulate` under `render_pass=post_lighting` (line 364-380);
// the pre_lighting path the current `decal_tinting_none` port emits
// already skips that branch.
//
// Pipeline blend state: src=One, dst=One (Halo `_alpha_blend_additive`,
// wired in `render_methods/pipeline_cache.rs::pick_blend_for_mode`).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    let rgb = color.xyz * decal_constants.fade;
    var a = color.w;
    if (!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA) {
        a = a * decal_constants.fade;
    }
    return vec4<f32>(rgb, a);
}

const BLEND_MODE_SELF_ILLUM: bool = true;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
