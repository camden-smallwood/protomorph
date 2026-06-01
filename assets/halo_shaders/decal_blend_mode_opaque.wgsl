// Engine `decal_fx.hlsl::fade_out` — blend_mode=opaque branch
// (line 384-421). Behavior:
//   - NOT additive/multiply/double_multiply → no .xyz scale by fade.
//   - BLEND_MODE_USES_SRC_ALPHA = true (HLSL line 142 includes opaque)
//     → `color.w *= fade` runs.
//   - NOT pre_multiplied_alpha → skip the premultiply block.
//
// `BLEND_MODE_SELF_ILLUM` for opaque = false (only additive /
// add_src_times_srcalpha set self-illum per HLSL line 67).
//
// Pipeline blend state: src=One, dst=Zero (replace). Alpha output is
// computed but the GPU blend stage discards it.

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.xyz, color.w * decal_constants.fade);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = true;
