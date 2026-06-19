// Engine `decal_fx.hlsl::fade_out` — blend_mode=opaque branch
// (line 384-421). Behavior:
//   - NOT additive/multiply/double_multiply → no .xyz scale by fade.
//   - BLEND_MODE_USES_SRC_ALPHA = false: the macro (line 141-148) EXCLUDES
//     opaque, so `color.w *= fade` runs only via the `!IS_FLAT_VERTEX`
//     compile-time branch (D3 fix: this const was wrongly `true`). So a
//     FLAT opaque decal leaves alpha unfaded; a BUMPED one fades RT1.w.
//   - NOT pre_multiplied_alpha → skip the premultiply block.
//
// `BLEND_MODE_SELF_ILLUM` for opaque = false (only additive /
// add_src_times_srcalpha set self-illum per HLSL line 67).
//
// Pipeline blend state: src=One, dst=Zero (replace). The RT0 alpha output
// is discarded by the GPU blend stage; RT1.w drives the bump-normal blend.

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    var a = color.w;
    if (!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA) {
        a = a * decal_constants.fade;
    }
    return vec4<f32>(color.xyz, a);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
