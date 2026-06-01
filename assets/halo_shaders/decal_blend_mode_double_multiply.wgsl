// Engine `decal_fx.hlsl::fade_out` — blend_mode=double_multiply branch
// (line 394-397):
//   color.xyz = lerp(0.5, color.xyz, fade.x);
// (fade=1 → use color, fade=0 → 0.5 which acts as "neutral" for
// double-multiply's 2*src*dst blend formula).
//
// BLEND_MODE_USES_SRC_ALPHA = false for double_multiply → no alpha scale.
//
// Pipeline blend state: src=Dst, dst=Src (Halo `_alpha_blend_double_multiply`).
// LDR_gamma2 / HDR_gamma2 forced to false (line 136-139).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        mix(vec3<f32>(0.5, 0.5, 0.5), color.xyz, decal_constants.fade),
        color.w,
    );
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
