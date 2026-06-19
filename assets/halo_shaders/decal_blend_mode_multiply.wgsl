// Engine `decal_fx.hlsl::fade_out` — blend_mode=multiply branch
// (line 390-393):
//   color.xyz = lerp(1.0, color.xyz, fade.x);
// (i.e. fade=1 → use color, fade=0 → identity (white) so the multiply
// becomes a no-op against the destination).
//
// BLEND_MODE_USES_SRC_ALPHA = false for multiply. The HLSL alpha clause
// (line 399-402) still fades `color.w` when `!IS_FLAT_VERTEX` (specular is
// always `leave` for decals) — a BUMPED multiply decal fades RT1.w.
//
// Pipeline blend state: src=Dst, dst=Zero (Halo `_alpha_blend_multiply`).
// LDR_gamma2 / HDR_gamma2 forced to false for multiply (HLSL line
// 136-139: "Don't apply gamma twice").

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    let rgb = mix(vec3<f32>(1.0, 1.0, 1.0), color.xyz, decal_constants.fade);
    var a = color.w;
    if (!IS_FLAT_VERTEX || BLEND_MODE_USES_SRC_ALPHA) {
        a = a * decal_constants.fade;
    }
    return vec4<f32>(rgb, a);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
