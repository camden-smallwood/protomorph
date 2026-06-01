// Engine `decal_fx.hlsl::fade_out` — blend_mode=inv_alpha_blend branch
// (line 384-421). Behavior matches alpha_blend in fade_out:
//   - NOT additive/multiply/double_multiply → no .xyz scale by fade.
//   - NOT pre_multiplied_alpha → skip the premultiply block.
//   - BLEND_MODE_USES_SRC_ALPHA macro (HLSL line 142-147) does NOT
//     list inv_alpha_blend, but the !IS_FLAT_VERTEX / !specular_leave
//     branches still fade alpha for most decal authoring — match the
//     alpha_blend behavior (`color.w *= fade`) since the GPU blend
//     stage consumes the inverted alpha.
//
// Pipeline blend state: src=OneMinusSrcAlpha, dst=SrcAlpha (the
// "inverse alpha" variant per `reference_halo_blend_modes.md`). Engine
// uses this for decals that should bias toward the destination color
// (e.g., shadow stains that darken less the brighter the underlying).
//
// `BLEND_MODE_SELF_ILLUM` for inv_alpha_blend = false (HLSL line 67
// only flags additive / add_src_times_srcalpha).
// `BLEND_MODE_USES_SRC_ALPHA` reflects the macro = false (engine
// authoring still routes alpha through `* fade` via the !IS_FLAT
// branch, but our WGSL fragment exposes the macro value verbatim).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.xyz, color.w * decal_constants.fade);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = false;
