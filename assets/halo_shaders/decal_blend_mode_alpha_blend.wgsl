// Engine `decal_fx.hlsl::fade_out` — blend_mode=alpha_blend branch
// (line 384-421). Hit list when blend_mode=alpha_blend:
//   - NOT additive/multiply/double_multiply → skip the .xyz scale.
//   - BLEND_MODE_USES_SRC_ALPHA = true → color.w *= fade.
//   - NOT pre_multiplied_alpha → skip the premultiply branch.
//
// `BLEND_MODE_SELF_ILLUM` for alpha_blend = false → no self-illum
// scaling needed in `tint_and_modulate`.
//
// Pipeline blend state: src=SrcAlpha, dst=OneMinusSrcAlpha (Halo
// `_alpha_blend_alpha_blend` per `reference_halo_blend_modes.md`).

fn fade_out(color: vec4<f32>) -> vec4<f32> {
    // Engine sources fade from `set_shader_constant(0x140000, fade)` —
    // a per-decal scalar from `lifespan_distance_fade * distance_fade`.
    // The orchestrator writes it into `decal_constants.fade` per draw.
    return vec4<f32>(color.xyz, color.w * decal_constants.fade);
}

const BLEND_MODE_SELF_ILLUM: bool = false;
const BLEND_MODE_USES_SRC_ALPHA: bool = true;
