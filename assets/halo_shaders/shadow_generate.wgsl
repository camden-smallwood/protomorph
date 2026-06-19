// Halo 3 `shadow_generate.fx` (depth-only) port — outline.
//
// Engine flow per `c_rasterizer::setup_targets_shadow_generate @
// 0x180671080`: bind `_surface_shadow_1` as depth-stencil, all 4 color
// RTs `_surface_none`, clear depth=1.0, viewport (w,h), nullify PS.
// The per-object / per-light driver invokes this with a fresh ortho or
// perspective camera in light-space and re-rasterizes shadow-casting
// geometry through it.
//
// Compatible with `shared.model_bgl` (one ModelUniforms with dynamic
// offset) and `ModelVertex` layout (8 attributes; we only consume the
// position at @location(0)).
//
// WGSL doesn't have a "no PS" option; we emit an empty PS (`fs_main`)
// that simply allows depth-write — this is the opaque caster path
// (engine `shadow_generate_ps` with `alpha_test = off`, which sets
// `output_alpha = 1.0` and never clips).
//
// `fs_alpha_test` is the alpha-tested caster path — a verbatim port of
// engine `shadow_generate_ps` (`shadow_generate_fx.hlsl:52-60`) with
// `calc_alpha_test_on_ps` (`alpha_test_fx.hlsl:16-25`): sample
// `alpha_test_map.a` at `transform_texcoord(texcoord, alpha_test_map_xform)`
// and `clip(alpha - 0.5)` (`discard`). Drawn for opaque parts whose
// material has `alpha_test != off` (grates, chain-link, foliage props) so
// they cast holey shadows instead of a solid silhouette. The alpha-test
// resources are bound at @group(1) by the per-material bind group the
// renderer builds at material-load time.

struct ShadowGenerateUniforms {
    /// World-to-light-clip matrix. Built per-caster in CPU code from the
    /// caster's projection_bounds OBB (per-object) or the light's
    /// frustum/ortho (per-light).
    world_to_shadow: mat4x4<f32>,
    /// Object-to-world for this caster. Folded into the per-caster slot
    /// (dynamic offset) so shadows aren't tied to the render_list.
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow_u: ShadowGenerateUniforms;

struct VertexInput {
    // Only `position` is needed for depth — match `ModelVertex` layout
    // so the same vertex buffer can feed both the opaque pipeline and
    // this shadow pipeline.
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec2<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent:  vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    // Engine `shadow_generate_vs` passes `texcoord = vertex.texcoord`
    // (TEXCOORD1) for the alpha-test PS. The opaque `fs_main` ignores it.
    @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = shadow_u.model * vec4<f32>(input.position, 1.0);
    out.clip = shadow_u.world_to_shadow * world_pos;
    out.texcoord = input.texcoord;
    return out;
}

@fragment
fn fs_main() {
    // Opaque path. Depth is written by the rasterizer automatically; wgpu
    // requires an entry. Engine sets PS=NULL / `alpha_test = off`.
}

// ── Alpha-tested caster path (engine `shadow_generate_ps` + ──
// `calc_alpha_test_on_ps`). @group(1) holds the material's alpha-test map
// + sampler + the `alpha_test_map_xform` (.xy = scale, .zw = translate).
@group(1) @binding(0) var alpha_test_map: texture_2d<f32>;
@group(1) @binding(1) var alpha_test_sampler: sampler;
struct AlphaTestParams {
    /// `material.alpha_test_map_xform` (alpha_test_fx.hlsl:4).
    xform: vec4<f32>,
}
@group(1) @binding(2) var<uniform> alpha_test: AlphaTestParams;

@fragment
fn fs_alpha_test(in: VertexOutput) {
    // `transform_texcoord(texcoord, xform)` = `texcoord * xform.xy + xform.zw`
    // (utilities.fx). Sample `.a`, clip(alpha - 0.5) → discard below 0.5.
    let uv = in.texcoord * alpha_test.xform.xy + alpha_test.xform.zw;
    let alpha = textureSample(alpha_test_map, alpha_test_sampler, uv).a;
    if (alpha - 0.5 < 0.0) {
        discard;
    }
}
