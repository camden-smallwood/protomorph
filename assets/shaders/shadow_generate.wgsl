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
// WGSL doesn't have a "no PS" option; we emit an empty PS that simply
// allows depth-write. Per-shader alpha-test variants (foliage, decals)
// are deferred — they would `discard` from this PS after sampling
// their alpha-test map.

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
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = shadow_u.model * vec4<f32>(input.position, 1.0);
    out.clip = shadow_u.world_to_shadow * world_pos;
    return out;
}

@fragment
fn fs_main() {
    // Depth is written by the rasterizer automatically. wgpu requires
    // an entry; engine sets PS=NULL.
}
