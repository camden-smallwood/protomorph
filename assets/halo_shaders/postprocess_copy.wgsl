// `copy_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/copy.hlsl`.
//
// Single-tap copy with `ps_postprocess_scale.xy` texcoord scaling.
// Engine body is one line: `sample2D(source_sampler, texcoord * scale.xy)`.
// Most callers pass `scale = (1, 1)` (raw passthrough) but the scale
// parameter is honored so non-1 callers (e.g. selective region copies)
// work engine-faithfully. Used by `c_screen_postprocess::copy` for
// per-frame surface-to-surface blits.

struct PostprocessParams {
    pixel_size: vec4<f32>,
    scale: vec4<f32>,
    intensity_vector: vec4<f32>,
    dark_color_multiplier: vec4<f32>,
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;
@group(0) @binding(2) var<uniform> u: PostprocessParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0,  1.0), vec2<f32>( 3.0,  1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
    );
    var out: VsOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    out.texcoord = uvs[idx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.texcoord * u.scale.xy);
}
