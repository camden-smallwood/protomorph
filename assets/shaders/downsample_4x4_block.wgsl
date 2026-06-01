// `downsample_4x4_block_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/downsample_4x4_block.hlsl`.
//
// Plain 4x4 box filter via 4 bilinear-trick samples at ±1 source
// pixel offsets. No bloom-curve threshold — used by
// `c_screen_postprocess::downsample_generate` (shader 49) to chain
// the multi-scale bloom pyramid (1/2 → 1/4 → 1/8).

struct PostprocessParams {
    pixel_size: vec4<f32>, // (1/w, 1/h, w, h) of the source surface
    scale: vec4<f32>,
    intensity_vector: vec4<f32>,
    dark_color_multiplier: vec4<f32>,
}

@group(0) @binding(0) var source_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> u: PostprocessParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
    );
    var out: VsOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    out.texcoord = uvs[idx];
    return out;
}

fn tex2D_offset(uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(source_sampler_tex, source_sampler,
                         uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color: vec4<f32> = vec4<f32>(0.0);
    color = color + tex2D_offset(in.texcoord, -1.0, -1.0);
    color = color + tex2D_offset(in.texcoord,  1.0, -1.0);
    color = color + tex2D_offset(in.texcoord, -1.0,  1.0);
    color = color + tex2D_offset(in.texcoord,  1.0,  1.0);
    return color / 4.0;
}
