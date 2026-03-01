// HDR + bloom composite with ACES tone mapping
// Reads the HDR lighting buffer and bloom texture, composites, tone maps, outputs to sRGB swapchain

@group(0) @binding(0) var t_base: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var s_filtering: sampler;

struct CompositeParams {
    bloom_strength: f32,
    exposure: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(3) var<uniform> params: CompositeParams;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// ACES Narkowicz approximation
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(t_base, s_filtering, in.tex_coords).rgb;
    let bloom = textureSample(t_bloom, s_filtering, in.tex_coords).rgb;

    let hdr = base + bloom * params.bloom_strength;
    let tonemapped = aces_tonemap(hdr * params.exposure);

    return vec4<f32>(tonemapped, 1.0);
}
