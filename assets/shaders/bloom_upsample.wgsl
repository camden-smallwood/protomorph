// 3x3 tent filter for bloom upsampling
// Additive blending with destination is done via pipeline blend state (src: One, dst: One)

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;

struct UpsampleParams {
    filter_radius: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};
@group(0) @binding(2) var<uniform> params: UpsampleParams;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;
    let r = params.filter_radius;

    // 3x3 tent kernel (weights sum to 1.0)
    // 1 2 1
    // 2 4 2  / 16
    // 1 2 1
    let a = textureSample(t_source, s_source, uv + vec2<f32>(-r, -r)).rgb;
    let b = textureSample(t_source, s_source, uv + vec2<f32>( 0.0, -r)).rgb;
    let c = textureSample(t_source, s_source, uv + vec2<f32>( r, -r)).rgb;

    let d = textureSample(t_source, s_source, uv + vec2<f32>(-r,  0.0)).rgb;
    let e = textureSample(t_source, s_source, uv).rgb;
    let f = textureSample(t_source, s_source, uv + vec2<f32>( r,  0.0)).rgb;

    let g = textureSample(t_source, s_source, uv + vec2<f32>(-r,  r)).rgb;
    let h = textureSample(t_source, s_source, uv + vec2<f32>( 0.0,  r)).rgb;
    let i = textureSample(t_source, s_source, uv + vec2<f32>( r,  r)).rgb;

    let color = (a + c + g + i) * (1.0 / 16.0)
              + (b + d + f + h) * (2.0 / 16.0)
              + e * (4.0 / 16.0);

    return vec4<f32>(color, 1.0);
}
