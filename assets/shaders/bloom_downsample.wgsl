// 13-tap downsample filter (Jimenez, SIGGRAPH 2014)
// Used for bloom prefilter (with soft-knee threshold) and subsequent downsample passes.

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;

struct BloomParams {
    threshold: f32,
    knee: f32,
    texel_size: vec2<f32>,
};
@group(0) @binding(2) var<uniform> params: BloomParams;

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

// 13-tap downsample: samples a 6x6 area with 13 bilinear taps
// Prevents firefly artifacts by distributing weight across a wide area
fn downsample_13tap(uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let x = texel_size.x;
    let y = texel_size.y;

    // Center cross (4 samples at +-1 texel)
    let a = textureSample(t_source, s_source, uv + vec2<f32>(-2.0*x, -2.0*y)).rgb;
    let b = textureSample(t_source, s_source, uv + vec2<f32>( 0.0,   -2.0*y)).rgb;
    let c = textureSample(t_source, s_source, uv + vec2<f32>( 2.0*x, -2.0*y)).rgb;

    let d = textureSample(t_source, s_source, uv + vec2<f32>(-2.0*x,  0.0)).rgb;
    let e = textureSample(t_source, s_source, uv).rgb;
    let f = textureSample(t_source, s_source, uv + vec2<f32>( 2.0*x,  0.0)).rgb;

    let g = textureSample(t_source, s_source, uv + vec2<f32>(-2.0*x,  2.0*y)).rgb;
    let h = textureSample(t_source, s_source, uv + vec2<f32>( 0.0,    2.0*y)).rgb;
    let i = textureSample(t_source, s_source, uv + vec2<f32>( 2.0*x,  2.0*y)).rgb;

    let j = textureSample(t_source, s_source, uv + vec2<f32>(-x, -y)).rgb;
    let k = textureSample(t_source, s_source, uv + vec2<f32>( x, -y)).rgb;
    let l = textureSample(t_source, s_source, uv + vec2<f32>(-x,  y)).rgb;
    let m = textureSample(t_source, s_source, uv + vec2<f32>( x,  y)).rgb;

    // Weighted combination (weights sum to 1.0)
    var color = e * 0.125;                          // center: 1/8
    color += (a + c + g + i) * 0.03125;             // corners: 4 * 1/32
    color += (b + d + f + h) * 0.0625;              // edges: 4 * 1/16
    color += (j + k + l + m) * 0.125;               // inner: 4 * 1/8

    return color;
}

// Soft-knee threshold function
fn soft_threshold(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let brightness = max(max(color.r, color.g), color.b);
    let soft = brightness - (threshold - knee);
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    let contribution = soft_clamped * soft_clamped / (4.0 * knee + 0.00001);
    let factor = max(contribution, brightness - threshold) / max(brightness, 0.0001);
    return color * max(factor, 0.0);
}

// Prefilter pass: 13-tap downsample + soft-knee threshold (pass 0 only)
@fragment
fn fs_prefilter(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = downsample_13tap(in.tex_coords, params.texel_size);
    let filtered = soft_threshold(color, params.threshold, params.knee);
    return vec4<f32>(filtered, 1.0);
}

// Downsample pass: 13-tap downsample without threshold (passes 1-4)
@fragment
fn fs_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = downsample_13tap(in.tex_coords, params.texel_size);
    return vec4<f32>(color, 1.0);
}
