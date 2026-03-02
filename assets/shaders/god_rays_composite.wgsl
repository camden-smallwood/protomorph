// God rays composite — additive blend half-res god rays onto lighting buffer
// Pipeline uses additive blend state (src: One, dst: One)

@group(0) @binding(0) var t_god_rays: texture_2d<f32>;
@group(0) @binding(1) var s_filtering: sampler;

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
    let color = textureSample(t_god_rays, s_filtering, in.tex_coords).rgb;
    return vec4(color, 1.0);
}
