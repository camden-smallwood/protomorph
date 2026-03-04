// Debug overlay: draws a single cubemap face as a small quad on screen

struct Params {
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    _pad: f32,
};

@group(0) @binding(0) var t_face: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;
@group(0) @binding(2) var<uniform> params: Params;

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
    let pos = in.position * params.scale + vec2<f32>(params.offset_x, params.offset_y);
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(t_face, s_linear, in.tex_coords);
    let lum = dot(hdr.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let tonemapped = hdr.rgb / (1.0 + lum);
    return vec4<f32>(tonemapped, 1.0);
}
