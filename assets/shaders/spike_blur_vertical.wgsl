// `spike_blur_vertical_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/spike_blur_vertical.hlsl`.
//
// Halo runs this for "vertical-ish" spike directions. Same 8-tap
// loop with per-channel geometric falloff as the horizontal variant,
// but the bilinear sampling is on the X axis instead of Y.
//
// Used by `c_screen_postprocess::bling_generate` via
// `c_rasterizer::set_explicit_shaders(4, ...)`.

struct SpikeBlurParams {
    source_pixel_size: vec4<f32>,
    offset_delta: vec4<f32>,
    initial_color: vec4<f32>,
    delta_color: vec4<f32>,
}

@group(0) @binding(0) var source_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> u: SpikeBlurParams;

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

/// `get_pixel_linear_x` (h:14-32). Linear-in-X, point-in-Y 2-tap.
fn get_pixel_linear_x(tex_coord_in: vec2<f32>) -> vec3<f32> {
    var tex_coord = tex_coord_in;
    tex_coord.x = (tex_coord.x / u.source_pixel_size.x) - 0.5;
    var texel0: vec2<f32> = vec2<f32>(floor(tex_coord.x), tex_coord.y);

    var blend: vec4<f32>;
    blend = vec4<f32>(tex_coord - texel0, 0.0, 0.0);
    blend = vec4<f32>(blend.x, blend.y, 1.0 - blend.x, 1.0 - blend.y);

    texel0.x = (texel0.x + 0.5) * u.source_pixel_size.x;

    var texel1: vec2<f32> = texel0;
    texel1.x = texel1.x + u.source_pixel_size.x;

    let s0 = textureSample(source_sampler_tex, source_sampler, texel0).rgb;
    let s1 = textureSample(source_sampler_tex, source_sampler, texel1).rgb;
    return blend.z * s0 + blend.x * s1;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var sample0: vec2<f32> = in.texcoord + u.offset_delta.xy;

    var color_scale: vec3<f32> = u.initial_color.rgb;
    var color: vec3<f32> = color_scale * get_pixel_linear_x(sample0);

    for (var y: i32 = 1; y < 8; y = y + 1) {
        color_scale = color_scale * u.delta_color.rgb;
        sample0 = sample0 + u.offset_delta.zw;
        color = color + color_scale * get_pixel_linear_x(sample0);
    }

    return vec4<f32>(color, 1.0);
}
