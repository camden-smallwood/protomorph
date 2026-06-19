// `vertical_gaussian_blur_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/vertical_gaussian_blur.hlsl`.
//
// 11-tap vertical gaussian using Pascal's triangle row 10 weights
// (1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1) / 1024. Mirror of
// `horizontal_gaussian_blur.wgsl` with the offset axis swapped.

struct PostprocessParams {
    pixel_size: vec4<f32>, // (1/w, 1/h, w, h)
}

@group(0) @binding(0) var target_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var target_sampler: sampler;
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
    var sample: vec2<f32> = in.texcoord;
    let pixel: vec2<f32> = u.pixel_size.xy;

    sample.y = sample.y - 5.0 * pixel.y;
    var color: vec3<f32> = (1.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (10.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (45.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (120.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (210.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (252.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (210.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (120.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (45.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (10.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    sample.y = sample.y + pixel.y;
    color = color + (1.0/1024.0) * textureSample(target_sampler_tex, target_sampler, sample).rgb;

    return vec4<f32>(color, 1.0);
}
