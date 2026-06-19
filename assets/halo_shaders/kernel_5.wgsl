// `kernel_5_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/kernel_5.hlsl`.
//
// 5-tap arbitrary-kernel sampler. `kernel[i].xy` = pixel offset,
// `kernel[i].z` = weight. Result × `ps_postprocess_scale` for
// per-channel tinting.
//
// Used as the V pass inside `c_screen_postprocess::gaussian_blur_fixed`
// (called via `copy(34, ...)`); the kernel coefficients get uploaded
// before each call via `set_shader_constant(0x330000, 5, kernel_vertical)`.

struct Kernel5Params {
    pixel_size: vec4<f32>,    // ps_postprocess_pixel_size
    scale: vec4<f32>,         // ps_postprocess_scale
    kernel: array<vec4<f32>, 5>, // (offset_x, offset_y, weight, _) × 5
}

@group(0) @binding(0) var target_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var target_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Kernel5Params;

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
    let sample: vec2<f32> = in.texcoord;
    let pixel: vec2<f32> = u.pixel_size.xy;

    let color: vec4<f32> =
        u.kernel[0].z * textureSample(target_sampler_tex, target_sampler, sample + u.kernel[0].xy * pixel) +
        u.kernel[1].z * textureSample(target_sampler_tex, target_sampler, sample + u.kernel[1].xy * pixel) +
        u.kernel[2].z * textureSample(target_sampler_tex, target_sampler, sample + u.kernel[2].xy * pixel) +
        u.kernel[3].z * textureSample(target_sampler_tex, target_sampler, sample + u.kernel[3].xy * pixel) +
        u.kernel[4].z * textureSample(target_sampler_tex, target_sampler, sample + u.kernel[4].xy * pixel);

    return color * u.scale;
}
