// `blur_11_horizontal_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/blur_11_horizontal.hlsl`.
//
// 5-sample bilinear-trick variant of an 11-tap gaussian blur. Each
// sample reads two adjacent texels weighted by bilinear filtering.
// The 0.5 vertical offset shifts the center half a pixel; combined
// with the V pass's 0.5 horizontal offset the result is a perfectly-
// centered 11×11 blur. (See engine HLSL header diagram.)
//
// Hard-coded kernel weights (Pascal row 10 with bilinear pairing):
//   pair sums = (1+9, 36+84, 126+126, 84+36, 9+1) = (10, 120, 252, 120, 10),
//   total = 512.

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
    let sample: vec2<f32> = in.texcoord;
    let pixel: vec2<f32> = u.pixel_size.xy;

    // const float2 offset[5] (HLSL):
    let offset0 = vec2<f32>(-4.0 - 1.0 / (1.0 + 9.0), 0.5);            //  -4.1
    let offset1 = vec2<f32>(-2.0 - 36.0 / (36.0 + 84.0), 0.5);         //  -2.3
    let offset2 = vec2<f32>( 0.0 - 126.0 / (126.0 + 126.0), 0.5);      //  -0.5
    let offset3 = vec2<f32>( 2.0 - 84.0 / (84.0 + 36.0), 0.5);         //  +1.3
    let offset4 = vec2<f32>( 4.0 - 9.0 / (1.0 + 9.0), 0.5);            //  +3.1

    let color: vec4<f32> =
        (1.0 + 9.0)     * textureSample(target_sampler_tex, target_sampler, sample + offset0 * pixel) +
        (36.0 + 84.0)   * textureSample(target_sampler_tex, target_sampler, sample + offset1 * pixel) +
        (126.0 + 126.0) * textureSample(target_sampler_tex, target_sampler, sample + offset2 * pixel) +
        (84.0 + 36.0)   * textureSample(target_sampler_tex, target_sampler, sample + offset3 * pixel) +
        (1.0 + 9.0)     * textureSample(target_sampler_tex, target_sampler, sample + offset4 * pixel);

    return color / 512.0;
}
