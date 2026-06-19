// `spike_blur_horizontal_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/spike_blur_horizontal.hlsl`.
//
// Halo runs this for "horizontal-ish" spike directions (angle bins
// where tan(angle) is well-behaved on the X axis). It walks 8 taps
// along the spike direction, doing 2-tap bilinear-Y point-X sampling
// per step, with per-channel geometric falloff giving chromatic
// dispersion (R/G/B attenuate at slightly different rates → lens-
// flare prismatic color shift).
//
// Used by `c_screen_postprocess::bling_generate` via
// `c_rasterizer::set_explicit_shaders(3, ...)`.

struct SpikeBlurParams {
    /// `source_pixel_size` (postprocess.fx PIXEL_SIZE) — `(1/w, 1/h, w, h)`
    /// of the **source** surface. The HLSL only reads `.x/.y`.
    source_pixel_size: vec4<f32>,
    /// `(start_offset.xy, step_delta.zw)`. start_offset is added to
    /// the texcoord before the first sample; step_delta is added each
    /// loop iteration.
    offset_delta: vec4<f32>,
    /// `(R, G, B, _)` starting weight for tap 0.
    initial_color: vec4<f32>,
    /// `(R, G, B, _)` per-step multiplicative falloff. After tap N the
    /// weight is `initial_color * delta_color^N`.
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

/// `get_pixel_linear_y` (h:14-32). Linear-in-Y, point-in-X 2-tap
/// bilinear. Mirrors HLSL exactly.
fn get_pixel_linear_y(tex_coord_in: vec2<f32>) -> vec3<f32> {
    var tex_coord = tex_coord_in;
    tex_coord.y = (tex_coord.y / u.source_pixel_size.y) - 0.5;
    var texel0: vec2<f32> = vec2<f32>(tex_coord.x, floor(tex_coord.y));

    var blend: vec4<f32>;
    blend = vec4<f32>(tex_coord - texel0, 0.0, 0.0);
    blend = vec4<f32>(blend.x, blend.y, 1.0 - blend.x, 1.0 - blend.y);

    texel0.y = (texel0.y + 0.5) * u.source_pixel_size.y;

    var texel1: vec2<f32> = texel0;
    texel1.y = texel1.y + u.source_pixel_size.y;

    let s0 = textureSample(source_sampler_tex, source_sampler, texel0).rgb;
    let s1 = textureSample(source_sampler_tex, source_sampler, texel1).rgb;
    return blend.w * s0 + blend.y * s1;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var sample0: vec2<f32> = in.texcoord + u.offset_delta.xy;

    var color_scale: vec3<f32> = u.initial_color.rgb;
    var color: vec3<f32> = color_scale * get_pixel_linear_y(sample0);

    for (var x: i32 = 1; x < 8; x = x + 1) {
        color_scale = color_scale * u.delta_color.rgb;
        sample0 = sample0 + u.offset_delta.zw;
        color = color + color_scale * get_pixel_linear_y(sample0);
    }

    return vec4<f32>(color, 1.0);
}
