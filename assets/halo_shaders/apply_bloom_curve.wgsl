// `apply_bloom_curve_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/apply_bloom_curve.hlsl` (single
// pass shader, no class hierarchy — dispatched by
// `c_screen_postprocess::render_bloom_apply_curve` or similar).
//
// Reads the HDR composite, applies the bloom-curve threshold, and
// writes the bright-pixel residue. `ps_postprocess_scale` carries
// `(bloom_point, bloom_inherent, _, _)` from `c_camera_fx_values`.
//
// Protomorph divergence from engine: `if (maximum > 0.0)` guard around
// the unconditional `color *= (overwhite/maximum)`. Engine HLSL has no
// guard — `0/0 = NaN` for all-zero pixels gets multiplied through.
// Defensive per `feedback_engine_hlsl_isnt_always_portable.md`: WGSL/Metal
// can propagate NaN where D3D9 floats it harmlessly to 0; explicit guard
// kept.

struct PostprocessParams {
    pixel_size: vec4<f32>,           // (1/w, 1/h, w, h)
    scale: vec4<f32>,                // (bloom_point, bloom_inherent, _, _)
    dark_color_multiplier: vec4<f32>,// (g_exposure.g, _, _, _)
}

@group(0) @binding(0) var source_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
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
    let sample0: vec2<f32> = in.texcoord;

    var color: vec4<f32> = textureSample(source_sampler_tex, source_sampler, sample0);
    color = vec4<f32>(color.rgb * u.dark_color_multiplier.x, color.a);

    // drop out visible values (no bloom on em)
    let maximum: f32 = max(max(color.r, color.g), color.b);

    let overwhite: f32 = max(maximum * u.scale.y, maximum - u.scale.x);
    if (maximum > 0.0) {
        color = color * (overwhite / maximum);
    }

    return color / u.dark_color_multiplier.x;
}
