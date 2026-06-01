// `downsample_4x4_block_bloom_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/downsample_4x4_block_bloom.hlsl`.
//
// 4-sample 4x4 box average via bilinear-trick offsets ±1 pixel,
// then bloom-curve threshold. Output is the bloom buffer at half
// the source resolution; alpha carries the per-texel intensity
// for downstream stages.
//
// `ps_postprocess_scale` = (bloom_point, bloom_inherent, _, _).
// `intensity_vector` defaults to NTSC weights (0.299, 0.587, 0.114).
// `DARK_COLOR_MULTIPLIER` = g_exposure.g (PC = 1.0).
//
// Engine writes LINEAR to RT1 on PC: `c_rasterizer::setup_render_target_globals_with_exposure
// @ 0x180670AD0` sets `LDR_gamma2 = HDR_gamma2 = 0`, so the conditional
// `safe_sqrt()` branches inside `convert_to_render_target.hlsl` never
// fire (verified against the decompiled DXBC for terrain static_sh,
// patchy_fog, etc. — they all `movc` the linear branch). The bloom-
// extract HLSL therefore sums LINEAR samples directly with no sqrt.
// An earlier port added an inline sqrt here based on a misreading of
// the conditional in render_target_fx.hlsl — that biased auto-exposure
// to half-stops (×0.5) and made the bloom_point threshold ~2× more
// permissive than authored cfxs values intended.

struct PostprocessParams {
    pixel_size: vec4<f32>,            // (1/w, 1/h, w, h) — source res
    scale: vec4<f32>,                 // ps_postprocess_scale
    intensity_vector: vec4<f32>,      // luminance weights
    dark_color_multiplier: vec4<f32>, // (g_exposure.g, _, _, _)
}

@group(0) @binding(0) var dark_source_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var dark_source_sampler: sampler;
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

// Mirrors HLSL's `tex2D_offset(s, uv, dx, dy)` =
// `sample2D(s, uv + (dx, dy) * PIXEL_SIZE.xy)`.
fn tex2D_offset(uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(dark_source_sampler_tex, dark_source_sampler,
                         uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color: vec3<f32> = vec3<f32>(0.00000001);  // hack to keep divide by zero from happening on the nVidia cards

    // Engine HLSL `downsample_4x4_block_bloom.hlsl` sums the four
    // dark_color samples directly. With HDR_gamma2=0 on PC the samples
    // are LINEAR scene radiance × DARK_COLOR_MULTIPLIER⁻¹; multiplying
    // back by DCM below recovers the linear scene value averaged over
    // the 4-tap neighborhood.
    var sample: vec4<f32> = tex2D_offset(in.texcoord, -1.0, -1.0);
    color = color + sample.rgb;

    sample = tex2D_offset(in.texcoord, 1.0, -1.0);
    color = color + sample.rgb;

    sample = tex2D_offset(in.texcoord, -1.0, 1.0);
    color = color + sample.rgb;

    sample = tex2D_offset(in.texcoord, 1.0, 1.0);
    color = color + sample.rgb;

    color = color * u.dark_color_multiplier.x / 4.0;

    // calculate 'intensity' (max or dot product?)
    let intensity: f32 = dot(color.rgb, u.intensity_vector.rgb);

    // calculate bloom curve intensity
    let bloom_intensity: f32 = max(intensity * u.scale.y, intensity - u.scale.x);

    // calculate bloom color
    let bloom_color: vec3<f32> = color * (bloom_intensity / intensity);

    return vec4<f32>(bloom_color.rgb, intensity);
}
