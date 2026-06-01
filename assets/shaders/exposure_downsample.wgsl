// `exposure_downsample_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/exposure_downsample.hlsl`.
// `c_screen_postprocess::downsample_generate` step 3 — explicit shader 35. Produces a single half-float average-luminance value
// in log2-stops by 32-tap weighted log/linear average of a 1/16-res
// HDR source (intensity-in-alpha format from the bloom downsample
// chain).
//
// Inputs (engine plumbing via `c_screen_postprocess::copy`):
//   - source_sampler (slot 0): the 1/16-res HDR with `.a` =
//     per-texel intensity (NTSC-weighted luminance, written by
//     `downsample_4x4_block_bloom`).
//   - weight_sampler (slot 1): `auto_exposure_weight` bitmap
//     (engine `default_textures[7]` — see `rasterizer_globals`).
//     18×10 single-channel center-weighted Gaussian. The shader
//     reads `.g` (replicated from the monochrome channel on PC).
//   - `pixel_size = (1/source_w, 1/source_h, 0, 0)` — engine sets
//     this in `c_screen_postprocess::copy` based on the source
//     surface dims.
//   - `scale.x = m_auto_exposure_sensitivity` — cfxs slider that
//     lerps between mean(log) and log(mean) interpretations of
//     average luminance.
//
// Output: 1×1 R16Float surface. Only `.r` matters; we replicate
// the scalar to all four channels to match `: SV_Target` shape.

struct PostprocessParams {
    pixel_size: vec4<f32>,
    scale: vec4<f32>,
}

@group(0) @binding(0) var source_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var weight_sampler_tex: texture_2d<f32>;
@group(0) @binding(3) var weight_sampler: sampler;
@group(0) @binding(4) var<uniform> u: PostprocessParams;

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

fn tex2D_offset_source(uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(source_sampler_tex, source_sampler,
                         uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

fn tex2D_offset_weight(uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(weight_sampler_tex, weight_sampler,
                         uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // weighted_sum(log(intensity)), log(weighted_sum(intensity)), total_weight
    var average: vec3<f32> = vec3<f32>(0.0);

    // Engine: `for (int x = -3; x <= 3; x += 2)` — 4 columns.
    var x: f32 = -3.0;
    loop {
        if (x > 3.0) { break; }
        // Engine: `for (int y = -7; y <= 7; y += 2)` — 8 rows.
        var y: f32 = -7.0;
        loop {
            if (y > 7.0) { break; }
            let weight: f32 = tex2D_offset_weight(in.texcoord, x, y).g;
            // Engine: `intensity = tex2D_offset(source_sampler, ..., x, y).a`
            // (bloom downsample wrote intensity into alpha).
            let intensity: f32 = tex2D_offset_source(in.texcoord, x, y).a;
            average = average + weight * vec3<f32>(
                log2(0.00001 + intensity),
                intensity,
                1.0,
            );
            y = y + 2.0;
        }
        x = x + 2.0;
    }

    // Engine: `average.xy /= average.z;`
    let xy: vec2<f32> = average.xy / average.z;
    // Engine: `average.y = log2(0.00001 + average.y);` (after division)
    let log_of_mean: f32 = log2(0.00001 + xy.y);
    let mean_of_log: f32 = xy.x;

    // Engine: `return (avg.y * scale.x + avg.x * (1 - scale.x));`
    // scale.x = m_auto_exposure_sensitivity.
    let result: f32 = log_of_mean * u.scale.x + mean_of_log * (1.0 - u.scale.x);
    return vec4<f32>(result, result, result, result);
}
