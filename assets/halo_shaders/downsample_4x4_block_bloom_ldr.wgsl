// `downsample_4x4_block_bloom_ldr_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/downsample_4x4_block_bloom_LDR.hlsl`.
//
// Engine shader 67. Called by `c_screen_postprocess::postprocess_player_view
// @ 0x1806B4160` immediately after shader 48 (`downsample_4x4_block_bloom`):
//
//   apply_binary_op_ex(48, composite_cubemap → aux_bloom,
//                          scale=(v23*bloom_point, bloom_inherent, 1, 1))
//   apply_binary_op_ex(67, aux_bloom + composite_depth → aux_exposure_5,
//                          scale=(v23*bloom_point, bloom_inherent, 1, 1))
//
// `aux_exposure_5` is the input handed to `postprocess_bloom_buffer` —
// the multi-scale pyramid starts from this 1/4-resolution surface,
// NOT from `aux_bloom` (1/2 res). Skipping this step (protomorph's
// previous behavior) made the pyramid operate ONE LEVEL HIGHER than
// engine, producing visibly sharper/brighter bloom.
//
// Body:
//   1. Box-sample the SCENE (`source_sampler`, slot 1) at 4 offset
//      positions → quarter-resolution average.
//   2. Apply the same bloom-curve threshold as shader 48
//      (`bloom_intensity = max(intensity × scale.y, intensity - scale.x)`).
//   3. Return `max(bloom_curve_result, point-sample of aux_bloom)`.
//      The point-sample at the same screen position pulls in whatever
//      shader 48 already extracted there — the MAX preserves any
//      locally-bright bloom that survives the quarter-res downsample.
//
// Slot mapping (mirrors engine `apply_binary_op_ex(67, source_0, source_1, dest)`):
//   - bloom_sampler  (slot 0) ← source_0 = aux_bloom
//   - source_sampler (slot 1) ← source_1 = composite_depth (scene HDR)
//
// Wgpu adaptation: WGSL has no `SCREEN_POSITION_INPUT` builtin, so we
// `textureLoad` from `bloom_sampler` at the matching integer pixel —
// mirrors engine `tex2D_offset_point(bloom_sampler, (screen_pos+0.5)*pixel_size.xy, 0, 0)`.

// Shared `PostprocessUniforms` layout used by every bgl_2tex consumer
// (`pixel_size`, `scale`, `intensity_vector`, `dark_color_multiplier`).
// The LDR variant doesn't read `dark_color_multiplier` (engine line 32
// is bare `color / 4.0`, no multiplier), but the struct shape has to
// match what `upload_uniforms` writes.
struct PostprocessParams {
    pixel_size: vec4<f32>,
    scale: vec4<f32>,
    intensity_vector: vec4<f32>,
    dark_color_multiplier: vec4<f32>,
}

@group(0) @binding(0) var bloom_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;
@group(0) @binding(2) var source_sampler_tex: texture_2d<f32>;
@group(0) @binding(3) var source_sampler: sampler;
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

fn tex2D_offset(uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(source_sampler_tex, source_sampler,
                         uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // HLSL: float3 color = 0.00000001f;
    var color: vec3<f32> = vec3<f32>(0.00000001);

    var sample: vec4<f32> = tex2D_offset(in.texcoord, -1.0, -1.0);
    color = color + sample.rgb;
    sample = tex2D_offset(in.texcoord, 1.0, -1.0);
    color = color + sample.rgb;
    sample = tex2D_offset(in.texcoord, -1.0, 1.0);
    color = color + sample.rgb;
    sample = tex2D_offset(in.texcoord, 1.0, 1.0);
    color = color + sample.rgb;
    // HLSL: color = color / 4.0  (no DARK_COLOR_MULTIPLIER — engine LDR
    // variant differs from shader 48 here; line 32 is bare `color/4.0`).
    color = color / 4.0;

    // HLSL: float intensity = dot(color.rgb, intensity_vector.rgb);
    let intensity: f32 = dot(color, u.intensity_vector.rgb);

    // HLSL: float bloom_intensity = max(intensity*scale.y, intensity-scale.x);
    let bloom_intensity: f32 = max(intensity * u.scale.y, intensity - u.scale.x);

    // HLSL: float3 bloom_color = color * (bloom_intensity / intensity);
    let bloom_color: vec3<f32> = color * (bloom_intensity / intensity);

    // HLSL: return max(float4(bloom_color.rgb, intensity),
    //                  tex2D_offset_point(bloom_sampler, (screen_pos+0.5)*pixel_size.xy, 0, 0));
    let bloom_dims = vec2<f32>(textureDimensions(bloom_sampler_tex, 0));
    let prev_bloom_pixel = vec2<i32>(clamp(in.texcoord * bloom_dims,
                                           vec2<f32>(0.0),
                                           bloom_dims - vec2<f32>(1.0)));
    let prev_bloom = textureLoad(bloom_sampler_tex, prev_bloom_pixel, 0);

    return max(vec4<f32>(bloom_color, intensity), prev_bloom);
}
