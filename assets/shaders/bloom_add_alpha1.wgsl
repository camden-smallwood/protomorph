// `bloom_add_alpha1_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/bloom_add_alpha1.hlsl`.
//
// Variant of `add` that forces alpha=1 and skips the `add.a`
// multiplier. Used by `apply_binary_op_ex(66, ...)` in
// `c_screen_postprocess::postprocess_bloom_buffer`.
//
// `color.rgb = ps_postprocess_scale.rgb * original.rgb + add.rgb`
// `color.a = 1.0`

struct PostprocessParams {
    pixel_size: vec4<f32>,
    scale: vec4<f32>, // ps_postprocess_scale
}

@group(0) @binding(0) var original_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var original_sampler: sampler;
@group(0) @binding(2) var add_sampler_tex: texture_2d<f32>;
@group(0) @binding(3) var add_sampler: sampler;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let original: vec4<f32> = textureSample(original_sampler_tex, original_sampler, in.texcoord);
    let add: vec4<f32> = textureSample(add_sampler_tex, add_sampler, in.texcoord);

    var color: vec4<f32>;
    color = vec4<f32>(
        u.scale.rgb * original.rgb + add.rgb,
        1.0,
    );

    return color;
}
