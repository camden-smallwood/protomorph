// protomorph host: bloom+bling combine extracted as a dedicated shader.
//
// Engine has no separate HLSL file for this — the combine is inlined in
// `c_screen_postprocess::copy_accumulation_target @ 0x1806B71E0`
// (screen_postprocess.cpp:451+ in Ares source). The inline expression is:
//   dest = bloom × bloom_large_color × bloom_intensity + star × bling_intensity
// which protomorph packs as:
//   scale.rgb = bloom_large_color × bloom_intensity  (precomputed on CPU)
//   scale.w   = bling_intensity                       (precomputed on CPU)
//   dest      = bloom × scale.rgb + star × scale.w
//
// Splitting this into its own shader lets protomorph's pass-per-shader
// model dispatch it cleanly. Behavior is engine-faithful even though the
// file layout diverges from the engine HLSL set.

struct PostprocessParams {
    pixel_size: vec4<f32>,
    /// (R, G, B, _) bloom large-tint × bloom_intensity, _ = bling_intensity.
    scale: vec4<f32>,
}

@group(0) @binding(0) var bloom_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;
@group(0) @binding(2) var star_tex: texture_2d<f32>;
@group(0) @binding(3) var star_sampler: sampler;
@group(0) @binding(4) var<uniform> u: PostprocessParams;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let bloom = textureSample(bloom_tex, bloom_sampler, in.texcoord);
    let star = textureSample(star_tex, star_sampler, in.texcoord);
    let result = bloom.rgb * u.scale.rgb + star.rgb * u.scale.w;
    return vec4<f32>(result, 1.0);
}
