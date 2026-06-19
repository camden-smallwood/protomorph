// `radial_blur_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/radial_blur.hlsl`.
//
// Shader index 98 in `c_screen_postprocess::render_lightshafts`.
// 16-tap log-spaced radial blur from `g_center_scale.xy` (sun pos in
// screen space). Tap stride is `pow(1 + scale, (i - 15) / 16)` —
// samples scatter from the center outward with the i=15 sample at the
// fragment itself and earlier samples pulled toward the sun.
//
// Run twice in succession by render_lightshafts:
//   - Pass 1: BLUR_WEIGHTS encode the dynamic kernel
//             (`weights[i] = (1 - x*x*sum) * (i+1)^2 / sum² ` —
//             quadratic-decay), `g_tint = (intensity*r, intensity*g, intensity*b, _)`.
//   - Pass 2: BLUR_WEIGHTS uniform `1/16` (= 0x3D800000 broadcast),
//             `g_tint = (1, 1, 1, 1)`. Smooths out the dynamic-kernel
//             striations.
//
// Cbuffer layout (engine slot → field):
//   0x420001 g_tint         = (r, g, b, _)
//   0x430001 g_center_scale = (sun.x, sun.y, scale.x, scale.y)
//   0x430002+0..3 BLUR_WEIGHTS0..3 = 16 sample weights packed as 4 vec4s

struct PostprocessParams {
    pixel_size: vec4<f32>,           // unused by this shader
    g_tint: vec4<f32>,               // (r, g, b, _) — engine slot 0x420001
    g_center_scale: vec4<f32>,       // (sun.x, sun.y, scale.x, scale.y) — repurposes intensity_vector
    _pad: vec4<f32>,                 // unused (was dark_color_multiplier)
}

struct BlurWeights {
    w0: vec4<f32>,
    w1: vec4<f32>,
    w2: vec4<f32>,
    w3: vec4<f32>,
}

@group(0) @binding(0) var color_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var color_sampler: sampler;
@group(0) @binding(2) var<uniform> u: PostprocessParams;
@group(0) @binding(3) var<uniform> w: BlurWeights;

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

fn weight_at(i: i32) -> f32 {
    // Pack the 4-vec4 weights table into a 16-entry lookup. WGSL has
    // no array-of-vec4 indexing-by-component, so unroll the case
    // manually.
    let group = i / 4;
    let comp = i % 4;
    var v: vec4<f32>;
    if (group == 0) { v = w.w0; }
    else if (group == 1) { v = w.w1; }
    else if (group == 2) { v = w.w2; }
    else { v = w.w3; }
    if (comp == 0) { return v.x; }
    if (comp == 1) { return v.y; }
    if (comp == 2) { return v.z; }
    return v.w;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let center: vec2<f32> = u.g_center_scale.xy;
    let scale: vec2<f32> = u.g_center_scale.zw;
    let offset: f32 = -15.0;
    let fade_offset: f32 = 1.0;

    let vec_to_frag: vec2<f32> = in.texcoord - center;

    var blurred: vec3<f32> = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 16; i = i + 1) {
        // Engine: `pow(1 + scale, float(i + offset) / sampleN)` — element-wise
        // pow on `1 + scale.xy` (asymmetric per-axis radial stretch).
        // WGSL: element-wise via vec2 pow.
        let exponent: f32 = (f32(i) + offset) / 16.0;
        let t: vec2<f32> = pow(vec2<f32>(1.0) + scale, vec2<f32>(exponent));
        let uv: vec2<f32> = center + vec_to_frag * t;
        let mask: f32 = saturate(12.0 * uv.x * (1.0 - uv.x) + fade_offset)
                     * saturate(12.0 * uv.y * (1.0 - uv.y) + fade_offset);
        blurred = blurred + textureSample(color_sampler_tex, color_sampler, uv).rgb
                          * weight_at(i) * mask;
    }

    return vec4<f32>(blurred * u.g_tint.rgb, 0.0);
}
