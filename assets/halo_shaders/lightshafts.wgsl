// `lightshafts_hlsl.hlsl::default_ps` — verbatim port.
// Engine source: `source/rasterizer/hlsl/lightshafts.hlsl`.
//
// Shader index 97 in `c_screen_postprocess::render_lightshafts`. First
// pass — extracts bright sun-cone pixels from the scene HDR with:
//   - outer falloff cone (`g_sun_pos.zw` scales)
//   - inner falloff cone (`g_inner_size.xy` scales)
//   - intensity clamp (`g_tint.w` luminance threshold)
//   - depth clamp (`g_inner_size.zw` scale + offset)
//
// `g_tint.xyz` is commented out in the shipping HLSL (line 36) so the
// tint applies later (radial_blur's per-pixel `g_tint.rgb` multiply).
//
// Cbuffer layout (engine slot → field):
//   0x420001 g_tint        = (_, _, _, intensity_clamp)
//   0x370001 g_sun_pos     = (sun.x, sun.y, outer_falloff.x, outer_falloff.y)
//   0x370002 g_inner_size  = (inner.x, inner.y, depth_clamp_scale, depth_clamp_bias)
//
// Wgpu cbuffer reuses the standard 4-vec4 `PostprocessParams` layout —
// the unused `intensity_vector` and `dark_color_multiplier` slots
// carry `g_sun_pos` and `g_inner_size` so this shader can dispatch
// via `apply_binary_op_ex` (which uses bgl_2tex).

struct PostprocessParams {
    pixel_size: vec4<f32>,
    g_tint: vec4<f32>,            // engine slot 0x420001
    g_sun_pos: vec4<f32>,         // engine slot 0x370001 (repurposes intensity_vector)
    g_inner_size: vec4<f32>,      // engine slot 0x370002 (repurposes dark_color_multiplier)
}

@group(0) @binding(0) var color_sampler_tex: texture_2d<f32>;
@group(0) @binding(1) var color_sampler: sampler;
// Depth surface is Depth32Float — declare as `texture_depth_2d` so
// the bind group's depth view (sample_type=Depth) matches.
@group(0) @binding(2) var depth_sampler_tex: texture_depth_2d;
@group(0) @binding(3) var depth_sampler: sampler;
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
    var color: vec3<f32> = textureSample(color_sampler_tex, color_sampler, in.texcoord).rgb;
    let depth: f32 = textureSample(depth_sampler_tex, depth_sampler, in.texcoord);

    var offs: vec2<f32> = in.texcoord - u.g_sun_pos.xy;
    let offs_inner: vec2<f32> = offs * u.g_inner_size.xy;
    offs = offs * u.g_sun_pos.zw;

    let falloff: f32 = pow(saturate(1.0 - dot(offs, offs)), 2.0);
    let inner_falloff: f32 = saturate(pow(saturate(dot(offs_inner, offs_inner)), 2.0));
    let lum: f32 = dot(color, vec3<f32>(0.299, 0.587, 0.114));

    // intensity clamp — `g_tint.w` is the cfxs `intensity_clamp` slider.
    color = color * saturate((lum - u.g_tint.w) * 10.0);

    // depth clamp — far pixels excluded.
    color = color * saturate(depth * u.g_inner_size.z + u.g_inner_size.w);

    // falloff clamps (outer × inner cones around the sun).
    color = color * falloff;
    color = color * inner_falloff;

    // Engine `g_tint.xyz` multiply is commented out (line 36 of HLSL)
    // — tint is applied in the radial_blur pass instead.

    return vec4<f32>(color, 0.0);
}
