// `ssao_blur_hlsl.hlsl::default_ps` — verbatim port (engine shader 96).
// Engine source: `source/rasterizer/hlsl/ssao_blur.hlsl`.
//
// Diagonal-cross 4-tap blur with depth-discontinuity rejection. Used
// twice by render_ssao:
//   - First pass: blur AO (output of ssao.wgsl) with depth=raw_depth
//   - Second pass: blend (multiply) the blurred AO into the scene HDR
//
// `sampleDiag` reads two opposite-corner samples (e.g., NW + SE) and
// the corresponding depth values. If the depth ratio test passes
// (similar depths on both sides), it averages all 3 (the two corner
// samples + the central sample-mask). Otherwise it picks the side
// whose depth is closer to center.
//
// The final pass mixes XY0 (NW-SE) and XY1 (NE-SW) diagonals.

struct BlurParams {
    pixel_size: vec4<f32>,    // (1/w, 1/h, w, h) of source surface
    _scale: vec4<f32>,        // unused
    _intensity: vec4<f32>,    // unused
    _dark: vec4<f32>,         // unused
}

// Depth surface is Depth32Float — declare as `texture_depth_2d` to
// match the bind-group layout (sample type Depth, not Float).
// textureSample on a depth texture returns f32 directly.
@group(0) @binding(0) var depth_sampler_tex: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var ssao_sampler_tex: texture_2d<f32>;
@group(0) @binding(3) var ssao_sampler: sampler;
@group(0) @binding(4) var<uniform> u: BlurParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex: vec2<f32>,
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
    out.tex = uvs[idx];
    return out;
}

fn tex2d_offset(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, ox: f32, oy: f32) -> vec4<f32> {
    return textureSample(tex, samp, uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

// Depth-specific offset sampler — `textureSample` on `texture_depth_2d`
// returns f32, no `.r` swizzle needed.
fn depth_offset(uv: vec2<f32>, ox: f32, oy: f32) -> f32 {
    return textureSample(depth_sampler_tex, depth_sampler, uv + vec2<f32>(ox, oy) * u.pixel_size.xy);
}

fn sample_diag(sm_mask: f32, depth: f32, tex_c: vec2<f32>, off0: vec2<f32>, off1: vec2<f32>) -> f32 {
    let d_v_x0 = depth_offset(tex_c, off0.x, off0.y);
    let d_v_x1 = depth_offset(tex_c, off1.x, off1.y);

    let dx0 = d_v_x0 - depth;
    let dx1 = depth - d_v_x1;

    let sm_x0 = tex2d_offset(ssao_sampler_tex, ssao_sampler, tex_c, off0.x, off0.y).r;
    let sm_x1 = tex2d_offset(ssao_sampler_tex, ssao_sampler, tex_c, off1.x, off1.y).r;

    var ret: f32;
    if (abs(1.0 - dx1 / dx0) < 0.5) {
        // Symmetric — both neighbors are at similar depth gradient.
        ret = (sm_x0 + sm_x1 + sm_mask * 0.5) / 2.5;
    } else if (abs(dx0) < abs(dx1)) {
        // Closer to dx0 side.
        ret = (sm_x0 + sm_mask) / 2.0;
    } else {
        ret = (sm_x1 + sm_mask) / 2.0;
    }
    return ret;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ssao = textureSample(ssao_sampler_tex, ssao_sampler, in.tex).r;
    let depth = textureSample(depth_sampler_tex, depth_sampler, in.tex);

    let diag_xy0 = sample_diag(ssao, depth, in.tex,
                               vec2<f32>(-1.0, -1.0),
                               vec2<f32>( 1.0,  1.0));
    let diag_xy1 = sample_diag(ssao, depth, in.tex,
                               vec2<f32>(-1.0,  1.0),
                               vec2<f32>( 1.0, -1.0));

    let result = (diag_xy0 + diag_xy1) * 0.5;
    return vec4<f32>(result, result, result, result);
}
