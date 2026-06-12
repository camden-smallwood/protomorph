// particle_distortion_apply.wgsl — full-screen warp that applies the
// accumulated particle distortion to the scene. Engine `displacement.hlsl`:
//   displacement = scale · (sample(accum).xy - 0.5)
//   frame_texcoords = uv + displacement
//   out = sample(scene, frame_texcoords)
// where the accumulation buffer was cleared to 0.5 and distortion particles
// added their per-pixel displacement deltas (so sample-0.5 = total offset).

@group(0) @binding(0) var scene_tex: texture_2d<f32>;   // copy of the scene color
@group(0) @binding(1) var accum_tex: texture_2d<f32>;   // distortion accumulation
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle.
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// Hard cap on the scene warp (UV fraction) — the primary safety bound. The
// engine's heat shimmer is SUBTLE: it ripples edges, it must not pull content
// away. 0.4% of the frame (~4px at 1080p) shimmers without erasing the blue
// antigrav jet behind it. (The engine bounds via max_displacement·1024/32767.)
const MAX_DISP: f32 = 0.015;

@fragment
fn fs_apply(in: VsOut) -> @location(0) vec4<f32> {
    var disp = textureSample(accum_tex, samp, in.uv).xy - vec2<f32>(0.5, 0.5);
    if (dot(disp, disp) <= 1.0e-10) {
        discard;
    }
    disp = clamp(disp, vec2<f32>(-MAX_DISP), vec2<f32>(MAX_DISP));
    let warped_uv = in.uv + disp;
    return textureSample(scene_tex, samp, warped_uv);
}
