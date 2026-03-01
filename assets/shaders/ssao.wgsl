// Screen-space ambient occlusion
// Port of C's ssao.fs
// Reads depth from position_depth.w (packed G-buffer)

@group(0) @binding(0) var t_normal: texture_2d<f32>;
@group(0) @binding(1) var t_position_depth: texture_2d<f32>;
@group(0) @binding(2) var t_noise: texture_2d<f32>;
@group(0) @binding(3) var s_nearest: sampler;

const NUM_KERNEL_SAMPLES: u32 = 32u;

struct SsaoParams {
    kernel_samples: array<vec4<f32>, 32>,
    strength: f32,
    falloff: f32,
    radius: f32,
    noise_scale_x: f32,
    noise_scale_y: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(4) var<uniform> params: SsaoParams;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let uv = in.tex_coords;

    let frag_normal = textureSample(t_normal, s_nearest, uv).xyz;
    let frag_depth = textureSample(t_position_depth, s_nearest, uv).w;

    let noise_scale = vec2<f32>(params.noise_scale_x, params.noise_scale_y);

    let noise_sample = normalize(textureSample(t_noise, s_nearest, uv * noise_scale).xyz * 2.0 - vec3<f32>(1.0));

    var occlusion = 0.0;

    for (var i = 0u; i < NUM_KERNEL_SAMPLES; i++) {
        // Reflect kernel sample by noise
        let ray = params.radius * reflect(params.kernel_samples[i].xyz, noise_sample);

        // Get occluder fragment
        let occluder_uv = uv + sign(dot(ray, frag_normal)) * ray.xy;
        let occluder_normal = textureSample(t_normal, s_nearest, occluder_uv).xyz;
        let occluder_depth = textureSample(t_position_depth, s_nearest, occluder_uv).w;

        // Depth difference (negative = occluder behind fragment)
        let depth_diff = frag_depth - occluder_depth;

        // Occlusion calculation (matching C exactly)
        occlusion += step(params.falloff, depth_diff)
            * (1.0 - dot(occluder_normal, frag_normal))
            * (1.0 - smoothstep(params.falloff, params.strength, depth_diff));
    }

    return 1.0 - (occlusion / f32(NUM_KERNEL_SAMPLES));
}
