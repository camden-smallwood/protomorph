// Screen-space ambient occlusion with bent normal output
// Port of C's ssao.fs
// Reads depth from depth buffer

@group(0) @binding(0) var t_normal: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_noise: texture_2d<f32>;
@group(0) @binding(3) var s_nearest: sampler;

const NUM_KERNEL_SAMPLES: u32 = 16u;

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

// Octahedral decoding: vec2 in [-1,1] -> unit vec3
fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    let frag_normal = oct_decode(textureSample(t_normal, s_nearest, uv).rg);
    let frag_depth = textureSample(t_depth, s_nearest, uv);

    let noise_scale = vec2<f32>(params.noise_scale_x, params.noise_scale_y);

    let raw_noise = textureSample(t_noise, s_nearest, uv * noise_scale).xyz * 2.0 - vec3<f32>(1.0);
    let noise_len = length(raw_noise);
    let noise_sample = select(vec3<f32>(0.0, 0.0, 1.0), raw_noise / noise_len, noise_len > 0.0001);

    var occlusion = 0.0;
    var bent_normal = vec3<f32>(0.0);

    for (var i = 0u; i < NUM_KERNEL_SAMPLES; i++) {
        // Reflect kernel sample by noise
        let ray = params.radius * reflect(params.kernel_samples[i].xyz, noise_sample);

        // Hemisphere direction (flip if below surface)
        let sample_dir = sign(dot(ray, frag_normal)) * ray;

        // Get occluder fragment
        let occluder_uv = uv + sample_dir.xy;
        let occluder_normal = oct_decode(textureSample(t_normal, s_nearest, occluder_uv).rg);
        let occluder_depth = textureSample(t_depth, s_nearest, occluder_uv);

        // Depth difference — reverse-Z: closer = larger depth, so flip subtraction
        let depth_diff = occluder_depth - frag_depth;

        // Occlusion calculation (matching C exactly)
        let sample_occlusion = step(params.falloff, depth_diff)
            * (1.0 - dot(occluder_normal, frag_normal))
            * (1.0 - smoothstep(params.falloff, params.strength, depth_diff));

        occlusion += sample_occlusion;

        // Accumulate bent normal — unoccluded directions contribute
        bent_normal += normalize(sample_dir) * (1.0 - sample_occlusion);
    }

    let ao = 1.0 - (occlusion / f32(NUM_KERNEL_SAMPLES));
    let bent_len = length(bent_normal);
    let safe_bent = select(frag_normal, bent_normal / bent_len, bent_len > 0.0001);
    bent_normal = normalize(mix(frag_normal, safe_bent, 1.0 - ao));

    return vec4<f32>(bent_normal, ao);
}
