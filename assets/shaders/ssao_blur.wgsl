// Bilateral blur for SSAO — 4x4 depth-aware kernel
// Preserves edges by weighting samples with depth similarity
// Blurs bent normal (xyz) + AO (w) together

@group(0) @binding(0) var t_ssao: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_nearest: sampler;

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

const SIGMA_DEPTH: f32 = 0.05;
const KERNEL_RADIUS: i32 = 2;
const SIGMA_SPATIAL: f32 = 2.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;
    let ssao_size = vec2<f32>(textureDimensions(t_ssao));
    let texel_size = 1.0 / ssao_size;

    let center_depth = textureSample(t_depth, s_nearest, uv);

    var total_weight = 0.0;
    var total_value = vec4<f32>(0.0);

    for (var y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
        for (var x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
            let offset = vec2<f32>(f32(x), f32(y));
            let sample_uv = uv + offset * texel_size;

            let sample_value = textureSample(t_ssao, s_nearest, sample_uv);
            let depth = textureSample(t_depth, s_nearest, sample_uv);

            let depth_diff = center_depth - depth;
            let spatial_dist = f32(x * x + y * y);
            let weight = exp(-spatial_dist / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL)
                             - (depth_diff * depth_diff) / (SIGMA_DEPTH * SIGMA_DEPTH));

            total_value += sample_value * weight;
            total_weight += weight;
        }
    }

    let result = total_value / max(total_weight, 0.0001);
    let bent_len = length(result.xyz);
    let safe_bent = select(vec3<f32>(0.0, 0.0, 1.0), result.xyz / bent_len, bent_len > 0.0001);
    return vec4<f32>(safe_bent, result.w);
}
