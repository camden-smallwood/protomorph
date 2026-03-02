// Bilateral blur for SSAO — 4x4 depth-aware kernel
// Preserves edges by weighting samples with depth similarity

@group(0) @binding(0) var t_ssao: texture_2d<f32>;
@group(0) @binding(1) var t_position_depth: texture_2d<f32>;
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let uv = in.tex_coords;
    let ssao_size = vec2<f32>(textureDimensions(t_ssao));
    let texel_size = 1.0 / ssao_size;

    let center_depth = textureSample(t_position_depth, s_nearest, uv).w;

    var total_weight = 0.0;
    var total_ao = 0.0;

    for (var y = -KERNEL_RADIUS; y < KERNEL_RADIUS; y++) {
        for (var x = -KERNEL_RADIUS; x < KERNEL_RADIUS; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) + 0.5;
            let sample_uv = uv + offset * texel_size;

            let ao = textureSample(t_ssao, s_nearest, sample_uv).r;
            let depth = textureSample(t_position_depth, s_nearest, sample_uv).w;

            let depth_diff = center_depth - depth;
            let weight = exp(-(depth_diff * depth_diff) / (SIGMA_DEPTH * SIGMA_DEPTH));

            total_ao += ao * weight;
            total_weight += weight;
        }
    }

    return total_ao / max(total_weight, 0.0001);
}
