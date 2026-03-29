// God rays — radial blur from sun screen position
// Reads depth buffer (sky = depth 1.0), outputs light shaft intensity

struct GodRayParams {
    sun_screen_pos: vec2<f32>,
    density: f32,
    weight: f32,
    decay: f32,
    exposure: f32,
    num_samples: f32,
    sun_visible: f32,
    sun_color: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var s_nearest: sampler;
@group(0) @binding(2) var<uniform> params: GodRayParams;
@group(0) @binding(3) var t_cloud: texture_2d<f32>;    // cloud buffer (A = transmittance)
@group(0) @binding(4) var s_cloud: sampler;             // filtering sampler for quarter-res

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
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (params.sun_visible < 0.001) {
        return vec4(0.0);
    }

    // Radial falloff: pixels far from the sun on screen produce weaker rays
    let to_sun = in.tex_coords - params.sun_screen_pos;
    let dist_to_sun = length(to_sun);
    let radial_falloff = max(1.0 - dist_to_sun * 1.2, 0.0);
    if (radial_falloff < 0.001) {
        return vec4(0.0);
    }

    let num_samples = min(i32(params.num_samples), 16);
    let delta_uv = to_sun * params.density / f32(num_samples);

    var uv = in.tex_coords;
    var illumination_decay = 1.0;
    var color = vec3<f32>(0.0);

    for (var i = 0; i < num_samples; i++) {
        uv -= delta_uv;

        // Clamp to [0,1] to avoid wrapping artifacts
        let clamped_uv = clamp(uv, vec2(0.001), vec2(0.999));

        // Sample depth buffer — sky pixels have depth==0.0 (reverse-Z clears to 0.0)
        let sample_depth = textureSample(t_depth, s_nearest, clamped_uv);

        // Sky contributes light, geometry and clouds occlude
        let is_sky = step(sample_depth, 0.001);
        let cloud_t = textureSample(t_cloud, s_cloud, clamped_uv).a; // cloud transmittance
        color += is_sky * cloud_t * params.weight * illumination_decay;
        illumination_decay *= params.decay;
    }

    color *= params.exposure * params.sun_color * params.sun_visible * radial_falloff;

    return vec4(color, 1.0);
}
