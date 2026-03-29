// HDR + bloom composite with ACES tone mapping, saturation control, film grain
// Reads the HDR lighting buffer and bloom texture, composites, tone maps, outputs to sRGB swapchain

@group(0) @binding(0) var t_base: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var s_filtering: sampler;

struct CompositeParams {
    bloom_strength: f32,
    exposure: f32,
    saturation: f32,
    grain_intensity: f32,
    vignette_strength: f32,
    sun_screen_x: f32,
    sun_screen_y: f32,
    sun_visible: f32,
};
@group(0) @binding(3) var<uniform> params: CompositeParams;
@group(0) @binding(4) var t_depth: texture_depth_2d;
@group(0) @binding(5) var s_nearest: sampler;

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

// ACES Narkowicz approximation
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Hash-based noise for film grain
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(t_base, s_filtering, in.tex_coords).rgb;
    let bloom = textureSample(t_bloom, s_filtering, in.tex_coords).rgb;

    let hdr = base + bloom * params.bloom_strength;
    var color = aces_tonemap(hdr * params.exposure);

    // Post-tonemap sun disc — analytical, ADDITIVE, masked by depth
    if (params.sun_visible > 0.001) {
        let pixel_depth = textureSample(t_depth, s_nearest, in.tex_coords);
        let pixel_is_sky = step(pixel_depth, 0.001); // 1.0 if sky, 0.0 if geometry

        let sun_uv = vec2(params.sun_screen_x, params.sun_screen_y);
        let to_sun = in.tex_coords - sun_uv;
        let dist = length(to_sun);

        // Hard bright core
        let disc = (1.0 - smoothstep(0.0, 0.012, dist)) * 0.5;
        // Medium glow
        let glow = exp(-dist * dist * 400.0) * 0.25;
        // Wide warm halo
        let halo = exp(-dist * dist * 60.0) * 0.08;

        let sun_add = (disc + glow + halo) * params.sun_visible * pixel_is_sky;
        color += vec3(sun_add, sun_add * 0.95, sun_add * 0.85);
        color = min(color, vec3(1.0));
    }

    // Saturation adjustment
    let luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luma), color, params.saturation);

    // Vignette — radial darkening toward screen edges
    if (params.vignette_strength > 0.0) {
        let uv_centered = in.tex_coords * 2.0 - 1.0;
        let vignette = 1.0 - params.vignette_strength * dot(uv_centered, uv_centered);
        color *= max(vignette, 0.0);
    }

    // Film grain
    if (params.grain_intensity > 0.0) {
        let noise = hash(in.clip_position.xy) * 2.0 - 1.0;
        let grain_amount = params.grain_intensity * (1.0 - luma * 0.5);
        color += vec3(noise * grain_amount);
    }

    return vec4(color, 1.0);
}
