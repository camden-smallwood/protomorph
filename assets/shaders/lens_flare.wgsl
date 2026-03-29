// Screen-space lens flare — procedural ghost discs along the flare axis
// Ghosts are placed explicitly along the sun→center→opposite line,
// with varying sizes, ring shapes, and colors (Halo-style).

@group(0) @binding(0) var t_bloom: texture_2d<f32>;
@group(0) @binding(1) var s_filtering: sampler;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var s_nearest: sampler;

struct FlareParams {
    sun_screen_pos: vec2<f32>,
    sun_visible: f32,
    aspect_ratio: f32,
    chroma_shift: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(4) var<uniform> params: FlareParams;

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

// Ghost disc shape — blend between filled disc and ring
fn ghost_shape(norm_d: f32, ring_factor: f32) -> f32 {
    let disc = 1.0 - smoothstep(0.6, 1.0, norm_d);
    let ring = smoothstep(0.4, 0.75, norm_d) * (1.0 - smoothstep(0.75, 1.0, norm_d));
    return mix(disc, ring, ring_factor);
}

const NUM_GHOSTS: i32 = 7;
// Position along flare axis: 0 = at sun, 1 = at screen center, 2 = opposite side
const GHOST_OFFSETS: array<f32, 7> = array<f32, 7>(
    0.2, 0.45, 0.7, 1.2, 1.5, 1.8, 2.15
);
// Radius in screen-height fraction
const GHOST_RADII: array<f32, 7> = array<f32, 7>(
    0.025, 0.04, 0.015, 0.055, 0.02, 0.07, 0.035
);
// 0 = filled disc, 1 = thin ring
const GHOST_RING: array<f32, 7> = array<f32, 7>(
    0.6, 0.4, 0.0, 0.8, 0.0, 0.7, 0.3
);
// Intensity per ghost
const GHOST_INTENSITY: array<f32, 7> = array<f32, 7>(
    0.08, 0.05, 0.12, 0.06, 0.10, 0.04, 0.05
);
// Tint colors
const GHOST_TINTS: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
    vec3(1.0, 0.85, 0.5),  // warm gold
    vec3(0.85, 0.55, 0.3), // brown/orange
    vec3(0.3, 0.9, 0.45),  // green
    vec3(0.75, 0.5, 0.35), // brown ring
    vec3(0.4, 0.8, 1.0),   // blue
    vec3(0.85, 0.6, 0.35), // amber ring
    vec3(0.6, 0.7, 1.0),   // lavender
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (params.sun_visible < 0.001) {
        return vec4(0.0);
    }

    // Sun occlusion check — multi-sample disc for smooth partial occlusion
    let sun_clamped = clamp(params.sun_screen_pos, vec2(0.01), vec2(0.99));
    let sample_radius = 0.015; // screen-space radius for occlusion sampling
    var occlusion_sum = 0.0;
    let offsets = array<vec2<f32>, 9>(
        vec2( 0.0,  0.0),
        vec2( 1.0,  0.0), vec2(-1.0,  0.0),
        vec2( 0.0,  1.0), vec2( 0.0, -1.0),
        vec2( 0.707,  0.707), vec2(-0.707,  0.707),
        vec2( 0.707, -0.707), vec2(-0.707, -0.707),
    );
    for (var s = 0; s < 9; s++) {
        let sample_uv = clamp(sun_clamped + offsets[s] * sample_radius, vec2(0.01), vec2(0.99));
        let d = textureSample(t_depth, s_nearest, sample_uv);
        occlusion_sum += step(d, 0.001);
    }
    let occlusion = occlusion_sum / 9.0;
    let visibility = params.sun_visible * occlusion;

    if (visibility < 0.001) {
        return vec4(0.0);
    }

    let uv = in.tex_coords;
    let center = vec2(0.5);
    let flare_axis = center - params.sun_screen_pos;
    let axis_len = length(flare_axis);

    // Sample bloom brightness at sun position to modulate overall intensity
    let sun_bloom = textureSample(t_bloom, s_filtering, sun_clamped).rgb;
    let bloom_intensity = max(max(sun_bloom.r, sun_bloom.g), sun_bloom.b);
    let bloom_factor = smoothstep(0.2, 1.5, bloom_intensity);

    var result = vec3<f32>(0.0);
    let chroma_dir = select(normalize(flare_axis), vec2(0.0), axis_len < 0.001);

    // --- Procedural ghost discs ---
    for (var i = 0; i < NUM_GHOSTS; i++) {
        let ghost_center = params.sun_screen_pos + flare_axis * GHOST_OFFSETS[i];

        // Aspect-corrected distance so ghosts are circular
        let delta = (uv - ghost_center) * vec2(params.aspect_ratio, 1.0);
        let d = length(delta);
        let radius = GHOST_RADII[i];

        // Shape
        let shape = ghost_shape(d / radius, GHOST_RING[i]);
        if (shape < 0.001) {
            continue;
        }

        // Chromatic aberration — shift R and B along flare axis
        let delta_r = (uv + chroma_dir * params.chroma_shift - ghost_center) * vec2(params.aspect_ratio, 1.0);
        let delta_b = (uv - chroma_dir * params.chroma_shift - ghost_center) * vec2(params.aspect_ratio, 1.0);
        let shape_r = ghost_shape(length(delta_r) / radius, GHOST_RING[i]);
        let shape_b = ghost_shape(length(delta_b) / radius, GHOST_RING[i]);

        let chroma_color = vec3(shape_r, shape, shape_b);

        // Fade ghosts near screen edges
        let edge_d = max(abs(ghost_center.x - 0.5), abs(ghost_center.y - 0.5));
        let edge_fade = 1.0 - smoothstep(0.35, 0.5, edge_d);

        result += chroma_color * GHOST_TINTS[i] * GHOST_INTENSITY[i] * edge_fade;
    }

    // Fade when looking directly at sun (ghosts overlap and look bad)
    let sun_center_dist = distance(params.sun_screen_pos, center);
    let direct_look_fade = smoothstep(0.02, 0.15, sun_center_dist);

    result *= visibility * direct_look_fade * bloom_factor;

    return vec4(result, 0.0);
}
