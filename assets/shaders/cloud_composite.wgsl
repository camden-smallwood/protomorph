// Cloud composite — non-linear transmittance + bilateral upscale + temporal reprojection
// Uses Unity HDRP technique: apply transmittance in tonemapped luminance space
// to preserve sun disc visibility through clouds.

@group(0) @binding(0) var cloud_quarter: texture_2d<f32>;
@group(0) @binding(1) var cloud_history: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var samp_filter: sampler;
@group(0) @binding(4) var samp_nearest: sampler;

struct CompositeParams {
    prev_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    quarter_texel_w: f32,
    quarter_texel_h: f32,
    frame_index: u32,
    _pad: vec2<f32>,
};
@group(0) @binding(5) var<uniform> params: CompositeParams;
@group(0) @binding(6) var scene_copy: texture_2d<f32>; // pre-cloud scene (lighting_base copy)
@group(0) @binding(7) var god_rays_tex: texture_2d<f32>; // god rays (half-res)

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct FragmentOutput {
    @location(0) scene: vec4<f32>,
    @location(1) history: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// Reinhard tonemap + inverse for non-linear transmittance
fn fast_tonemap(x: f32) -> f32 { return x / (1.0 + x); }
fn fast_tonemap_inv(x: f32) -> f32 { return x / (1.0 - x); }

// Non-linear transmittance (Unity HDRP technique)
// Applies transmittance in tonemapped luminance space so bright features (sun)
// maintain perceptual contrast through clouds.
fn non_linear_transmittance(scene_color: vec3<f32>, linear_t: f32) -> f32 {
    let lum = dot(scene_color, vec3(0.2126, 0.7152, 0.0722));
    if (lum < 0.001) { return linear_t; }

    let tonemapped_lum = fast_tonemap(lum);
    let attenuated_lum = tonemapped_lum * linear_t;
    let restored_lum = fast_tonemap_inv(clamp(attenuated_lum, 0.0, 0.999));
    let perceptual_t = max(restored_lum / lum, pow(linear_t, 6.0));

    return mix(linear_t, perceptual_t, 0.8);
}

// Bilateral upsample from quarter-res
fn bilateral_upsample(uv: vec2<f32>) -> vec4<f32> {
    let texel = vec2(params.quarter_texel_w, params.quarter_texel_h);
    let offsets = array<vec2<f32>, 4>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5), vec2(0.5, 0.5),
    );
    var total = vec4(0.0);
    for (var i = 0; i < 4; i++) {
        total += textureSample(cloud_quarter, samp_filter, uv + offsets[i] * texel);
    }
    return total * 0.25;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let uv = in.tex_coords;
    var out: FragmentOutput;

    let depth = textureSample(depth_tex, samp_nearest, uv);
    let current_cloud = bilateral_upsample(uv);

    // Read pre-cloud scene for non-linear transmittance
    let scene_color = textureSample(scene_copy, samp_filter, uv).rgb;

    // God rays (added to all pixels)
    let god_rays = textureSample(god_rays_tex, samp_filter, uv).rgb;

    // Geometry pixels: no cloud compositing
    if (depth > 0.001) {
        out.scene = vec4(scene_color + god_rays, 1.0);
        out.history = current_cloud;
        return out;
    }

    // --- Temporal reprojection ---
    var history_cloud = current_cloud;
    var blend_factor = 0.9;

    {
        let ndc_full = vec4(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0, 1.0);
        let world_far = params.inverse_view_projection * ndc_full;
        let ray_dir = normalize(world_far.xyz / world_far.w - params.camera_position);
        let far_point = params.camera_position + ray_dir * 100000.0;
        let prev_clip = params.prev_view_projection * vec4(far_point, 1.0);

        if (prev_clip.w > 0.0) {
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            let prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, -prev_ndc.y * 0.5 + 0.5);

            if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
                let history_sample = textureSample(cloud_history, samp_filter, prev_uv);
                let c_min = current_cloud - vec4(0.1);
                let c_max = current_cloud + vec4(0.1);
                history_cloud = clamp(history_sample, c_min, c_max);
            } else {
                blend_factor = 0.0;
            }
        } else {
            blend_factor = 0.0;
        }
    }

    if (params.frame_index == 0u) { blend_factor = 0.0; }

    let blended = mix(current_cloud, history_cloud, blend_factor);
    let scattered = blended.rgb;
    let linear_transmittance = blended.a;

    // Non-linear transmittance: preserves sun disc through clouds
    let adjusted_t = non_linear_transmittance(scene_color, linear_transmittance);

    // Composite: scattered + scene * adjusted_transmittance
    let composited = scattered + scene_color * adjusted_t;

    out.scene = vec4(composited + god_rays, 1.0);
    out.history = blended;

    return out;
}
