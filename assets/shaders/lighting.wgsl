// Deferred lighting pass — reads compact G-buffer, Cook-Torrance (Halo 3), up to 16 lights
// Port of C's lighting.fs
// G-buffer layout: position_depth, normal (w=emissive_luma), albedo_specular, material, ssao

const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// Shadow constants — must match gpu_types.rs
const SHADOW_MAP_SIZE_F: f32 = 1024.0;
const SHADOW_PCF_SPREAD: f32 = 3.0;
const SHADOW_NORMAL_BIAS_SCALE: f32 = 4.0;

// Group 0: G-buffer textures (5 textures + sampler)
@group(0) @binding(0) var t_position_depth: texture_2d<f32>;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var t_albedo_specular: texture_2d<f32>;
@group(0) @binding(3) var t_material: texture_2d<f32>;
@group(0) @binding(4) var t_ssao: texture_2d<f32>;
@group(0) @binding(5) var s_nearest: sampler;

// Group 1: Lighting uniforms
struct GpuLightData {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    constant_atten: f32,
    diffuse_color: vec3<f32>,
    linear_atten: f32,
    ambient_color: vec3<f32>,
    quadratic_atten: f32,
    specular_color: vec3<f32>,
    inner_cutoff: f32,
    outer_cutoff: f32,
    shadow_index: i32,
    _pad1: f32,
    _pad2: f32,
};

const MAX_LIGHTS: u32 = 16u;

struct LightingUniforms {
    camera_position: vec3<f32>,
    light_count: u32,
    camera_direction: vec3<f32>,
    _pad: f32,
    lights: array<GpuLightData, 16>,
};
@group(1) @binding(0) var<uniform> lighting: LightingUniforms;

// Group 2: Shadow maps
@group(2) @binding(0) var t_shadow_cube_0: texture_depth_cube;
@group(2) @binding(1) var t_shadow_cube_1: texture_depth_cube;
@group(2) @binding(2) var t_shadow_2d_0: texture_depth_2d;
@group(2) @binding(3) var t_shadow_2d_1: texture_depth_2d;
@group(2) @binding(4) var s_shadow_compare: sampler_comparison;

struct ShadowData {
    point_params: array<vec4<f32>, 2>,
    spot_view_proj: array<mat4x4<f32>, 2>,
    spot_params: array<vec4<f32>, 2>,
};
@group(2) @binding(5) var<uniform> shadow_data: ShadowData;

// 8-sample Poisson disk for PCF (unit disk, values in [-1, 1])
const POISSON_DISK: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
);

// Per-pixel pseudo-random rotation angle to break up PCF banding
fn random_angle(screen_pos: vec2<f32>) -> f32 {
    return fract(sin(dot(screen_pos, vec2(12.9898, 78.233))) * 43758.5453) * 6.283185;
}

fn rotate_offset(offset: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
}

// Push shadow lookup position along surface normal to prevent self-shadowing.
// Surfaces nearly parallel to the light get more offset; surfaces facing the light get less.
fn apply_normal_offset(
    frag_pos: vec3<f32>,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    dist: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let texel_world = 2.0 * dist / SHADOW_MAP_SIZE_F;
    return frag_pos + frag_normal * (1.0 - n_dot_l) * texel_world * SHADOW_NORMAL_BIAS_SCALE;
}

// Fullscreen quad vertex
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

fn calculate_point_shadow(
    frag_pos: vec3<f32>,
    light_pos: vec3<f32>,
    shadow_idx: u32,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
) -> f32 {
    let params = shadow_data.point_params[shadow_idx];
    let near = params.x;
    let far = params.y;

    // Normal offset bias
    let dist = length(frag_pos - light_pos);
    let biased_pos = apply_normal_offset(frag_pos, frag_normal, light_dir, dist);
    let d = biased_pos - light_pos;

    let max_comp = max(max(abs(d.x), abs(d.y)), abs(d.z));
    let reference = far * (max_comp - near) / (max_comp * (far - near));

    // Build tangent frame perpendicular to the cubemap lookup direction
    let d_norm = normalize(d);
    var up = vec3(0.0, 1.0, 0.0);
    if (abs(d_norm.y) > 0.99) {
        up = vec3(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(d_norm, up));
    let bitangent = cross(d_norm, tangent);

    // PCF disk radius scales with distance so softness is consistent in world space
    let disk_radius = dist * SHADOW_PCF_SPREAD / SHADOW_MAP_SIZE_F;
    let rotation = random_angle(screen_pos);

    var shadow = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = rotate_offset(POISSON_DISK[i], rotation);
        let sample_d = d + (tangent * p.x + bitangent * p.y) * disk_radius;
        if (shadow_idx == 0u) {
            shadow += textureSampleCompareLevel(t_shadow_cube_0, s_shadow_compare, sample_d, reference);
        } else {
            shadow += textureSampleCompareLevel(t_shadow_cube_1, s_shadow_compare, sample_d, reference);
        }
    }
    return shadow / 8.0;
}

fn calculate_spot_shadow(
    frag_pos: vec3<f32>,
    light_pos: vec3<f32>,
    shadow_idx: u32,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
) -> f32 {
    let view_proj = shadow_data.spot_view_proj[shadow_idx];

    // Normal offset bias
    let dist = length(light_pos - frag_pos);
    let biased_pos = apply_normal_offset(frag_pos, frag_normal, light_dir, dist);

    let frag_light_space = view_proj * vec4(biased_pos, 1.0);

    // Fragment behind the light — no shadow
    if (frag_light_space.w <= 0.0) {
        return 1.0;
    }

    let proj = frag_light_space.xyz / frag_light_space.w;

    // NDC XY [-1,1] -> UV [0,1], flip Y for wgpu texture coords
    let uv = vec2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);

    // Out of range = not in shadow
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    // Rotated Poisson disk PCF
    let texel_size = 1.0 / SHADOW_MAP_SIZE_F;
    let rotation = random_angle(screen_pos);

    var shadow = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = rotate_offset(POISSON_DISK[i], rotation);
        let offset_uv = uv + p * SHADOW_PCF_SPREAD * texel_size;
        if (shadow_idx == 0u) {
            shadow += textureSampleCompareLevel(t_shadow_2d_0, s_shadow_compare, offset_uv, proj.z);
        } else {
            shadow += textureSampleCompareLevel(t_shadow_2d_1, s_shadow_compare, offset_uv, proj.z);
        }
    }
    return shadow / 8.0;
}

// Beckmann NDF — cook_torrance.fx lines 280-284
fn beckmann_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let r2 = roughness * roughness;
    let c2 = n_dot_h * n_dot_h;
    let c4 = c2 * c2;
    return exp((c2 - 1.0) / (r2 * c2)) / (r2 * c4 + 0.00001);
}

// Cook-Torrance geometry — cook_torrance.fx line 271
// Note: saturate(v_dot_h) in denominator, not raw v_dot_h
fn ct_geometry(n_dot_h: f32, n_dot_v: f32, n_dot_l: f32, v_dot_h: f32) -> f32 {
    return 2.0 * n_dot_h * min(n_dot_v, n_dot_l) / (saturate(v_dot_h) + 0.00001);
}

// Exact Fresnel — cook_torrance.fx lines 273-278
// Derives IOR from f0, computes full Fresnel (not Schlick)
fn fresnel_exact(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let sqrt_f0 = sqrt(clamp(f0, vec3(0.0), vec3(0.999)));
    let n = (vec3(1.0) + sqrt_f0) / (vec3(1.0) - sqrt_f0);
    let g = sqrt(max(n * n + vec3(v_dot_h * v_dot_h) - vec3(1.0), vec3(0.0)));
    let gpc = g + vec3(v_dot_h);
    let gmc = g - vec3(v_dot_h);
    let r = (vec3(v_dot_h) * gpc - vec3(1.0)) / (vec3(v_dot_h) * gmc + vec3(1.0));
    return clamp(0.5 * ((gmc * gmc) / (gpc * gpc + vec3(0.00001))) * (vec3(1.0) + r * r), vec3(0.0), vec3(1.0));
}

// Single HDR output — bloom prefilter handles brightness extraction
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    let frag_position = textureSample(t_position_depth, s_nearest, uv).rgb;
    let normal_sample = textureSample(t_normal, s_nearest, uv);
    let frag_normal = normal_sample.rgb;
    let emissive_luma = normal_sample.w;  // emissive luminance stored in normal.w (f16, HDR range)
    let albedo_specular = textureSample(t_albedo_specular, s_nearest, uv);

    let mat_sample = textureSample(t_material, s_nearest, uv);
    let material_ambient_amount = mat_sample.r;
    let material_specular_amount = mat_sample.g;
    let roughness = max(mat_sample.b, 0.05);  // Halo 3: max(roughness, 0.05)

    let ambient_occlusion = textureSample(t_ssao, s_nearest, uv).r;

    // Standard per-fragment view direction (correct for all light types)
    let view_direction = normalize(lighting.camera_position - frag_position);
    let screen_pos = in.clip_position.xy;

    var light_color = vec3<f32>(0.0);

    for (var i = 0u; i < lighting.light_count; i++) {
        let light = lighting.lights[i];

        var light_direction: vec3<f32>;
        if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
            light_direction = normalize(-light.direction);
        } else {
            light_direction = normalize(light.position - frag_position);
        }

        let halfway_direction = normalize(light_direction + view_direction);

        // Ambient (per-light, scaled by AO)
        var ambient_color = material_ambient_amount * light.ambient_color * albedo_specular.rgb * ambient_occlusion;

        // Diffuse — unchanged
        let diffuse_amount = max(dot(light_direction, frag_normal), 0.0);
        var diffuse_color = diffuse_amount * albedo_specular.rgb * light.diffuse_color;

        // Specular — Cook-Torrance (Halo 3 cook_torrance.fx)
        var specular_color = vec3<f32>(0.0);
        let n_dot_l = dot(frag_normal, light_direction);
        let n_dot_v = dot(frag_normal, view_direction);
        if (min(n_dot_l, n_dot_v) > 0.0) {
            let n_dot_h = max(dot(frag_normal, halfway_direction), 0.0);
            let v_dot_h = max(dot(view_direction, halfway_direction), 0.0);

            let D = beckmann_ndf(n_dot_h, roughness);
            let G = ct_geometry(n_dot_h, n_dot_v, n_dot_l, v_dot_h);
            let F = fresnel_exact(vec3(0.04), v_dot_h);  // f0=0.04 dielectric

            // Halo 3 denominator: pi * NdotV (not standard 4 * NdotL * NdotV)
            let ct = D * saturate(G) / (3.14159 * n_dot_v + 0.00001) * F;

            // Anti-shadow clamp — cook_torrance.fx line 298
            let clamped = min(ct, vec3(n_dot_l + 1.0));

            // specular_amount and spec_tex scale the result (NOT used as f0)
            specular_color = material_specular_amount * albedo_specular.a * clamped * light.specular_color;
        }

        // Attenuation for non-directional lights
        if (light.light_type != LIGHT_TYPE_DIRECTIONAL) {
            let light_distance = length(light.position - frag_position);
            var light_attenuation = 1.0 / (light.constant_atten + light.linear_atten * light_distance + light.quadratic_atten * (light_distance * light_distance));

            // Spot light cone falloff
            if (light.light_type == LIGHT_TYPE_SPOT) {
                let light_theta = dot(light_direction, normalize(-light.direction));
                let light_epsilon = light.inner_cutoff - light.outer_cutoff;
                let light_intensity = clamp((light_theta - light.outer_cutoff) / light_epsilon, 0.0, 1.0);
                light_attenuation *= light_intensity;
            }

            ambient_color *= light_attenuation;
            diffuse_color *= light_attenuation;
            specular_color *= light_attenuation;
        }

        // Shadow (affects diffuse + specular, not ambient)
        var shadow_factor = 1.0;
        if (light.shadow_index >= 0) {
            let si = u32(light.shadow_index);
            if (light.light_type == LIGHT_TYPE_POINT) {
                shadow_factor = calculate_point_shadow(frag_position, light.position, si, frag_normal, light_direction, screen_pos);
            } else if (light.light_type == LIGHT_TYPE_SPOT) {
                shadow_factor = calculate_spot_shadow(frag_position, light.position, si, frag_normal, light_direction, screen_pos);
            }
        }
        diffuse_color *= shadow_factor;
        specular_color *= shadow_factor;

        light_color += ambient_color + diffuse_color + specular_color;
    }

    // Emissive: luminance from normal.w, tinted by albedo color (added once)
    // HDR boost so emissive exceeds bloom threshold and produces visible glow
    let emissive_color = albedo_specular.rgb * emissive_luma * 5.0;
    light_color += emissive_color;

    return vec4<f32>(light_color, 1.0);
}
