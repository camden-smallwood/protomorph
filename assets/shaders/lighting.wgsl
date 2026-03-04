// Deferred lighting pass — reads compact G-buffer, Cook-Torrance (Halo 3), up to 16 lights
// Port of C's lighting.fs
// G-buffer layout: position_depth, normal (w=emissive_luma), albedo_specular, material, ssao

const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// Shadow constants — must match gpu_types.rs
const SHADOW_MAP_SIZE_F: f32 = 1024.0;
const SHADOW_PCF_SPREAD: f32 = 3.0;
const SHADOW_NORMAL_BIAS_SCALE: f32 = 3.0;

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

struct AtmosphereData {
    sun_direction: vec3<f32>,
    atmosphere_enable: f32,
    rayleigh_coefficients: vec3<f32>,
    rayleigh_height_scale: f32,
    mie_coefficient: f32,
    mie_height_scale: f32,
    mie_g: f32,
    max_fog_thickness: f32,
    inscatter_scale: f32,
    reference_height: f32,
    _pad: vec2<f32>,
};
@group(1) @binding(1) var<uniform> atmosphere: AtmosphereData;

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
    cascade_view_proj: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    cascade_texel_sizes: vec4<f32>,
};
@group(2) @binding(5) var<uniform> shadow_data: ShadowData;
@group(2) @binding(6) var t_shadow_cascade: texture_depth_2d_array;

// Group 3: Environment cubemap
@group(3) @binding(0) var t_env_cubemap: texture_cube<f32>;
@group(3) @binding(1) var s_env_filtering: sampler;

struct EnvProbeData {
    probe_position: vec3<f32>,
    env_roughness_scale: f32,
    env_specular_contribution: f32,
    env_mip_count: f32,
    env_intensity: f32,
    env_diffuse_intensity: f32,
};
@group(3) @binding(2) var<uniform> env_probe: EnvProbeData;

struct SHCoefficients {
    coefficients: array<vec4<f32>, 9>,
};
@group(3) @binding(3) var<uniform> sh_data: SHCoefficients;

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
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    return frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;
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
    let params = shadow_data.spot_params[shadow_idx];
    let spot_texel_size = params.y; // 1.0 / spot_shadow_map_size
    let view_proj = shadow_data.spot_view_proj[shadow_idx];

    // Normal offset bias — use actual spot map texel size
    let dist = length(light_pos - frag_pos);
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let texel_world = 2.0 * dist * spot_texel_size;
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    let biased_pos = frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;

    let frag_light_space = view_proj * vec4(biased_pos, 1.0);

    // Per-pixel slope bias replacing hardware slope_scale (smooth across triangle boundaries)
    let tan_theta = sqrt(1.0 - n_dot_l * n_dot_l) / max(n_dot_l, 0.001);
    let depth_bias = 0.001 + min(tan_theta, 10.0) * 0.0005;

    // Fragment behind the light — no shadow
    if (frag_light_space.w <= 0.0) {
        return 1.0;
    }

    let proj = frag_light_space.xyz / frag_light_space.w;

    // NDC XY [-1,1] -> UV [0,1], flip Y for wgpu texture coords
    let uv = vec2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
    // Shift reference depth toward light to fight acne at grazing angles
    let ref_z = proj.z - depth_bias;

    // Out of range = not in shadow
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ref_z < 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    // Rotated Poisson disk PCF
    let rotation = random_angle(screen_pos);

    var shadow = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = rotate_offset(POISSON_DISK[i], rotation);
        let offset_uv = uv + p * SHADOW_PCF_SPREAD * spot_texel_size;
        if (shadow_idx == 0u) {
            shadow += textureSampleCompareLevel(t_shadow_2d_0, s_shadow_compare, offset_uv, ref_z);
        } else {
            shadow += textureSampleCompareLevel(t_shadow_2d_1, s_shadow_compare, offset_uv, ref_z);
        }
    }
    return shadow / 8.0;
}

const CSM_MAP_SIZE_F: f32 = 2048.0;
const CSM_BLEND_FRACTION: f32 = 0.1; // blend in last 10% before each split boundary

fn sample_cascade_pcf(
    frag_pos: vec3<f32>,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
    cascade_idx: u32,
) -> f32 {
    let texel_world = shadow_data.cascade_texel_sizes[cascade_idx];
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    let biased_pos = frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;

    let view_proj = shadow_data.cascade_view_proj[cascade_idx];
    let frag_light_space = view_proj * vec4(biased_pos, 1.0);
    let proj = frag_light_space.xyz / frag_light_space.w;
    let uv = vec2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    // Rotated Poisson disk PCF
    let texel_size = 1.0 / CSM_MAP_SIZE_F;
    let rotation = random_angle(screen_pos);
    var shadow = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = rotate_offset(POISSON_DISK[i], rotation);
        let offset_uv = uv + p * SHADOW_PCF_SPREAD * texel_size;
        shadow += textureSampleCompareLevel(
            t_shadow_cascade, s_shadow_compare,
            offset_uv, cascade_idx, proj.z
        );
    }
    return shadow / 8.0;
}

fn calculate_directional_shadow(
    frag_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
) -> f32 {
    // Select cascade based on camera distance
    let dist = length(frag_pos - camera_pos);
    var cascade_idx = 0u;
    if (dist > shadow_data.cascade_splits.x) { cascade_idx = 1u; }
    if (dist > shadow_data.cascade_splits.y) { cascade_idx = 2u; }

    // Beyond last cascade — no shadow
    if (dist > shadow_data.cascade_splits.z) { return 1.0; }

    let shadow_current = sample_cascade_pcf(frag_pos, frag_normal, light_dir, screen_pos, cascade_idx);

    // Blend at cascade transitions: in the last 10% before each split boundary,
    // sample both current and next cascade and mix between them
    let splits = shadow_data.cascade_splits;
    var split_end = splits.x;
    if (cascade_idx == 1u) { split_end = splits.y; }
    if (cascade_idx == 2u) { split_end = splits.z; }

    let blend_start = split_end * (1.0 - CSM_BLEND_FRACTION);
    if (cascade_idx < 2u && dist > blend_start) {
        let shadow_next = sample_cascade_pcf(frag_pos, frag_normal, light_dir, screen_pos, cascade_idx + 1u);
        let blend_t = (dist - blend_start) / (split_end - blend_start);
        return mix(shadow_current, shadow_next, blend_t);
    }

    return shadow_current;
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

// Halo 3 lighting constants
const ANALYTICAL_SPECULAR_CONTRIBUTION: f32 = 0.5;  // other half from SH area specular
const AREA_SPECULAR_CONTRIBUTION: f32 = 0.5;        // SH-convolved BRDF contribution
const ALBEDO_BLEND: f32 = 0.0;  // 0=white specular, 1=albedo-tinted (metallic)
const ANTI_SHADOW_CONTROL: f32 = 0.1;  // 0=no attenuation, 1=aggressive shadow-edge kill

// Halo 3 rim fresnel — cook_torrance.fx lines 1403-1475
const RIM_FRESNEL_COEFFICIENT: f32 = 0.5;
const RIM_FRESNEL_POWER: f32 = 3.0;
const RIM_FRESNEL_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const RIM_FRESNEL_ALBEDO_BLEND: f32 = 0.3;

// Diffuse cosine-lobe ZH transfer coefficients (applied at eval time, not baked)
const A0_TRANSFER: f32 = 3.14159265;   // π
const A1_TRANSFER: f32 = 2.09439510;   // 2π/3
const A2_TRANSFER: f32 = 0.78539816;   // π/4

// SH basis normalization constants
const SH_Y00: f32 = 0.282095;   // 1/(2√π)
const SH_Y1X: f32 = 0.488603;   // √3/(2√π)
const SH_Y20: f32 = 0.315392;   // √5/(4√π) * (1/2)
const SH_Y21: f32 = 1.092548;   // √15/(2√π)
const SH_Y22: f32 = 0.546274;   // √15/(4√π)

// Evaluate L2 SH irradiance at a given normal direction.
// Applies diffuse cosine-lobe transfer (A_l) at evaluation time.
fn evaluate_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    // L0
    result += sh_data.coefficients[0].rgb * SH_Y00 * A0_TRANSFER;
    // L1
    result += sh_data.coefficients[1].rgb * SH_Y1X * n.y * A1_TRANSFER;
    result += sh_data.coefficients[2].rgb * SH_Y1X * n.z * A1_TRANSFER;
    result += sh_data.coefficients[3].rgb * SH_Y1X * n.x * A1_TRANSFER;
    // L2
    result += sh_data.coefficients[4].rgb * SH_Y21 * n.x * n.y * A2_TRANSFER;
    result += sh_data.coefficients[5].rgb * SH_Y21 * n.y * n.z * A2_TRANSFER;
    result += sh_data.coefficients[6].rgb * SH_Y20 * (3.0 * n.z * n.z - 1.0) * A2_TRANSFER;
    result += sh_data.coefficients[7].rgb * SH_Y21 * n.x * n.z * A2_TRANSFER;
    result += sh_data.coefficients[8].rgb * SH_Y22 * (n.x * n.x - n.y * n.y) * A2_TRANSFER;

    return max(result, vec3(0.0));
}

// Dominant light extraction from SH — entry_points.fx:752-755, spherical_harmonics.fx:130-150
// Returns (dominant_direction, dominant_intensity)
fn extract_dominant_light_from_sh() -> array<vec3<f32>, 2> {
    // Luminance-weighted L1 direction (Rec. 709 weights)
    // L1 basis order: [1]=Y1,-1(y), [2]=Y1,0(z), [3]=Y1,1(x)
    let lum_y = sh_data.coefficients[1].r * 0.212656 + sh_data.coefficients[1].g * 0.715158 + sh_data.coefficients[1].b * 0.0721856;
    let lum_z = sh_data.coefficients[2].r * 0.212656 + sh_data.coefficients[2].g * 0.715158 + sh_data.coefficients[2].b * 0.0721856;
    let lum_x = sh_data.coefficients[3].r * 0.212656 + sh_data.coefficients[3].g * 0.715158 + sh_data.coefficients[3].b * 0.0721856;

    let l1_lum = vec3<f32>(lum_x, lum_y, lum_z);
    let l1_len = length(l1_lum);

    if (l1_len < 0.0001) {
        return array<vec3<f32>, 2>(vec3(0.0, 0.0, 1.0), vec3(0.0));
    }

    let dominant_dir = normalize(l1_lum);

    // Estimate intensity: L1 magnitude relates to directional component
    // L1 coefficient for a directional light: L * Y1x * d_component
    // The magnitude of the luminance L1 vector ~ L * Y1x
    // So L ~ |L1_lum| / Y1x, and the dominant intensity per channel:
    let scale = 1.0 / SH_Y1X;
    let dominant_intensity = vec3<f32>(
        length(vec3<f32>(sh_data.coefficients[1].r, sh_data.coefficients[2].r, sh_data.coefficients[3].r)) * scale,
        length(vec3<f32>(sh_data.coefficients[1].g, sh_data.coefficients[2].g, sh_data.coefficients[3].g)) * scale,
        length(vec3<f32>(sh_data.coefficients[1].b, sh_data.coefficients[2].b, sh_data.coefficients[3].b)) * scale,
    );

    return array<vec3<f32>, 2>(dominant_dir, dominant_intensity);
}

// Diffuse SH with dominant light subtract+re-add — spherical_harmonics.fx:130-150
// Subtracts the dominant light's SH contribution, evaluates residual ambient,
// then re-adds dominant light as max(N.L, 0) * intensity * 0.281.
fn evaluate_sh_irradiance_with_dominant_light(
    n: vec3<f32>,
    dominant_dir: vec3<f32>,
    dominant_intensity: vec3<f32>,
) -> array<vec3<f32>, 2> {
    // Subtract dominant light from L0 and L1 coefficients
    // dir_eval = -Y1x * dominant_dir components (matching basis order)
    let dir_eval_y = -SH_Y1X * dominant_dir.y;
    let dir_eval_z = -SH_Y1X * dominant_dir.z;
    let dir_eval_x = -SH_Y1X * dominant_dir.x;

    // Modified coefficients (L0 + L1 only, per Halo's ravi_order_2_with_dominant_light)
    // Halo subtracts per-channel: constants[ch].xyz -= dir_eval.zxy * dominant_intensity[ch]
    // Our layout is transposed: [basis] = (R,G,B)
    var c0 = sh_data.coefficients[0].rgb - SH_Y00 * dominant_intensity;
    var c1 = sh_data.coefficients[1].rgb - vec3<f32>(dir_eval_y) * dominant_intensity;  // Y1,-1(y)
    var c2 = sh_data.coefficients[2].rgb - vec3<f32>(dir_eval_z) * dominant_intensity;  // Y1,0(z)
    var c3 = sh_data.coefficients[3].rgb - vec3<f32>(dir_eval_x) * dominant_intensity;  // Y1,1(x)

    // Evaluate residual ambient (L0+L1 only, Halo's order-2 evaluation)
    // Uses Ravi Ramamoorthi's constants: c4=0.886227, c2_ravi=0.511664
    let c4_ravi = 0.886227;
    let c2_ravi = 0.511664;

    var x1 = vec3<f32>(0.0);
    // Per-channel dot(normal, L1_per_channel) — transposed access
    x1 = vec3<f32>(
        n.y * c1.r + n.z * c2.r + n.x * c3.r,
        n.y * c1.g + n.z * c2.g + n.x * c3.g,
        n.y * c1.b + n.z * c2.b + n.x * c3.b,
    );

    let residual_ambient = (c4_ravi * c0 + (-2.0 * c2_ravi) * x1) / 3.14159265;

    // Re-add dominant light with sharp N.L falloff
    let n_dot_l = max(dot(dominant_dir, n), 0.0);
    let dominant_diffuse = n_dot_l * dominant_intensity * 0.281;

    let total_diffuse = max(residual_ambient + dominant_diffuse, vec3(0.0));

    return array<vec3<f32>, 2>(total_diffuse, dominant_diffuse);
}

// Halo 3 area specular — spherical_harmonics.fx:568-611
// Evaluates SH-convolved specular BRDF at the reflection direction.
// Uses roughness-dependent polynomial-fitted transfer coefficients.
//
// Protomorph stores SH as [basis_i] = (R,G,B,pad) while Halo expects
// per-channel [channel] = (basis values...). We transpose the access.
fn evaluate_sh_area_specular(reflection_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let roughness_sq = roughness * roughness;

    let c_dc = 0.282095;
    let c_linear = -(0.5128945834 + (-0.1407369526) * roughness + (-0.002660066620) * roughness_sq) * 0.60;
    let c_quadratic = -(0.7212524717 + (-0.5541015389) * roughness + 0.07960539966 * roughness_sq) * 0.5;

    // L0: x0 = coefficients[0].rgb (per-basis, already RGB)
    let x0 = sh_data.coefficients[0].rgb;

    // L1: Halo does dot(reflection_dir, per_channel_L1[ch].xyz)
    // Our layout: [1]=(R,G,B) for Y1,-1(y), [2] for Y1,0(z), [3] for Y1,1(x)
    // Per-channel linear: x1.r = r.y*coeff[1].r + r.z*coeff[2].r + r.x*coeff[3].r
    let x1 = vec3<f32>(
        reflection_dir.y * sh_data.coefficients[1].r + reflection_dir.z * sh_data.coefficients[2].r + reflection_dir.x * sh_data.coefficients[3].r,
        reflection_dir.y * sh_data.coefficients[1].g + reflection_dir.z * sh_data.coefficients[2].g + reflection_dir.x * sh_data.coefficients[3].g,
        reflection_dir.y * sh_data.coefficients[1].b + reflection_dir.z * sh_data.coefficients[2].b + reflection_dir.x * sh_data.coefficients[3].b,
    );

    // L2 quadratic: cross-component products for Y2,-2, Y2,-1, Y2,1
    let quadratic_a = reflection_dir.xyz * reflection_dir.yzx;
    // Per-channel: x2.r = dot(quadratic_a, (coeff[4].r, coeff[5].r, coeff[7].r))
    // Basis indices: [4]=Y2,-2(xy), [5]=Y2,-1(yz), [7]=Y2,1(xz)
    let x2 = vec3<f32>(
        quadratic_a.x * sh_data.coefficients[4].r + quadratic_a.y * sh_data.coefficients[5].r + quadratic_a.z * sh_data.coefficients[7].r,
        quadratic_a.x * sh_data.coefficients[4].g + quadratic_a.y * sh_data.coefficients[5].g + quadratic_a.z * sh_data.coefficients[7].g,
        quadratic_a.x * sh_data.coefficients[4].b + quadratic_a.y * sh_data.coefficients[5].b + quadratic_a.z * sh_data.coefficients[7].b,
    );

    // Squared terms for Y2,0 and Y2,2
    let quadratic_b = vec4<f32>(
        reflection_dir.x * reflection_dir.x,
        reflection_dir.y * reflection_dir.y,
        reflection_dir.z * reflection_dir.z,
        1.0 / 3.0,
    );
    // Basis indices: [8]=Y2,2(x²-y²), [6]=Y2,0(3z²-1)
    // Halo packs these as (x², y², -z²*sqrt3, z²*sqrt3) — but in the new_phong_3 path
    // it's simpler: just dot(quadratic_b, per_channel_constants[7..9])
    // For our layout: x3.r = r.x²*coeff[8].r + r.y²*(-coeff[8].r) + stuff for coeff[6]
    // Actually, Halo's pack_constants_texture_array maps:
    //   constants[7] = (-sh[8].r, sh[8].r, -sh[6].r*sqrt3, sh[6].r*sqrt3)
    //   etc. for g, b channels
    // But calculate_area_specular_new_phong_3 uses a DIFFERENT layout than sh_glossy_ct.
    // It directly dots quadratic_b=(x²,y²,z²,1/3) against constants[7..9].
    // In Halo's pack_constants_texture_array:
    //   constants[7].rgba = (-sh8.r, sh8.r, -sh6.r*√3, sh6.r*√3)
    // So: dot(quadratic_b, constants[7]) = -sh8.r*x² + sh8.r*y² - sh6.r*√3*z² + sh6.r*√3/3
    //   = sh8.r*(y²-x²) + sh6.r*√3*(1/3 - z²)
    //   = -sh8.r*(x²-y²) - sh6.r*√3*(z² - 1/3)
    // Our basis: coeff[8] stores Y22 = 0.546274*(x²-y²), coeff[6] stores Y20 = 0.315392*(3z²-1)
    // The raw SH coefficient for Y22 basis is coeff[8]/Y22_norm, similarly for Y20.
    // But since we store L*Y_lm, the coefficients already include the basis function.
    // Let me just directly compute the L2 quadratic contribution using our basis:
    let x3 = vec3<f32>(
        sh_data.coefficients[6].r * (3.0 * reflection_dir.z * reflection_dir.z - 1.0)
            + sh_data.coefficients[8].r * (reflection_dir.x * reflection_dir.x - reflection_dir.y * reflection_dir.y),
        sh_data.coefficients[6].g * (3.0 * reflection_dir.z * reflection_dir.z - 1.0)
            + sh_data.coefficients[8].g * (reflection_dir.x * reflection_dir.x - reflection_dir.y * reflection_dir.y),
        sh_data.coefficients[6].b * (3.0 * reflection_dir.z * reflection_dir.z - 1.0)
            + sh_data.coefficients[8].b * (reflection_dir.x * reflection_dir.x - reflection_dir.y * reflection_dir.y),
    );

    return max(x0 * c_dc + x1 * c_linear + x2 * c_quadratic + x3 * c_quadratic, vec3(0.0));
}

// Environment cubemap sampling — environment_mapping.fx
fn sample_environment(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    fresnel_f0: f32,
) -> vec3<f32> {
    let R = reflect(-view_dir, normal);
    let lod = roughness * env_probe.env_roughness_scale * (env_probe.env_mip_count - 1.0);
    let env_color = textureSampleLevel(t_env_cubemap, s_env_filtering, R, lod).rgb;

    // Schlick approximation for environment fresnel
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = fresnel_f0 + (1.0 - fresnel_f0) * pow(1.0 - n_dot_v, 5.0);

    return env_color * fresnel * env_probe.env_specular_contribution * env_probe.env_intensity;
}

// Atmospheric scattering — atmosphere.fx lines 31-107
fn compute_scattering(camera_pos: vec3<f32>, world_pos: vec3<f32>) -> array<vec3<f32>, 2> {
    let ray = world_pos - camera_pos;
    var dist = length(ray);
    dist = min(dist, atmosphere.max_fog_thickness);

    if (dist < 0.001) {
        return array<vec3<f32>, 2>(vec3(1.0), vec3(0.0));
    }

    let ray_dir = normalize(ray);

    // Height-based density (atmosphere.fx lines 45-85)
    let h_cam = max(camera_pos.z - atmosphere.reference_height, 0.0);
    let h_frag = max(world_pos.z - atmosphere.reference_height, 0.0);
    let diff = h_cam - h_frag;

    var rayleigh_depth: f32;
    var mie_depth: f32;

    if (abs(diff) > 0.001) {
        // Analytic integral of exponential density along ray
        rayleigh_depth = (exp(-h_frag / atmosphere.rayleigh_height_scale)
                       - exp(-h_cam / atmosphere.rayleigh_height_scale))
                       * dist * atmosphere.rayleigh_height_scale / diff;
        mie_depth = (exp(-h_frag / atmosphere.mie_height_scale)
                  - exp(-h_cam / atmosphere.mie_height_scale))
                  * dist * atmosphere.mie_height_scale / diff;
    } else {
        // Nearly horizontal ray — use average height
        rayleigh_depth = exp(-h_cam / atmosphere.rayleigh_height_scale) * dist;
        mie_depth = exp(-h_cam / atmosphere.mie_height_scale) * dist;
    }

    // Extinction (Beer's law)
    let extinction = exp(-(atmosphere.rayleigh_coefficients * rayleigh_depth
                         + vec3(atmosphere.mie_coefficient) * mie_depth));

    // Phase functions
    let cos_theta = dot(ray_dir, normalize(atmosphere.sun_direction));

    // Rayleigh: (3/16π)(1 + cos²θ)
    let rayleigh_phase = 0.05968 * (1.0 + cos_theta * cos_theta);

    // Mie: Henyey-Greenstein
    let g = atmosphere.mie_g;
    let g2 = g * g;
    let mie_phase = 0.07958 * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);

    // Inscatter
    let scatter = vec3(1.0) - extinction;
    let inscatter = (atmosphere.rayleigh_coefficients * rayleigh_phase
                   + vec3(atmosphere.mie_coefficient) * mie_phase)
                   * scatter * atmosphere.inscatter_scale;

    return array<vec3<f32>, 2>(extinction, inscatter);
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
    let fresnel_f0 = mat_sample.a;

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
            let F = fresnel_exact(vec3(fresnel_f0), v_dot_h);

            // Standard Cook-Torrance denominator: 4 * NdotV (post NdotL cancellation)
            let ct = D * saturate(G) / (4.0 * n_dot_v + 0.00001) * F;

            // Anti-shadow clamp — cook_torrance.fx line 298, tunable control
            let clamped = min(ct, vec3(n_dot_l + (1.0 - ANTI_SHADOW_CONTROL)));

            // Albedo blend: interpolate specular tint between white and surface albedo
            let spec_tint = mix(vec3(1.0), albedo_specular.rgb, ALBEDO_BLEND);

            specular_color = ANALYTICAL_SPECULAR_CONTRIBUTION * material_specular_amount * albedo_specular.a * (clamped * spec_tint) * light.specular_color;
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
            } else if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
                shadow_factor = calculate_directional_shadow(
                    frag_position, lighting.camera_position,
                    frag_normal, light_direction, screen_pos
                );
            }
        }
        diffuse_color *= shadow_factor;
        specular_color *= shadow_factor;

        light_color += ambient_color + diffuse_color + specular_color;
    }

    // Environment cubemap specular
    if (env_probe.env_specular_contribution > 0.0) {
        let envmap = sample_environment(frag_normal, view_direction, roughness, fresnel_f0);
        light_color += envmap * material_specular_amount * albedo_specular.a;
    }

    // Area specular — SH-convolved BRDF (the other half of frequency-decomposed specular)
    if (env_probe.env_intensity > 0.0) {
        let reflection = reflect(-view_direction, frag_normal);
        let area_spec = evaluate_sh_area_specular(reflection, roughness);
        let spec_tint = mix(vec3(1.0), albedo_specular.rgb, ALBEDO_BLEND);
        light_color += area_spec * AREA_SPECULAR_CONTRIBUTION * material_specular_amount
                     * albedo_specular.a * spec_tint;
    }

    // Diffuse indirect — SH with dominant light subtract+re-add
    // Extracts the dominant directional light from SH, subtracts it, evaluates
    // residual ambient, then re-adds with sharp N.L for crisper light/shadow.
    var sh_irradiance = vec3<f32>(0.0);
    if (env_probe.env_diffuse_intensity > 0.0) {
        let dominant = extract_dominant_light_from_sh();
        let dominant_dir = dominant[0];
        let dominant_intensity = dominant[1];

        let sh_result = evaluate_sh_irradiance_with_dominant_light(
            frag_normal, dominant_dir, dominant_intensity
        );
        sh_irradiance = sh_result[0];
        light_color += sh_irradiance * albedo_specular.rgb * ambient_occlusion
                     * env_probe.env_diffuse_intensity;

        // Dominant light Cook-Torrance specular — same BRDF as analytical lights
        let dom_n_dot_l = dot(frag_normal, dominant_dir);
        let dom_n_dot_v = dot(frag_normal, view_direction);
        if (min(dom_n_dot_l, dom_n_dot_v) > 0.0) {
            let dom_half = normalize(dominant_dir + view_direction);
            let dom_n_dot_h = max(dot(frag_normal, dom_half), 0.0);
            let dom_v_dot_h = max(dot(view_direction, dom_half), 0.0);

            let dom_D = beckmann_ndf(dom_n_dot_h, roughness);
            let dom_G = ct_geometry(dom_n_dot_h, dom_n_dot_v, dom_n_dot_l, dom_v_dot_h);
            let dom_F = fresnel_exact(vec3(fresnel_f0), dom_v_dot_h);

            let dom_ct = dom_D * saturate(dom_G) / (4.0 * dom_n_dot_v + 0.00001) * dom_F;
            let dom_clamped = min(dom_ct, vec3(dom_n_dot_l + (1.0 - ANTI_SHADOW_CONTROL)));
            let dom_spec_tint = mix(vec3(1.0), albedo_specular.rgb, ALBEDO_BLEND);

            // Scale by dominant intensity and the 0.281 factor matching diffuse re-add
            light_color += dom_clamped * dom_spec_tint * dominant_intensity * 0.281
                         * material_specular_amount * albedo_specular.a
                         * env_probe.env_diffuse_intensity;
        }
    }

    // Rim fresnel — modulated by SH irradiance (approximates area light at grazing angles)
    let n_dot_v_rim = max(dot(frag_normal, view_direction), 0.0);
    let rim_factor = pow(1.0 - n_dot_v_rim, RIM_FRESNEL_POWER);
    let rim_color = mix(RIM_FRESNEL_COLOR, albedo_specular.rgb, RIM_FRESNEL_ALBEDO_BLEND);
    let rim_fresnel = RIM_FRESNEL_COEFFICIENT * material_specular_amount * albedo_specular.a * rim_color * rim_factor;
    light_color += rim_fresnel * sh_irradiance;

    // Emissive: luminance from normal.w, tinted by albedo color (added once)
    // HDR boost so emissive exceeds bloom threshold and produces visible glow
    let emissive_color = albedo_specular.rgb * emissive_luma * 5.0;
    light_color += emissive_color;

    // Atmospheric scattering — atmosphere.fx + entry_points.fx
    if (atmosphere.atmosphere_enable > 0.5) {
        let scattering = compute_scattering(lighting.camera_position, frag_position);
        light_color = light_color * scattering[0] + scattering[1];
    }

    return vec4<f32>(light_color, 1.0);
}
