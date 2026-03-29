// Deferred lighting pass — fullscreen fragment shader
// Replaces forward_base_lighting + forward_dynamic_lighting with a single pass.
// Reads GBuffer, evaluates all lights, shadows, env probe, SH, atmosphere.

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

struct SkyParams {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
};

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
    sun_luminance: f32,
    sun_angular_radius: f32,
    sun_edge_softness: f32,
    sun_disc_intensity: f32,
    sun_inner_glow_intensity: f32,
    sun_air_mass_scale: f32,
    sun_tint: vec3<f32>,
    zenith_air_mass_factor: f32,
    horizon_fade_start: f32,
    horizon_fade_end: f32,
    _pad2: vec2<f32>,
};

struct EnvProbeData {
    probe_position: vec3<f32>, env_roughness_scale: f32,
    env_specular_contribution: f32, env_mip_count: f32, env_intensity: f32, env_diffuse_intensity: f32,
};

struct SHCoefficients { coefficients: array<vec4<f32>, 9> };

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

struct LightingUniforms {
    camera_position: vec3<f32>,
    light_count: u32,
    camera_direction: vec3<f32>,
    specular_occlusion_enable: f32,
    lights: array<GpuLightData, 16>,
};

struct ShadowData {
    point_params: array<vec4<f32>, 4>,
    spot_view_proj: array<mat4x4<f32>, 4>,
    spot_params: array<vec4<f32>, 4>,
    cascade_view_proj: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    cascade_texel_sizes: vec4<f32>,
    num_point_casters: u32,
    num_spot_casters: u32,
    _pad: vec2<f32>,
    cascade_atlas: array<vec4<f32>, 3>,
    spot_atlas: array<vec4<f32>, 4>,
};

// ---------------------------------------------------------------------------
// Bind groups
// ---------------------------------------------------------------------------

// Group 0 — GBuffer + SSAO
@group(0) @binding(0) var t_gb_normal: texture_2d<f32>;
@group(0) @binding(1) var t_gb_albedo_specular: texture_2d<f32>;
@group(0) @binding(2) var t_gb_material: texture_2d<f32>;
@group(0) @binding(3) var t_depth: texture_depth_2d;
@group(0) @binding(4) var t_ssao: texture_2d<f32>;
@group(0) @binding(5) var s_nearest: sampler;
@group(0) @binding(6) var s_filtering: sampler;
@group(0) @binding(7) var t_gb_emissive: texture_2d<f32>;

// Group 1 — Shadow
@group(1) @binding(0) var t_shadow_cubes: texture_depth_cube_array;
@group(1) @binding(1) var t_shadow_atlas: texture_depth_2d;
@group(1) @binding(2) var s_shadow_compare: sampler_comparison;
@group(1) @binding(3) var<uniform> shadow_data: ShadowData;

// Group 2 — Lighting + Env Probe + Atmosphere
@group(2) @binding(0) var<uniform> lighting: LightingUniforms;
@group(2) @binding(1) var<uniform> atmosphere: AtmosphereData;
@group(2) @binding(2) var t_env_cubemap: texture_cube<f32>;
@group(2) @binding(3) var s_env_filtering: sampler;
@group(2) @binding(4) var<uniform> env_probe: EnvProbeData;
@group(2) @binding(5) var<uniform> sh_data: SHCoefficients;
@group(2) @binding(6) var<uniform> sky: SkyParams;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;
const MAX_LIGHTS: u32 = 16u;
const SHADOW_MAP_SIZE_F: f32 = 1024.0;
const SHADOW_PCF_SPREAD: f32 = 3.0;
const SHADOW_NORMAL_BIAS_SCALE: f32 = 3.0;
const CSM_MAP_SIZE_F: f32 = 2048.0;
const CSM_BLEND_FRACTION: f32 = 0.1;
const ANALYTICAL_SPEC_BASE: f32 = 0.4;
const ANALYTICAL_SPEC_METAL: f32 = 0.8;
const ANTI_SHADOW_CONTROL: f32 = 0.1;
const AREA_SPEC_BASE: f32 = 0.6;
const AREA_SPEC_METAL: f32 = 0.2;
const ENV_SPEC_BASE: f32 = 1.0;
const ENV_SPEC_METAL: f32 = 1.5;
const RIM_FRESNEL_COEFFICIENT: f32 = 0.5;
const RIM_FRESNEL_POWER: f32 = 3.0;
const RIM_FRESNEL_COLOR: vec3<f32> = vec3(1.0, 1.0, 1.0);
const RIM_FRESNEL_ALBEDO_BLEND: f32 = 0.3;

// SH constants
const SH_L0: f32 = 0.886227;
const SH_L1: f32 = 1.023327;
const SH_L2_21: f32 = 0.858086;
const SH_L2_20: f32 = 0.247708;
const SH_L2_22: f32 = 0.429043;

// ---------------------------------------------------------------------------
// Vertex — fullscreen quad
// ---------------------------------------------------------------------------

struct QuadVertex {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: QuadVertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) { n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z); }
    return normalize(n);
}

fn fresnel_exact(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let f0c = min(f0, vec3(0.999));
    let sqrt_f0 = sqrt(f0c);
    let n = (vec3(1.0) + sqrt_f0) / (vec3(1.0) - sqrt_f0);
    let g = sqrt(n * n + vec3(v_dot_h * v_dot_h - 1.0));
    let gpc = g + vec3(v_dot_h);
    let gmc = g - vec3(v_dot_h);
    let r = (vec3(v_dot_h) * gpc - vec3(1.0)) / (vec3(v_dot_h) * gmc + vec3(1.0));
    return 0.5 * ((gmc * gmc) / (gpc * gpc + vec3(0.00001))) * (vec3(1.0) + r * r);
}

fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return saturate(pow(n_dot_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
}

fn evaluate_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    var r = sh_data.coefficients[0].rgb * SH_L0;
    r += (sh_data.coefficients[1].rgb * n.y + sh_data.coefficients[2].rgb * n.z + sh_data.coefficients[3].rgb * n.x) * SH_L1;
    r += (sh_data.coefficients[4].rgb * (n.x * n.y) + sh_data.coefficients[5].rgb * (n.y * n.z) + sh_data.coefficients[7].rgb * (n.x * n.z)) * SH_L2_21;
    r += sh_data.coefficients[6].rgb * (3.0 * n.z * n.z - 1.0) * SH_L2_20;
    r += sh_data.coefficients[8].rgb * (n.x * n.x - n.y * n.y) * SH_L2_22;
    return max(r, vec3(0.0));
}

fn evaluate_sh_area_specular(rd: vec3<f32>, roughness: f32) -> vec3<f32> {
    let rs = roughness * roughness;
    let c_dc = 0.282095;
    let c_l = -(0.5128945834 + (-0.1407369526) * roughness + (-0.002660066620) * rs) * 0.60;
    let c_q = -(0.7212524717 + (-0.5541015389) * roughness + 0.07960539966 * rs) * 0.5;
    let x0 = sh_data.coefficients[0].rgb;
    let x1 = sh_data.coefficients[1].rgb * rd.y + sh_data.coefficients[2].rgb * rd.z + sh_data.coefficients[3].rgb * rd.x;
    let x2 = sh_data.coefficients[4].rgb * (rd.x * rd.y) + sh_data.coefficients[5].rgb * (rd.y * rd.z) + sh_data.coefficients[7].rgb * (rd.x * rd.z);
    let x3 = sh_data.coefficients[6].rgb * (3.0 * rd.z * rd.z - 1.0) + sh_data.coefficients[8].rgb * (rd.x * rd.x - rd.y * rd.y);
    return max(x0 * c_dc + x1 * c_l + x2 * c_q + x3 * c_q, vec3(0.0));
}

fn sample_environment(normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32, fresnel_f0: f32) -> vec3<f32> {
    let R = reflect(-view_dir, normal);
    let lod = roughness * env_probe.env_roughness_scale * (env_probe.env_mip_count - 1.0);
    let env_color = textureSampleLevel(t_env_cubemap, s_env_filtering, R, lod).rgb;
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let F = fresnel_exact(vec3(fresnel_f0), n_dot_v);
    return env_color * F * env_probe.env_specular_contribution * env_probe.env_intensity;
}

// ---------------------------------------------------------------------------
// Atmospheric scattering — full (extinction + inscatter)
// ---------------------------------------------------------------------------

fn compute_scattering(cam: vec3<f32>, wp: vec3<f32>) -> array<vec3<f32>, 2> {
    let ray = wp - cam;
    var dist = min(length(ray), atmosphere.max_fog_thickness);
    if (dist < 0.001) { return array<vec3<f32>, 2>(vec3(1.0), vec3(0.0)); }
    let rd = normalize(ray);
    let hc = max(cam.z - atmosphere.reference_height, 0.0);
    let hf = max(wp.z - atmosphere.reference_height, 0.0);
    let diff = hc - hf;
    var ray_d: f32; var mie_d: f32;
    if (abs(diff) > 0.001) {
        ray_d = (exp(-hf / atmosphere.rayleigh_height_scale) - exp(-hc / atmosphere.rayleigh_height_scale)) * dist * atmosphere.rayleigh_height_scale / diff;
        mie_d = (exp(-hf / atmosphere.mie_height_scale) - exp(-hc / atmosphere.mie_height_scale)) * dist * atmosphere.mie_height_scale / diff;
    } else {
        ray_d = exp(-hc / atmosphere.rayleigh_height_scale) * dist;
        mie_d = exp(-hc / atmosphere.mie_height_scale) * dist;
    }
    let ext = exp(-(atmosphere.rayleigh_coefficients * ray_d + vec3(atmosphere.mie_coefficient) * mie_d));
    let sd_len = length(atmosphere.sun_direction);
    let sd = select(vec3(0.0, 0.0, 1.0), atmosphere.sun_direction / sd_len, sd_len > 0.0001);
    let ct = dot(rd, sd);
    let rp = 0.05968 * (1.0 + ct * ct);
    let g = atmosphere.mie_g; let g2 = g * g;
    let mp = 0.07958 * (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * ct, 0.00001), 1.5);
    let sse = max(sd.z, 0.01);
    let sam = atmosphere.sun_air_mass_scale / sse;
    let srd = exp(-hc / atmosphere.rayleigh_height_scale) * atmosphere.rayleigh_height_scale * sam;
    let smd = exp(-hc / atmosphere.mie_height_scale) * atmosphere.mie_height_scale * sam;
    let se = exp(-(atmosphere.rayleigh_coefficients * srd + vec3(atmosphere.mie_coefficient) * smd));
    let sc = vec3(1.0) - ext;
    let ins = (atmosphere.rayleigh_coefficients * rp + vec3(atmosphere.mie_coefficient) * mp) * sc * se * atmosphere.inscatter_scale;
    return array<vec3<f32>, 2>(ext, ins);
}

// Extinction-only scattering for per-light atmospheric attenuation
fn compute_extinction_only(camera_pos: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let ray = world_pos - camera_pos;
    var dist = length(ray);
    dist = min(dist, atmosphere.max_fog_thickness);
    if (dist < 0.001) { return vec3(1.0); }
    let h_cam = max(camera_pos.z - atmosphere.reference_height, 0.0);
    let h_frag = max(world_pos.z - atmosphere.reference_height, 0.0);
    let diff = h_cam - h_frag;
    var rayleigh_depth: f32; var mie_depth: f32;
    if (abs(diff) > 0.001) {
        rayleigh_depth = (exp(-h_frag / atmosphere.rayleigh_height_scale) - exp(-h_cam / atmosphere.rayleigh_height_scale)) * dist * atmosphere.rayleigh_height_scale / diff;
        mie_depth = (exp(-h_frag / atmosphere.mie_height_scale) - exp(-h_cam / atmosphere.mie_height_scale)) * dist * atmosphere.mie_height_scale / diff;
    } else {
        rayleigh_depth = exp(-h_cam / atmosphere.rayleigh_height_scale) * dist;
        mie_depth = exp(-h_cam / atmosphere.mie_height_scale) * dist;
    }
    let extinction = exp(-(atmosphere.rayleigh_coefficients * rayleigh_depth + vec3(atmosphere.mie_coefficient) * mie_depth));
    return extinction;
}

// ---------------------------------------------------------------------------
// Cook-Torrance BRDF
// ---------------------------------------------------------------------------

fn beckmann_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let r2 = roughness * roughness;
    let c2 = n_dot_h * n_dot_h;
    let c4 = c2 * c2;
    return exp((c2 - 1.0) / (r2 * c2)) / (r2 * c4 + 0.00001);
}

fn ct_geometry(n_dot_h: f32, n_dot_v: f32, n_dot_l: f32, v_dot_h: f32) -> f32 {
    return 2.0 * n_dot_h * min(n_dot_v, n_dot_l) / (saturate(v_dot_h) + 0.00001);
}

// ---------------------------------------------------------------------------
// Shadow functions
// ---------------------------------------------------------------------------

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

fn random_angle(screen_pos: vec2<f32>) -> f32 {
    return fract(sin(dot(screen_pos, vec2(12.9898, 78.233))) * 43758.5453) * 6.283185;
}

fn rotate_offset(offset: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
}

fn apply_normal_offset(frag_pos: vec3<f32>, frag_normal: vec3<f32>, light_dir: vec3<f32>, dist: f32) -> vec3<f32> {
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let texel_world = 2.0 * dist / SHADOW_MAP_SIZE_F;
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    return frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;
}

fn calculate_point_shadow(frag_pos: vec3<f32>, light_pos: vec3<f32>, shadow_idx: u32, frag_normal: vec3<f32>, light_dir: vec3<f32>, screen_pos: vec2<f32>) -> f32 {
    let params = shadow_data.point_params[shadow_idx];
    let near = params.x;
    let far = params.y;
    let dist = length(frag_pos - light_pos);
    let biased_pos = apply_normal_offset(frag_pos, frag_normal, light_dir, dist);
    let d = biased_pos - light_pos;
    let max_comp = max(max(abs(d.x), abs(d.y)), abs(d.z));
    let reference = far * (max_comp - near) / max(max_comp * (far - near), 0.00001);
    let d_norm = normalize(d);
    var up = vec3(0.0, 1.0, 0.0);
    if (abs(d_norm.y) > 0.99) { up = vec3(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(d_norm, up));
    let bitangent = cross(d_norm, tangent);
    let disk_radius = dist * SHADOW_PCF_SPREAD / SHADOW_MAP_SIZE_F;
    let rotation = random_angle(screen_pos);
    var shadow = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let p = rotate_offset(POISSON_DISK[i], rotation);
        let sample_d = d + (tangent * p.x + bitangent * p.y) * disk_radius;
        shadow += textureSampleCompareLevel(t_shadow_cubes, s_shadow_compare, sample_d, shadow_idx, reference);
    }
    return shadow / 4.0;
}

fn calculate_spot_shadow(frag_pos: vec3<f32>, light_pos: vec3<f32>, shadow_idx: u32, frag_normal: vec3<f32>, light_dir: vec3<f32>, screen_pos: vec2<f32>) -> f32 {
    let params = shadow_data.spot_params[shadow_idx];
    let spot_texel_size = params.y;
    let view_proj = shadow_data.spot_view_proj[shadow_idx];
    let dist = length(light_pos - frag_pos);
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let texel_world = 2.0 * dist * spot_texel_size;
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    let biased_pos = frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;
    let frag_light_space = view_proj * vec4(biased_pos, 1.0);
    let tan_theta = sqrt(1.0 - n_dot_l * n_dot_l) / max(n_dot_l, 0.001);
    let depth_bias = 0.001 + min(tan_theta, 10.0) * 0.0005;
    if (frag_light_space.w <= 0.0) { return 1.0; }
    let proj = frag_light_space.xyz / frag_light_space.w;
    let uv = vec2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
    let ref_z = proj.z - depth_bias;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ref_z < 0.0 || proj.z > 1.0) { return 1.0; }
    let tile = shadow_data.spot_atlas[shadow_idx];
    let spot_atlas_size = f32(textureDimensions(t_shadow_atlas).x);
    let spot_half_texel = 0.5 / spot_atlas_size;
    let atlas_uv = clamp(uv * tile.z + tile.xy, tile.xy + spot_half_texel, tile.xy + tile.z - spot_half_texel);
    let texel_uv = atlas_uv * spot_atlas_size - 0.5;
    let f = fract(texel_uv);
    let g = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv, ref_z);
    let shadow = mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y);
    return shadow;
}

fn sample_cascade_pcf(frag_pos: vec3<f32>, frag_normal: vec3<f32>, light_dir: vec3<f32>, screen_pos: vec2<f32>, cascade_idx: u32) -> f32 {
    let texel_world = shadow_data.cascade_texel_sizes[cascade_idx];
    let n_dot_l = max(dot(frag_normal, light_dir), 0.0);
    let bias_factor = max(1.0 - n_dot_l, 0.2);
    let biased_pos = frag_pos + frag_normal * bias_factor * texel_world * SHADOW_NORMAL_BIAS_SCALE;
    let view_proj = shadow_data.cascade_view_proj[cascade_idx];
    let frag_light_space = view_proj * vec4(biased_pos, 1.0);
    let proj = frag_light_space.xyz / frag_light_space.w;
    let uv = vec2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) { return 1.0; }
    let tile = shadow_data.cascade_atlas[cascade_idx];
    let atlas_size = f32(textureDimensions(t_shadow_atlas).x);
    let texel_size = 1.0 / atlas_size;
    let half_texel = texel_size * 1.5;
    let atlas_uv = clamp(uv * tile.z + tile.xy, tile.xy + half_texel, tile.xy + tile.z - half_texel);
    let g00 = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv + vec2(-1.0, -1.0) * texel_size, proj.z);
    let g10 = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv + vec2( 1.0, -1.0) * texel_size, proj.z);
    let g01 = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv + vec2(-1.0,  1.0) * texel_size, proj.z);
    let g11 = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv + vec2( 1.0,  1.0) * texel_size, proj.z);
    let total = g00.x + g00.y + g00.z + g00.w + g10.x + g10.y + g10.z + g10.w + g01.x + g01.y + g01.z + g01.w + g11.x + g11.y + g11.z + g11.w;
    return total / 16.0;
}

fn calculate_directional_shadow(frag_pos: vec3<f32>, camera_pos: vec3<f32>, frag_normal: vec3<f32>, light_dir: vec3<f32>, screen_pos: vec2<f32>) -> f32 {
    let dist = length(frag_pos - camera_pos);
    var cascade_idx = 0u;
    if (dist > shadow_data.cascade_splits.x) { cascade_idx = 1u; }
    if (dist > shadow_data.cascade_splits.y) { cascade_idx = 2u; }
    if (dist > shadow_data.cascade_splits.z) { return 1.0; }
    let shadow_current = sample_cascade_pcf(frag_pos, frag_normal, light_dir, screen_pos, cascade_idx);
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

// ---------------------------------------------------------------------------
// Sky rendering
// ---------------------------------------------------------------------------

fn compute_sky(uv: vec2<f32>) -> vec4<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0, 1.0);
    let world_pos = sky.inverse_view_projection * ndc;
    let ray_dir = normalize(world_pos.xyz / world_pos.w - sky.camera_position);

    let sun_dir_len = length(atmosphere.sun_direction);
    let sun_dir = select(vec3<f32>(0.0, 0.0, 1.0), atmosphere.sun_direction / sun_dir_len, sun_dir_len > 0.0001);

    let h_cam = max(sky.camera_position.z - atmosphere.reference_height, 0.0);
    let sin_elev = max(ray_dir.z, 0.02);
    let air_mass = 1.0 / sin_elev;

    let rayleigh_density_at_cam = exp(-h_cam / atmosphere.rayleigh_height_scale);
    let mie_density_at_cam = exp(-h_cam / atmosphere.mie_height_scale);
    let rayleigh_depth = rayleigh_density_at_cam * atmosphere.rayleigh_height_scale * air_mass;
    let mie_depth = mie_density_at_cam * atmosphere.mie_height_scale * air_mass;

    let extinction = exp(-(atmosphere.rayleigh_coefficients * rayleigh_depth
                         + vec3(atmosphere.mie_coefficient) * mie_depth));

    let sun_sin_elev = max(sun_dir.z, 0.01);
    let base_sun_air_mass = atmosphere.sun_air_mass_scale / sun_sin_elev;
    let zenith_lerp = smoothstep(0.0, 0.5, max(ray_dir.z, 0.0));
    let sun_air_mass = base_sun_air_mass * mix(1.0, atmosphere.zenith_air_mass_factor, zenith_lerp);

    let sun_rayleigh_depth = rayleigh_density_at_cam * atmosphere.rayleigh_height_scale * sun_air_mass;
    let sun_mie_depth = mie_density_at_cam * atmosphere.mie_height_scale * sun_air_mass;
    let sun_extinction = exp(-(atmosphere.rayleigh_coefficients * sun_rayleigh_depth
                             + vec3(atmosphere.mie_coefficient) * sun_mie_depth));

    let cos_theta = dot(ray_dir, sun_dir);
    let rayleigh_phase = 0.05968 * (1.0 + cos_theta * cos_theta);

    let g = atmosphere.mie_g;
    let g2 = g * g;
    let mie_base_sky = max(1.0 + g2 - 2.0 * g * cos_theta, 0.00001);
    let mie_phase = 0.07958 * (1.0 - g2) / pow(mie_base_sky, 1.5);

    let scatter = vec3(1.0) - extinction;
    var sky_color = (atmosphere.rayleigh_coefficients * rayleigh_phase
                   + vec3(atmosphere.mie_coefficient) * mie_phase)
                   * scatter * sun_extinction * atmosphere.inscatter_scale * atmosphere.sun_luminance;

    let sun_cos_angle = dot(ray_dir, sun_dir);
    let angular_dist = acos(clamp(sun_cos_angle, -1.0, 1.0));

    let disc_mask = 1.0 - smoothstep(atmosphere.sun_angular_radius - atmosphere.sun_edge_softness,
                                      atmosphere.sun_angular_radius + atmosphere.sun_edge_softness,
                                      angular_dist);
    let mu = saturate(1.0 - angular_dist / atmosphere.sun_angular_radius);
    let limb_dark = 1.0 - 0.6 * (1.0 - sqrt(mu));
    let sun_disc = disc_mask * limb_dark * atmosphere.sun_disc_intensity;

    let glow_inner = pow(max(sun_cos_angle, 0.0), 256.0) * atmosphere.sun_inner_glow_intensity;

    sky_color += (sun_disc + glow_inner) * atmosphere.sun_tint * sun_extinction * extinction;

    let horizon_fade = smoothstep(atmosphere.horizon_fade_start, atmosphere.horizon_fade_end, ray_dir.z);
    sky_color *= horizon_fade;

    return vec4<f32>(sky_color, 1.0);
}

// ---------------------------------------------------------------------------
// Fragment — deferred lighting
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;
    let screen_pos = in.clip_position.xy;

    // Sample depth (reverse-Z: 0 = far, 1 = near)
    let depth = textureLoad(t_depth, vec2<i32>(in.clip_position.xy), 0);

    // Sky pixel — reverse-Z means far plane is near 0
    if (depth < 0.0001) {
        if (atmosphere.atmosphere_enable > 0.5) {
            return compute_sky(uv);
        }
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // -----------------------------------------------------------------------
    // Geometry pixel — read GBuffer
    // -----------------------------------------------------------------------

    let frag_normal = oct_decode(textureLoad(t_gb_normal, vec2<i32>(in.clip_position.xy), 0).rg);
    let albedo_specular = textureLoad(t_gb_albedo_specular, vec2<i32>(in.clip_position.xy), 0);
    let mat = textureLoad(t_gb_material, vec2<i32>(in.clip_position.xy), 0);

    let material_ambient_amount = mat.r;
    let metallic = mat.g;
    let roughness = max(mat.b, 0.05);
    let fresnel_f0 = mat.a;

    let analytical_spec_weight = mix(ANALYTICAL_SPEC_BASE, ANALYTICAL_SPEC_METAL, metallic);
    let area_w = mix(AREA_SPEC_BASE, AREA_SPEC_METAL, metallic);
    let env_w = mix(ENV_SPEC_BASE, ENV_SPEC_METAL, metallic);
    let albedo_blend = metallic;

    // Reconstruct world position from depth
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
    let wp4 = sky.inverse_view_projection * ndc;
    let world_position = wp4.xyz / wp4.w;

    // View direction
    let V = normalize(lighting.camera_position - world_position);
    let n_dot_v = max(dot(frag_normal, V), 0.0);

    // SSAO — inline 5x5 bilateral blur of raw half-res SSAO
    let ssao_texel = 1.0 / vec2<f32>(textureDimensions(t_ssao));
    let ssao_inv_sigma_sq = 1.0 / (0.05 * 0.05);
    let sg = array<f32, 3>(0.3829, 0.2417, 0.0606);
    var ssao_tw = 0.0;
    var ssao_tv = vec4<f32>(0.0);
    for (var sy = -2; sy <= 2; sy++) {
        for (var sx = -2; sx <= 2; sx++) {
            let suv = uv + vec2<f32>(f32(sx), f32(sy)) * ssao_texel;
            let sv = textureSample(t_ssao, s_nearest, suv);
            let sd = textureSample(t_depth, s_nearest, suv);
            let dd = depth - sd;
            let w = sg[abs(sx)] * sg[abs(sy)] * exp(-(dd * dd) * ssao_inv_sigma_sq);
            ssao_tv += sv * w;
            ssao_tw += w;
        }
    }
    let ao = (ssao_tv / max(ssao_tw, 0.0001)).w;

    // Specular occlusion (GTSO)
    var spec_occ = 1.0;
    if (lighting.specular_occlusion_enable > 0.5) {
        spec_occ = specular_occlusion(n_dot_v, ao, roughness);
    }

    // -----------------------------------------------------------------------
    // Indirect / ambient lighting (modulated by SSAO)
    // -----------------------------------------------------------------------

    var color = vec3<f32>(0.0);

    // Env probe cubemap specular
    if (env_probe.env_specular_contribution > 0.0) {
        let envmap = sample_environment(frag_normal, V, roughness, fresnel_f0);
        color += envmap * albedo_specular.a * spec_occ * env_w;
    }

    // SH area specular
    if (env_probe.env_intensity > 0.0 && albedo_specular.a > 0.01) {
        let R = reflect(-V, frag_normal);
        let area_spec = evaluate_sh_area_specular(R, roughness);
        let spec_tint = mix(vec3(1.0), albedo_specular.rgb, albedo_blend);
        color += area_spec * area_w * albedo_specular.a * spec_tint * spec_occ;
    }

    // SH diffuse irradiance
    var sh_irr = vec3<f32>(0.0);
    if (env_probe.env_diffuse_intensity > 0.0) {
        sh_irr = evaluate_sh_irradiance(frag_normal);
        color += sh_irr * albedo_specular.rgb * ao * env_probe.env_diffuse_intensity;
    }

    // H3 rim fresnel
    let rim = pow(1.0 - n_dot_v, RIM_FRESNEL_POWER);
    let rim_color = mix(RIM_FRESNEL_COLOR, albedo_specular.rgb, RIM_FRESNEL_ALBEDO_BLEND);
    color += RIM_FRESNEL_COEFFICIENT * albedo_specular.a * rim_color * rim * sh_irr;

    // -----------------------------------------------------------------------
    // Direct lighting — loop over ALL lights
    // -----------------------------------------------------------------------

    // Pre-compute atmospheric extinction for per-light attenuation
    var atmo_extinction = vec3<f32>(1.0);
    var atmo_fade = 0.0;
    if (atmosphere.atmosphere_enable > 0.5) {
        let dist_to_frag = length(world_position - lighting.camera_position);
        atmo_fade = smoothstep(2.0, 10.0, dist_to_frag);
        if (atmo_fade > 0.001) {
            atmo_extinction = compute_extinction_only(lighting.camera_position, world_position);
        }
    }

    for (var i = 0u; i < lighting.light_count; i++) {
        let light = lighting.lights[i];

        var light_direction: vec3<f32>;
        var light_vec = light.position - world_position;
        var light_distance = length(light_vec);
        if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
            let dir_len = length(light.direction);
            light_direction = select(vec3<f32>(0.0, 0.0, -1.0), -light.direction / dir_len, dir_len > 0.0001);
        } else {
            light_direction = light_vec / max(light_distance, 0.00001);
        }

        let halfway_direction = normalize(light_direction + V);

        // Ambient (per-light)
        var ambient_color = material_ambient_amount * light.ambient_color * albedo_specular.rgb;

        // Diffuse — H3 5% ambient floor
        let diffuse_amount = max(dot(light_direction, frag_normal), 0.05);
        var diffuse_color = diffuse_amount * albedo_specular.rgb * light.diffuse_color;

        // Specular — Cook-Torrance
        var specular_color = vec3<f32>(0.0);
        let n_dot_l = dot(frag_normal, light_direction);
        let n_dot_v_l = dot(frag_normal, V);
        if (min(n_dot_l, n_dot_v_l) > 0.0) {
            let n_dot_h = max(dot(frag_normal, halfway_direction), 0.0);
            let v_dot_h = max(dot(V, halfway_direction), 0.0);
            let D = beckmann_ndf(n_dot_h, roughness);
            let G = ct_geometry(n_dot_h, n_dot_v_l, n_dot_l, v_dot_h);
            let F = fresnel_exact(vec3(fresnel_f0), v_dot_h);
            let ct = D * saturate(G) / (4.0 * n_dot_v_l + 0.00001) * F;
            let clamped = min(ct, vec3(n_dot_l + (1.0 - ANTI_SHADOW_CONTROL)));
            let spec_tint = mix(vec3(1.0), albedo_specular.rgb, albedo_blend);
            specular_color = analytical_spec_weight * albedo_specular.a * (clamped * spec_tint) * light.specular_color;
        }

        // Attenuation
        if (light.light_type != LIGHT_TYPE_DIRECTIONAL) {
            let light_dist2 = light_distance * light_distance;
            var atten = saturate(1.0 / (light.constant_atten + light_dist2));
            if (light.light_type == LIGHT_TYPE_SPOT) {
                let spot_dir_len = length(light.direction);
                let spot_dir = select(vec3<f32>(0.0, 0.0, -1.0), -light.direction / spot_dir_len, spot_dir_len > 0.0001);
                let light_theta = dot(light_direction, spot_dir);
                let light_epsilon = light.inner_cutoff - light.outer_cutoff;
                let intensity = clamp((light_theta - light.outer_cutoff) / max(light_epsilon, 0.00001), 0.0, 1.0);
                atten *= intensity;
            }
            ambient_color *= atten;
            diffuse_color *= atten;
            specular_color *= atten;
        }

        // Shadow — H3 double-squared darkening
        var shadow_factor = 1.0;
        if (light.shadow_index >= 0) {
            let si = u32(light.shadow_index);
            if (light.light_type == LIGHT_TYPE_POINT) {
                shadow_factor = calculate_point_shadow(world_position, light.position, si, frag_normal, light_direction, screen_pos);
            } else if (light.light_type == LIGHT_TYPE_SPOT) {
                shadow_factor = calculate_spot_shadow(world_position, light.position, si, frag_normal, light_direction, screen_pos);
            } else if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
                shadow_factor = calculate_directional_shadow(world_position, lighting.camera_position, frag_normal, light_direction, screen_pos);
            }
            // H3 double-squared shadow darkening
            shadow_factor *= shadow_factor;
        }
        diffuse_color *= shadow_factor;
        specular_color *= shadow_factor;
        ambient_color *= shadow_factor;

        var light_color = ambient_color + diffuse_color + specular_color;

        // Atmospheric extinction on per-light contribution (no inscatter)
        if (atmo_fade > 0.001) {
            light_color *= mix(vec3(1.0), atmo_extinction, atmo_fade);
        }

        color += light_color;
    }

    // -----------------------------------------------------------------------
    // Emissive (from GBuffer MRT 3)
    // -----------------------------------------------------------------------

    let emissive = textureLoad(t_gb_emissive, vec2<i32>(in.clip_position.xy), 0).rgb;
    color += emissive;

    // -----------------------------------------------------------------------
    // Full atmospheric scattering (extinction + inscatter) on total color
    // -----------------------------------------------------------------------

    if (atmosphere.atmosphere_enable > 0.5) {
        let d = length(world_position - lighting.camera_position);
        let f = smoothstep(2.0, 10.0, d);
        if (f > 0.001) {
            let s = compute_scattering(lighting.camera_position, world_position);
            color = mix(color, color * s[0] + s[1], f);
        }
    }

    return vec4(clamp(color, vec3(0.0), vec3(500.0)), 1.0);
}
