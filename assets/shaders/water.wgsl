// Water forward pass — Gerstner waves, texture bump maps, Beer's law, SSS

// Group 0: Camera
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Group 1: Water uniforms
struct WaterUniforms {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
    water_height: f32,
    absorption_depth: f32,
    edge_softness: f32,
    specular_roughness: f32,
    deep_water_color: vec3<f32>,
    refraction_strength: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    _pad: f32,
};
@group(1) @binding(0) var<uniform> water: WaterUniforms;

// Group 2: Scene textures + normal maps
@group(2) @binding(0) var t_depth: texture_depth_2d;
@group(2) @binding(1) var t_lighting_copy: texture_2d<f32>;
@group(2) @binding(2) var t_env_cubemap: texture_cube<f32>;
@group(2) @binding(3) var s_nearest: sampler;
@group(2) @binding(4) var s_filtering: sampler;
@group(2) @binding(5) var t_bump0: texture_2d<f32>;
@group(2) @binding(6) var t_bump1: texture_2d<f32>;

// Group 3: Shadow maps (atlas: cascades + spots in single 2D texture)
@group(3) @binding(0) var t_shadow_cubes: texture_depth_cube_array;
@group(3) @binding(1) var t_shadow_atlas: texture_depth_2d;
@group(3) @binding(2) var s_shadow_compare: sampler_comparison;

struct ShadowData {
    point_params: array<vec4<f32>, 4>,
    spot_view_proj: array<mat4x4<f32>, 4>,
    spot_params: array<vec4<f32>, 4>,
    cascade_view_proj: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    cascade_texel_sizes: vec4<f32>,
    num_point_casters: u32,
    num_spot_casters: u32,
    _shadow_pad: vec2<f32>,
    cascade_atlas: array<vec4<f32>, 3>,
    spot_atlas: array<vec4<f32>, 4>,
};
@group(3) @binding(3) var<uniform> shadow_data: ShadowData;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ABSORPTION_COEFFS = vec3<f32>(3.0, 0.6, 0.3);

const BUMP_STRENGTH: f32 = 0.25;

const CSM_MAP_SIZE_F: f32 = 2048.0;
const SHADOW_NORMAL_BIAS_SCALE: f32 = 3.0;
const SHADOW_PCF_SPREAD: f32 = 3.0;

const CAUSTIC_INTENSITY: f32 = 0.6;
const CAUSTIC_DEPTH_FADE: f32 = 0.4;

const SHADOW_SCATTER_RATE: f32 = 0.35;

// ---------------------------------------------------------------------------
// Integer hash (used by sparkle)
// ---------------------------------------------------------------------------

fn hash_cell(ix: i32, iy: i32) -> f32 {
    var n = bitcast<u32>(ix) * 0x27d4eb2du + bitcast<u32>(iy) * 0x85ebca6bu;
    n ^= n >> 15u;
    n *= 0x1b873593u;
    n ^= n >> 13u;
    return f32(n >> 8u) / 16777216.0;
}

// ---------------------------------------------------------------------------
// Gerstner wave parameters (4 waves)
// ---------------------------------------------------------------------------

struct GerstnerWave {
    direction: vec2<f32>,
    steepness: f32,
    wavelength: f32,
    amplitude: f32,
    speed: f32,
};

const NUM_WAVES: u32 = 4u;

fn get_wave(i: u32) -> GerstnerWave {
    var w: GerstnerWave;
    switch i {
        case 0u: {
            w.direction = normalize(vec2<f32>(1.0, 0.6));
            w.steepness = 0.3;
            w.wavelength = 8.0;
            w.amplitude = 0.08;
            w.speed = 1.2;
        }
        case 1u: {
            w.direction = normalize(vec2<f32>(0.3, 1.0));
            w.steepness = 0.25;
            w.wavelength = 5.0;
            w.amplitude = 0.05;
            w.speed = 0.8;
        }
        case 2u: {
            w.direction = normalize(vec2<f32>(-0.7, 0.4));
            w.steepness = 0.2;
            w.wavelength = 3.0;
            w.amplitude = 0.035;
            w.speed = 1.5;
        }
        default: {
            w.direction = normalize(vec2<f32>(0.5, -0.8));
            w.steepness = 0.15;
            w.wavelength = 2.0;
            w.amplitude = 0.02;
            w.speed = 2.0;
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Vertex shader — grid + Gerstner displacement
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) grid_position: vec2<f32>,
    @location(3) wave_position: vec3<f32>,  // unscaled, for visual computations
};

struct GerstnerResult {
    position: vec3<f32>,
    normal: vec3<f32>,
};

fn gerstner(pos: vec2<f32>) -> GerstnerResult {
    var result: GerstnerResult;
    result.position = vec3<f32>(pos.x, pos.y, water.water_height);
    result.normal = vec3<f32>(0.0, 0.0, 1.0);

    for (var i = 0u; i < NUM_WAVES; i++) {
        let w = get_wave(i);
        let frequency = 6.283185 / w.wavelength;
        let theta = dot(w.direction, pos) * frequency + w.speed * water.time;
        let c = cos(theta);
        let s = sin(theta);
        let wa = frequency * w.amplitude;

        result.position.x += w.steepness * w.amplitude * w.direction.x * c;
        result.position.y += w.steepness * w.amplitude * w.direction.y * c;
        result.position.z += w.amplitude * s;

        result.normal.x -= w.direction.x * wa * c;
        result.normal.y -= w.direction.y * wa * c;
        result.normal.z -= w.steepness * wa * s;
    }

    result.normal = normalize(result.normal);
    return result;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let wave = gerstner(in.position.xy);
    out.wave_position = wave.position;
    out.world_position = wave.position * 0.5;
    out.normal = wave.normal;
    out.grid_position = in.position.xy;

    out.clip_position = camera.projection * camera.view * vec4<f32>(wave.position * 0.5, 1.0);

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
    let world_pos = water.inverse_view_projection * ndc;
    return world_pos.xyz / world_pos.w;
}

// Reoriented Normal Mapping — UE5 BlendAngleCorrectedNormals equivalent.
fn blend_rnm(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) / t.z - u);
}

// Per-pixel Gerstner normal — evaluates 4 main waves at exact fragment position
fn compute_water_normal(pos: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(0.0, 0.0, 1.0);
    for (var i = 0u; i < NUM_WAVES; i++) {
        let w = get_wave(i);
        let frequency = 6.283185 / w.wavelength;
        let theta = dot(w.direction, pos) * frequency + w.speed * water.time;
        let wa = frequency * w.amplitude;
        n.x -= w.direction.x * wa * cos(theta);
        n.y -= w.direction.y * wa * cos(theta);
        n.z -= w.steepness * wa * sin(theta);
    }
    return normalize(n);
}

// Procedural caustic pattern (Dave Hoskins / Shadertoy)
fn caustic_pattern(uv: vec2<f32>, time: f32) -> f32 {
    let TAU = 6.283185;
    var p = uv * TAU;
    var i = p;
    let inten = 0.005;
    var c = 0.0;
    for (var n = 0; n < 4; n++) {
        let t = time * (1.0 - 3.5 / (f32(n) + 1.0));
        i = p + vec2<f32>(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
        c += 1.0 / max(length(vec2<f32>(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten))), 0.001);
    }
    c /= 4.0;
    c = 1.17 - pow(c, 1.4);
    return pow(abs(c), 8.0);
}

// GGX normal distribution
fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.141593 * d * d);
}

// ---------------------------------------------------------------------------
// Shadow sampling (cascade shadow maps)
// ---------------------------------------------------------------------------

fn sample_water_cascade_pcf(
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

    let tile = shadow_data.cascade_atlas[cascade_idx];
    let atlas_size = f32(textureDimensions(t_shadow_atlas).x);
    let w_half_texel = 0.5 / atlas_size;
    let atlas_uv = clamp(uv * tile.z + tile.xy, tile.xy + w_half_texel, tile.xy + tile.z - w_half_texel);
    let texel_uv = atlas_uv * atlas_size - 0.5;
    let f = fract(texel_uv);
    let g = textureGatherCompare(t_shadow_atlas, s_shadow_compare, atlas_uv, proj.z);
    let shadow = mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y);
    return shadow;
}

fn calculate_water_shadow(
    frag_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    screen_pos: vec2<f32>,
) -> f32 {
    let dist = length(frag_pos - camera_pos);
    var cascade_idx = 0u;
    if (dist > shadow_data.cascade_splits.x) { cascade_idx = 1u; }
    if (dist > shadow_data.cascade_splits.y) { cascade_idx = 2u; }

    if (dist > shadow_data.cascade_splits.z) { return 1.0; }

    let shadow_current = sample_water_cascade_pcf(frag_pos, frag_normal, light_dir, screen_pos, cascade_idx);

    let splits = shadow_data.cascade_splits;
    var split_end = splits.x;
    if (cascade_idx == 1u) { split_end = splits.y; }
    if (cascade_idx == 2u) { split_end = splits.z; }

    let blend_start = split_end * 0.9;
    if (cascade_idx < 2u && dist > blend_start) {
        let shadow_next = sample_water_cascade_pcf(frag_pos, frag_normal, light_dir, screen_pos, cascade_idx + 1u);
        let blend_t = (dist - blend_start) / (split_end - blend_start);
        return mix(shadow_current, shadow_next, blend_t);
    }

    return shadow_current;
}

fn sample_cascade_single_tap(
    frag_pos: vec3<f32>,
    frag_normal: vec3<f32>,
    light_dir: vec3<f32>,
    camera_pos: vec3<f32>,
) -> f32 {
    let dist = length(frag_pos - camera_pos);
    var cascade_idx = 0u;
    if (dist > shadow_data.cascade_splits.x) { cascade_idx = 1u; }
    if (dist > shadow_data.cascade_splits.y) { cascade_idx = 2u; }
    if (dist > shadow_data.cascade_splits.z) { return 1.0; }

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

    let tile = shadow_data.cascade_atlas[cascade_idx];
    let atlas_uv = uv * tile.z + tile.xy;
    return textureSampleCompareLevel(t_shadow_atlas, s_shadow_compare, atlas_uv, proj.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_depth);
    let screen_uv = in.clip_position.xy / vec2<f32>(dims);

    let scene_depth = textureLoad(t_depth, vec2<i32>(in.clip_position.xy), 0);
    let water_ndc_depth = in.clip_position.z;

    // Underwater distance for absorption and shore fade
    var underwater_distance: f32;
    var floor_world_pos = vec3<f32>(0.0, 0.0, 0.0);
    var has_floor = false;
    if scene_depth < 0.0001 {
        underwater_distance = water.absorption_depth;
    } else {
        floor_world_pos = reconstruct_world_pos(screen_uv, scene_depth);
        has_floor = true;
        let z_diff = max(in.world_position.z - floor_world_pos.z, 0.0);
        let view_dir = normalize(in.world_position - water.camera_position);
        underwater_distance = z_diff / max(abs(view_dir.z), 0.05);
    }

    // --- Per-pixel Gerstner normal (smooth base, hides mesh faces) ---
    let base_normal = compute_water_normal(in.grid_position);

    // --- Texture-based normal maps (two scrolling layers) ---
    let bump_uv1 = in.wave_position.xy * 1.0 + vec2<f32>(0.03, 0.02) * water.time;
    let bump_uv2 = in.wave_position.xy * 2.5 + vec2<f32>(-0.02, 0.035) * water.time;

    let n1 = textureSample(t_bump0, s_filtering, bump_uv1).rgb * 2.0 - 1.0;
    let n2 = textureSample(t_bump1, s_filtering, bump_uv2).rgb * 2.0 - 1.0;
    let bump = normalize(n1 + n2);

    // Additive bump on per-pixel Gerstner base
    let N = normalize(base_normal + (bump - vec3<f32>(0.0, 0.0, 1.0)) * BUMP_STRENGTH);
    let V = normalize(water.camera_position - in.world_position);

    // --- Refraction (with chromatic aberration) ---
    let depth_scale = saturate(underwater_distance / 0.5);
    let refract_offset = N.xy * water.refraction_strength * depth_scale;
    var refract_uv = screen_uv + refract_offset;

    let refract_texel = clamp(vec2<i32>(vec2<f32>(dims) * refract_uv), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let refract_depth = textureLoad(t_depth, refract_texel, 0);
    if refract_depth > water_ndc_depth {
        refract_uv = screen_uv;
    }
    refract_uv = clamp(refract_uv, vec2<f32>(0.001), vec2<f32>(0.999));

    let chroma_offset = N.xy * 0.004;
    let ref_r = textureSample(t_lighting_copy, s_filtering,
        clamp(refract_uv + chroma_offset, vec2<f32>(0.001), vec2<f32>(0.999))).r;
    let ref_g = textureSample(t_lighting_copy, s_filtering, refract_uv).g;
    let ref_b = textureSample(t_lighting_copy, s_filtering,
        clamp(refract_uv - chroma_offset, vec2<f32>(0.001), vec2<f32>(0.999))).b;
    var refraction_color = vec3<f32>(ref_r, ref_g, ref_b);

    // --- Caustics (procedural, projected onto floor) ---
    if has_floor {
        let water_depth = max(in.world_position.z - floor_world_pos.z, 0.0);
        let depth_fade = exp(-water_depth * CAUSTIC_DEPTH_FADE);
        let n_dot_l_floor = saturate(water.sun_direction.z); // floor faces up

        let caustic = caustic_pattern(floor_world_pos.xy * 0.5, water.time * 0.4);

        // Single-tap floor shadow for caustic masking (cheap)
        let floor_shadow = sample_cascade_single_tap(
            floor_world_pos,
            vec3<f32>(0.0, 0.0, 1.0),
            normalize(water.sun_direction),
            water.camera_position,
        );

        refraction_color += caustic * CAUSTIC_INTENSITY * depth_fade * n_dot_l_floor
                            * floor_shadow * water.sun_color * water.sun_intensity;
    }

    // --- Shadow depth attenuation ---
    if has_floor {
        let water_depth = max(in.world_position.z - floor_world_pos.z, 0.0);
        let scatter_atten = exp(-water_depth * SHADOW_SCATTER_RATE * ABSORPTION_COEFFS / ABSORPTION_COEFFS.z);
        let avg_lum = dot(refraction_color, vec3<f32>(0.2126, 0.7152, 0.0722));
        let scattered = vec3<f32>(avg_lum * 0.3, avg_lum * 0.7, avg_lum);
        refraction_color = mix(scattered, refraction_color, scatter_atten);
    }

    // --- Beer's law absorption (per-channel) ---
    let abs_coeffs = ABSORPTION_COEFFS / water.absorption_depth;
    let transmittance = exp(-abs_coeffs * underwater_distance);
    let water_color = refraction_color * transmittance + water.deep_water_color * (vec3<f32>(1.0) - transmittance);

    // --- Reflection ---
    let R = reflect(-V, N);
    let reflection_color = textureSampleLevel(t_env_cubemap, s_filtering, R, 0.0).rgb;

    // --- Fresnel ---
    let f0 = 0.02;
    let n_dot_v = max(dot(N, V), 0.0);
    let fresnel = min(f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0), 0.6);

    // --- Shadow (surface) ---
    let L = normalize(water.sun_direction);
    let shadow = calculate_water_shadow(
        in.world_position,
        water.camera_position,
        N,
        L,
        in.clip_position.xy,
    );

    var color = mix(water_color, reflection_color, fresnel);

    // --- Subsurface scattering approximation ---
    let sss_distortion = 0.3;
    let H_sss = normalize(L + N * sss_distortion);
    let sss = pow(saturate(dot(-V, H_sss)), 4.0) * 0.15;
    let sss_shadow = mix(0.7, 1.0, shadow);
    let sss_color = vec3<f32>(0.02, 0.12, 0.1) * sss * sss_shadow * water.sun_intensity;
    color += sss_color;

    // --- Sun specular (hard shadow — fully gated) ---
    let H = normalize(L + V);
    let n_dot_h = max(dot(N, H), 0.0);
    let n_dot_l = max(dot(N, L), 0.0);

    let sparkle_uv = in.wave_position.xy + water.time * vec2<f32>(0.01, 0.015);
    let sp = vec2<i32>(floor(sparkle_uv * 13.0));
    let sparkle = pow(hash_cell(sp.x, sp.y), 4.0);

    // Geometric specular anti-aliasing (Tokuyoshi-Kaplanyan 2019)
    let du = dpdx(N);
    let dv = dpdy(N);
    let sigma2 = clamp(0.15 * (dot(du, du) + dot(dv, dv)), 0.0, 0.18);
    let aa_roughness = sqrt(water.specular_roughness * water.specular_roughness + sigma2);

    // Gate specular by geometric normal — prevents bump-map ripples from
    // creating specular on macro-surface faces that point away from the sun
    let geo_gate = smoothstep(0.0, 0.1, dot(base_normal, L));

    let spec_d = ggx_d(n_dot_h, aa_roughness);
    let specular = spec_d * fresnel * n_dot_l * geo_gate * water.sun_intensity * water.sun_color;
    let spec_shadow = mix(0.15, 1.0, shadow);
    color += specular * spec_shadow * (0.3 + sparkle * 0.7);

    // --- Shore edge blend ---
    let alpha = saturate(underwater_distance / water.edge_softness);

    return vec4<f32>(color * alpha, alpha);
}
