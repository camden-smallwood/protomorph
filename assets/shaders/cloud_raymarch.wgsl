// Volumetric cloud raymarching — IQ-style inline FBM with HZD-inspired lighting
// Based on Inigo Quilez's "Clouds" (Shadertoy XslGRr) + Schneider's HZD techniques.

struct CloudParams {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    cloud_bottom: f32,
    cloud_top: f32,
    coverage: f32,
    wind_x: f32,
    wind_z: f32,
    wind_speed: f32,
    base_noise_scale: f32,
    extinction_coeff: f32,
    frame_index: u32,
    quarter_w: u32,
    quarter_h: u32,
    _pad: vec2<u32>,
    rayleigh_coefficients: vec3<f32>,
    mie_coefficient: f32,
    mie_g: f32,
    inscatter_scale: f32,
    reference_height: f32,
    sun_air_mass_scale: f32,
    rayleigh_height_scale: f32,
    mie_height_scale: f32,
    cloud_phase_g_forward: f32,
    cloud_phase_g_back: f32,
    cloud_phase_blend: f32,
    cloud_optical_depth_scale: f32,
    cloud_light_sample_dist: f32,
    _pad3: f32,
    cloud_albedo: vec3<f32>,
    _pad4: f32,
    cloud_sky_ambient_day: vec3<f32>,
    _pad5: f32,
    cloud_sky_ambient_sunset: vec3<f32>,
    _pad6: f32,
    cloud_ground_ambient_day: vec3<f32>,
    _pad7: f32,
    cloud_ground_ambient_sunset: vec3<f32>,
    _pad8: f32,
    cloud_bg_day: vec3<f32>,
    _pad9: f32,
    cloud_bg_sunset: vec3<f32>,
    _pad10: f32,
};

@group(0) @binding(0) var<uniform> params: CloudParams;
@group(0) @binding(1) var noise_3d: texture_3d<f32>;       // 128^3 Perlin-Worley base shape
@group(0) @binding(2) var weather_map: texture_2d<f32>;
@group(0) @binding(3) var blue_noise: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var depth_tex: texture_depth_2d;
@group(0) @binding(6) var nearest_samp: sampler;
@group(0) @binding(7) var detail_3d: texture_3d<f32>;      // 32^3 high-freq Worley erosion

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const GOLDEN_RATIO: f32 = 1.61803398875;
const PI: f32 = 3.14159265;

// --- 3D value noise (hash-based — proven to produce good cloud shapes) ---

fn hash_3d(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(mix(hash_3d(i + vec3(0.0, 0.0, 0.0)),
                       hash_3d(i + vec3(1.0, 0.0, 0.0)), u.x),
                   mix(hash_3d(i + vec3(0.0, 1.0, 0.0)),
                       hash_3d(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
               mix(mix(hash_3d(i + vec3(0.0, 0.0, 1.0)),
                       hash_3d(i + vec3(1.0, 0.0, 1.0)), u.x),
                   mix(hash_3d(i + vec3(0.0, 1.0, 1.0)),
                       hash_3d(i + vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z) * 2.0 - 1.0;
}

fn fbm5(p: vec3<f32>) -> f32 {
    var q = p;
    var f = 0.5000 * noise(q); q = q * 2.02;
    f    += 0.2500 * noise(q); q = q * 2.03;
    f    += 0.1250 * noise(q); q = q * 2.01;
    f    += 0.0625 * noise(q); q = q * 2.04;
    f    += 0.03125 * noise(q);
    return f;
}

fn fbm3(p: vec3<f32>) -> f32 {
    var q = p;
    var f = 0.5000 * noise(q); q = q * 2.02;
    f    += 0.2500 * noise(q); q = q * 2.03;
    f    += 0.1250 * noise(q);
    return f;
}

// --- Height gradient for cumulus cloud profile ---

fn cumulus_gradient(h: f32) -> f32 {
    return smoothstep(0.0, 0.1, h) * (1.0 - smoothstep(0.6, 1.0, h));
}

// --- Spherical height fraction ---

fn sphere_height_frac(p: vec3<f32>) -> f32 {
    let earth_center = vec3(0.0, 0.0, -EARTH_RADIUS);
    let altitude = length(p - earth_center) - EARTH_RADIUS;
    return clamp((altitude - params.cloud_bottom) / (params.cloud_top - params.cloud_bottom), 0.0, 1.0);
}

// --- Cloud density: inline FBM for base shape + texture detail erosion ---

fn cloud_density(p: vec3<f32>, detail_level: bool) -> f32 {
    let height_frac = sphere_height_frac(p);
    if (height_frac <= 0.0 || height_frac >= 1.0) { return 0.0; }

    let height_grad = cumulus_gradient(height_frac);
    let wind = vec3(params.wind_x, params.wind_z, 0.0) * params.wind_speed * params.time;
    let sample_pos = (p + wind) * params.base_noise_scale;

    // Base shape from inline FBM
    let f = select(fbm3(sample_pos), fbm5(sample_pos), detail_level);
    let base = params.coverage + 0.6 * f;
    var den = clamp(base * height_grad, 0.0, 1.0) * 0.5;

    // Detail erosion from precomputed 32^3 Worley texture (smooth, no grid artifacts)
    if (detail_level && den > 0.01) {
        let detail_uvw = sample_pos * 0.15; // scale for 32^3 texture
        let detail_sample = textureSampleLevel(detail_3d, samp, detail_uvw, 0.0);
        let detail_fbm = detail_sample.r * 0.625 + detail_sample.g * 0.25 + detail_sample.b * 0.125;

        // Erode edges: more at top (wispy), less at bottom (dense base)
        let erosion = mix(1.0 - detail_fbm, detail_fbm, smoothstep(0.3, 0.7, height_frac));
        den = clamp(den - erosion * 0.2, 0.0, 1.0);
    }

    return den;
}

// --- Cornette-Shanks phase function (better Mie approximation than HG) ---

fn cs_phase(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    let norm = 3.0 / (8.0 * PI * (2.0 + g2));
    return norm * (1.0 - g2) * (1.0 + cos_theta * cos_theta) / (denom * sqrt(max(denom, 0.0001)));
}

fn cloud_phase(cos_theta: f32) -> f32 {
    let forward = cs_phase(params.cloud_phase_g_forward, cos_theta);
    let back = cs_phase(params.cloud_phase_g_back, cos_theta);
    return mix(forward, back, params.cloud_phase_blend);
}

// --- Beer-Powder light transmittance (HZD) with height-dependent powder ---
// Lower powder at cloud base lets more light penetrate undersides (warm sunset bellies).
// Higher powder at cloud top gives fluffy bright-edge appearance.

fn beer_powder(optical_depth: f32, height_frac: f32) -> f32 {
    let beer = exp(-optical_depth);
    let powder_exp = mix(0.5, 2.0, smoothstep(0.3, 0.85, height_frac));
    let powder = 1.0 - exp(-2.0 * pow(max(optical_depth, 0.001), powder_exp));
    return beer * powder;
}

// --- Multi-scattering approximation (Wrenninge/Frostbite) ---
// Three octaves: each successive bounce halves extinction & energy, trends toward isotropic.

fn cloud_lighting(light_density: f32, cos_theta: f32, sun_ext: vec3<f32>, height_frac: f32) -> vec3<f32> {
    let optical_depth = max(light_density * params.cloud_optical_depth_scale, 0.0);

    var scatter = 0.0;
    var ext_mult = 1.0;
    var energy_mult = 1.0;
    var phase_iso = 0.0; // 0 = full directional, 1 = full isotropic

    for (var ms = 0; ms < 3; ms++) {
        let phase_i = mix(cloud_phase(cos_theta), 1.0 / (4.0 * PI), phase_iso);
        let trans_i = beer_powder(optical_depth * ext_mult, height_frac);
        scatter += trans_i * phase_i * energy_mult;
        ext_mult *= 0.5;
        energy_mult *= 0.5;
        phase_iso = 1.0 - (1.0 - phase_iso) * 0.5; // 0 → 0.5 → 0.75
    }

    // Apply atmosphere-filtered sun color (warm orange at sunset)
    return params.sun_color * sun_ext * params.sun_intensity * scatter;
}

// --- Integration with improved lighting ---

fn integrate(sum: vec4<f32>, density: f32, height_frac: f32, light_density: f32,
             cos_theta: f32, t: f32, bg_color: vec3<f32>, sun_ext: vec3<f32>) -> vec4<f32> {

    // Direct sun with multi-scattering + atmospheric sun filtering
    let direct = cloud_lighting(light_density, cos_theta, sun_ext, height_frac);

    // Height-based ambient that adapts to time of day via sun extinction.
    // sun_ext.r >> sun_ext.b at sunset → warm ambient; sun_ext ≈ uniform at noon → cool blue.
    let sun_warmth = saturate((sun_ext.r - sun_ext.b) * 2.0);
    let sky_ambient = mix(params.cloud_sky_ambient_day, params.cloud_sky_ambient_sunset, sun_warmth);
    let ground_ambient = mix(params.cloud_ground_ambient_day, params.cloud_ground_ambient_sunset, sun_warmth);
    let ambient = mix(ground_ambient, sky_ambient, smoothstep(0.0, 0.6, height_frac));

    let lighting = ambient + direct;

    // Cloud albedo
    var col = vec4(params.cloud_albedo * lighting, density);

    // Aerial perspective — fade toward atmosphere-tinted sky at distance
    let aerial_blend = 1.0 - exp(-t * 0.00003);
    let aerial_color = mix(bg_color, bg_color * 1.3, smoothstep(0.0, 0.5, max(0.0, cos_theta)));
    col = vec4(mix(col.rgb, aerial_color, aerial_blend), col.a);

    // Alpha dampening — distance-dependent to prevent horizon buildup
    let alpha_damp = mix(0.1, 0.02, smoothstep(10000.0, 40000.0, t));
    col = vec4(col.rgb, col.a * alpha_damp);

    // Pre-multiplied alpha compositing (front-to-back)
    col = vec4(col.rgb * col.a, col.a);
    return sum + col * (1.0 - sum.a);
}

// --- Ray-sphere intersection ---
// Returns (t_near, t_far) or (-1, -1) if no intersection.

const EARTH_RADIUS: f32 = 6371000.0;

fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return vec2(-1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2(-b - s, -b + s);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.quarter_w || id.y >= params.quarter_h) {
        return;
    }

    // Reconstruct world-space ray
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.quarter_w), f32(params.quarter_h));
    let ndc = vec4(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0, 1.0);
    let world_pos = params.inverse_view_projection * ndc;
    let ray_dir = normalize(world_pos.xyz / world_pos.w - params.camera_position);

    // Intersect ray with spherical cloud shell (earth curvature)
    // Earth center is below us at (0, 0, -EARTH_RADIUS)
    let earth_center = vec3(0.0, 0.0, -EARTH_RADIUS);
    let inner_radius = EARTH_RADIUS + params.cloud_bottom;
    let outer_radius = EARTH_RADIUS + params.cloud_top;

    let hit_inner = ray_sphere_intersect(params.camera_position, ray_dir, earth_center, inner_radius);
    let hit_outer = ray_sphere_intersect(params.camera_position, ray_dir, earth_center, outer_radius);

    var t_start: f32;
    var t_end: f32;

    let cam_height = params.camera_position.z;

    if (cam_height < params.cloud_bottom) {
        // Below cloud layer — need to hit inner sphere going up
        if (hit_inner.x < 0.0 && hit_inner.y < 0.0) {
            textureStore(output, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
            return;
        }
        // Enter at far side of inner sphere, exit at far side of outer sphere
        t_start = max(hit_inner.y, 0.0);
        if (hit_outer.y < 0.0) {
            textureStore(output, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
            return;
        }
        t_end = hit_outer.y;
    } else if (cam_height > params.cloud_top) {
        // Above cloud layer — need to hit outer sphere going down
        if (hit_outer.x < 0.0) {
            textureStore(output, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
            return;
        }
        t_start = hit_outer.x;
        // Exit at inner sphere (or far side of outer if we miss inner)
        if (hit_inner.x > 0.0) {
            t_end = hit_inner.x;
        } else {
            t_end = hit_outer.y;
        }
    } else {
        // Inside cloud layer
        t_start = 0.0;
        // Exit at whichever sphere we hit first (going outward)
        var t_exit = 100000.0;
        if (hit_outer.y > 0.0) { t_exit = min(t_exit, hit_outer.y); }
        if (hit_inner.x > 0.0) { t_exit = min(t_exit, hit_inner.x); }
        t_end = t_exit;
    }

    if (t_start < 0.0 || t_end <= t_start) {
        textureStore(output, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Cap max march distance
    t_end = min(t_end, t_start + 50000.0);

    // Precompute sun-ray angle for phase function
    let cos_theta = dot(ray_dir, params.sun_direction);

    // Atmospheric sun extinction — how much sunlight is filtered before reaching clouds
    let sun_sin_elev = max(params.sun_direction.z, 0.01);
    let cloud_sun_air_mass = params.sun_air_mass_scale / sun_sin_elev;
    let sun_ext = exp(-(params.rayleigh_coefficients * params.rayleigh_height_scale * cloud_sun_air_mass
                       + vec3(params.mie_coefficient) * params.mie_height_scale * cloud_sun_air_mass));

    // Sky-tinted fog color for aerial perspective — adapts to sunset
    let sun_warmth = saturate((sun_ext.r - sun_ext.b) * 2.0);
    let bg_color = mix(params.cloud_bg_day, params.cloud_bg_sunset, sun_warmth);

    // Blue noise dithered start
    let noise_uv = vec2<f32>(id.xy) / 256.0;
    let blue = textureSampleLevel(blue_noise, samp, noise_uv, 0.0).r;
    let jitter = fract(blue + f32(params.frame_index % 16u) * GOLDEN_RATIO);

    var sum = vec4(0.0);
    var t = t_start;

    let min_step = max((t_end - t_start) / 48.0, 10.0);
    t += jitter * min_step;

    for (var i = 0; i < 48; i++) {
        if (t > t_end || sum.a > 0.99) {
            break;
        }

        let pos = params.camera_position + ray_dir * t;

        // Bounds check using spherical altitude
        let height_frac = sphere_height_frac(pos);
        if (height_frac < 0.0 || height_frac > 1.0) {
            t += max(min_step, 0.02 * t);
            continue;
        }

        // LOD: detail erosion close, base shape only far
        let use_detail = t < 15000.0;
        var den = cloud_density(pos, use_detail);

        // Distance attenuation — prevents solid horizon band
        let dist_atten = 1.0 / (1.0 + t * 0.00005);
        den *= dist_atten;

        if (den > 0.08) {
            // Light sample: density toward sun for Beer-Powder shadowing
            let sun_offset = params.sun_direction * params.cloud_light_sample_dist;
            let light_den = cloud_density(pos + sun_offset, false);

            sum = integrate(sum, den, height_frac, light_den, cos_theta, t, bg_color, sun_ext);
        }

        t += max(min_step, 0.02 * t);
    }

    let transmittance = 1.0 - sum.a;
    let scattered = sum.rgb;

    textureStore(output, id.xy, vec4(scattered, transmittance));
}
