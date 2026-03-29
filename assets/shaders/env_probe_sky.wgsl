// Procedural atmospheric sky for env probe cubemap faces.
// Same scattering logic as the merged sky in lighting.wgsl.

struct SkyParams {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> sky: SkyParams;

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
@group(0) @binding(1) var<uniform> atmosphere: AtmosphereData;

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

const PI: f32 = 3.14159265;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    // Reconstruct world-space ray direction from UV via inverse VP
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0, 1.0);
    let world_pos = sky.inverse_view_projection * ndc;
    let ray_dir = normalize(world_pos.xyz / world_pos.w - sky.camera_position);

    let sun_dir_len = length(atmosphere.sun_direction);
    let sun_dir = select(vec3<f32>(0.0, 0.0, 1.0), atmosphere.sun_direction / sun_dir_len, sun_dir_len > 0.0001);

    let h_cam = max(sky.camera_position.z - atmosphere.reference_height, 0.0);
    let rayleigh_density_at_cam = exp(-h_cam / atmosphere.rayleigh_height_scale);
    let mie_density_at_cam = exp(-h_cam / atmosphere.mie_height_scale);

    // View ray air mass
    let sin_elev = max(ray_dir.z, 0.02);
    let view_air_mass = 1.0 / sin_elev;

    let rayleigh_depth = rayleigh_density_at_cam * atmosphere.rayleigh_height_scale * view_air_mass;
    let mie_depth = mie_density_at_cam * atmosphere.mie_height_scale * view_air_mass;

    // View extinction
    let extinction = exp(-(atmosphere.rayleigh_coefficients * rayleigh_depth
                         + vec3(atmosphere.mie_coefficient) * mie_depth));

    // Sun extinction — atmospheric filtering of sunlight before it scatters into view.
    // View-dependent: horizon scatter points get full filtering (warm sunset),
    // zenith scatter points get reduced filtering (preserves blue sky above).
    let sun_sin_elev = max(sun_dir.z, 0.01);
    let base_sun_air_mass = atmosphere.sun_air_mass_scale / sun_sin_elev;
    let zenith_lerp = smoothstep(0.0, 0.5, max(ray_dir.z, 0.0));
    let sun_air_mass = base_sun_air_mass * mix(1.0, atmosphere.zenith_air_mass_factor, zenith_lerp);

    let sun_rayleigh_depth = rayleigh_density_at_cam * atmosphere.rayleigh_height_scale * sun_air_mass;
    let sun_mie_depth = mie_density_at_cam * atmosphere.mie_height_scale * sun_air_mass;
    let sun_extinction = exp(-(atmosphere.rayleigh_coefficients * sun_rayleigh_depth
                             + vec3(atmosphere.mie_coefficient) * sun_mie_depth));

    // Phase functions
    let cos_theta = dot(ray_dir, sun_dir);
    let rayleigh_phase = 0.05968 * (1.0 + cos_theta * cos_theta);

    let g = atmosphere.mie_g;
    let g2 = g * g;
    let mie_base = max(1.0 + g2 - 2.0 * g * cos_theta, 0.00001);
    let mie_phase = 0.07958 * (1.0 - g2) / pow(mie_base, 1.5);

    // Inscatter — modulated by sun_extinction (warm sky at low sun angles)
    let scatter = vec3(1.0) - extinction;
    var sky_color = (atmosphere.rayleigh_coefficients * rayleigh_phase
                   + vec3(atmosphere.mie_coefficient) * mie_phase)
                   * scatter * sun_extinction * atmosphere.inscatter_scale * atmosphere.sun_luminance;

    // Sun disc with limb darkening + glow layers
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

    // Darken below horizon
    let horizon_fade = smoothstep(atmosphere.horizon_fade_start, atmosphere.horizon_fade_end, ray_dir.z);
    sky_color *= horizon_fade;

    // Sanitize output — env probe writes to Rgba16Float
    sky_color = max(sky_color, vec3(0.0));
    sky_color = min(sky_color, vec3(500.0));
    return vec4<f32>(sky_color, 1.0);
}
