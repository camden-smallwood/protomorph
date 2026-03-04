// Procedural atmospheric sky for env probe cubemap faces.
// Same scattering logic as sky.wgsl but fills every pixel (no G-buffer discard).

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
    _pad: vec2<f32>,
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

const SUN_ANGULAR_RADIUS: f32 = 0.00935;
const SUN_EDGE_SOFTNESS: f32 = 0.00015;
const SUN_INTENSITY: f32 = 50.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    // Reconstruct world-space ray direction from UV via inverse VP
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), 1.0, 1.0);
    let world_pos = sky.inverse_view_projection * ndc;
    let ray_dir = normalize(world_pos.xyz / world_pos.w - sky.camera_position);

    let sun_dir = normalize(atmosphere.sun_direction);

    let h_cam = max(sky.camera_position.z - atmosphere.reference_height, 0.0);
    let sin_elev = max(ray_dir.z, 0.02);
    let air_mass = 1.0 / sin_elev;

    let rayleigh_density_at_cam = exp(-h_cam / atmosphere.rayleigh_height_scale);
    let mie_density_at_cam = exp(-h_cam / atmosphere.mie_height_scale);
    let rayleigh_depth = rayleigh_density_at_cam * atmosphere.rayleigh_height_scale * air_mass;
    let mie_depth = mie_density_at_cam * atmosphere.mie_height_scale * air_mass;

    // Extinction (Beer's law)
    let extinction = exp(-(atmosphere.rayleigh_coefficients * rayleigh_depth
                         + vec3(atmosphere.mie_coefficient) * mie_depth));

    // Phase functions
    let cos_theta = dot(ray_dir, sun_dir);

    // Rayleigh: (3/16π)(1 + cos²θ)
    let rayleigh_phase = 0.05968 * (1.0 + cos_theta * cos_theta);

    // Mie: Henyey-Greenstein
    let g = atmosphere.mie_g;
    let g2 = g * g;
    let mie_phase = 0.07958 * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);

    // Inscatter
    let scatter = vec3(1.0) - extinction;
    let sun_luminance = 20.0;
    var sky_color = (atmosphere.rayleigh_coefficients * rayleigh_phase
                   + vec3(atmosphere.mie_coefficient) * mie_phase)
                   * scatter * atmosphere.inscatter_scale * sun_luminance;

    // Sun disk
    let sun_cos_angle = dot(ray_dir, sun_dir);
    let sun_edge = smoothstep(cos(SUN_ANGULAR_RADIUS + SUN_EDGE_SOFTNESS),
                              cos(SUN_ANGULAR_RADIUS - SUN_EDGE_SOFTNESS),
                              sun_cos_angle);
    sky_color += vec3<f32>(SUN_INTENSITY) * sun_edge;

    // Darken below horizon
    let horizon_fade = smoothstep(-0.05, 0.0, ray_dir.z);
    sky_color *= horizon_fade;

    return vec4<f32>(sky_color, 1.0);
}
