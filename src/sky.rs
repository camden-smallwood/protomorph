use glam::Vec3;

/// Central configuration for all sky, atmosphere, and cloud visual parameters.
/// Construct this in one place and everything flows to the GPU via uniform buffers.
pub struct SkyConfig {
    // --- Atmosphere scattering ---
    pub rayleigh_coefficients: [f32; 3],
    pub rayleigh_height_scale: f32,
    pub mie_coefficient: f32,
    pub mie_height_scale: f32,
    pub mie_g: f32,
    pub max_fog_thickness: f32,
    pub inscatter_scale: f32,
    pub reference_height: f32,

    // --- Sun direction & light colors ---
    pub sun_direction: Vec3,
    pub sun_diffuse_color: Vec3,
    pub sun_ambient_color: Vec3,
    pub sun_specular_color: Vec3,

    // --- Sun disc appearance ---
    pub sun_luminance: f32,
    pub sun_disc_intensity: f32,
    pub sun_inner_glow_intensity: f32,
    pub sun_angular_radius: f32,
    pub sun_edge_softness: f32,
    pub sun_tint: [f32; 3],
    pub sun_air_mass_scale: f32,
    pub zenith_air_mass_factor: f32,
    pub horizon_fade_start: f32,
    pub horizon_fade_end: f32,

    // --- Cloud shape ---
    pub cloud_bottom: f32,
    pub cloud_top: f32,
    pub cloud_coverage: f32,
    pub cloud_wind_x: f32,
    pub cloud_wind_z: f32,
    pub cloud_wind_speed: f32,
    pub cloud_noise_scale: f32,
    pub cloud_extinction: f32,
    pub cloud_sun_intensity: f32,

    // --- Cloud lighting ---
    pub cloud_albedo: [f32; 3],
    pub cloud_phase_g_forward: f32,
    pub cloud_phase_g_back: f32,
    pub cloud_phase_blend: f32,
    pub cloud_optical_depth_scale: f32,
    pub cloud_light_sample_dist: f32,

    // --- Cloud ambient ---
    pub cloud_sky_ambient_day: [f32; 3],
    pub cloud_sky_ambient_sunset: [f32; 3],
    pub cloud_ground_ambient_day: [f32; 3],
    pub cloud_ground_ambient_sunset: [f32; 3],
    pub cloud_bg_day: [f32; 3],
    pub cloud_bg_sunset: [f32; 3],
}

impl Default for SkyConfig {
    fn default() -> Self {
        Self {
            // Atmosphere scattering
            rayleigh_coefficients: [0.02, 0.05, 0.1],
            rayleigh_height_scale: 20.0,
            mie_coefficient: 0.003,
            mie_height_scale: 8.0,
            mie_g: 0.76,
            max_fog_thickness: 50.0,
            inscatter_scale: 1.0,
            reference_height: 0.0,

            // Sun direction & light colors (early sunset, ~10° elevation)
            sun_direction: Vec3::new(-0.7, 0.4, -0.15).normalize(),
            sun_diffuse_color: Vec3::new(1.2, 0.9, 0.6),
            sun_ambient_color: Vec3::new(0.12, 0.11, 0.1),
            sun_specular_color: Vec3::new(1.2, 0.9, 0.6),

            // Sun disc appearance
            sun_luminance: 20.0,
            sun_disc_intensity: 150.0,
            sun_inner_glow_intensity: 1.0,
            sun_angular_radius: 0.025,
            sun_edge_softness: 0.003,
            sun_tint: [1.0, 0.95, 0.9],
            sun_air_mass_scale: 0.2,
            zenith_air_mass_factor: 0.25,
            horizon_fade_start: -0.05,
            horizon_fade_end: 0.0,

            // Cloud shape
            cloud_bottom: 1500.0,
            cloud_top: 5000.0,
            cloud_coverage: 0.15,
            cloud_wind_x: 1.0,
            cloud_wind_z: 0.3,
            cloud_wind_speed: 20.0,
            cloud_noise_scale: 0.001,
            cloud_extinction: 0.02,
            cloud_sun_intensity: 15.0,

            // Cloud lighting
            cloud_albedo: [1.0, 0.98, 0.95],
            cloud_phase_g_forward: 0.88,
            cloud_phase_g_back: -0.4,
            cloud_phase_blend: 0.25,
            cloud_optical_depth_scale: 2.0,
            cloud_light_sample_dist: 300.0,

            // Cloud ambient
            cloud_sky_ambient_day: [0.55, 0.6, 0.75],
            cloud_sky_ambient_sunset: [0.6, 0.4, 0.3],
            cloud_ground_ambient_day: [0.25, 0.22, 0.18],
            cloud_ground_ambient_sunset: [0.55, 0.3, 0.12],
            cloud_bg_day: [0.4, 0.55, 0.8],
            cloud_bg_sunset: [0.5, 0.35, 0.25],
        }
    }
}
