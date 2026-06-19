//! `c_lights_view : c_world_view` — dynamic lights pass. Mirrors
//! Ares `render/views/render_view.h:90-156`.

use crate::halo::rasterizer::Surface;
use glam::Vec3;

/// `c_lights_view::s_simple_light` (Ares 80 bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleLight {
    pub position: Vec3,
    pub light_source_size: f32,
    pub inv_direction: Vec3,
    pub sphere: f32,
    pub color: Vec3,
    pub cone_smooth: f32,
    pub distance_scale: f32,
    pub cone_scale: f32,
    pub distance_offset: f32,
    pub cone_offset: f32,
    pub bounding_radius2: f32,
    pub _padding: [f32; 3],
}

/// `c_lights_view`. Engine inherits `c_world_view`; in protomorph
/// the world/view trait scaffolding was removed in the 2026-05-09
/// audit since no other view needed it. The fields used by our live
/// lighting path (simple_lights array, `initialize_simple_light`,
/// the few surface refs) live below.
#[derive(Debug, Clone, Default)]
pub struct LightsView {
    // 0x2A0+ in Ares
    pub ldr_surface: Surface,
    pub hdr_surface: Surface,
    pub depth_surface: Surface,
    pub texture_projection_matrix: [[f32; 4]; 4],
    pub light_orthogonal: bool,
    pub light_near_width: f32,
    pub light_near_height: f32,
    pub light_near_depth: f32,
    pub light_far_depth: f32,
    pub shadow_res_x: i32,
    pub shadow_res_y: i32,
    pub user_index: i32,
    pub simple_light_count: i32,
    pub local_light: SimpleLight,
    pub simple_lights: [SimpleLight; 8],
    pub light_intensity_scale: f32,
}

impl LightsView {
    /// Halo `c_lights_view::initialize_simple_light @ 0x1806c7c50`.
    /// Encodes a light into the shader-friendly form:
    ///   - inv_direction = -direction (cheaper than negating in PS)
    ///   - color / size  (Halo divides color by light_source_size for
    ///     a 1/r^2 attenuation falloff)
    ///   - cone_scale / cone_offset derived from cone_angle so the PS
    ///     can do a single mad to evaluate the cone falloff
    ///   - bounding_radius2 = max_dist^2 (PS rejects via squared
    ///     distance to skip sqrt)
    ///
    /// Verbatim port of dllcache `c_lights_view::initialize_simple_light
    /// @ 0x1806c7c50`:
    ///   bounding_radius2  = max_dist²
    ///   color             = authored_color / max(size, 1e-6)
    ///   cone_scale        = (1 - sphere)^(1/smoothness) / (1 - cos(angle/2))
    ///   cone_offset       = -cone_scale × cos(angle/2)
    ///   distance_scale    = (size + max_dist²) × size / max_dist²
    ///   distance_offset   = -size / max_dist²
    /// Designed so the HLSL evaluation `falloff.x = 1/(size+d²) ×
    /// distance_scale + distance_offset` produces 1.0 at d=0 and 0.0
    /// at d=max_dist, monotonically decreasing.
    pub fn initialize_simple_light(
        out: &mut SimpleLight,
        position: Vec3,
        color: Vec3,
        size: f32,
        max_dist: f32,
        direction: Vec3,
        cone_angle_radians: f32,
        cone_smoothness: f32,
        sphere_percentage: f32,
    ) {
        out.position = position;
        out.light_source_size = size;
        out.inv_direction = -direction;
        out.sphere = sphere_percentage;
        let size_clamped = if size > 1.0e-6 { size } else { 1.0e-6 };
        out.color = color / size_clamped;
        out.cone_smooth = cone_smoothness;
        let max_dist2 = max_dist * max_dist;
        out.bounding_radius2 = max_dist2;
        if cone_angle_radians >= 0.000099999997 {
            let cos_half = (cone_angle_radians * 0.5).cos();
            let cone_scale = (1.0 - sphere_percentage).powf(1.0 / cone_smoothness)
                / (1.0 - cos_half);
            out.cone_scale = cone_scale;
            out.cone_offset = -(cone_scale * cos_half);
        } else {
            // dllcache writes (0.0, 1.0) here so the cone term collapses
            // to a constant 1.0 — `pow(1.0, smoothness) + sphere` then
            // saturates to 1.0 in the HLSL combined-falloff product.
            out.cone_scale = 0.0;
            out.cone_offset = 1.0;
        }
        // dllcache always writes these (no `max_dist > epsilon` guard);
        // a max_dist of 0 would already have produced a 0 bounding radius
        // and the in-shader cull skips before this matters.
        let inv_max_dist2 = 1.0 / max_dist2;
        out.distance_scale = (size + max_dist2) * size * inv_max_dist2;
        out.distance_offset = -(size * inv_max_dist2);
    }

    /// Halo `c_lights_view::set_light_intensity_scale @ 0x18068e670`.
    pub fn set_light_intensity_scale(&mut self, scale: f32) {
        self.light_intensity_scale = scale;
    }
}

