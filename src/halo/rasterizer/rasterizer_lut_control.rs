//! Mirror of `Ares/source/rasterizer/rasterizer_lut_control.h`
//! (56 lines).
//!
//! `c_rasterizer_LUT_control` packs the gamma + screen-brightness +
//! color-balance + black-level constants the rasterizer pumps into
//! the LUT generator. Authored from the cfxs tag's
//! `color_grading_parameter` block plus user display settings.

use glam::Vec3;

/// `c_rasterizer::e_buffer_gamma_mode`. Halo offers a few gamma
/// curves; sRGB is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum BufferGammaMode {
    #[default]
    Srgb = 0,
    Power22 = 1,
    Linear = 2,
}

/// `c_rasterizer_LUT_control` (rasterizer_lut_control.h:11-23, 40B).
#[derive(Debug, Clone, Copy)]
pub struct RasterizerLutControl {
    pub buffer_gamma_curve: BufferGammaMode,
    pub buffer_gamma: f32,
    pub screen_gamma: f32,
    pub display_brightness: i32,
    pub color_balance: Vec3,
    pub black_level: Vec3,
}

impl RasterizerLutControl {
    /// `c_rasterizer_LUT_control` constructor (h:20). Halo defaults:
    /// display_brightness=2, color_balance=(1,1,1), black_level=zero.
    pub fn new(buffer_gamma: f32, screen_gamma: f32, curve: BufferGammaMode) -> Self {
        Self {
            buffer_gamma_curve: curve,
            buffer_gamma,
            screen_gamma,
            display_brightness: 2,
            color_balance: Vec3::ONE,
            black_level: Vec3::ZERO,
        }
    }

    /// `same_curve_as @ h:21`.
    pub fn same_curve_as(&self, other: &Self) -> bool {
        self.buffer_gamma_curve == other.buffer_gamma_curve
            && (self.buffer_gamma - other.buffer_gamma).abs() < 1.0e-4
            && (self.screen_gamma - other.screen_gamma).abs() < 1.0e-4
    }
}

impl Default for RasterizerLutControl {
    fn default() -> Self {
        Self::new(2.2, 2.2, BufferGammaMode::Srgb)
    }
}
