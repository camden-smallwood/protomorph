//! Mirror of `Ares/source/rasterizer/rasterizer_hue_saturation.h`
//! (58 lines).
//!
//! `c_hue_saturation_control` — runtime hue/saturation/contrast/per-
//! channel filter values. Drives the LUT generator's hue rotation
//! + saturation matrix + RGB filter pass alongside
//! `c_rasterizer_LUT_control`.

use glam::Vec3;

/// `c_hue_saturation_control` (h:18-38, 20B).
#[derive(Debug, Clone, Copy)]
pub struct HueSaturationControl {
    pub current_hue: f32,
    pub current_saturation: f32,
    pub current_filter_red: f32,
    pub current_filter_green: f32,
    pub current_filter_blue: f32,
}

impl HueSaturationControl {
    /// Halo `c_hue_saturation_control::reset @ h:22`. Resets to
    /// neutral (hue=0, sat=1, filters=1).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn set_hue(&mut self, hue: f32) {
        self.current_hue = hue;
    }

    pub fn set_saturation(&mut self, saturation: f32) {
        self.current_saturation = saturation;
    }

    pub fn set_color_filter_red(&mut self, red: f32) {
        self.current_filter_red = red;
    }
    pub fn set_color_filter_green(&mut self, green: f32) {
        self.current_filter_green = green;
    }
    pub fn set_color_filter_blue(&mut self, blue: f32) {
        self.current_filter_blue = blue;
    }

    /// Halo: `set_hue_saturaton_and_color_filters` (Bungie typo
    /// "saturaton" preserved in canonical name; we use "saturation"
    /// in Rust). Combines all settings into a final
    /// (color_scale, color_offset) pair the shader applies.
    pub fn set_all(
        &mut self,
        hue_bias: f32,
        saturation_bias: f32,
        contrast_enhance: f32,
        out_color_scale: &mut Vec3,
        out_color_offset: &mut Vec3,
    ) {
        self.current_hue = hue_bias;
        self.current_saturation = saturation_bias;
        // contrast_enhance maps to a (scale, offset) per channel:
        //   value' = (value - 0.5) * contrast + 0.5
        // → scale = contrast, offset = 0.5*(1 - contrast)
        let contrast = 1.0 + contrast_enhance;
        let offset = 0.5 * (1.0 - contrast);
        *out_color_scale = Vec3::new(
            self.current_filter_red * contrast,
            self.current_filter_green * contrast,
            self.current_filter_blue * contrast,
        );
        *out_color_offset = Vec3::splat(offset);
    }
}

impl Default for HueSaturationControl {
    fn default() -> Self {
        Self {
            current_hue: 0.0,
            current_saturation: 1.0,
            current_filter_red: 1.0,
            current_filter_green: 1.0,
            current_filter_blue: 1.0,
        }
    }
}
