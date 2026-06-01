//! Mirror of Halo's lightprobe types from
//! `Ares/source/math/color_math.h:288-364`.
//!
//! Halo packs spherical harmonics light probes in several sizes:
//!   - mini = SH order 1 (4 coefficients per channel)  —  48B real / 24B half
//!   - full = SH order 2 (9 coefficients per channel)  —  108B real / 56B half
//!   - with-dominant-light = SH2 + dominant direction + intensity — 72B half
//!
//! `lightprobe_definitions.h` itself is empty — these structs live
//! in `color_math.h` because they're used by both renderer and
//! lightmapper. We pull them into `halo/geometry/lightprobe.rs` so
//! the render-path imports read `halo::geometry::lightprobe::...`.
//!
//! `half` = `f16` in Halo. Rust's stable `f16` lands sometime; until
//! then we represent the on-disk shape as `u16` and provide unpack
//! helpers via the `half` crate when needed (already used by the
//! BC-texture path).

/// `real_rgb_mini_lightprobe` (color_math.h:288-294, 48B). SH1 RGB.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealRgbMiniLightprobe {
    pub red_terms: [f32; 4],
    pub green_terms: [f32; 4],
    pub blue_terms: [f32; 4],
}

/// `half_rgb_mini_lightprobe` (color_math.h:296-302, 24B). On-disk
/// half-precision SH1 RGB.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfRgbMiniLightprobe {
    pub red_terms: [u16; 4],
    pub green_terms: [u16; 4],
    pub blue_terms: [u16; 4],
}

/// `real_rgb_lightprobe` (color_math.h:304-310, 108B). SH2 (9 coef) RGB.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealRgbLightprobe {
    pub red_terms: [f32; 9],
    pub green_terms: [f32; 9],
    pub blue_terms: [f32; 9],
}

/// `real_ycbcr_lightprobe` (color_math.h:320-324, 108B). SH2 in
/// Y/Cb/Cr space — used by the lightmapper for compression.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealYcbcrLightprobe {
    pub terms_y: [f32; 9],
    pub terms_cb: [f32; 9],
    pub terms_cr: [f32; 9],
}

/// `half_rgb_lightprobe` (color_math.h:335-342, 56B). On-disk
/// half-precision SH2 RGB. Includes 16-bit pad to round to a 4-byte
/// boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfRgbLightprobe {
    pub red_terms: [u16; 9],
    pub green_terms: [u16; 9],
    pub blue_terms: [u16; 9],
    pub _pad: u16,
}

/// `half_linear_rgb_color` (color_math.h:326-333, 8B).
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfLinearRgbColor {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub pad: u16,
}

/// `half_vector3d` — 3 × f16. Used by the dominant-light direction
/// in `half_rgb_lightprobe_with_dominant_light`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfVector3d {
    pub i: u16,
    pub j: u16,
    pub k: u16,
}

/// `half_rgb_lightprobe_with_dominant_light` (color_math.h:344-351,
/// 72B). The struct that
/// `s_scenario_lightmap_airprobe`/`scenery_probe`/`device_machine_probe`
/// store. Combines a directional dominant light with the SH2 probe
/// for the surrounding ambient.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfRgbLightprobeWithDominantLight {
    pub dominant_light_direction: HalfVector3d,
    pub _pad: u16,
    pub dominant_light_intensity: HalfLinearRgbColor,
    pub quadratic_probe: HalfRgbLightprobe,
}

/// `s_faux_logluv_lightprobe` (color_math.h:361-365, 36B). LogLuv-
/// packed SH2 used by the SH atlas texture format on PC.
#[derive(Debug, Clone, Copy, Default)]
pub struct FauxLogluvLightprobe {
    pub terms: [u32; 9],
}
