//! Spherical-harmonics (SH3) + dominant-light extraction math.
//!
//! These are general lighting math helpers (engine `c_lighting_interface`
//! SH path) moved out of the scenario loader, which only happened to host
//! them. NOTE: `render/lighting_interface.rs` carries its OWN, distinct
//! copies of `find_dominant_light_direction` /
//! `calculate_dominant_light_from_lightprobe` (different signatures /
//! call conventions) — these are deliberately kept separate and are NOT
//! the same functions.

use blam_tags::math::RealVector3d;

/// dllcache `find_dominant_light_direction`. Computes direction TO the
/// dominant light from SH3 L1 coefficients using ITU-R BT.709 luminance
/// weighting (0.213 R + 0.715 G + 0.072 B).
///
/// The L1 components of the SH probe encode the luminance gradient:
/// `sh[1]` carries the y-axis sensitivity, `sh[2]` z, `sh[3]` x. Halo's
/// sign convention negates x and y on extraction (so the returned
/// direction points TOWARD the source).
pub fn find_dominant_light_direction(probe_r: &[f32; 9], probe_g: &[f32; 9], probe_b: &[f32; 9])
    -> RealVector3d
{
    let lum = |i: usize| 0.21265601 * probe_r[i] + 0.71515799 * probe_g[i] + 0.072185598 * probe_b[i];
    let mut x = -lum(3);
    let mut y = -lum(1);
    let mut z = lum(2);
    let len = (x*x + y*y + z*z).sqrt();
    if len > 1e-6 { x /= len; y /= len; z /= len; }
    RealVector3d { i: x, j: y, k: z }
}

/// dllcache `sh_eval_direction(out, order=3, dir)`. Evaluates the SH3
/// basis functions at a given direction. Halo's sign convention
/// negates the m≠0 components of the L1/L2 bands.
pub fn sh_eval_direction_order3(dir: &RealVector3d) -> [f32; 9] {
    let x = dir.i; let y = dir.j; let z = dir.k;
    let inv_sqrt_pi = 1.0 / std::f32::consts::PI.sqrt();
    let sqrt3 = 3.0_f32.sqrt();
    let sqrt5 = 5.0_f32.sqrt();
    let sqrt15 = 15.0_f32.sqrt();
    let l1 = sqrt3 * inv_sqrt_pi;
    let l2 = sqrt15 * inv_sqrt_pi;
    let mz = sqrt5 * inv_sqrt_pi;
    [
        inv_sqrt_pi * 0.5,                       // 1/(2√π)
        l1 * y * -0.5,                           // -y √(3/π)/2
        l1 * z *  0.5,                           //  z √(3/π)/2
        l1 * x * -0.5,                           // -x √(3/π)/2
        l2 * (y * x) *  0.5,                     //  xy √(15/π)/2
        l2 * (y * z) * -0.5,                     // -yz √(15/π)/2
        mz * (3.0 * z * z - 1.0) * 0.25,         // (3z²-1) √(5/π)/4
        l2 * (z * x) * -0.5,                     // -xz √(15/π)/2
        l2 * (x*x - y*y) * 0.25,                 // (x²-y²) √(15/π)/4
    ]
}

/// dllcache `calculate_dominant_light_from_lightprobe`. Extracts a
/// dominant light direction + intensity from an SH3 probe by:
/// 1) finding the brightest direction (luminance-weighted L1),
/// 2) evaluating the SH3 basis at that direction,
/// 3) projecting each channel's probe onto the basis (dot/self_dot).
///
/// Returns `(direction, intensity_rgb)` in the same units as the input
/// probe — so for a probe with sh[0]≈1.2 the intensity is also ~1ish.
pub fn calculate_dominant_light_from_lightprobe(
    probe_r: &[f32; 9], probe_g: &[f32; 9], probe_b: &[f32; 9],
) -> (RealVector3d, RealVector3d) {
    let dir = find_dominant_light_direction(probe_r, probe_g, probe_b);
    let basis = sh_eval_direction_order3(&dir);

    let mut self_dot = 0.0f32;
    for v in basis.iter() { self_dot += v * v; }
    let denom = if self_dot <= 1e-4 { 1e-4 } else { self_dot };

    let project = |probe: &[f32; 9]| -> f32 {
        let mut s = 0.0f32;
        for i in 0..9 { s += probe[i] * basis[i]; }
        if s < 0.0 { 0.0 } else { s / denom }
    };

    let intensity = RealVector3d {
        i: project(probe_r),
        j: project(probe_g),
        k: project(probe_b),
    };
    (dir, intensity)
}
