//! `c_lighting_interface` — line-by-line port from dllcache.
//!
//! Source-of-truth functions:
//! - `c_lighting_interface::setup_default_lighting`              `@ 0x1806AA190`
//! - `calculate_dominant_light_from_lightprobe`                  `@ 0x18048DEC0`
//! - `find_dominant_light_direction`                             `@ 0x18048FBA0`
//! - `reconstruct_quadratic_lightprobe_from_linear_and_intensity` `@ 0x18048F900`
//! - `sh_eval_direction`                                         `@ 0x18051B880`
//! - `c_lighting_interface::calculate_and_set_ravi_constants`    `@ 0x18048E000` (caller of `_internal`)
//! - `c_lighting_interface::set_object_dominant_light_direction_and_intensity`
//!
//! These produce the per-frame "default lighting" state that flows into
//! the engine cbuffer extern table (slot 0x24 dominant light, slot 0x30
//! SH probe). See `reference_h3_extern_table.md` for the full slot map.

use glam::{Vec3, Vec4};

/// ITU-R BT.709 luminance weights — used to fold the per-channel L1
/// SH bands into a single dominant direction.
pub const LUMA_R: f32 = 0.21265601;
pub const LUMA_G: f32 = 0.71515799;
pub const LUMA_B: f32 = 0.072185598;

/// `find_dominant_light_direction(sr, sg, sb, &out)` from
/// dllcache `@ 0x18048FBA0`.
///
/// Folds the L1 (m=-1, 0, +1) SH bands across R/G/B into a luma-weighted
/// 3D direction, then negates X and Y (Z stays positive — Halo's
/// game-world basis). Returns the **unnormalized** length squared via
/// the normalize call's return; we drop it and just normalize.
///
/// `probe_*[1] = L1 m=-1 (Y axis component)`,
/// `probe_*[2] = L1 m=0  (Z axis component)`,
/// `probe_*[3] = L1 m=+1 (X axis component)`.
pub fn find_dominant_light_direction(
    probe_r: &[f32; 9],
    probe_g: &[f32; 9],
    probe_b: &[f32; 9],
) -> Vec3 {
    let x = -(probe_r[3] * LUMA_R + probe_g[3] * LUMA_G + probe_b[3] * LUMA_B);
    let y = -(probe_r[1] * LUMA_R + probe_g[1] * LUMA_G + probe_b[1] * LUMA_B);
    let z = probe_r[2] * LUMA_R + probe_g[2] * LUMA_G + probe_b[2] * LUMA_B;
    let v = Vec3::new(x, y, z);
    let len2 = v.length_squared();
    if len2 > 1.0e-10 {
        v / len2.sqrt()
    } else {
        Vec3::Z
    }
}

/// `sh_eval_direction(out, order, &dir)` from dllcache `@ 0x18051B880`.
///
/// Evaluates the real spherical harmonic basis at `dir`. Always fills
/// indices 0..3 (order 2) and 4..8 (order 3). Order 4 (indices 9..15)
/// is filled only when `order > 3`.
///
/// We store all 16 slots; callers pick the prefix.
///
/// Constants:
/// - `v6 = 1/√π`
/// - `v7 = √(3/π) = √3 * v6`
/// - `v8 = √15` (combined with v6 below)
pub fn sh_eval_direction(order: u32, dir: Vec3) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let v6 = 1.0 / std::f32::consts::PI.sqrt();
    let v7 = 3.0_f32.sqrt() * v6;

    out[0] = v6 * 0.5;
    out[1] = v7 * y * -0.5;
    out[2] = v7 * z * 0.5;
    out[3] = v7 * x * -0.5;

    if order <= 2 {
        return out;
    }

    let v8 = 15.0_f32.sqrt();
    out[4] = (y * x) * (v8 * v6) * 0.5;
    out[5] = (v8 * v6) * y * z * -0.5;
    out[6] = 5.0_f32.sqrt() * (z * z * 3.0 - 1.0) * v6 * 0.25;
    out[7] = (z * x) * (v8 * v6) * -0.5;
    out[8] = (x * x - y * y) * v8 * v6 * 0.25;

    if order > 3 {
        let xx = x * x;
        let yy = y * y;
        let zz5 = z * z * 5.0;
        let v11 = 2.0_f32.sqrt();
        let v12 = 35.0_f32.sqrt() * v11;
        let v13 = 105.0_f32.sqrt();
        let v14 = 21.0_f32.sqrt();

        out[9] = (xx * 3.0 - yy) * (v12 * y) * v6 * -0.125;
        out[10] = (v13 * v6) * (y * x) * z * 0.5;
        out[11] = (v14 * v11) * v6 * y * (zz5 - 1.0) * -0.125;
        out[12] = 7.0_f32.sqrt() * z * (zz5 - 3.0) * v6 * 0.25;
        out[13] = (v6 * x) * (v14 * v11) * (zz5 - 1.0) * -0.125;
        out[14] = (v6 * z) * (v13 * (xx - yy)) * 0.25;
        out[15] = (xx - yy * 3.0) * (v12 * x) * v6 * -0.125;
    }

    out
}

/// `calculate_dominant_light_from_lightprobe` from dllcache `@ 0x18048DEC0`.
///
/// Returns `(direction, intensity, contrast)`:
/// - `direction` — luma-weighted dominant from L1 (via
///   `find_dominant_light_direction`), normalized
/// - `intensity` — per-channel `(1/Σbasis²) · Σ(basis[i] · probe[i])`
/// - `contrast` — `(1 - residual_g/total_g)^4 × 10`, gates SH
///   reconstruction quality (used by detail SH renderer)
///
/// Decompile reference: `/tmp/calc_dominant.txt` (189 lines).
pub fn calculate_dominant_light_from_lightprobe(
    probe_r: &[f32; 9],
    probe_g: &[f32; 9],
    probe_b: &[f32; 9],
) -> (Vec3, Vec3, f32) {
    let dir = find_dominant_light_direction(probe_r, probe_g, probe_b);

    // sh_eval at dominant direction (we read 9 slots — order 3).
    let basis = sh_eval_direction(3, dir);

    // Σ(basis[i] · probe_X[i]) for X in {R, G, B}
    let mut acc_r = 0.0_f32;
    let mut acc_g = 0.0_f32;
    let mut acc_b = 0.0_f32;
    let mut sum_basis_sq = 0.0_f32;
    for i in 0..9 {
        acc_r += basis[i] * probe_r[i];
        acc_g += basis[i] * probe_g[i];
        acc_b += basis[i] * probe_b[i];
        sum_basis_sq += basis[i] * basis[i];
    }
    if acc_r < 0.0 { acc_r = 0.0; }
    if acc_g < 0.0 { acc_g = 0.0; }
    if acc_b < 0.0 { acc_b = 0.0; }

    // Halo guards against degenerate basis; matches `<= 9.9999997e-5`.
    let inv = if sum_basis_sq <= 9.999_9997e-5 {
        1.0 / 9.999_9997e-5
    } else {
        1.0 / sum_basis_sq
    };

    let intensity = Vec3::new(acc_r * inv, acc_g * inv, acc_b * inv);

    // contrast — green-channel reconstruction residual:
    //   numerator   = Σ(intensity_g · basis[i] - probe_g[i])²
    //   denominator = 2 · Σ(probe_g[i])²
    //   contrast    = ((1 - num/den)^4) · 10
    // Source uses probe_g specifically (luma proxy).
    let inten_g = intensity.y;
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..9 {
        let diff = inten_g * basis[i] - probe_g[i];
        num += diff * diff;
        den += probe_g[i] * probe_g[i] * 2.0;
    }
    let contrast = if den <= 9.999_9997e-5 {
        10.0
    } else {
        let f = 1.0 - num / den;
        f * f * f * f * 10.0
    };

    (dir, intensity, contrast)
}

/// `reconstruct_quadratic_lightprobe_from_linear_and_intensity` from
/// dllcache `@ 0x18048F900`.
///
/// Fills SH bands `[4..9]` (the L2 / quadratic terms) of each channel
/// by evaluating the SH basis at the dominant direction and scaling
/// per-channel by `dom_intensity`. The L0/L1 bands `[0..4]` are
/// untouched.
///
/// Engine call sites (per `c_geometry_sampler::sample @ 0x18048A450`):
/// - Per-vertex priority branch: after interpolating per-vertex SH
///   and copying the baked `m_interpolated_dominant_light_intensity`
///   into `m_dominant_light_intensity`.
/// - Per-pixel atlas branch: inside `sample_lightprobe_textures`
///   after decoding the BC3 SH + intensity bitmaps.
///
/// Without this, ravi SH eval at normals pointing away from L1's
/// dominant direction returns near-zero — receivers shading those
/// normals go black despite the probe having real L1 energy.
///
/// Engine also re-derives `dom_dir` via `find_dominant_light_direction`
/// from the L1 bands (in case the input direction is stale), then
/// reuses the freshly-computed direction for both basis eval and
/// contrast residual.
///
/// `contrast` quantifies how directional the SH is — `(1 - residual/total)^4 * 10`
/// on the green channel. Used downstream to gate detail-SH rendering
/// (high contrast = "almost a directional light"; low contrast =
/// "mostly ambient"). Sentinel `10.0` when the SH energy is degenerate.
pub fn reconstruct_quadratic_lightprobe_from_linear_and_intensity(
    sh_red: &mut [f32; 9],
    sh_green: &mut [f32; 9],
    sh_blue: &mut [f32; 9],
    dom_intensity: Vec3,
    dom_dir: &mut Vec3,
) -> f32 {
    *dom_dir = find_dominant_light_direction(sh_red, sh_green, sh_blue);
    let basis = sh_eval_direction(3, *dom_dir);

    for k in 4..9 {
        sh_red[k] = basis[k] * dom_intensity.x;
        sh_green[k] = basis[k] * dom_intensity.y;
        sh_blue[k] = basis[k] * dom_intensity.z;
    }

    let inten_g = dom_intensity.y;
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for k in 0..9 {
        let diff = inten_g * basis[k] - sh_green[k];
        num += diff * diff;
        den += sh_green[k] * sh_green[k] * 2.0;
    }
    if den <= 9.999_9997e-5 {
        10.0
    } else {
        let f = 1.0 - num / den;
        f * f * f * f * 10.0
    }
}

// =============================================================================
// sh_add + sh_eval_directional_light — Ares `math/spherical_harmonics.cpp`
// =============================================================================

/// `sh_add(out, order, a, b)` @ dllcache `0x18051b590`. Element-wise add
/// over `order * order` floats — for `order=3` that's 9 elements (SH3).
pub fn sh_add(out: &mut [f32], a: &[f32], b: &[f32], order: u32) {
    let n = (order as usize) * (order as usize);
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
}

/// `sh_eval_directional_light(order, dir, r_intensity, g_intensity, b_intensity, ...)`
/// @ dllcache `0x18051bd90`. Generates SH coefficients representing a
/// directional light at `dir` with per-channel intensity.
///
/// Engine body is ~190 LoC of dense math. The encoding is the standard
/// SH directional-light formula: `coef[k] = basis[k](dir) × intensity ×
/// band_scale[band(k)]` where `band_scale` are the Lambertian cosine-lobe
/// integrals per SH band:
///
/// | Band | Indices | A_l (= π × normalization) |
/// |---|---|---|
/// | 0 (L0)   | `[0]`     | `π`        |
/// | 1 (L1)   | `[1..4]`  | `2π/3`     |
/// | 2 (L2)   | `[4..9]`  | `π/4`      |
/// | 3 (L3+)  | `[9..16]` | `0` (windowed out for cosine projection) |
///
/// Used by `c_geometry_sampler::sample`'s bounce-light addition (engine
/// line 467-475) — wired in **Phase 7b.6c** when `sample_diffuse_textures`
/// lands.
pub fn sh_eval_directional_light(
    order: u32,
    dir: Vec3,
    r_intensity: f32,
    g_intensity: f32,
    b_intensity: f32,
    r_out: &mut [f32; 9],
    g_out: &mut [f32; 9],
    b_out: &mut [f32; 9],
) {
    let basis = sh_eval_direction(order, dir);
    let band_scale = [
        std::f32::consts::PI,             // L0
        2.0 * std::f32::consts::PI / 3.0, // L1
        std::f32::consts::PI / 4.0,       // L2
    ];
    for k in 0..9 {
        let band = if k == 0 { 0 } else if k < 4 { 1 } else { 2 };
        let s = band_scale[band];
        r_out[k] = basis[k] * r_intensity * s;
        g_out[k] = basis[k] * g_intensity * s;
        b_out[k] = basis[k] * b_intensity * s;
    }
}

/// `c_lighting_interface` — host-side mirror of the engine's static
/// per-frame lighting state.
///
/// Populated by `setup_default_lighting` (sky-tag fallback) and
/// `select_instance_entry_point` (per-instance single-probe overrides
/// via `calculate_and_set_ravi_constants` +
/// `set_object_dominant_light_direction_and_intensity`).
///
/// Read by externs 11 (atlas — defers to F1), 27..34 (dominant light
/// dir/intensity, 4 redundant routings).
#[derive(Debug, Clone, Default)]
pub struct LightingInterface {
    /// `m_dominant_light_dir` — vec4, w=1. World-space direction TO
    /// the dominant light. Routed to engine cbuffer slot 0x240000.
    pub dominant_light_dir: Vec4,
    /// `m_dominant_light_intensity` — RGB intensity. Routed to engine
    /// cbuffer slot 0x240001.
    pub dominant_light_intensity: Vec3,
    /// SH3 R coefficients (9 floats). Routed to engine cbuffer slot
    /// 0x300000.. via `calculate_and_set_ravi_constants`. Order-3 SH
    /// in Halo's basis ordering. Default sky probe sourced from
    /// `default_sky_lightprobe` block (offsets 61, 77, 93 floats per
    /// channel — see `reference_player_view_pass_setup.md`).
    pub sh_probe_r: [f32; 9],
    /// SH3 G coefficients.
    pub sh_probe_g: [f32; 9],
    /// SH3 B coefficients.
    pub sh_probe_b: [f32; 9],
}

impl LightingInterface {
    /// `setup_default_lighting` deepest-fallback path (dllcache LABEL_24).
    ///
    /// Mirrors the engine's behavior when no scenario sky chains
    /// resolve: reads `c_geometry_sampler::m_default_lightprobe_sample`
    /// (a hard-baked default probe) plus hardcoded globals for
    /// dir/intensity. We don't have the binary's static initializers,
    /// so this is an analytical neutral analog: top-down sun + warm-white
    /// intensity + ambient-gray L0 + +z L1 lobe.
    pub fn fallback() -> Self {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[lighting] LightingInterface::fallback() in use — no scenario sky chain \
                 resolved; rendering with the neutral warm-white floor (warned once)",
            );
        }
        // Y_00 = 1/(2√π). SH[0] is the constant term — solid ambient.
        const Y00_INV: f32 = 3.5449078; // 2 * √π
        // Y_1,0 = √(3/(4π)) z. SH[2] is the z directional lobe.
        const Y10_INV: f32 = 2.0466534; // 1 / √(3/(4π))
        let ambient = 0.3 * Y00_INV;
        let z_lobe = 0.5 * Y10_INV;

        let make_channel = |level: f32| {
            let mut sh = [0.0f32; 9];
            sh[0] = ambient * level;
            sh[2] = z_lobe * level;
            sh
        };

        Self {
            dominant_light_dir: Vec4::new(0.0, 0.0, 1.0, 1.0),
            dominant_light_intensity: Vec3::new(1.0, 0.97, 0.92),
            sh_probe_r: make_channel(1.0),
            sh_probe_g: make_channel(0.97),
            sh_probe_b: make_channel(0.92),
        }
    }

    /// `setup_default_lighting` happy path (dllcache lines 71-93):
    /// when a sky tag's `render_model.lightprobe` is reachable, run
    /// `calculate_dominant_light_from_lightprobe(probe_r, probe_g,
    /// probe_b, &dir, &intensity, &contrast)` and route through.
    ///
    /// `calculate_and_set_ravi_constants(probe_r, probe_g, probe_b, 1.0)`
    /// uploads the SH bands to engine cbuffer slot 0x30.
    /// `set_object_dominant_light_direction_and_intensity(&dir, &intensity)`
    /// uploads to engine cbuffer slot 0x24.
    pub fn from_lightprobe(
        probe_r: [f32; 9],
        probe_g: [f32; 9],
        probe_b: [f32; 9],
    ) -> Self {
        let (dir, intensity, _contrast) =
            calculate_dominant_light_from_lightprobe(&probe_r, &probe_g, &probe_b);
        Self {
            dominant_light_dir: dir.extend(1.0),
            dominant_light_intensity: intensity,
            sh_probe_r: probe_r,
            sh_probe_g: probe_g,
            sh_probe_b: probe_b,
        }
    }

    /// Analytical SH3 projection of (directional sun + ambient).
    /// Same math as `env_probe_pass::compute_analytical_sh`, just
    /// split per-channel. Used when only `(sun_dir, sun_color,
    /// ambient_color)` are available — no baked probe.
    ///
    /// `sun_dir` is the direction TO the sun.
    pub fn from_sun_and_ambient(sun_dir: Vec3, sun_color: Vec3, ambient_color: Vec3) -> Self {
        // SH basis constants (matches env_probe_pass)
        const Y00: f32 = 0.282095;   // 1/(2√π)
        const Y1X: f32 = 0.488603;   // √3/(2√π)
        const Y20: f32 = 0.315392;   // √5/(4√π) * (1/2)
        const Y21: f32 = 1.092548;   // √15/(2√π)
        const Y22: f32 = 0.546274;   // √15/(4√π)

        let d = sun_dir.normalize_or_zero();

        // L0 — ambient + directional constant term
        let l0 = (sun_color + ambient_color) * Y00;
        // L1 (Halo convention — see spherical_harmonics_fx.hlsl
        // `ravi_order_2_with_dominant_light`'s `dir_eval`).
        let l1_m1 = -sun_color * Y1X * d.y;
        let l1_0 =   sun_color * Y1X * d.z;
        let l1_1 =  -sun_color * Y1X * d.x;
        // L2
        let l2_m2 = sun_color * Y21 * d.x * d.y;
        let l2_m1 = sun_color * Y21 * d.y * d.z;
        let l2_0 = sun_color * Y20 * (3.0 * d.z * d.z - 1.0);
        let l2_1 = sun_color * Y21 * d.x * d.z;
        let l2_2 = sun_color * Y22 * (d.x * d.x - d.y * d.y);

        let pack = |c: fn(Vec3) -> f32| -> [f32; 9] {
            [
                c(l0),
                c(l1_m1), c(l1_0), c(l1_1),
                c(l2_m2), c(l2_m1), c(l2_0), c(l2_1), c(l2_2),
            ]
        };

        Self {
            dominant_light_dir: Vec4::new(d.x, d.y, d.z, 1.0),
            dominant_light_intensity: sun_color,
            sh_probe_r: pack(|v| v.x),
            sh_probe_g: pack(|v| v.y),
            sh_probe_b: pack(|v| v.z),
        }
    }
}
