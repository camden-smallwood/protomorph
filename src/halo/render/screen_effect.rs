//! Screen-effect (`sefc` / `area_screen_effect`) color grade — the engine's
//! per-frame screen-space color filter (the mainmenu blue tint, damage
//! flashes, etc.).
//!
//! Engine path (dllcache + Ares `effects/screen_effect.cpp`,
//! `rasterizer/rasterizer_hue_saturation.cpp`):
//!   1. `screen_effect_sample @ 0x1803A4E90` starts from
//!      `s_screen_effect_settings::set_defaults` and `accumulate_settings`
//!      (`0x1803A4600`) each active effect. The scenario's
//!      `global screen effect` is accumulated at falloff = 1.0 every frame.
//!   2. `c_screen_postprocess::postprocess_player_view @ 0x1806B4160` maps the
//!      accumulated settings:
//!        hue_bias        = hue_right - hue_left
//!        saturation_bias = saturate  - desaturate
//!        contrast        = contrast_enhance
//!        color_scale     = color_filter   (per-channel multiply)
//!        color_offset    = color_floor    (per-channel add)
//!        gamma           = clamp((gamma_enhance + 1) - gamma_reduce, 0.01, 10)
//!      then `set_hue_saturaton_and_color_filters @ 0x1806956B0` builds a 3×4
//!      color matrix (shader constant `0x420002`) and gamma (`0x420005`),
//!      consumed in `copy_accumulation_target` (= our `final_composite_pass`).
//!   3. The matrix builder is a VERBATIM port of Paul Haeberli's "Matrix
//!      Operations for Image Processing" (SGI 1993) — the engine's
//!      `build_saturation_matrix`/`build_hue_matrix`/`xrotatemat`/`yrotatemat`/
//!      `zrotatemat`/`zshearmat` are Haeberli's functions by name. We reproduce
//!      that algorithm with Haeberli's luminance weights.

use blam_tags::area_screen_effect::{AreaScreenEffect, ScreenEffectSettings};

// Haeberli luminance weights (RLUM/GLUM/BLUM), verbatim.
const RLUM: f32 = 0.3086;
const GLUM: f32 = 0.6094;
const BLUM: f32 = 0.0820;

/// The composite-ready color grade derived from the accumulated screen-effect
/// settings: a column-major 4×4 color matrix (`out.rgb = M·(in.rgb, 1)`,
/// engine `ps_postprocess_hue_saturation_matrix` / const 0x420002) and the
/// luminance-gamma exponent (engine `ps_postprocess_contrast` / const 0x420005;
/// the final-composite shader applies `blend *= pow(luminance, gamma)/luminance`).
#[derive(Debug, Clone, Copy)]
pub struct ScreenEffectGrade {
    pub matrix: [[f32; 4]; 4],
    /// Engine `ps_postprocess_contrast.x` — despite the register name this
    /// carries the screen-effect GAMMA `clamp((gamma_enhance+1)-gamma_reduce,
    /// 0.01, 10)`. 1.0 = no change. (contrast_enhance is folded into `matrix`.)
    pub gamma: f32,
}

impl ScreenEffectGrade {
    /// Identity grade — no color shift, gamma 1.0.
    pub const IDENTITY: Self = Self {
        matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        gamma: 1.0,
    };
}

/// `screen_effect_sample` for the scenario's GLOBAL screen effect: start from
/// engine defaults and `accumulate_settings` every effect at falloff = 1.0.
/// (Per-area / triggered effects with real distance/time/angle falloff are a
/// gameplay-event system — not wired here; the global path is what mainmenu and
/// most ambient tints use.)
pub fn sample_global(ase: &AreaScreenEffect) -> ScreenEffectSettings {
    let mut acc = ScreenEffectSettings::defaults();
    for eff in &ase.effects {
        if eff.debug_disabled() {
            continue;
        }
        accumulate_settings(&mut acc, &eff.settings, 1.0);
    }
    acc
}

/// `s_screen_effect_settings::accumulate_settings @ 0x1803A4600`. Most fields
/// take `max(current, falloff·setting)`; color_filter takes
/// `min(current, falloff·setting + (1-falloff))` (starts at 1.0, the most-
/// tinting effect wins); color_floor takes the max.
pub fn accumulate_settings(
    acc: &mut ScreenEffectSettings,
    s: &ScreenEffectSettings,
    falloff: f32,
) {
    let f = falloff.clamp(0.0, 1.0);
    let inv = 1.0 - f;
    acc.exposure_boost = acc.exposure_boost.max(f * s.exposure_boost);
    acc.hue_left = acc.hue_left.max(f * s.hue_left);
    acc.hue_right = acc.hue_right.max(f * s.hue_right);
    acc.saturate = acc.saturate.max(f * s.saturate);
    acc.desaturate = acc.desaturate.max(f * s.desaturate);
    acc.contrast_enhance = acc.contrast_enhance.max(f * s.contrast_enhance);
    acc.gamma_enhance = acc.gamma_enhance.max(f * s.gamma_enhance);
    acc.gamma_reduce = acc.gamma_reduce.max(f * s.gamma_reduce);
    acc.color_filter.red = acc.color_filter.red.min(f * s.color_filter.red + inv);
    acc.color_filter.green = acc.color_filter.green.min(f * s.color_filter.green + inv);
    acc.color_filter.blue = acc.color_filter.blue.min(f * s.color_filter.blue + inv);
    acc.color_floor.red = acc.color_floor.red.max(f * s.color_floor.red);
    acc.color_floor.green = acc.color_floor.green.max(f * s.color_floor.green);
    acc.color_floor.blue = acc.color_floor.blue.max(f * s.color_floor.blue);
}

/// Build the composite color grade from accumulated settings, mirroring
/// `postprocess_player_view`'s field mapping + `set_hue_saturaton_and_color_
/// filters`. The renderer has no gameplay hue/sat/filter base state, so the
/// `c_hue_saturation_control` base = `reset()` defaults (hue 0, saturation 1,
/// filter (1,1,1)) and the screen-effect biases apply on top.
pub fn build_grade(s: &ScreenEffectSettings) -> ScreenEffectGrade {
    let hue = s.hue_right - s.hue_left; // + base hue 0
    let saturation = 1.0 + (s.saturate - s.desaturate); // + base saturation 1
    let scale = [s.color_filter.red, s.color_filter.green, s.color_filter.blue];
    let offset = [s.color_floor.red, s.color_floor.green, s.color_floor.blue];
    let gamma = ((s.gamma_enhance + 1.0) - s.gamma_reduce).clamp(0.01, 10.0);

    // Haeberli composition: M = colorfilter(scale) · saturation(s) · hue(h).
    let sat = saturation_matrix(saturation);
    let hue_m = hue_matrix(hue);
    let mut m3 = mat3_mul(&sat, &hue_m);
    // Per-channel color filter (diagonal scale on the OUTPUT rgb).
    for r in 0..3 {
        for c in 0..3 {
            m3[r][c] *= scale[r];
        }
    }
    // Fold contrast_enhance into the matrix (set_hue_saturaton folds contrast
    // into 0x420002, NOT the shader's pow-contrast which carries gamma).
    // Standard linear contrast about mid-gray 0.5: out = (1+c)·x - 0.5c.
    // Identity at c=0 (the common case incl. mainmenu). NOTE: the engine's exact
    // SSE folding isn't byte-reversed; this is the standard contrast, exact at c=0.
    let cm = 1.0 + s.contrast_enhance;
    let mut off = [offset[0] * scale[0], offset[1] * scale[1], offset[2] * scale[2]];
    for r in 0..3 {
        for c in 0..3 {
            m3[r][c] *= cm;
        }
        off[r] = cm * off[r] - 0.5 * s.contrast_enhance;
    }

    // Pack into a column-major 4×4 with the offset in the translation column
    // (out.rgb = M·in.rgb + off).
    let mut matrix = [[0.0f32; 4]; 4];
    for r in 0..3 {
        for c in 0..3 {
            matrix[c][r] = m3[r][c]; // column-major: matrix[col][row]
        }
        matrix[3][r] = off[r];
    }
    matrix[3][3] = 1.0;

    ScreenEffectGrade { matrix, gamma }
}

/// Haeberli `build_saturation_matrix(s)` — blend each channel toward the
/// luminance of the pixel by `(1-s)`. `s=1` identity, `s=0` grayscale.
fn saturation_matrix(s: f32) -> [[f32; 3]; 3] {
    let is = 1.0 - s;
    [
        [is * RLUM + s, is * GLUM, is * BLUM],
        [is * RLUM, is * GLUM + s, is * BLUM],
        [is * RLUM, is * GLUM, is * BLUM + s],
    ]
}

/// Haeberli `build_hue_matrix(hue_degrees)` — rotate the color cube about the
/// gray (1,1,1) axis by `hue` degrees, preserving luminance via the
/// shear (`zshearmat`) before the gray-axis rotation. Identity at hue=0.
fn hue_matrix(hue_deg: f32) -> [[f32; 3]; 3] {
    if hue_deg == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    // Haeberli: rotate the gray vector (1,1,1) into +Z, shear so the luminance
    // plane is horizontal, rotate by `hue` about Z, then undo shear + rotations.
    // `mat = rot · mat` accumulation (row-vector convention).
    let mut m = ident4();
    // xrotate: gray into the x=0 plane. mag = sqrt(2); xrs = xrc = 1/mag.
    let xinv = 1.0 / (2.0f32).sqrt();
    m = mat4_mul(&xrotatemat(xinv, xinv), &m);
    // yrotate: gray onto +Z. mag = sqrt(3); yrs = -1/mag; yrc = sqrt(2)/mag.
    let m3 = (3.0f32).sqrt();
    let yrs = -1.0 / m3;
    let yrc = (2.0f32).sqrt() / m3;
    m = mat4_mul(&yrotatemat(yrs, yrc), &m);
    // Shear so the luminance vector becomes horizontal (Haeberli: lx/lz, ly/lz).
    let lx = RLUM * m[0][0] + GLUM * m[1][0] + BLUM * m[2][0];
    let ly = RLUM * m[0][1] + GLUM * m[1][1] + BLUM * m[2][1];
    let lz = RLUM * m[0][2] + GLUM * m[1][2] + BLUM * m[2][2];
    let zsx = lx / lz;
    let zsy = ly / lz;
    m = mat4_mul(&zshearmat(zsx, zsy), &m);
    // The actual hue rotation about Z.
    let rad = hue_deg.to_radians();
    m = mat4_mul(&zrotatemat(rad.sin(), rad.cos()), &m);
    // Undo: shear back, yrotate back, xrotate back.
    m = mat4_mul(&zshearmat(-zsx, -zsy), &m);
    m = mat4_mul(&yrotatemat(-yrs, yrc), &m);
    m = mat4_mul(&xrotatemat(-xinv, xinv), &m);
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

fn ident4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
fn xrotatemat(rs: f32, rc: f32) -> [[f32; 4]; 4] {
    let mut m = ident4();
    m[1][1] = rc; m[1][2] = rs; m[2][1] = -rs; m[2][2] = rc;
    m
}
fn yrotatemat(rs: f32, rc: f32) -> [[f32; 4]; 4] {
    let mut m = ident4();
    m[0][0] = rc; m[0][2] = -rs; m[2][0] = rs; m[2][2] = rc;
    m
}
fn zrotatemat(rs: f32, rc: f32) -> [[f32; 4]; 4] {
    let mut m = ident4();
    m[0][0] = rc; m[0][1] = rs; m[1][0] = -rs; m[1][1] = rc;
    m
}
fn zshearmat(dx: f32, dy: f32) -> [[f32; 4]; 4] {
    let mut m = ident4();
    m[0][2] = dx; m[1][2] = dy;
    m
}
fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut o = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}
