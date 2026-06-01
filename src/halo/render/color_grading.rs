//! Engine-faithful port of `c_screen_postprocess::update_color_grading`
//! (dllcache `0x?` — 753-line per-texel 16³ LUT bake).
//!
//! The engine builds the postprocess color-grading LUT each frame the
//! settings change. Per-texel pipeline:
//!
//!   1. Sweep `(R, G, B) = (ix/16, iy/16, iz/16)` for ix,iy,iz ∈ [0, 16).
//!   2. **Curves editor** (`c_function_definition::evaluate_scalar` per
//!      channel) — *not yet ported* (skipped; matches identity).
//!   3. **Brightness, contrast** — additive brightness + linear contrast.
//!   4. **Color balance** (`SetColorBalanceParams` builds a Tones[3][3]
//!      matrix; per-channel pow remap) — *not yet ported* (skipped).
//!   5. RGB → HSL (degrees-based hue).
//!   6. **HSLV** — hue offset + saturation offset + skin-tone-aware
//!      saturation multiplier (engine "vibrance") + lightness offset.
//!   7. **Colorize** — H/S/L lerp toward authored target, with L target
//!      = luminance(R,G,B) biased toward 0 or 1.
//!   8. HSL → RGB (engine's m1/m2 formulation, hue ∈ [0, 360)).
//!   9. **Selective color** (RGB → naive CMYK → `ProcessSelectiveColor`
//!      → CMYK → RGB) — *not yet ported* (skipped).
//!  10. Clamp to [0, 1] and store as RGBA8.
//!
//! The skipped effects (curves editor, color balance, selective color)
//! depend on opaque helpers (`c_function_definition::evaluate_scalar`,
//! `SetColorBalanceParams`, `ProcessSelectiveColor`) whose bodies
//! aren't in the dllcache region we have. Their flag bits get
//! surfaced via [`ColorGradingBlock`] so callers can warn the user
//! when an authored cfxs would be visually incorrect.

use blam_tags::camera_fx_settings::{
    BrightnessContrast, CmybBand, ColorBalanceBlock, ColorGradingBlock, ColorizeBlock,
    CurvesEditorBlock, HslvBlock, SelectiveColorBlock,
};
use blam_tags::tag_function::TagFunction;

/// 3D LUT side length. Engine uses 16 (`FLOAT_0_0625` = 1/16 input
/// step in `update_color_grading`).
pub const LUT_DIM: u32 = 16;

/// Bytes per LUT (16 × 16 × 16 × RGBA8).
pub const LUT_BYTES: usize = (LUT_DIM as usize).pow(3) * 4;

/// Hue in degrees `[0, 360)`, saturation/lightness in `[0, 1]`. Engine
/// stores HSL as a `rendHSL { float H, S, L }` and operates on hue in
/// degrees throughout `update_color_grading`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hsl {
    pub h: f32,
    pub s: f32,
    pub l: f32,
}

/// `FromRGBToHSL` (declared at `camera_fx_settings.h:356`; body opaque
/// — this implementation matches the standard CIE HSL formulation
/// the engine's downstream code expects).
pub fn from_rgb_to_hsl(r: f32, g: f32, b: f32) -> Hsl {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let l = (max + min) * 0.5;

    if delta < 1.0e-6 {
        return Hsl { h: 0.0, s: 0.0, l };
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if r >= max {
        60.0 * (g - b) / delta
    } else if g >= max {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };

    Hsl { h, s, l }
}

/// One channel of the engine's HSL→RGB unwrap (lines 480-598 of the
/// `update_color_grading` decompile). Identical math for R, G, B —
/// only the hue offset differs (R: H+120, G: H, B: H-120).
fn hsl_channel(hue_in: f32, m1: f32, m2: f32) -> f32 {
    // Engine wraps with a one-step add/sub — values from the call
    // sites stay within (-120, 480), so a single ±360 is enough.
    let h = if hue_in > 360.0 {
        hue_in - 360.0
    } else if hue_in < 0.0 {
        hue_in + 360.0
    } else {
        hue_in
    };
    if h < 60.0 {
        m1 + (m2 - m1) * h * (1.0 / 60.0)
    } else if h < 180.0 {
        m2
    } else if h < 240.0 {
        m1 + (m2 - m1) * (240.0 - h) * (1.0 / 60.0)
    } else {
        m1
    }
}

/// `FromHSLtoRGB` — engine's degrees-based unwrap. R uses hue+120,
/// G uses hue, B uses hue-120. Saturation < ~1e-6 → grayscale at L.
pub fn from_hsl_to_rgb(hsl: Hsl) -> (f32, f32, f32) {
    if hsl.s.abs() < 1.0e-6 {
        return (hsl.l, hsl.l, hsl.l);
    }
    let m2 = if hsl.l > 0.5 {
        hsl.l + hsl.s - hsl.l * hsl.s
    } else {
        (hsl.s + 1.0) * hsl.l
    };
    let m1 = 2.0 * hsl.l - m2;
    (
        hsl_channel(hsl.h + 120.0, m1, m2),
        hsl_channel(hsl.h, m1, m2),
        hsl_channel(hsl.h - 120.0, m1, m2),
    )
}

/// One-texel evaluator. Mirrors the per-iteration body of
/// `update_color_grading`'s 3-deep loop (decompile lines 198-697).
/// `cg.flags & 2 == 0` → returns input unchanged (engine's
/// `LABEL_137` fast path).
pub fn evaluate(input: [f32; 3], cg: &ColorGradingBlock) -> [f32; 3] {
    if !cg.is_enabled() {
        return input;
    }
    let [mut r, mut g, mut b] = input;

    // Curves editor (decompile lines 212-269). Mode 1 = brightness
    // curve applied to all 3 channels; mode 0 = per-channel R/G/B
    // curves. Engine passes `range = 1.0` to evaluate_scalar (line 234).
    if let Some(ce) = cg.curves_editor.as_ref() {
        if ce.flags != 0 {
            apply_curves_editor(&mut r, &mut g, &mut b, ce);
        }
    }

    // Brightness / contrast (decompile lines 272-318).
    if let Some(bc) = cg.brightness_contrast.as_ref() {
        if bc.flags != 0 {
            apply_brightness_contrast(&mut r, &mut g, &mut b, bc);
        }
    }

    // Color balance (decompile lines 325-360). `Tones` is built once
    // per LUT bake by `set_color_balance_params`; the per-texel work
    // here is just the 3-channel pow-remap.
    if let Some(cb) = cg.color_balance.as_ref() {
        if cb.flags != 0 {
            let tones = set_color_balance_params(cb);
            r = apply_tones_channel(r, tones[0]);
            g = apply_tones_channel(g, tones[1]);
            b = apply_tones_channel(b, tones[2]);
        }
    }

    let hsl_in = from_rgb_to_hsl(r, g, b);
    let mut h = hsl_in.h;
    let mut s = hsl_in.s;
    let mut l = hsl_in.l;

    // HSLV (decompile lines 388-411).
    if let Some(hslv) = cg.hslv.as_ref() {
        if hslv.flags != 0 {
            apply_hslv(&mut h, &mut s, &mut l, hslv);
        }
    }

    // Colorize (decompile lines 413-454). Reads pre-HSLV `r/g/b` for
    // the luminance term — engine takes them from v30/v31/v34 which
    // are post-brightness/contrast but pre-HSL.
    if let Some(cz) = cg.colorize.as_ref() {
        if cz.flags != 0 {
            apply_colorize(&mut h, &mut s, &mut l, cz, [r, g, b]);
        }
    }

    let (mut rr, mut gg, mut bb) = from_hsl_to_rgb(Hsl { h, s, l });

    // Selective color (decompile lines 600-651).
    if let Some(sc) = cg.selective_color.as_ref() {
        if sc.flags != 0 {
            let mut cmyk = rgb_to_naive_cmyk(rr, gg, bb);
            process_selective_color(&mut cmyk, [rr, gg, bb], sc);
            let (nr, ng, nb) = naive_cmyk_to_rgb(cmyk);
            rr = nr;
            gg = ng;
            bb = nb;
        }
    }

    rr = rr.clamp(0.0, 1.0);
    gg = gg.clamp(0.0, 1.0);
    bb = bb.clamp(0.0, 1.0);
    [rr, gg, bb]
}

fn apply_curves_editor(r: &mut f32, g: &mut f32, b: &mut f32, ce: &CurvesEditorBlock) {
    let eval = |fnd: &Option<TagFunction>, x: f32| -> f32 {
        // Engine: `c_function_definition::evaluate_scalar(this, input,
        // range=1.0, _)`. blam-tags' `TagFunction::evaluate(input,
        // range)` mirrors this (legacy + map_to_output_range).
        // Missing/empty curves → identity.
        match fnd {
            Some(f) => f.evaluate(x, 1.0),
            None => x,
        }
    };
    if ce.mode == 1 {
        // Brightness mode: single curve applied to all 3 channels.
        *r = eval(&ce.brightness, *r);
        *g = eval(&ce.brightness, *g);
        *b = eval(&ce.brightness, *b);
    } else {
        // RGB mode: per-channel curves.
        *r = eval(&ce.red, *r);
        *g = eval(&ce.green, *g);
        *b = eval(&ce.blue, *b);
    }
}

fn apply_brightness_contrast(r: &mut f32, g: &mut f32, b: &mut f32, bc: &BrightnessContrast) {
    let br = bc.brightness;
    if br >= 0.0 {
        *r += (1.0 - *r) * br;
        *g += (1.0 - *g) * br;
        *b += (1.0 - *b) * br;
    } else {
        let m = br + 1.0;
        *r *= m;
        *g *= m;
        *b *= m;
    }
    let cm = bc.contrast + 1.0;
    *r = cm * (*r - 0.5) + 0.5;
    *g = cm * (*g - 0.5) + 0.5;
    *b = cm * (*b - 0.5) + 0.5;
}

fn apply_hslv(h: &mut f32, s: &mut f32, l: &mut f32, hslv: &HslvBlock) {
    *h += hslv.hue_offset_deg;
    *s = (*s + hslv.saturation_offset).max(0.0);
    // Skin-tone protect: pull vibrance back when the hue is in the
    // red-orange band (25.5° < H < 127.5° fails — outside is the
    // protected zone) AND when saturation is already high. Engine
    // constants verbatim: 0.55 / 0.9 hue threshold, 0.48 / 0.10
    // multiplier (decompile lines 396-405).
    let hue_thresh = if *h > 127.5 || *h < 25.5 { 0.55 } else { 0.9 };
    let sat_strength = if *s <= hue_thresh { 0.48 } else { 0.1 };
    *s *= sat_strength * hslv.vibrance + 1.0;
    *l += hslv.lightness_offset;
}

fn apply_colorize(h: &mut f32, s: &mut f32, l: &mut f32, cz: &ColorizeBlock, rgb_pre: [f32; 3]) {
    let inv_blend = 1.0 - cz.blendfactor;
    let orig_h = *h;
    let orig_s = *s;
    let orig_l = *l;

    // Hue lerp toward target (degrees).
    *h = (orig_h - cz.target_hue_deg) * inv_blend + cz.target_hue_deg;
    // Saturation lerp toward target.
    *s = (orig_s - cz.target_saturation) * inv_blend + cz.target_saturation;

    // L target: luminance biased toward 0/1 by `target_lightness`.
    // Engine: `0.333 * (R+G+B)`. Note the literal 0.333 (not 1/3) —
    // matches the constant the dllcache emits.
    let luminance = (rgb_pre[0] + rgb_pre[1] + rgb_pre[2]) * 0.333;
    let lt = cz.target_lightness;
    let l_target = if lt > 0.0 {
        (1.0 - lt) * luminance + lt
    } else if lt < 0.0 {
        luminance * (lt + 1.0)
    } else {
        luminance
    };
    *l = (orig_l - l_target) * inv_blend + l_target;
}

/// `SetColorBalanceParams @ 0x1806cb650` — line-by-line port. Builds
/// the per-channel `Tones[3][3]` matrix the per-texel pow-remap reads:
/// row R/G/B × column (lo, gamma, hi). Shadows feed `lo` (via HSL
/// hue rotation), midtones feed `gamma` (sign-aware 0.7/0.8 mixing
/// of the 3 sliders), highlights feed `hi` (sign-aware 1.2× mixing,
/// with min 0.001 to avoid division explosions).
pub fn set_color_balance_params(cb: &ColorBalanceBlock) -> [[f32; 3]; 3] {
    let s = cb.shadows;
    let m = cb.midtones;
    let h = cb.highlights;

    // === Midtones → gamma row (decompile lines 106-147) ===
    // Each slider's contribution is sign-asymmetric: positive vs
    // negative pick (0.7, 0.8) vs (-0.8, 0.7) etc. so positive and
    // negative sliders darken/brighten by slightly different amounts.
    let v21 = m.cyan_red;
    let (v22, v23) = if v21 < 0.0 {
        (v21 * -0.80000001, v21 * 0.69999999)
    } else {
        (v21 * -0.69999999, v21 * 0.80000001)
    };
    let v24 = m.magenta_green;
    let (v25, v26) = if v24 < 0.0 {
        (v24 * 0.80000001, v24 * 0.69999999)
    } else {
        (v24 * 0.69999999, v24 * 0.80000001)
    };
    let v27 = v22 - v25;
    let v28 = v23 - v25;
    let v29 = m.yellow_blue;
    let v30 = v22 + v26;
    let (v31, v32, v33) = if v29 < 0.0 {
        (v28 - v29 * 0.80000001, v30 - v29 * 0.80000001, v29 * 0.69999999)
    } else {
        (v28 - v29 * 0.69999999, v30 - v29 * 0.69999999, v29 * 0.80000001)
    };
    let r_gamma = v31 + 1.0;
    let g_gamma = v32 + 1.0;
    let b_gamma = (v27 + v33) + 1.0;

    // === Shadows → lo row, via HSL hue rotation (lines 148-235) ===
    // The shadows sliders accumulate complementary amounts into
    // (v13, v14, v15) treated as RGB, get converted to HSL, hue
    // rotated 180°, converted back, then scaled by 0.333.
    let mut v13 = 0.0;
    let mut v14 = 0.0;
    let mut v15 = 0.0;
    if s.cyan_red <= 0.0 {
        let neg = -s.cyan_red;
        v14 = neg;
        v15 = neg;
    } else {
        v13 = s.cyan_red;
    }
    if s.magenta_green <= 0.0 {
        v13 -= s.magenta_green;
        v15 -= s.magenta_green;
    } else {
        v14 += s.magenta_green;
    }
    if s.yellow_blue <= 0.0 {
        v13 -= s.yellow_blue;
        v14 -= s.yellow_blue;
    } else {
        v15 += s.yellow_blue;
    }

    let hsl = from_rgb_to_hsl(v13, v14, v15);
    let h_rot = if hsl.h <= 180.0 { hsl.h + 180.0 } else { hsl.h - 180.0 };

    // HSL → RGB with rotated hue (engine inlines QqhToRgb here, lines
    // 207-234). Saturation < eps falls back to L.
    let (r_lo_pre, g_lo_pre, b_lo_pre) = if hsl.s.abs() >= 1.0e-6 {
        let m2 = if hsl.l > 0.5 {
            hsl.l + hsl.s - hsl.l * hsl.s
        } else {
            (hsl.s + 1.0) * hsl.l
        };
        let m1 = 2.0 * hsl.l - m2;
        // Engine call order: L = QqhToRgb(m1, m2, h+120) → R channel,
        // then G with h, then B with h-120. The decompile names the
        // first result `L` because of an IDA naming collision.
        (
            hsl_channel(h_rot + 120.0, m1, m2),
            hsl_channel(h_rot, m1, m2),
            hsl_channel(h_rot - 120.0, m1, m2),
        )
    } else {
        (hsl.l, hsl.l, hsl.l)
    };
    let r_lo = r_lo_pre * 0.333;
    let g_lo = g_lo_pre * 0.333;
    let b_lo = b_lo_pre * 0.333;

    // === Highlights → hi row (lines 241-278) ===
    // Positive sliders scale the original channel by 1.2× contribution;
    // negative sliders contribute |amount| to the OTHER two channels.
    let mut v16 = 0.0_f32;
    let mut v17 = 0.0_f32;
    let mut v19 = 0.0_f32;
    if h.cyan_red <= 0.0 {
        v17 = -h.cyan_red;
        v19 = -h.cyan_red;
    } else {
        v16 = h.cyan_red * 1.2;
    }
    if h.magenta_green <= 0.0 {
        v16 -= h.magenta_green;
        v19 -= h.magenta_green;
    } else {
        v17 += h.magenta_green * 1.2;
    }
    if h.yellow_blue <= 0.0 {
        v17 -= h.yellow_blue;
        v16 -= h.yellow_blue;
    } else {
        v19 += h.yellow_blue * 1.2;
    }
    let r_hi = (1.0 - v16).max(0.001);
    let g_hi = (1.0 - v17).max(0.001);
    let b_hi = (1.0 - v19).max(0.001);

    [
        [r_lo, r_gamma, r_hi],
        [g_lo, g_gamma, g_hi],
        [b_lo, b_gamma, b_hi],
    ]
}

/// Per-channel pow-remap from the per-texel loop in
/// `update_color_grading:327-359`. `tones` row is `[lo, gamma, hi]`.
fn apply_tones_channel(input: f32, tones: [f32; 3]) -> f32 {
    let lo = tones[0];
    let gamma = tones[1];
    let hi = tones[2];
    let span = hi - lo;
    let mut v = input - lo;
    if span.abs() >= 1.0e-6 {
        v = (input - lo) / span;
    }
    v = v.clamp(0.0, 1.0);
    if gamma.abs() >= 1.0e-6 {
        v = v.powf(1.0 / gamma);
    }
    v
}

/// `rendNAIVE_CMYK` (engine type). Naive CMYK = `K_half = 0.5 * min(1-R, 1-G, 1-B)`,
/// `C = (1-R - K_half)/(1 - K_half)`, etc. The "0.5×" makes K's range
/// effectively [0, 0.5] which is intentional — see `update_color_grading
/// :617-633`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveCmyk {
    pub c: f32,
    pub m: f32,
    pub y: f32,
    pub k: f32,
}

/// Forward conversion (`update_color_grading:610-633`). Note the
/// engine-specific `K_half = 0.5 * min(...)` — not standard CMYK.
pub fn rgb_to_naive_cmyk(r: f32, g: f32, b: f32) -> NaiveCmyk {
    let one_m_r = 1.0 - r;
    let one_m_g = 1.0 - g;
    let one_m_b = 1.0 - b;
    let mut min_inv = if one_m_r < 1.0 { one_m_r } else { 1.0 };
    if one_m_g < min_inv {
        min_inv = one_m_g;
    }
    if one_m_b < min_inv {
        min_inv = one_m_b;
    }
    let k_half = min_inv * 0.5;
    if k_half >= 1.0 {
        return NaiveCmyk { c: 0.0, m: 0.0, y: 0.0, k: k_half };
    }
    let denom = 1.0 / (1.0 - k_half);
    NaiveCmyk {
        c: (one_m_r - k_half) * denom,
        m: (one_m_g - k_half) * denom,
        y: (one_m_b - k_half) * denom,
        k: k_half,
    }
}

/// Reverse conversion (`update_color_grading:635-649`).
pub fn naive_cmyk_to_rgb(cmyk: NaiveCmyk) -> (f32, f32, f32) {
    if cmyk.k >= 1.0 {
        return (0.0, 0.0, 0.0);
    }
    let inv_k = 1.0 - cmyk.k;
    let r_inv = inv_k * cmyk.c + cmyk.k;
    let g_inv = inv_k * cmyk.m + cmyk.k;
    let b_inv = inv_k * cmyk.y + cmyk.k;
    (1.0 - r_inv, 1.0 - g_inv, 1.0 - b_inv)
}

/// `ProcessSelectiveColor @ 0x1806cb200` — line-by-line port. Mutates
/// `cmyk` in place. Walks the 9 bands accumulating CMYK adjustments
/// weighted by per-band hue/tonal-distance factors `dFactors[0..8]`,
/// then clamps to `[0, 1]`. Negative `black` slider gets a 2× factor
/// (asymmetric K).
pub fn process_selective_color(
    cmyk: &mut NaiveCmyk,
    rgb: [f32; 3],
    sc: &SelectiveColorBlock,
) {
    let [r, g, b] = rgb;

    // 6 hue-zone factors via RGB ordering (decompile lines 67-104).
    let mut d = [0.0f32; 9];
    if r > g && g > b {
        d[0] = r - g;
        d[1] = g - b;
    } else if g > r && r > b {
        d[1] = r - b;
        d[2] = g - r;
    } else if g > b && b > r {
        d[2] = g - b;
        d[3] = b - r;
    } else if b > g && g > r {
        d[4] = b - g;
        d[3] = g - r;
    } else if b > r && r > g {
        d[5] = r - g;
        d[4] = b - r;
    } else if r > b && b > g {
        d[0] = r - b;
        d[5] = b - g;
    }

    // Tonal weighting via luminance (lines 106-127). v13 is BT.709-ish
    // luminance scaled by 0.94117 — yields ~[0, 0.94] across the input
    // range. v14/v15/v16 form whites/blacks/neutrals zone strengths.
    let v13 = (r * 0.23999999 + g * 0.68599999 + b * 0.075000003) * 0.94116998;
    let mut v14 = ((1.0 - v13 * 0.30000001 * 2.5) * (v13 * 0.30000001 * 2.5)) * 0.84167993 - 0.15000001;
    if v14 <= 0.0 {
        v14 = 0.0;
    }
    let mut v15 = ((0.94116998 - v13).abs() * 0.175 * 1.7636685 * 1.020304) - 0.22499999;
    if v15 <= 0.0 {
        v15 = 0.0;
    }
    let mut v16 = ((1.0 - (0.47058499 - v13).abs() * 1.0625073) * 0.19 * 1.7636685 * 2.0408163) - 0.115;
    if v16 <= 0.0 {
        v16 = 0.0;
    }
    // Lines 134-136: tonal weights normalized via `x / (x+1)` falloff.
    d[6] = (v14 * 3.5999999) / (v14 + 1.0);
    d[7] = (v16 * 2.0) / (v16 + 1.0);
    d[8] = (v15 * 1.8) / (v15 + 1.0);

    // 9-band accumulation (lines 137-198). One iteration per row of 3
    // bands; inside, 3 sub-iterations apply (cyan, magenta, yellow,
    // black) with the i-th dFactor.
    for iter in 0..3 {
        for sub in 0..3 {
            let band_idx = iter * 3 + sub;
            let factor = d[band_idx];
            let band: &CmybBand = &sc.bands[band_idx];
            let k_sign = if band.black >= 0.0 { 1.0 } else { 2.0 };
            cmyk.c += factor * band.cyan;
            cmyk.m += factor * band.magenta;
            cmyk.y += factor * band.yellow;
            cmyk.k += 2.0 * factor * k_sign * band.black;
        }
    }

    // Clamp to [0, 1] (lines 199-237).
    cmyk.c = cmyk.c.clamp(0.0, 1.0);
    cmyk.m = cmyk.m.clamp(0.0, 1.0);
    cmyk.y = cmyk.y.clamp(0.0, 1.0);
    cmyk.k = cmyk.k.clamp(0.0, 1.0);
}

/// Bake a 16³ RGBA8 LUT to `dst` (must be exactly [`LUT_BYTES`]).
/// Texel layout matches what the engine writes to its staging volume:
/// X → R input axis, Y → G, Z → B; storage stride
/// `dst[((iz * 16 + iy) * 16 + ix) * 4 + 0..3] = (R_out, G_out, B_out)`.
/// Identity LUT when `cg` is `None` or has the enable bit clear.
pub fn bake_lut(cg: Option<&ColorGradingBlock>, dst: &mut [u8]) {
    assert_eq!(dst.len(), LUT_BYTES);

    // Engine input step is 1/16, not 1/15 — `FLOAT_0_0625 * v15` per
    // decompile line 187. Sampling at uvw=1.0 ends up reading the
    // last texel under wgpu's `address_mode = ClampToEdge`, so the
    // top end of the range collapses correctly.
    const STEP: f32 = 1.0 / LUT_DIM as f32;

    for iz in 0..LUT_DIM {
        for iy in 0..LUT_DIM {
            for ix in 0..LUT_DIM {
                let r_in = ix as f32 * STEP;
                let g_in = iy as f32 * STEP;
                let b_in = iz as f32 * STEP;
                let out = match cg {
                    Some(cg) => evaluate([r_in, g_in, b_in], cg),
                    None => [r_in, g_in, b_in],
                };
                let idx = (((iz * LUT_DIM) + iy) * LUT_DIM + ix) as usize * 4;
                dst[idx] = (out[0] * 255.0) as u8;
                dst[idx + 1] = (out[1] * 255.0) as u8;
                dst[idx + 2] = (out[2] * 255.0) as u8;
                dst[idx + 3] = 255;
            }
        }
    }
}

/// Allocate + bake. Convenience for one-shot use.
pub fn bake_lut_to_vec(cg: Option<&ColorGradingBlock>) -> Vec<u8> {
    let mut buf = vec![0u8; LUT_BYTES];
    bake_lut(cg, &mut buf);
    buf
}

/// Allocate a 16³ RGBA8 3D texture for use as a color grading LUT.
/// Mirrors `c_screen_postprocess::color_grading_init` (per-texture
/// alloc — engine pairs this with a staging texture; in wgpu we use
/// `queue.write_texture` which handles staging internally).
pub fn allocate_lut_texture(device: &wgpu::Device, label: &'static str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: LUT_DIM,
            height: LUT_DIM,
            depth_or_array_layers: LUT_DIM,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Upload a baked LUT to a 16³ RGBA8 3D texture allocated by
/// [`allocate_lut_texture`].
pub fn upload_lut(queue: &wgpu::Queue, texture: &wgpu::Texture, data: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LUT_DIM * 4),
            rows_per_image: Some(LUT_DIM),
        },
        wgpu::Extent3d {
            width: LUT_DIM,
            height: LUT_DIM,
            depth_or_array_layers: LUT_DIM,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passthrough_when_disabled() {
        // Default ColorGradingBlock has flags=0 → fast path.
        let cg = ColorGradingBlock::default();
        for (r, g, b) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.5)] {
            assert_eq!(evaluate([r, g, b], &cg), [r, g, b]);
        }
    }

    #[test]
    fn hsl_round_trip_pure_red() {
        let hsl = from_rgb_to_hsl(1.0, 0.0, 0.0);
        assert!((hsl.h - 0.0).abs() < 1e-3, "H = {}", hsl.h);
        assert!((hsl.s - 1.0).abs() < 1e-3, "S = {}", hsl.s);
        assert!((hsl.l - 0.5).abs() < 1e-3, "L = {}", hsl.l);
        let (r, g, b) = from_hsl_to_rgb(hsl);
        assert!((r - 1.0).abs() < 1e-3 && g.abs() < 1e-3 && b.abs() < 1e-3);
    }

    #[test]
    fn hsl_round_trip_pure_green() {
        let hsl = from_rgb_to_hsl(0.0, 1.0, 0.0);
        assert!((hsl.h - 120.0).abs() < 1e-3, "H = {}", hsl.h);
        let (r, g, b) = from_hsl_to_rgb(hsl);
        assert!(r.abs() < 1e-3 && (g - 1.0).abs() < 1e-3 && b.abs() < 1e-3);
    }

    #[test]
    fn hsl_round_trip_pure_blue() {
        let hsl = from_rgb_to_hsl(0.0, 0.0, 1.0);
        assert!((hsl.h - 240.0).abs() < 1e-3, "H = {}", hsl.h);
        let (r, g, b) = from_hsl_to_rgb(hsl);
        assert!(r.abs() < 1e-3 && g.abs() < 1e-3 && (b - 1.0).abs() < 1e-3);
    }

    #[test]
    fn lut_size_matches_const() {
        assert_eq!(LUT_BYTES, 16 * 16 * 16 * 4);
    }

    #[test]
    fn identity_lut_round_trip() {
        let buf = bake_lut_to_vec(None);
        // First texel: (0,0,0), last texel: (15/16, 15/16, 15/16) ≈ 239.
        assert_eq!(&buf[0..4], &[0, 0, 0, 255]);
        let last = LUT_BYTES - 4;
        let expected = (15.0 / 16.0 * 255.0) as u8;
        assert_eq!(buf[last + 0], expected);
        assert_eq!(buf[last + 1], expected);
        assert_eq!(buf[last + 2], expected);
    }
}
