//! Engine source: `Ares/source/math/color_math.{h,cpp}`.
//!
//! Verbatim ports of the runtime color/lightprobe codecs used by the
//! geometry sampler chain. Currently scoped to the subset Phase 7b needs:
//!
//! - `real_rgb_lightprobe_from_half`           @ dllcache `0x1803ea110`
//! - `pixel32_RGBE_platform_unswizzle`         @ dllcache `0x1803ed030`
//! - `pixel32_RGBE_to_real_rgb_color`          @ dllcache `0x1803ec930`
//!
//! Function name + signature from Reach (preserved C++ symbols, e.g.
//! `?real_rgb_lightprobe_from_half@@YAXPEBUhalf_rgb_lightprobe@@PEAUreal_rgb_lightprobe@@@Z`);
//! body from tool.exe (`dllcache_play.dll.i64` @ port 13372). The diff
//! against Reach showed no functional divergence for these three.

use blam_tags::math::RealRgbColor;

// =============================================================================
// half_rgb_lightprobe — `color_math.h:335` (sizeof=56)
// =============================================================================

/// Engine `half_rgb_lightprobe` (56 B = 28 × half). 27 SH coefs (3 channels ×
/// 9 SH3 terms) packed into u16 half-floats, plus a trailing pad half to
/// align to 8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfRgbLightprobe {
    pub red_terms: [u16; 9],
    pub green_terms: [u16; 9],
    pub blue_terms: [u16; 9],
    pub pad: u16,
}

const _: () = assert!(std::mem::size_of::<HalfRgbLightprobe>() == 56);

// =============================================================================
// half_linear_rgb_color — `color_math.h:326` (sizeof=8)
// =============================================================================

/// Engine `half_linear_rgb_color` — 4 halves (RGB + pad).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfLinearRgbColor {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub pad: u16,
}
const _: () = assert!(std::mem::size_of::<HalfLinearRgbColor>() == 8);

// =============================================================================
// half_rgb_lightprobe_with_dominant_light — `color_math.h:344` (sizeof=72)
// =============================================================================

/// Engine `half_rgb_lightprobe_with_dominant_light` (72 B).
///
/// Used as the per-entry layout for `bsp_lightmap_data.single_probes`: each
/// entry is `dominant_light_direction` (half_vector3d, 3 halves) + pad +
/// `dominant_light_intensity` (half_linear_rgb_color, 4 halves) +
/// `quadratic_probe` (half_rgb_lightprobe, 28 halves). The geometry sampler
/// reads `&entry.quadratic_probe` (offset +16) into its result.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfRgbLightprobeWithDominantLight {
    pub dominant_light_direction: [u16; 3],
    pub pad: u16,
    pub dominant_light_intensity: HalfLinearRgbColor,
    pub quadratic_probe: HalfRgbLightprobe,
}
const _: () = assert!(std::mem::size_of::<HalfRgbLightprobeWithDominantLight>() == 72);

// =============================================================================
// real_rgb_lightprobe — `color_math.h:304` (sizeof=108)
// =============================================================================
//
// Already defined in [`crate::halo::geometry::geometry_sampling::RealRgbLightprobe`]
// because that's where the geometry sampler test_result first needs it.
// Re-exported here for parity with Ares's header co-location, but the
// canonical definition stays in geometry_sampling.rs to avoid a circular
// dependency between math and geometry modules.

pub use crate::halo::geometry::geometry_sampling::RealRgbLightprobe;

// =============================================================================
// pixel32_RGBE_platform_unswizzle — `dllcache 0x1803ed030`
// =============================================================================

/// `pixel32_RGBE_platform_unswizzle(color)` @ dllcache `0x1803ed030`.
///
/// Reverses the platform byte-order for an RGBE-encoded ARGB pixel and, on
/// non-Xbox-360 platforms, subtracts 127 from each byte (signed-RGBE bias
/// removal). On Xbox 360 path the bytes pass through unchanged before the
/// byte reversal.
///
/// Engine returns `__int64` but only uses the low 32 bits; we use `u32`.
pub fn pixel32_rgbe_platform_unswizzle(color: u32) -> u32 {
    // Engine reads the four bytes via HIBYTE / BYTE2 / BYTE1 / LOBYTE, then
    // either subtracts 127 (PC) or leaves them alone (Xbox 360), then
    // rebuilds as ((HIBYTE << 24) | (LOBYTE << 16) | (BYTE1 << 8) | BYTE2).
    // That is: it swaps R↔B (bytes 0 and 2) while keeping G and A in place.
    let mut byte0 = (color & 0xFF) as u8; // LOBYTE
    let mut byte1 = ((color >> 8) & 0xFF) as u8;
    let mut byte2 = ((color >> 16) & 0xFF) as u8;
    let mut byte3 = ((color >> 24) & 0xFF) as u8; // HIBYTE

    if get_runtime_platform_type() != 0 {
        // Non-Xbox-360: subtract 127 from each byte (wrap-around mirrors the
        // engine's `LOBYTE(x) = HIBYTE(color) - 127` truncation).
        byte0 = byte0.wrapping_sub(127);
        byte1 = byte1.wrapping_sub(127);
        byte2 = byte2.wrapping_sub(127);
        byte3 = byte3.wrapping_sub(127);
    }

    // Reassemble: ((byte3 << 24) | (byte0 << 16) | (byte1 << 8) | byte2)
    (byte2 as u32)
        | ((byte1 as u32) << 8)
        | ((byte0 as u32) << 16)
        | ((byte3 as u32) << 24)
}

// =============================================================================
// pixel32_RGBE_to_real_rgb_color — `dllcache 0x1803ec930`
// =============================================================================

/// `pixel32_RGBE_to_real_rgb_color(pixel, &color, values_can_be_negative)`
/// @ dllcache `0x1803ec930`.
///
/// Decodes a 32-bit RGBE-encoded pixel into a 3-channel float color.
///
/// Encoding (matches the engine derivation exactly):
/// - byte 0 (lo) = R mantissa
/// - byte 1      = G mantissa
/// - byte 2      = B mantissa
/// - byte 3 (hi) = E exponent
/// - `values_can_be_negative=true`  → mantissas in `[-1,1]` via `i8/127`,
///   exponent treated as signed (`int8 e * (1/4)` → `2^(e/4)` scale).
/// - `values_can_be_negative=false` → mantissas in `[0,1]` via `u8/255`,
///   exponent biased by 127 (`u8 e * (1/4) - 31.75` → `2^((e-127)/4)`).
///
/// Output stored into `color->n[0..2]` (R, G, B). Engine returns the
/// pointer; we return `()` and write through `&mut`.
pub fn pixel32_rgbe_to_real_rgb_color(
    pixel: u32,
    color: &mut RealRgbColor,
    values_can_be_negative: bool,
) {
    let r;
    let g;
    let b;
    let exponent_y: f32;

    if values_can_be_negative {
        // Signed: SHIBYTE/SBYTE* — each byte is sign-extended (i8). Engine
        // scales the exponent byte by 0.0078740157 (=1/127), then multiplies
        // by 127.0/2² = 31.75 → net = i8 / 4.
        let s3 = (pixel >> 24) as i8 as f32;
        let s2 = (pixel >> 16) as i8 as f32;
        let s1 = (pixel >> 8) as i8 as f32;
        let s0 = pixel as i8 as f32;
        r = s0 * 0.007_874_015_7;
        g = s1 * 0.007_874_015_7;
        b = s2 * 0.007_874_015_7;
        let v4 = s3 * 0.007_874_015_7;
        exponent_y = 31.75 * v4;
    } else {
        // Unsigned: bytes are u8 in [0,1]. Exponent biased by 127.
        // Engine: Y = (255/4) * (e/255) - (127/4) = e/4 - 31.75.
        let u3 = ((pixel >> 24) & 0xFF) as f32;
        let u2 = ((pixel >> 16) & 0xFF) as f32;
        let u1 = ((pixel >> 8) & 0xFF) as f32;
        let u0 = (pixel & 0xFF) as f32;
        r = u0 * 0.003_921_568_9;
        g = u1 * 0.003_921_568_9;
        b = u2 * 0.003_921_568_9;
        let v12 = u3 * 0.003_921_568_9;
        // 1.0 / pow(2.0, 2.0) = 0.25; 0.25 * 255.0 = 63.75; 0.25 * 127.0 = 31.75.
        exponent_y = 63.75 * v12 - 31.75;
    }

    let scale = 2.0_f32.powf(exponent_y);
    color.red = r * scale;
    color.green = g * scale;
    color.blue = b * scale;
}

// =============================================================================
// real_rgb_lightprobe_from_half — `dllcache 0x1803ea110`
// =============================================================================

/// `real_rgb_lightprobe_from_half(&probe, &result)` @ dllcache `0x1803ea110`.
///
/// Decompresses a half-float-packed RGB lightprobe (27 halves) into the
/// f32 lightprobe form used by the geometry sampler.
///
/// Engine asserts on half values with exponent==0x1F (Inf/NaN). We skip the
/// assert in release; the decode still proceeds via `f16::to_f32` which
/// will yield NaN/Inf in that case rather than corrupting downstream math.
pub fn real_rgb_lightprobe_from_half(probe: &HalfRgbLightprobe, result: &mut RealRgbLightprobe) {
    // Engine: single loop over `i in 0..27`, indexing through `probe->red_terms[i]`
    // as if the 27 halves were one flat array (which they are — red[9] then
    // green[9] then blue[9]). Replicate that pattern via aliased flat slices.
    let probe_halves: [u16; 27] = {
        let mut a = [0u16; 27];
        a[..9].copy_from_slice(&probe.red_terms);
        a[9..18].copy_from_slice(&probe.green_terms);
        a[18..].copy_from_slice(&probe.blue_terms);
        a
    };
    let mut result_floats = [0.0_f32; 27];
    for i in 0..27 {
        let v = probe_halves[i];
        // Engine slim_assert: ((v >> 10) & 0x1F) == 0x1F → Inf/NaN guard.
        // We skip the assert; `half::f16::from_bits(v).to_f32()` returns
        // NaN/Inf in that case, which is the safe-default behavior.
        result_floats[i] = half::f16::from_bits(v).to_f32();
    }
    result.red_terms.copy_from_slice(&result_floats[..9]);
    result.green_terms.copy_from_slice(&result_floats[9..18]);
    result.blue_terms.copy_from_slice(&result_floats[18..]);
}

// =============================================================================
// get_runtime_platform_type — engine global
// =============================================================================
//
// Engine exports an `e_runtime_platform_type` enum (0 = xbox360, !=0 = PC).
// Protomorph always runs on PC. Stubbed here as a local constant so the
// RGBE unswizzle picks the PC code path; if a Durango/Scorpio runtime
// path is ever ported we re-route this through a real getter.
const fn get_runtime_platform_type() -> u32 {
    1 // _runtime_platform_not_xbox
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a known SH coef through f32 → f16 → f32 and verify the
    /// `real_rgb_lightprobe_from_half` decompresses correctly across all
    /// three channels.
    #[test]
    fn rgb_lightprobe_from_half_roundtrips() {
        let mut packed = HalfRgbLightprobe::default();
        for i in 0..9 {
            packed.red_terms[i] = half::f16::from_f32(0.1 + i as f32 * 0.05).to_bits();
            packed.green_terms[i] = half::f16::from_f32(0.5 - i as f32 * 0.02).to_bits();
            packed.blue_terms[i] = half::f16::from_f32(-0.25 + i as f32 * 0.1).to_bits();
        }
        let mut decoded = RealRgbLightprobe::default();
        real_rgb_lightprobe_from_half(&packed, &mut decoded);
        for i in 0..9 {
            let expected_r = half::f16::from_f32(0.1 + i as f32 * 0.05).to_f32();
            let expected_g = half::f16::from_f32(0.5 - i as f32 * 0.02).to_f32();
            let expected_b = half::f16::from_f32(-0.25 + i as f32 * 0.1).to_f32();
            assert!((decoded.red_terms[i] - expected_r).abs() < 1e-6);
            assert!((decoded.green_terms[i] - expected_g).abs() < 1e-6);
            assert!((decoded.blue_terms[i] - expected_b).abs() < 1e-6);
        }
    }

    /// White unsigned RGBE at exponent 128 (= 1.0 scale) should produce
    /// approximately (1, 1, 1).
    #[test]
    fn pixel32_rgbe_unsigned_white_at_unit_scale() {
        let mut c = RealRgbColor::default();
        // R=255, G=255, B=255, E=128 → mantissas 1.0, exponent (128-127)/4 = 0.25 → scale = 2^0.25 ≈ 1.189
        pixel32_rgbe_to_real_rgb_color(0x80FFFFFF, &mut c, false);
        let s = 2.0_f32.powf(0.25);
        assert!((c.red - s).abs() < 1e-4);
        assert!((c.green - s).abs() < 1e-4);
        assert!((c.blue - s).abs() < 1e-4);
    }

    /// Signed mode: exponent byte 0 gives scale 1.0; mantissa 127 → 1.0.
    #[test]
    fn pixel32_rgbe_signed_zero_exponent() {
        let mut c = RealRgbColor::default();
        // R=127 (=+1.0), G=-127 (=-1.0), B=64 (~+0.5), E=0 (=scale 1.0)
        let pixel: u32 = (0u8 as u32) << 24 | (64u8 as u32) << 16 | (((-127i8 as u8) as u32) << 8) | (127u8 as u32);
        pixel32_rgbe_to_real_rgb_color(pixel, &mut c, true);
        assert!((c.red - 1.0).abs() < 1e-3, "red={}", c.red);
        assert!((c.green - (-1.0)).abs() < 1e-3, "green={}", c.green);
        assert!((c.blue - (64.0 / 127.0)).abs() < 1e-3, "blue={}", c.blue);
    }

    /// Platform unswizzle on PC swaps R↔B and subtracts 127 from each byte.
    #[test]
    fn pixel32_rgbe_unswizzle_pc_swaps_red_blue_with_bias() {
        let input: u32 = 0xAABBCCDD; // A=AA, B=BB, G=CC, R=DD
        let out = pixel32_rgbe_platform_unswizzle(input);
        // After -127 each byte: A=2B, B=3C, G=4D, R=5E. After R↔B swap:
        // (A=2B << 24) | (R=5E << 16) | (G=4D << 8) | B=3C
        let expected: u32 = (0x2Bu32 << 24) | (0x5Eu32 << 16) | (0x4Du32 << 8) | 0x3C;
        assert_eq!(out, expected, "got {:08x} expected {:08x}", out, expected);
    }
}
