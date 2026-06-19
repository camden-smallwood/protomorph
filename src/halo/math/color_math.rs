//! Engine source: `Ares/source/math/color_math.{h,cpp}`.
//!
//! Verbatim ports of the runtime color/lightprobe codecs used by the
//! geometry sampler chain. Currently scoped to the subset Phase 7b needs:
//!
//! - `real_rgb_lightprobe_from_half`           @ dllcache `0x1803ea110`
//!
//! Function name + signature from Reach (preserved C++ symbols, e.g.
//! `?real_rgb_lightprobe_from_half@@YAXPEBUhalf_rgb_lightprobe@@PEAUreal_rgb_lightprobe@@@Z`);
//! body from tool.exe (`dllcache_play.dll.i64` @ port 13372). The diff
//! against Reach showed no functional divergence.

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
}
