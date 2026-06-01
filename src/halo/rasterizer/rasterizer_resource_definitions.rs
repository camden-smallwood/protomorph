//! Engine source: `Ares/source/rasterizer/rasterizer_resource_definitions.{h,cpp}`.
//!
//! Resource-definition helpers — small free functions that interpret packed
//! vertex/index compression formats. Currently scoped to the subset that
//! `c_geometry_sampler::geometry_test_triangle` needs to interpolate per-
//! vertex lightmap UVs.
//!
//! Function name + signature from Reach (preserved symbol
//! `?uncompress_word_to_real_normalized@@YAMG@Z`); body from tool.exe
//! (dllcache `0x1806d0730`).

/// `uncompress_word_to_real_normalized(value)` @ dllcache `0x1806d0730`.
///
/// Engine body: `_mm_cvtepi32_ps((__m128i)value).m128_f32[0] * 0.000015259022`.
/// `0.000015259022 ≈ 1/65535` — straightforward unorm16 → float decode.
///
/// Note: this is **unsigned-normalized** decode, output range `[0.0, 1.0]`.
/// The signed-normalized variant (which would map `0xFFFF → -1` ... `0x7FFF → 1`
/// or similar) is not used by the lightmap-UV path so we don't port it
/// here.
#[inline]
pub fn uncompress_word_to_real_normalized(value: u16) -> f32 {
    (value as f32) * 0.000_015_259_022
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_endpoints() {
        assert!(uncompress_word_to_real_normalized(0).abs() < 1e-6);
        assert!((uncompress_word_to_real_normalized(0xFFFF) - 1.0).abs() < 1e-4);
        assert!((uncompress_word_to_real_normalized(0x8000) - 0.5).abs() < 1e-3);
    }
}
