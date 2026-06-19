//! Engine source: `Ares/source/math/vector_math.{h,cpp}`.
//!
//! Vector3d helpers from the dllcache. Engine functions follow the
//! `(input_ptr, …, result_ptr)` convention and return `result_ptr`; Rust
//! ports take the input by value and return the result by value (no out-
//! param) since glam-style ergonomics fit better and the engine's pointer
//! convention exists only to dodge the calling-convention overhead of
//! returning a struct by value in MSVC x64.

use blam_tags::math::RealVector3d;

/// `scale_vector3d(a, c, result)` @ dllcache `0x18006A5B0`.
///
/// ```c
/// real_vector3d *scale_vector3d(const real_vector3d *a, float c, real_vector3d *result) {
///     result->n[0] = c * a->n[0];
///     result->n[1] = c * a->n[1];
///     result->n[2] = c * a->n[2];
///     return result;
/// }
/// ```
///
/// Thin engine-named alias over `RealVector3d`'s `Mul<f32>` (the canonical
/// implementation in `blam_tags::math`) — kept for call-site parity with the
/// dllcache, but no longer re-derives the component multiply.
#[inline]
pub fn scale_vector3d(a: RealVector3d, c: f32) -> RealVector3d {
    a * c
}
