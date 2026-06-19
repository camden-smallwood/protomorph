//! Engine source: `Ares/source/physics/bsp3d.{h,cpp}`.
//!
//! 3D kd-tree walker — descends from a root node following plane tests until a
//! leaf is reached. Engine uses this as the OUTER loop of
//! `collision_bsp_test_vector_recursive`; we'll use it directly for point
//! lookups (e.g. "which BSP leaf contains this position").
//!
//! ## Engine struct mapping
//!
//! Ares header (`physics/bsp3d.h`):
//!
//! ```cpp
//! struct bsp3d        { s_tag_block nodes; s_tag_block planes; };   // 24 B
//! struct bsp3d_node   {                                              // 8 B, bit-packed
//!     long long plane_index       : 16;
//!     long long below_child_index  : 24;
//!     long long above_child_index  : 24;
//! };
//! ```
//!
//! `bsp3d_node` data lives inside `blam_tags::structure_bsp::Bsp3d::nodes`. The
//! blam-tags parser already promotes child indices from the bit-packed source
//! to canonical `i32` with bit 31 = leaf (vs the engine's bit 23). Walker uses
//! `< 0` for the leaf test as a result.
//!
//! ## Child-index encoding
//!
//! Same convention as bsp2d:
//! - `c == -1`               → empty / no leaf
//! - `c & 0x80000000 != 0`   → leaf; leaf index is `c & 0x7FFFFFFF`
//! - `c & 0x80000000 == 0`   → interior node; recurse with child as next index
//!
//! Engine's raw form uses 24-bit children (`c & 0x7FFFFF`, leaf bit 23 = `0x800000`).
//! blam-tags has already sign-extended to 32-bit before we see it.

use blam_tags::math::{RealPlane3d, RealPoint3d};

/// `plane3d_distance_to_point_safe` @ dllcache `0x1801887c0`.
///
/// Engine returns `n·p − d`. The "safe" variant is the null-pointer-tolerant
/// form used by walkers that look up a plane through a possibly-invalid
/// `plane_index`. Returns `0.0` when `plane is None` (degenerate node — engine
/// reads through a null pointer which IDA shows as zeros).
///
/// Verbatim port of:
///
/// ```c
/// return plane->n.n[1] * point->n[1]
///      + plane->n.n[0] * point->n[0]
///      + plane->n.n[2] * point->n[2]
///      - plane->d;
/// ```
#[inline]
pub fn plane3d_distance_to_point_safe(plane: Option<&RealPlane3d>, point: RealPoint3d) -> f32 {
    match plane {
        Some(p) => p.i * point.x + p.j * point.y + p.k * point.z - p.d,
        None => 0.0,
    }
}
