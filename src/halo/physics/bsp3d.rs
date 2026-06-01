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
use blam_tags::structure_bsp::{Bsp3d, Bsp3dNode};

/// Sentinel returned when the walk lands in an empty leaf (`node_index == -1`).
/// Mirrors engine `NONE` / `0xFFFFFFFF` return.
pub const NONE: u32 = u32::MAX;

/// `bsp3d_test_point` @ dllcache `0x1803342e0`.
///
/// Iteratively descends from `node_index` (which may be a leaf designator on
/// entry) until a leaf is reached. Returns the leaf index, or [`NONE`] for
/// empty leaves.
///
/// Verbatim port of:
///
/// ```c
/// do {
///     v6 = &bsp.nodes[node_index & 0x7FFFFF];      // engine bit 23 mask
///     count_1 = v6->plane_index;
///     plane = (count_1 in range) ? &bsp.planes[count_1] : nullptr;
///     node_index = v6->below_child_index;
///     if (plane3d_distance_to_point_safe(plane, point) >= 0.0)
///         node_index = v6->above_child_index;
/// } while ((node_index & 0x800000) == 0);
/// return (node_index == -1) ? NONE : (node_index & 0x7FFFFF);
/// ```
///
/// Our blam-tags promoted child encoding to bit 31, so the `& 0x800000` loop
/// guard becomes `>= 0` and the leaf-index mask is `& 0x7FFFFFFF`.
pub fn test_point(bsp: &Bsp3d, mut node_index: i32, point: RealPoint3d) -> u32 {
    // Engine `do/while` — at least one descent. On entry, `node_index` could
    // already be a leaf designator; that's caught by the loop guard on the
    // first iteration's tail.
    loop {
        let node = if node_index >= 0 {
            bsp.nodes.get(node_index as usize)
        } else {
            None
        };
        // Engine's safe lookup: if the index is out-of-range, the read returns
        // a zero-filled node (null pointer + offset, which IDA shows as zeros).
        // Mirror by treating as `below = -1, above = -1, plane = None`.
        let node = node.copied().unwrap_or(Bsp3dNode {
            plane_index: -1,
            below_child: -1,
            above_child: -1,
        });

        let plane = if node.plane_index >= 0 {
            bsp.planes.get(node.plane_index as usize).copied()
        } else {
            None
        };

        // Default to "below" child; if the point is on the positive side of
        // the plane, swap to "above".
        node_index = if plane3d_distance_to_point_safe(plane.as_ref(), point) >= 0.0 {
            node.above_child
        } else {
            node.below_child
        };

        // Loop until we hit a leaf (engine bit 23 → blam-tags bit 31 = `< 0`).
        if node_index < 0 {
            break;
        }
    }
    if node_index == -1 {
        NONE
    } else {
        (node_index & 0x7FFF_FFFF) as u32
    }
}

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

/// `plane3d_distance_to_point` @ dllcache `0x180194f10`.
///
/// Identical body to the `_safe` variant — engine has both so the type system
/// can distinguish "definitely valid plane" call sites. We expose the same
/// pair so call sites in higher phases (collision_bsp_test_vector_recursive)
/// can mirror the engine 1:1.
#[inline]
pub fn plane3d_distance_to_point(plane: &RealPlane3d, point: RealPoint3d) -> f32 {
    plane.i * point.x + plane.j * point.y + plane.k * point.z - plane.d
}

/// `bsp3d_get_plane_from_designator` @ Ares prototype (no dllcache decompile
/// pulled yet — small enough to inline). Mirrors the standard pattern:
///
/// ```c
/// long plane_idx = plane_designator & 0x7FFFFFFF;
/// real_plane3d const* p = &bsp.planes[plane_idx];
/// if (plane_designator & 0x80000000)  // negate flag
///     result = { -p->i, -p->j, -p->k, -p->d };
/// else
///     result = *p;
/// return &result;
/// ```
///
/// Engine encodes a "plane index + negate" in a single `long` so that surfaces
/// can share the same plane storage with opposing normals.
pub fn get_plane_from_designator(bsp: &Bsp3d, plane_designator: i32) -> Option<RealPlane3d> {
    let index = (plane_designator & 0x7FFF_FFFF) as usize;
    let plane = bsp.planes.get(index).copied()?;
    if (plane_designator as u32) & 0x8000_0000 != 0 {
        Some(RealPlane3d {
            i: -plane.i,
            j: -plane.j,
            k: -plane.k,
            d: -plane.d,
        })
    } else {
        Some(plane)
    }
}

/// `bsp3d_plane_test_point` @ Ares prototype. Three-way classification of a
/// point against a plane (back / front / on-plane within epsilon). Engine
/// constants from `physics/bsp3d.h`:
///
/// ```cpp
/// enum {
///     _bsp3d_plane_test_back = 0,
///     _bsp3d_plane_test_front,
///     _bsp3d_plane_test_on,
/// };
/// ```
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneTest {
    Back = 0,
    Front = 1,
    On = 2,
}

/// Verbatim form of `bsp3d_plane_test_point` Ares prototype. `d_reference` is
/// the signed-distance output engine sets via a pointer; we return it in the
/// tuple for the same effect.
pub fn plane_test_point(plane: &RealPlane3d, point: RealPoint3d, epsilon: f32) -> (PlaneTest, f32) {
    let d = plane3d_distance_to_point(plane, point);
    let cls = if d > epsilon {
        PlaneTest::Front
    } else if d < -epsilon {
        PlaneTest::Back
    } else {
        PlaneTest::On
    };
    (cls, d)
}
