//! Mirror of `Ares/source/physics/bsp3d.cpp` — runtime BSP3D walker.
//!
//! Used by [`scenario_location_from_point`] (Phase C2) to convert a
//! world-space point into the leaf index that wraps it. The leaf
//! index then drives [`StructureBsp::leaves`] →
//! `cluster_index` → `ClusterReference`.
//!
//! The tag-side data ([`blam_tags::structure_bsp::Bsp3d`]) is parsed
//! by blam-tags; this module only owns the recursive plane-test
//! logic.

use blam_tags::math::{RealPlane3d, RealPoint3d};
use blam_tags::structure_bsp::{Bsp3d, Bsp3dNode};

// =============================================================================
// Constants — Ares `physics/bsp3d.h:13-21`
// =============================================================================

pub const MAXIMUM_BSP3D_DEPTH: usize = 128;
pub const BSP3D_ROOT_NODE_INDEX: i32 = 0;

// =============================================================================
// `bsp3d_get_plane_from_designator @ 0x1803347C0`
// =============================================================================

/// Resolve a `plane_designator` into an actual plane equation. The
/// sign bit (bit 31 of the i32, parser-normalized from bit-15 for
/// SMALL schemas) means "use the negated plane equation" — engine
/// convention for half-space inversion.
///
/// Returns `RealPlane3d::default()` if the index is out of range
/// (matches engine's tolerance for missing data).
pub fn bsp3d_get_plane_from_designator(bsp: &Bsp3d, plane_designator: i32) -> RealPlane3d {
    let index = (plane_designator & 0x7FFF_FFFF) as usize;
    let Some(plane) = bsp.planes.get(index) else {
        return RealPlane3d::default();
    };
    if plane_designator < 0 {
        RealPlane3d {
            i: -plane.i,
            j: -plane.j,
            k: -plane.k,
            d: -plane.d,
        }
    } else {
        *plane
    }
}

#[inline]
fn plane3d_distance_to_point(plane: RealPlane3d, point: RealPoint3d) -> f32 {
    point.x * plane.i + point.y * plane.j + point.z * plane.k - plane.d
}

// =============================================================================
// `bsp3d_test_point @ 0x1803342E0`
// =============================================================================

/// Recursive plane-test descent. Returns the index of the leaf the
/// point falls into, or `None` if the walk exits the tree (engine
/// returns `0xFFFFFFFFLL` = -1 in that case).
///
/// `node_index` is the starting node — pass [`BSP3D_ROOT_NODE_INDEX`]
/// (= 0) for the standard top-down walk. Recursion is bounded by
/// [`MAXIMUM_BSP3D_DEPTH`]; engine's iterative form does the same
/// implicitly through the `& 0x800000` leaf check.
pub fn bsp3d_test_point(bsp: &Bsp3d, mut node_index: i32, point: RealPoint3d) -> Option<i32> {
    // Canonical encoding (post-parse): child < 0 = leaf with index =
    // child & 0x7FFFFFFF; child == -1 = "outside the tree" sentinel
    // → None. Mirrors `bsp3d_test_point @ 0x1803342E0`.
    let mut depth = 0usize;
    loop {
        depth += 1;
        if depth > MAXIMUM_BSP3D_DEPTH {
            return None;
        }
        if node_index < 0 {
            return None;
        }
        let Some(node) = bsp.nodes.get(node_index as usize) else {
            return None;
        };
        let plane_idx = node.plane_index();
        let plane = if plane_idx >= 0 && (plane_idx as usize) < bsp.planes.len() {
            bsp.planes[plane_idx as usize]
        } else {
            RealPlane3d::default()
        };
        let next = if plane3d_distance_to_point(plane, point) >= 0.0 {
            node.above_child_index()
        } else {
            node.below_child_index()
        };
        if Bsp3dNode::child_is_leaf(next) {
            if next == -1 {
                return None;
            }
            return Some(Bsp3dNode::child_leaf_index(next));
        }
        node_index = next;
    }
}
