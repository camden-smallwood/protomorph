//! Engine source: `Ares/source/physics/bsp2d.{h,cpp}`.
//!
//! Per-surface 2D kd-tree walker. After [`crate::halo::physics::collision_bsp`]'s
//! BSP3D walk locates a candidate leaf, each leaf's `bsp2d_references` points at
//! a per-plane 2D surface tree rooted in `Bsp3d::bsp2d_nodes` (from blam-tags).
//! This module walks that 2D tree to find the surface index whose 2D footprint
//! contains a projected point.
//!
//! ## Engine struct mapping
//!
//! Ares header (`physics/bsp2d.h`):
//!
//! ```cpp
//! struct bsp2d        { s_tag_block nodes; };            // 12 B; just a node table
//! struct bsp2d_node       { real_plane2d plane; short child_indices[2]; };  // 16 B (SMALL)
//! struct large_bsp2d_node { real_plane2d plane; long  child_indices[2]; };  // 20 B (LARGE)
//! ```
//!
//! `bsp2d_node` data lives inside `blam_tags::structure_bsp::Bsp3d::bsp2d_nodes`
//! (we don't store a free-standing `bsp2d` struct — the nodes slice IS the bsp2d).
//! blam-tags normalises both SMALL (i16 child) and LARGE (i32 child) schemas to
//! the canonical i32-with-bit-31-leaf form, so this walker uses bit 31 instead of
//! the SMALL-runtime bit 15 the dllcache decompile shows.
//!
//! ## Child-index encoding
//!
//! For a child value `c`:
//! - `c == -1`               → empty leaf / no surface (engine `NONE`)
//! - `c & 0x80000000 != 0`   → leaf; surface index is `c & 0x7FFFFFFF`
//! - `c & 0x80000000 == 0`   → interior node; recurse with child as next node index

use blam_tags::math::RealPlane2d;
use blam_tags::structure_bsp::CollisionBsp2dNode;
use glam::Vec2;

/// Sentinel returned when the walk lands in an empty leaf (`child == -1`).
/// Mirrors engine `NONE` / `0xFFFFFFFF` return.
pub const NONE: u32 = u32::MAX;

/// `bsp2d_test_point` @ dllcache `0x180576a50`.
///
/// Walks the 2D tree starting at `child_index` (which may itself be a leaf
/// designator). Returns the surface index of the leaf the point lands in,
/// or [`NONE`] for an empty leaf.
///
/// Verbatim port — iterative descent, dot-product plane test, no early exit
/// before reaching a leaf.
pub fn test_point(nodes: &[CollisionBsp2dNode], point: Vec2, mut child_index: i32) -> u32 {
    // Engine loop condition: `(child_index & 0x8000) == 0`. Our blam-tags
    // promotes SMALL i16 children to canonical i32; both schemas converge on
    // `bit 31 set = leaf`. So the test becomes `child_index >= 0`.
    while child_index >= 0 {
        let node = match nodes.get(child_index as usize) {
            Some(n) => n,
            // Out-of-range interior index — engine sets the lookup pointer to
            // null and re-reads, which yields all-zeros and walks the "below"
            // child. Mirror by treating as empty leaf.
            None => return NONE,
        };
        child_index = pick_child(node, point);
    }
    if child_index == -1 {
        NONE
    } else {
        // Mask off the leaf bit; what remains is the surface index.
        (child_index & 0x7FFF_FFFF) as u32
    }
}

/// Inner step of both walkers: evaluate the node's 2D plane at `point` and
/// return the appropriate child index.
///
/// Engine math:
///
/// ```text
/// d = ny * py + nx * px - distance
/// child = (d >= 0) ? node.child_indices[1] : node.child_indices[0]
/// ```
///
/// Mirrors `*((__int16 *)v4 + ((float)... >= 0.0) + 6)` from the dllcache
/// decompile — `+ 6` is the byte offset of `child_indices[0]` past
/// `real_plane2d plane` (12 B) divided by `sizeof(short) == 2`.
#[inline]
fn pick_child(node: &CollisionBsp2dNode, point: Vec2) -> i32 {
    let RealPlane2d { i: nx, j: ny, d } = node.plane;
    let signed_distance = ny * point.y + nx * point.x - d;
    if signed_distance >= 0.0 {
        node.right_child
    } else {
        node.left_child
    }
}
