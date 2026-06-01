//! Mirror of `Ares/source/visibility/visibility_test.cpp`.
//!
//! The point/sphere/box/triangle tests against a [`VisibilityVolume`].
//! All tests return one of the [`VolumeTestResult`] discriminants:
//!
//! - `0` (`Outside`) — primitive is fully outside the volume.
//! - `1` (`Intersect`) — primitive crosses the volume boundary.
//! - `2` (`Inside`) — primitive is fully contained.
//!
//! The constants are exposed as `i32`/`u8` rather than a Rust enum so
//! call sites can OR-aggregate per-volume results the same way the
//! engine does (see `visibility_region_test_sphere` in
//! [`super::visibility_projections_and_volumes`]).

use blam_tags::math::RealPoint3d;

use crate::halo::visibility::{RealRectangle3d, VisibilityVolume};

// =============================================================================
// `e_volume_test_result` — Ares engine convention (return code from
// `visibility_volume_test_*`).
// =============================================================================

/// Outside the volume; do not draw.
pub const VOLUME_OUTSIDE: i32 = 0;
/// Crosses the volume boundary; draw and clip per primitive as needed.
pub const VOLUME_INTERSECT: i32 = 1;
/// Fully contained inside the volume; can skip per-primitive clipping.
pub const VOLUME_INSIDE: i32 = 2;

// =============================================================================
// Vertex outcode bits — Ares `visibility_test.cpp:16-26`
//
// Each bit = "vertex is on the OUTSIDE of plane N". Note the
// non-sequential mapping: planes are stored in
// `world_planes[]` as left/right/TOP/BOTTOM/near/far, but bits map
// left=0, right=1, BOTTOM=2, TOP=3, near=4, far=5 — i.e. planes 2/3
// have their bit positions swapped vs natural ordering.
// =============================================================================

pub const VERTEX_OUT_LEFT_BIT: u8 = 1 << 0; // 0x01
pub const VERTEX_OUT_RIGHT_BIT: u8 = 1 << 1; // 0x02
pub const VERTEX_OUT_BOTTOM_BIT: u8 = 1 << 2; // 0x04 — world_planes[3]
pub const VERTEX_OUT_TOP_BIT: u8 = 1 << 3; // 0x08 — world_planes[2]
pub const VERTEX_OUT_NEAR_BIT: u8 = 1 << 4; // 0x10
pub const VERTEX_OUT_FAR_BIT: u8 = 1 << 5; // 0x20
pub const VERTEX_OUT_ALL_BITS: u8 = 0x3F;

// =============================================================================
// Helpers
// =============================================================================

#[inline(always)]
fn signed_distance(plane: blam_tags::math::RealPlane3d, point: RealPoint3d) -> f32 {
    point.x * plane.i + point.y * plane.j + point.z * plane.k - plane.d
}

#[inline(always)]
fn point_outside_plane(volume: &VisibilityVolume, plane_idx: usize, point: RealPoint3d) -> bool {
    signed_distance(volume.world_planes[plane_idx], point) > 0.0
}

// =============================================================================
// `visibility_volume_build_vertex_flags @ 0x180574B30` (B1)
// =============================================================================

/// Build the 6-bit outcode for `point` vs the 6 world planes of
/// `volume`. Each bit is set when the point is on the OUTSIDE of the
/// corresponding plane.
///
/// `test_against_all_planes` controls whether near/far (planes 4/5)
/// participate; engine's triangle/box paths often pass `false` to
/// skip them when only side planes matter.
pub fn visibility_volume_build_vertex_flags(
    volume: &VisibilityVolume,
    point: RealPoint3d,
    test_against_all_planes: bool,
) -> u8 {
    let mut flags = 0u8;
    if point_outside_plane(volume, 0, point) {
        flags |= VERTEX_OUT_LEFT_BIT;
    }
    if point_outside_plane(volume, 1, point) {
        flags |= VERTEX_OUT_RIGHT_BIT;
    }
    if point_outside_plane(volume, 2, point) {
        flags |= VERTEX_OUT_TOP_BIT;
    }
    if point_outside_plane(volume, 3, point) {
        flags |= VERTEX_OUT_BOTTOM_BIT;
    }
    if test_against_all_planes {
        if point_outside_plane(volume, 4, point) {
            flags |= VERTEX_OUT_NEAR_BIT;
        }
        if point_outside_plane(volume, 5, point) {
            flags |= VERTEX_OUT_FAR_BIT;
        }
    }
    flags
}

// =============================================================================
// `visibility_volume_test_point @ 0x1805743B0` (B2)
// =============================================================================

/// Test a single point. Returns `VOLUME_INSIDE` if all 6 planes accept
/// the point, `VOLUME_OUTSIDE` otherwise. Never returns `INTERSECT`
/// (a point has no extent).
pub fn visibility_volume_test_point(volume: &VisibilityVolume, point: RealPoint3d) -> i32 {
    if visibility_volume_build_vertex_flags(volume, point, true) != 0 {
        VOLUME_OUTSIDE
    } else {
        VOLUME_INSIDE
    }
}

// =============================================================================
// `visibility_volume_test_sphere @ 0x18049AC30` (B3)
//
// Engine SSE path calls `IntersectSphere6Planes2(&center, &radius,
// &vector_planes[0], &vector_planes[1])`. Algorithm: per-plane signed
// distance from sphere center; reject if dist > radius (fully on
// outside side); track if all dist < -radius (fully inside). Scalar
// equivalent below uses `world_planes` (same plane equations as
// `vector_planes`, just AoS instead of SoA).
// =============================================================================

/// Sphere-vs-volume intersection. Returns `OUTSIDE` / `INTERSECT` /
/// `INSIDE`. The `OUTSIDE` early-exits as soon as any plane proves
/// the sphere fully on its outside; `INSIDE` is returned only when
/// every plane has the sphere fully on its inside.
pub fn visibility_volume_test_sphere(
    volume: &VisibilityVolume,
    center: RealPoint3d,
    radius: f32,
) -> i32 {
    let mut all_inside = true;
    for i in 0..6 {
        let dist = signed_distance(volume.world_planes[i], center);
        if dist > radius {
            return VOLUME_OUTSIDE;
        }
        if dist > -radius {
            all_inside = false;
        }
    }
    if all_inside { VOLUME_INSIDE } else { VOLUME_INTERSECT }
}

/// `visibility_volume_test_sphere_old @ 0x1805743E0` — legacy scalar
/// path with an additional AABB pre-cull. Same algebra as the SSE
/// path; alias forwards after a fast `world_bounds` reject so callers
/// that prefer the early-exit-heavy version still get it.
pub fn visibility_volume_test_sphere_old(
    volume: &VisibilityVolume,
    center: RealPoint3d,
    radius: f32,
) -> i32 {
    // World-bounds AABB-vs-sphere reject (matches engine pseudocode
    // line-for-line; engine returns 0 here without per-plane testing).
    let b = &volume.world_bounds;
    if center.x - radius > b.x1
        || center.y - radius > b.y1
        || center.z - radius > b.z1
        || center.x + radius < b.x0
        || center.y + radius < b.y0
        || center.z + radius < b.z0
    {
        return VOLUME_OUTSIDE;
    }
    visibility_volume_test_sphere(volume, center, radius)
}

// =============================================================================
// `visibility_volume_test_box @ 0x180574630` (B4)
// =============================================================================

/// AABB-vs-volume intersection. Engine algorithm:
///
/// 1. Early-out if `bounds` doesn't intersect `volume.world_bounds`.
/// 2. Build the 8 corners of `bounds`.
/// 3. For each corner, compute its vertex_flags (LEFT/RIGHT/TOP/BOTTOM
///    bits — near/far excluded since they don't always apply).
/// 4. AND-aggregate (`v14`): bits all corners share (= "all corners
///    on outside of plane N").
/// 5. OR-aggregate (`v17`): any-corner-out bits.
/// 6. If `v17 == 0` → fully inside → `INSIDE`.
/// 7. If `v14 != 0` → all corners share an outside-bit → `OUTSIDE`.
/// 8. Otherwise → `INTERSECT`.
///
/// `test_frustum_against_cube` (when `true`) also tests the volume's
/// frustum vertices against the cube — used by light passes for a
/// tighter cull. Default callers (decorators, instances, clusters)
/// pass `false`.
pub fn visibility_volume_test_box(
    volume: &VisibilityVolume,
    bounds: &RealRectangle3d,
    test_frustum_against_cube: bool,
) -> i32 {
    // ---- Step 1: world_bounds AABB reject (engine lines 90-105) ----
    let vb = &volume.world_bounds;
    if bounds.x0 > vb.x1
        || bounds.y0 > vb.y1
        || bounds.z0 > vb.z1
        || bounds.x1 < vb.x0
        || bounds.y1 < vb.y0
        || bounds.z1 < vb.z0
    {
        return VOLUME_OUTSIDE;
    }

    // ---- Step 2: build 8 corners ----
    let corners = [
        RealPoint3d { x: bounds.x0, y: bounds.y0, z: bounds.z0 },
        RealPoint3d { x: bounds.x1, y: bounds.y0, z: bounds.z0 },
        RealPoint3d { x: bounds.x0, y: bounds.y1, z: bounds.z0 },
        RealPoint3d { x: bounds.x1, y: bounds.y1, z: bounds.z0 },
        RealPoint3d { x: bounds.x0, y: bounds.y0, z: bounds.z1 },
        RealPoint3d { x: bounds.x1, y: bounds.y0, z: bounds.z1 },
        RealPoint3d { x: bounds.x0, y: bounds.y1, z: bounds.z1 },
        RealPoint3d { x: bounds.x1, y: bounds.y1, z: bounds.z1 },
    ];

    // ---- Steps 3-7: aggregate corner outcodes ----
    let mut and_flags: u8 = VERTEX_OUT_ALL_BITS;
    let mut or_flags: u8 = 0;
    for c in corners {
        // Engine passes `0` for `test_against_all_planes` here (only
        // 4 side planes considered, near/far skipped).
        let f = visibility_volume_build_vertex_flags(volume, c, false);
        and_flags &= f;
        or_flags |= f;
    }

    if or_flags == 0 {
        return VOLUME_INSIDE;
    }
    if and_flags != 0 {
        return VOLUME_OUTSIDE;
    }

    // ---- Step 8 + optional frustum-against-cube ----
    if test_frustum_against_cube {
        // Optional tighter test — symmetric outcode of frustum
        // vertices against the cube faces. Engine implements this
        // inline (visibility_volume_test_box pseudocode lines
        // ~190-260). Port deferred to Phase E (consumed only by light
        // pass / lightmap shadow path; current callers all pass false).
        // TODO(phase-e): port frustum-against-cube refinement.
    }

    VOLUME_INTERSECT
}

// =============================================================================
// `visibility_volume_test_triangle @ 0x180574AB0` (B5)
// =============================================================================

/// Triangle-vs-volume test. Returns `false` (outside) if all three
/// vertices share an outside-bit (separating plane theorem on the 6
/// side planes); `true` otherwise.
///
/// Engine signature returns `bool`: the result is binary (treats
/// inside == intersect for the consumers — mainly transparency
/// per-triangle culling and per-surface frustum binning).
pub fn visibility_volume_test_triangle(
    volume: &VisibilityVolume,
    p0: RealPoint3d,
    p1: RealPoint3d,
    p2: RealPoint3d,
) -> bool {
    // Engine pseudocode:
    //   return !flags(p0) || !flags(p1) || (f2 = flags(p2)) == 0 || (f2 & combined) == 0;
    // — i.e. NOT outside iff any vertex is fully inside OR the AND of
    // all three flags is 0 (no separating plane).
    let f0 = visibility_volume_build_vertex_flags(volume, p0, false);
    if f0 == 0 {
        return true;
    }
    let f1 = visibility_volume_build_vertex_flags(volume, p1, false);
    if f1 == 0 {
        return true;
    }
    let f2 = visibility_volume_build_vertex_flags(volume, p2, false);
    if f2 == 0 {
        return true;
    }
    (f0 & f1 & f2) == 0
}
