//! Mirror of `Ares/source/visibility/visibility_projections_and_volumes.cpp`.
//!
//! - [`visibility_volume_build`] (B6) — populate a [`VisibilityVolume`]
//!   from a [`VisibilityProjection`] + 2D frustum bounds.
//! - [`visibility_region_test_sphere`] (B7) — per-cluster sphere test
//!   against all per-projection volumes.
//! - [`visibility_region_test_box`] (B8) — per-cluster AABB test.
//! - [`visibility_volumes_intersect`] (B9) and the project_* helpers
//!   (B10) are stubbed; bodies land in Phase E (light/shadow paths)
//!   and the occlusion view (lens flares).

use blam_tags::math::{RealMatrix4x3, RealPlane3d, RealPoint3d, RealVector3d};

use crate::halo::visibility::{
    RealRectangle2d, RealRectangle3d, VisibilityProjection, VisibilityVolume,
};

// =============================================================================
// Math helpers (real_math.h equivalents)
//
// Engine uses `matrix4x3_transform_points`, `matrix4x3_transform_plane`,
// `normalize3d`, `real_rectangle3d_enclose_points`. These mirror those
// 1:1 but live here (the visibility module owns its math
// dependencies for now; promote to a shared math module if reused).
// =============================================================================

use crate::halo::math::matrix_math::{
    matrix4x3_transform_plane, matrix4x3_transform_point, normalize3d,
};

/// Public alias for [`matrix4x3_transform_point`] used by the
/// transformed-portal cache (Phase E2). Engine equivalent of
/// `matrix4x3_transform_points` for a single point.
#[inline]
pub fn matrix4x3_transform_point_pub(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    matrix4x3_transform_point(m, p)
}

#[inline]
fn matrix4x3_transform_points_in_place(m: &RealMatrix4x3, points: &mut [RealPoint3d]) {
    for p in points.iter_mut() {
        *p = matrix4x3_transform_point(m, *p);
    }
}

#[inline]
fn real_rectangle3d_empty() -> RealRectangle3d {
    // Engine uses a global "starting empty rect" (`*off_1810D5170`)
    // with x0=+inf etc; equivalent here.
    RealRectangle3d {
        x0: f32::INFINITY,
        x1: f32::NEG_INFINITY,
        y0: f32::INFINITY,
        y1: f32::NEG_INFINITY,
        z0: f32::INFINITY,
        z1: f32::NEG_INFINITY,
    }
}

#[inline]
fn real_rectangle3d_enclose_points(bounds: &mut RealRectangle3d, points: &[RealPoint3d]) {
    for p in points {
        if p.x < bounds.x0 { bounds.x0 = p.x; }
        if p.x > bounds.x1 { bounds.x1 = p.x; }
        if p.y < bounds.y0 { bounds.y0 = p.y; }
        if p.y > bounds.y1 { bounds.y1 = p.y; }
        if p.z < bounds.z0 { bounds.z0 = p.z; }
        if p.z > bounds.z1 { bounds.z1 = p.z; }
    }
}

// =============================================================================
// `visibility_volume_build @ 0x18050C8F0` (B6)
// =============================================================================

/// Populate `volume` from a projection + 2D frustum bounds.
///
/// Returns `true` on success, `false` if `frustum_bounds` is
/// degenerate (engine returns 0 in that case without touching the
/// volume's contents).
///
/// Steps (mirrors engine pseudocode line-for-line):
/// 1. Validate inputs + non-degenerate bounds.
/// 2. Place 4 far-plane corners in basis space:
///    `(far*x0, far*y1, -far)`, `(far*x0, far*y0, -far)`,
///    `(far*x1, far*y1, -far)`, `(far*x1, far*y0, -far)`.
/// 3. Transform in-place by `basis_to_world` → world-space far corners.
/// 4. `world_vertices[4]` = projection origin (camera position).
/// 5. `world_bounds` = AABB enclosing all 5 world vertices.
/// 6. Build 6 BASIS-space planes (LEFT/RIGHT/TOP/BOTTOM/NEAR/FAR),
///    normalize, transform each by `basis_to_world`.
/// 7. Compute 4 `world_edge_vectors` = far_corner[i] − origin.
/// 8. Pack `vector_planes[6]` SoA from `world_planes` (each entry =
///    `(n.i, n.j, n.k, d)`).
pub fn visibility_volume_build(
    projection: &VisibilityProjection,
    projection_index: i16,
    frustum_bounds: &RealRectangle2d,
    volume: &mut VisibilityVolume,
) -> bool {
    // Step 1
    if frustum_bounds.x1 <= frustum_bounds.x0 || frustum_bounds.y1 <= frustum_bounds.y0 {
        return false;
    }

    volume.projection_index = projection_index;
    volume.frustum_bounds = *frustum_bounds;

    // Engine asserts: far_bounded_flag set + basis_to_world.scale == 1.0.
    // Skip the asserts in release; in debug we validate.
    debug_assert!(
        projection.far_bounded_flag,
        "visibility_volume_build: projection not far-bounded"
    );
    debug_assert!(
        (projection.basis_to_world.scale - 1.0).abs() < f32::EPSILON,
        "visibility_volume_build: basis_to_world.scale != 1.0"
    );

    // Step 2: 4 basis-space far-plane corners.
    // Engine code reads odd at first glance — note `_xmm` is the
    // SSE sign-flip mask = XOR with 0x80000000 (negate float).
    let far = projection.far_distance;
    let x0 = frustum_bounds.x0;
    let x1 = frustum_bounds.x1;
    let y0 = frustum_bounds.y0;
    let y1 = frustum_bounds.y1;
    volume.world_vertices[0] = RealPoint3d { x: far * x0, y: far * y1, z: -far };
    volume.world_vertices[1] = RealPoint3d { x: far * x0, y: far * y0, z: -far };
    volume.world_vertices[2] = RealPoint3d { x: far * x1, y: far * y1, z: -far };
    volume.world_vertices[3] = RealPoint3d { x: far * x1, y: far * y0, z: -far };

    // Step 3
    matrix4x3_transform_points_in_place(
        &projection.basis_to_world,
        &mut volume.world_vertices[0..4],
    );

    // Step 4
    volume.world_vertices[4] = projection.basis_to_world.position;

    // Step 5
    let mut bounds = real_rectangle3d_empty();
    real_rectangle3d_enclose_points(&mut bounds, &volume.world_vertices);
    volume.world_bounds = bounds;

    // Step 6: build 4 side planes in basis space, normalize, then
    // transform each by basis_to_world. Plane normals point INTO the
    // frustum (so signed_distance > 0 means OUTSIDE that plane,
    // matching `point_outside_plane`).
    let make_plane = |i: f32, j: f32, k: f32, d: f32| -> RealPlane3d {
        let mut n = RealVector3d { i, j, k };
        normalize3d(&mut n);
        RealPlane3d { i: n.i, j: n.j, k: n.k, d }
    };
    let basis = &projection.basis_to_world;
    volume.world_planes[0] = matrix4x3_transform_plane(basis, make_plane(-1.0, 0.0, -x0, 0.0));
    volume.world_planes[1] = matrix4x3_transform_plane(basis, make_plane(1.0, 0.0, x1, 0.0));
    volume.world_planes[2] = matrix4x3_transform_plane(basis, make_plane(0.0, -1.0, -y0, 0.0));
    volume.world_planes[3] = matrix4x3_transform_plane(basis, make_plane(0.0, 1.0, y1, 0.0));

    // Plane[4] (NEAR): when projection is near-bounded, copy near_plane
    // and NEGATE it (engine flips i/j/k/d via _xmm sign mask). Else
    // build (0, 0, 1) at d=0 in basis and transform.
    if projection.near_bounded_flag {
        volume.world_planes[4] = RealPlane3d {
            i: -projection.near_plane.i,
            j: -projection.near_plane.j,
            k: -projection.near_plane.k,
            d: -projection.near_plane.d,
        };
    } else {
        let p_near = RealPlane3d { i: 0.0, j: 0.0, k: 1.0, d: 0.0 };
        volume.world_planes[4] = matrix4x3_transform_plane(&projection.basis_to_world, p_near);
    }

    // Plane[5] (FAR): (0, 0, -1) at d=far_distance in basis, transform.
    let p_far = RealPlane3d { i: 0.0, j: 0.0, k: -1.0, d: projection.far_distance };
    volume.world_planes[5] = matrix4x3_transform_plane(&projection.basis_to_world, p_far);

    // Step 7: world edge vectors = far_corner[i] − origin.
    let origin = volume.world_vertices[4];
    for i in 0..4 {
        let v = volume.world_vertices[i];
        volume.world_edge_vectors[i] = RealVector3d {
            i: v.x - origin.x,
            j: v.y - origin.y,
            k: v.z - origin.z,
        };
    }

    // Step 8: pack vector_planes (SoA) from world_planes.
    for i in 0..6 {
        let p = volume.world_planes[i];
        volume.vector_planes[i] = [p.i, p.j, p.k, p.d];
    }

    true
}

