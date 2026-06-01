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

use crate::halo::visibility::visibility_test::{
    visibility_volume_test_box, visibility_volume_test_sphere, VOLUME_INSIDE, VOLUME_OUTSIDE,
};
use crate::halo::visibility::{
    visibility_region_get_cluster, RealRectangle2d, RealRectangle3d,
    SVisibilityRegion, VisibilityProjection, VisibilityVolume,
    VisibilityVolumeIntersection,
};

// =============================================================================
// Math helpers (real_math.h equivalents)
//
// Engine uses `matrix4x3_transform_points`, `matrix4x3_transform_plane`,
// `normalize3d`, `real_rectangle3d_enclose_points`. These mirror those
// 1:1 but live here (the visibility module owns its math
// dependencies for now; promote to a shared math module if reused).
// =============================================================================

#[inline]
fn normalize3d(v: &mut RealVector3d) {
    let len_sq = v.i * v.i + v.j * v.j + v.k * v.k;
    if len_sq > 0.0 {
        let inv = 1.0 / len_sq.sqrt();
        v.i *= inv;
        v.j *= inv;
        v.k *= inv;
    }
}

/// Public alias for [`matrix4x3_transform_point`] used by the
/// transformed-portal cache (Phase E2). Engine equivalent of
/// `matrix4x3_transform_points` for a single point.
#[inline]
pub fn matrix4x3_transform_point_pub(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    matrix4x3_transform_point(m, p)
}

#[inline]
fn matrix4x3_transform_point(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    // Halo basis layout: basis vectors are rows-of-3-floats in
    // `forward/left/up`; transform is
    //   w = scale*(p.x*forward + p.y*left + p.z*up) + position
    // (engine asserts scale == 1.0 in `visibility_volume_build`, but
    // honour it here in case of future callers.)
    let s = m.scale;
    RealPoint3d {
        x: s * (p.x * m.forward.i + p.y * m.left.i + p.z * m.up.i) + m.position.x,
        y: s * (p.x * m.forward.j + p.y * m.left.j + p.z * m.up.j) + m.position.y,
        z: s * (p.x * m.forward.k + p.y * m.left.k + p.z * m.up.k) + m.position.z,
    }
}

#[inline]
fn matrix4x3_transform_points_in_place(m: &RealMatrix4x3, points: &mut [RealPoint3d]) {
    for p in points.iter_mut() {
        *p = matrix4x3_transform_point(m, *p);
    }
}

#[inline]
fn matrix4x3_transform_plane(m: &RealMatrix4x3, plane: RealPlane3d) -> RealPlane3d {
    // For scale=1 + orthonormal basis: n_world = R * n_basis,
    //   d_world = d_basis + dot(n_world, position).
    let nx = plane.i * m.forward.i + plane.j * m.left.i + plane.k * m.up.i;
    let ny = plane.i * m.forward.j + plane.j * m.left.j + plane.k * m.up.j;
    let nz = plane.i * m.forward.k + plane.j * m.left.k + plane.k * m.up.k;
    let d = plane.d + (nx * m.position.x + ny * m.position.y + nz * m.position.z);
    RealPlane3d { i: nx, j: ny, k: nz, d }
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

// =============================================================================
// `visibility_region_test_sphere @ 0x18050D570` (B7)
// =============================================================================

/// Test sphere against all per-projection volumes of a region cluster.
///
/// Returns the `VOLUME_*` discriminant ORed across volumes (so any
/// intersect returns intersect; any inside-or-intersect returns
/// non-zero). `fully_contained` is set to `true` only when at least
/// one volume fully contains the sphere — engine semantic preserved.
///
/// This is the direct entry point used by
/// `c_structure_renderer::render_decorators @ 0x1806901A0` to cull
/// decorator-cluster runs against the camera region (fixes decorator
/// pop in Phase I.1).
pub fn visibility_region_test_sphere(
    region: &SVisibilityRegion,
    region_cluster_index: i32,
    bounding_center: RealPoint3d,
    bounding_radius: f32,
    fully_contained: &mut bool,
) -> u8 {
    *fully_contained = false;
    let Some(cluster) = visibility_region_get_cluster(region, region_cluster_index) else {
        return 0;
    };
    debug_assert!(
        region.projection_count >= 0 && region.projection_count <= 6,
        "region.projection_count out of range"
    );

    let mut result_any: u8 = 0;
    let mut found_inside = false;
    'outer: for proj in 0..region.projection_count as usize {
        let count = cluster.volume_counts[proj] as i32;
        let first = cluster.first_volume_indices[proj] as i32;
        for i in 0..count {
            let volume_idx = first + i;
            if volume_idx < 0 || volume_idx >= region.volume_count as i32 {
                continue;
            }
            let volume = &region.volumes[volume_idx as usize];
            let result = visibility_volume_test_sphere(volume, bounding_center, bounding_radius);
            if result != VOLUME_OUTSIDE {
                result_any = 1;
            }
            if result == VOLUME_INSIDE {
                found_inside = true;
                break 'outer;
            }
        }
    }
    *fully_contained = found_inside;
    result_any
}

// =============================================================================
// `visibility_region_test_box @ 0x18050D7A0` (B8)
// =============================================================================

/// Test AABB against all per-projection volumes of a region cluster.
/// Returns `true` if any volume returns intersect/inside.
///
/// `test_frustum_against_cube` is forwarded to `volume_test_box` —
/// most callers pass `false`; light/shadow paths pass `true` for the
/// tighter cull (Phase E refinement).
pub fn visibility_region_test_box(
    region: &SVisibilityRegion,
    region_cluster_index: i32,
    bounding_box: &RealRectangle3d,
    test_frustum_against_cube: bool,
) -> bool {
    let Some(cluster) = visibility_region_get_cluster(region, region_cluster_index) else {
        return false;
    };
    debug_assert!(
        region.projection_count >= 0 && region.projection_count <= 6,
        "region.projection_count out of range"
    );

    let mut any_intersect = false;
    for proj in 0..region.projection_count as usize {
        let count = cluster.volume_counts[proj] as i32;
        let first = cluster.first_volume_indices[proj] as i32;
        for i in 0..count {
            let volume_idx = first + i;
            if volume_idx < 0 || volume_idx >= region.volume_count as i32 {
                continue;
            }
            let volume = &region.volumes[volume_idx as usize];
            let result =
                visibility_volume_test_box(volume, bounding_box, test_frustum_against_cube);
            if result != VOLUME_OUTSIDE {
                any_intersect = true;
                break; // engine breaks inner loop on first hit per projection
            }
        }
    }
    any_intersect
}

// =============================================================================
// `visibility_volumes_intersect` (B9) — light × camera region intersection
// `visibility_project_*` (B10) — projection helpers for occlusion / lens flares
//
// Stubs only: bodies depend on `c_visibility_collection::intersect_*`
// (Phase F) and the occlusion view (Phase G.5). Returning conservative
// "intersect/cannot-cull" results so any caller wiring up early sees
// nothing being culled rather than nothing being drawn.
// =============================================================================

/// `visibility_volumes_intersect` — Phase F stub. Always reports an
/// intersection; output structure is filled with zeros.
pub fn visibility_volumes_intersect(
    _light_projection: &VisibilityProjection,
    _light_volume: &VisibilityVolume,
    _camera_projection: &VisibilityProjection,
    _camera_volume: &VisibilityVolume,
    output_intersection: &mut VisibilityVolumeIntersection,
    _create_convex_hull: bool,
) -> bool {
    *output_intersection = VisibilityVolumeIntersection::default();
    // TODO(phase-f): port real intersection math from
    // `Ares/source/visibility/visibility_projections_and_volumes.cpp`
    // (used by c_visibility_collection::intersect_light_region_with_camera_region).
    true
}

/// `visibility_project_sphere` — Phase G stub (occlusion view).
pub fn visibility_project_sphere(
    _projection: &VisibilityProjection,
    _center: RealPoint3d,
    _radius: f32,
    projected_bounds: &mut RealRectangle3d,
) -> bool {
    *projected_bounds = real_rectangle3d_empty();
    // TODO(phase-g): port — needed by `c_occlusion_view::compute_visibility`.
    false
}

/// `visibility_project_box` — Phase G stub.
pub fn visibility_project_box(
    _projection: &VisibilityProjection,
    _bound: &RealRectangle3d,
    projected_bounds: &mut RealRectangle3d,
) -> bool {
    *projected_bounds = real_rectangle3d_empty();
    false
}
