//! Engine source: `Ares/source/math/matrix_math.{h,cpp}`.
//!
//! Engine-runtime matrix helpers operating on [`RealMatrix4x3`]. Currently
//! scoped to the subset Phase 7b.4c needs.
//!
//! Function name + signature from Reach (preserved symbol
//! `?matrix4x3_inverse_transform_point@@YAPEATreal_point3d@@PEBUreal_matrix4x3@@PEBT1@PEAT1@@Z`);
//! body from tool.exe (dllcache `0x1802c4ca0`).

use blam_tags::math::{RealMatrix4x3, RealPlane3d, RealPoint3d, RealQuaternion, RealVector3d};
use glam::Quat;

/// Engine `FLOAT_0_000099999997` / `FLOAT_N0_000099999997` — the
/// near-zero scale guard used by every `matrix4x3_inverse_transform_*`.
pub const SCALE_EPS: f32 = 9.999_999_7e-5;

/// Clamp `scale` away from zero **preserving its sign**, exactly like
/// the decompiled `matrix4x3_inverse_transform_point @ 0x1802c4ca0`:
/// negative scales clamp toward `-eps`, positive toward `+eps`.
///
/// This is the engine-faithful behavior. It is NOT `scale.abs().max(eps)`
/// (which would flip the sign of a mirrored / negative-scale instance,
/// silently mirroring its local projection) and NOT `scale.max(eps)`
/// (which collapses any negative scale to `+eps`, producing a huge
/// wrong-sign `1/s` — the B7 bug).
#[inline]
pub fn clamp_scale_signed(scale: f32) -> f32 {
    if scale < 0.0 {
        if scale > -SCALE_EPS {
            -SCALE_EPS
        } else {
            scale
        }
    } else if scale <= SCALE_EPS {
        SCALE_EPS
    } else {
        scale
    }
}

/// 3D cross product `a × b`.
#[inline]
pub fn cross(a: RealVector3d, b: RealVector3d) -> RealVector3d {
    RealVector3d {
        i: a.j * b.k - a.k * b.j,
        j: a.k * b.i - a.i * b.k,
        k: a.i * b.j - a.j * b.i,
    }
}

/// Normalize a vector in place. Zero-length vectors are left unchanged
/// (engine `normalize3d` returns the magnitude and no-ops on zero).
#[inline]
pub fn normalize3d(v: &mut RealVector3d) {
    let len_sq = v.i * v.i + v.j * v.j + v.k * v.k;
    if len_sq > 0.0 {
        let inv = len_sq.sqrt().recip();
        v.i *= inv;
        v.j *= inv;
        v.k *= inv;
    }
}

/// Normalize a vector, returning a fresh value (zero-length unchanged).
#[inline]
pub fn normalize3d_value(mut v: RealVector3d) -> RealVector3d {
    normalize3d(&mut v);
    v
}

/// Convert a tag-authored `RealQuaternion` (i,j,k,w) into a glam `Quat`
/// (x,y,z,w), normalizing with a zero-guard.
///
/// Tag-authored quaternions aren't guaranteed unit-length; feeding a
/// denormalized (or all-zero) quaternion straight into
/// `Mat4::from_rotation_translation` yields a scaled/sheared transform. This
/// is the single canonical conversion — node bind poses, marker rotations,
/// and decorator placements all route through it.
#[inline]
pub fn quat_from_real(q: RealQuaternion) -> Quat {
    let v = Quat::from_xyzw(q.i, q.j, q.k, q.w);
    if v.length_squared() > 1.0e-4 {
        v.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// `matrix4x3_inverse_transform_point(matrix, point, result)` @ dllcache
/// `0x1802c4ca0`.
///
/// Maps a world-space point into the instance-local space defined by
/// `matrix` (scale + 3×3 rotation rows + translation). Used by the
/// geometry sampler instance branch to put a world collision_point into
/// instance-local coords for the point-in-triangle test (since instance
/// mesh vertices are stored in mesh-local coords).
///
/// Engine layout: `matrix.n[0..2]` are rotation rows (each `RealVector3d`),
/// `matrix.n[3]` is translation (`RealPoint3d`). In protomorph's
/// [`RealMatrix4x3`], these correspond to `forward / left / up / position`
/// respectively, with `scale` at offset 0 (engine reads it the same way).
///
/// Algorithm:
/// 1. `diff = point - matrix.position`
/// 2. If scale != 1, divide diff by scale (clamped to avoid div-by-zero
///    near `±1e-4`).
/// 3. `result = transpose(rotation) * diff`. Engine multiplies row-wise,
///    which is mathematically `R^T · diff` since the rows ARE rotation
///    columns when forward/left/up are world-space directions for the
///    local +X/+Y/+Z axes.
///
/// Returns the result pointer (engine returns `real_point3d*`); we return
/// `()` and write through `&mut`.
pub fn matrix4x3_inverse_transform_point(
    matrix: &RealMatrix4x3,
    point: &RealPoint3d,
    result: &mut RealPoint3d,
) {
    // Engine line 11-14: diff = point - matrix.position.
    let mut v4 = point.x - matrix.position.x;
    let mut v3 = point.y - matrix.position.y;
    let mut v6 = point.z - matrix.position.z;

    // Engine line 15-26: scale-divide with near-zero guard (sign-preserving).
    if matrix.scale != 1.0 {
        let inv = 1.0 / clamp_scale_signed(matrix.scale);
        v4 *= inv;
        v3 *= inv;
        v6 *= inv;
    }

    // Engine line 27-30: result = (diff.x, diff.y, diff.z) · row of rotation.
    // Engine indexes `matrix->n[i][j]` where i=row, j=column. Our
    // RealMatrix4x3 stores rows as forward/left/up; field [j] = (i, j, k).
    // - matrix.n[0] = forward (= row 0 of rotation, world-space)
    // - matrix.n[1] = left
    // - matrix.n[2] = up
    //
    // The engine code writes:
    //   result.x = v3 * m[0][1] + v4 * m[0][0] + v6 * m[0][2]
    //            = v4 * m[0][0] + v3 * m[0][1] + v6 * m[0][2]
    //            = (v4, v3, v6) · m[0]
    //            = (v4, v3, v6) · forward
    // and likewise for result.y/z with left/up.
    result.x = v4 * matrix.forward.i + v3 * matrix.forward.j + v6 * matrix.forward.k;
    result.y = v4 * matrix.left.i + v3 * matrix.left.j + v6 * matrix.left.k;
    result.z = v4 * matrix.up.i + v3 * matrix.up.j + v6 * matrix.up.k;
}

/// Value-returning form of [`matrix4x3_inverse_transform_point`] —
/// `p_local = (M^{-1})(p_world)`. Same engine body, ergonomic for
/// callers that compose results.
#[inline]
pub fn matrix4x3_inverse_transform_point_value(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    let mut r = RealPoint3d::default();
    matrix4x3_inverse_transform_point(m, &p, &mut r);
    r
}

/// `matrix4x3_inverse_transform_vector @ 0x1802c4d90`. Same as
/// [`matrix4x3_inverse_transform_point`] but ignores translation:
/// `v_local = R^T · (v / scale)` with the sign-preserving scale clamp.
#[inline]
pub fn matrix4x3_inverse_transform_vector(m: &RealMatrix4x3, v: [f32; 3]) -> [f32; 3] {
    let (mut vx, mut vy, mut vz) = (v[0], v[1], v[2]);
    if m.scale != 1.0 {
        let inv = 1.0 / clamp_scale_signed(m.scale);
        vx *= inv;
        vy *= inv;
        vz *= inv;
    }
    [
        vx * m.forward.i + vy * m.forward.j + vz * m.forward.k,
        vx * m.left.i + vy * m.left.j + vz * m.left.k,
        vx * m.up.i + vy * m.up.j + vz * m.up.k,
    ]
}

/// Rotation-only inverse: `v_local = R^T · v_world`. No translation and
/// **no scale-divide** — used to pull pure DIRECTION basis vectors (which
/// are not scaled by the instance scale) into local space. Same as
/// [`matrix4x3_inverse_transform_normal`].
#[inline]
pub fn matrix4x3_inverse_rotate_vector(m: &RealMatrix4x3, v: RealVector3d) -> RealVector3d {
    RealVector3d {
        i: m.forward.i * v.i + m.forward.j * v.j + m.forward.k * v.k,
        j: m.left.i * v.i + m.left.j * v.j + m.left.k * v.k,
        k: m.up.i * v.i + m.up.j * v.j + m.up.k * v.k,
    }
}

/// `matrix4x3_inverse_transform_normal @ 0x1802c4dd0` — pure rotation
/// transpose `n_local = R^T · n_world` (translation + scale ignored;
/// engine treats normals as directions). Alias of
/// [`matrix4x3_inverse_rotate_vector`].
#[inline]
pub fn matrix4x3_inverse_transform_normal(m: &RealMatrix4x3, n: RealVector3d) -> RealVector3d {
    matrix4x3_inverse_rotate_vector(m, n)
}

/// `matrix4x3_inverse_transform_plane @ 0x1802c4f40` — inverse of
/// `matrix4x3_transform_plane`. World plane `(n, d)` (`n·p = d`) maps to
/// local plane: normal rotates by `R^T`, and
/// `d_local = (d - n·t) / scale` with the sign-preserving clamp.
#[inline]
pub fn matrix4x3_inverse_transform_plane(m: &RealMatrix4x3, p: RealPlane3d) -> RealPlane3d {
    let n = matrix4x3_inverse_rotate_vector(m, RealVector3d { i: p.i, j: p.j, k: p.k });
    let s = clamp_scale_signed(m.scale);
    let d_local =
        (p.d - (p.i * m.position.x + p.j * m.position.y + p.k * m.position.z)) / s;
    RealPlane3d { i: n.i, j: n.j, k: n.k, d: d_local }
}

/// `matrix4x3_transform_point` — apply basis + scale + translation:
/// `w = scale·(p.x·forward + p.y·left + p.z·up) + position`.
#[inline]
pub fn matrix4x3_transform_point(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    let s = m.scale;
    RealPoint3d {
        x: s * (p.x * m.forward.i + p.y * m.left.i + p.z * m.up.i) + m.position.x,
        y: s * (p.x * m.forward.j + p.y * m.left.j + p.z * m.up.j) + m.position.y,
        z: s * (p.x * m.forward.k + p.y * m.left.k + p.z * m.up.k) + m.position.z,
    }
}

/// `[f32; 3]` form of [`matrix4x3_transform_point`].
#[inline]
pub fn matrix4x3_transform_point_arr(m: &RealMatrix4x3, p: [f32; 3]) -> [f32; 3] {
    let r = matrix4x3_transform_point(m, RealPoint3d { x: p[0], y: p[1], z: p[2] });
    [r.x, r.y, r.z]
}

/// `matrix4x3_transform_normal` — apply basis + scale, no translation.
#[inline]
pub fn matrix4x3_transform_normal(m: &RealMatrix4x3, n: RealVector3d) -> RealVector3d {
    let s = m.scale;
    RealVector3d {
        i: s * (n.i * m.forward.i + n.j * m.left.i + n.k * m.up.i),
        j: s * (n.i * m.forward.j + n.j * m.left.j + n.k * m.up.j),
        k: s * (n.i * m.forward.k + n.j * m.left.k + n.k * m.up.k),
    }
}

/// `matrix4x3_transform_plane` — for scale=1 + orthonormal basis:
/// `n_world = R · n_basis`, `d_world = d_basis + n_world·position`.
#[inline]
pub fn matrix4x3_transform_plane(m: &RealMatrix4x3, plane: RealPlane3d) -> RealPlane3d {
    let nx = plane.i * m.forward.i + plane.j * m.left.i + plane.k * m.up.i;
    let ny = plane.i * m.forward.j + plane.j * m.left.j + plane.k * m.up.j;
    let nz = plane.i * m.forward.k + plane.j * m.left.k + plane.k * m.up.k;
    let d = plane.d + (nx * m.position.x + ny * m.position.y + nz * m.position.z);
    RealPlane3d { i: nx, j: ny, k: nz, d }
}

/// Inverse-compose `M_inst^{-1} ∘ M_world`. The result's scale is
/// `M_world.scale / clamp(M_inst.scale)` (sign-preserving), basis vectors
/// are inverse-ROTATED (directions, no scale-divide), and the position is
/// inverse-transformed (with scale-divide). Used to pull a world-space
/// projection into an instance's local space.
#[inline]
pub fn matrix4x3_inverse_compose(m_inst: &RealMatrix4x3, m_world: &RealMatrix4x3) -> RealMatrix4x3 {
    RealMatrix4x3 {
        scale: m_world.scale / clamp_scale_signed(m_inst.scale),
        forward: matrix4x3_inverse_rotate_vector(m_inst, m_world.forward),
        left: matrix4x3_inverse_rotate_vector(m_inst, m_world.left),
        up: matrix4x3_inverse_rotate_vector(m_inst, m_world.up),
        position: matrix4x3_inverse_transform_point_value(m_inst, m_world.position),
    }
}

/// Full orthonormal inverse (`matrix4x3_inverse`). Fast inverse of an
/// affine transform with **scale == 1** and an orthonormal basis: the
/// rotation transposes and the translation negates + re-rotates. Asserts
/// unit scale in debug builds.
#[inline]
pub fn matrix4x3_inverse_orthonormal(m: &RealMatrix4x3) -> RealMatrix4x3 {
    debug_assert!((m.scale - 1.0).abs() < 1e-4, "non-unit scale");
    let f = m.forward;
    let l = m.left;
    let u = m.up;
    let p = m.position;
    RealMatrix4x3 {
        scale: 1.0,
        forward: RealVector3d { i: f.i, j: l.i, k: u.i },
        left: RealVector3d { i: f.j, j: l.j, k: u.j },
        up: RealVector3d { i: f.k, j: l.k, k: u.k },
        position: RealPoint3d {
            x: -(p.x * f.i + p.y * f.j + p.z * f.k),
            y: -(p.x * l.i + p.y * l.j + p.z * l.k),
            z: -(p.x * u.i + p.y * u.j + p.z * u.k),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::math::RealVector3d;

    /// Identity rotation + zero translation + scale 1 → result equals point.
    #[test]
    fn identity_passthrough() {
        let m = RealMatrix4x3 {
            scale: 1.0,
            forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
            left: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d::default(),
        };
        let p = RealPoint3d { x: 3.0, y: 4.0, z: 5.0 };
        let mut r = RealPoint3d::default();
        matrix4x3_inverse_transform_point(&m, &p, &mut r);
        assert!((r.x - 3.0).abs() < 1e-5);
        assert!((r.y - 4.0).abs() < 1e-5);
        assert!((r.z - 5.0).abs() < 1e-5);
    }

    /// Translation subtraction: world point (10, 0, 0), instance at (3, 0, 0),
    /// no rotation/scale → local (7, 0, 0).
    #[test]
    fn translation_subtracts() {
        let m = RealMatrix4x3 {
            scale: 1.0,
            forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
            left: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d { x: 3.0, y: 0.0, z: 0.0 },
        };
        let p = RealPoint3d { x: 10.0, y: 0.0, z: 0.0 };
        let mut r = RealPoint3d::default();
        matrix4x3_inverse_transform_point(&m, &p, &mut r);
        assert!((r.x - 7.0).abs() < 1e-5);
        assert!(r.y.abs() < 1e-5);
        assert!(r.z.abs() < 1e-5);
    }

    /// Scale-2 instance: world (4, 0, 0), instance at origin → local (2, 0, 0).
    #[test]
    fn scale_divides() {
        let m = RealMatrix4x3 {
            scale: 2.0,
            forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
            left: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d::default(),
        };
        let p = RealPoint3d { x: 4.0, y: 0.0, z: 0.0 };
        let mut r = RealPoint3d::default();
        matrix4x3_inverse_transform_point(&m, &p, &mut r);
        assert!((r.x - 2.0).abs() < 1e-5);
    }

    /// 90° rotation about Z (forward = +Y, left = -X): world +X → local -Y.
    /// In Halo's row-stored rotation, forward = (0,1,0) means "local +X is
    /// world +Y", so to map world +X back to local we need (1,0,0) → (?,?,?).
    /// Inverse-rotate world (1, 0, 0) under this basis → (0, -1, 0):
    /// because local +X is world +Y, world +X is local -Y.
    #[test]
    fn rotation_inverse_rotates() {
        let m = RealMatrix4x3 {
            scale: 1.0,
            forward: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            left: RealVector3d { i: -1.0, j: 0.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d::default(),
        };
        let p = RealPoint3d { x: 1.0, y: 0.0, z: 0.0 };
        let mut r = RealPoint3d::default();
        matrix4x3_inverse_transform_point(&m, &p, &mut r);
        assert!(r.x.abs() < 1e-5, "r.x={}", r.x);
        assert!((r.y - (-1.0)).abs() < 1e-5, "r.y={}", r.y);
        assert!(r.z.abs() < 1e-5);
    }

    fn identity_scaled(scale: f32) -> RealMatrix4x3 {
        RealMatrix4x3 {
            scale,
            forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
            left: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d::default(),
        }
    }

    /// B7 regression: a NEGATIVE (mirrored) scale must PRESERVE its sign
    /// through the inverse transform. With `scale = -2`, `inv = -0.5`, so
    /// world (4,0,0) → local (-2,0,0). The old `.max(eps)` bug would clamp
    /// -2 up to +1e-4 → inv ≈ +10000 → garbage (huge, wrong sign).
    #[test]
    fn negative_scale_preserves_sign() {
        let m = identity_scaled(-2.0);
        let p = RealPoint3d { x: 4.0, y: 0.0, z: 0.0 };
        let r = matrix4x3_inverse_transform_point_value(&m, p);
        assert!((r.x - (-2.0)).abs() < 1e-5, "r.x={}", r.x);

        // Same for the vector form.
        let v = matrix4x3_inverse_transform_vector(&m, [4.0, 0.0, 0.0]);
        assert!((v[0] - (-2.0)).abs() < 1e-5, "v.x={}", v[0]);
    }

    /// The signed clamp itself: tiny negative → -eps, tiny positive → +eps,
    /// large magnitudes pass through unchanged with their sign.
    #[test]
    fn clamp_scale_signed_behavior() {
        assert_eq!(clamp_scale_signed(-1e-9), -SCALE_EPS);
        assert_eq!(clamp_scale_signed(1e-9), SCALE_EPS);
        assert!((clamp_scale_signed(-3.0) - (-3.0)).abs() < 1e-9);
        assert!((clamp_scale_signed(3.0) - 3.0).abs() < 1e-9);
    }

    /// `transform_point` then `inverse_transform_point` round-trips,
    /// including a negative scale.
    #[test]
    fn transform_inverse_roundtrip_negative_scale() {
        let m = RealMatrix4x3 {
            scale: -1.5,
            forward: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            left: RealVector3d { i: -1.0, j: 0.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d { x: 5.0, y: -2.0, z: 1.0 },
        };
        let local = RealPoint3d { x: 1.0, y: 2.0, z: 3.0 };
        let world = matrix4x3_transform_point(&m, local);
        let back = matrix4x3_inverse_transform_point_value(&m, world);
        assert!((back.x - local.x).abs() < 1e-4, "x {}", back.x);
        assert!((back.y - local.y).abs() < 1e-4, "y {}", back.y);
        assert!((back.z - local.z).abs() < 1e-4, "z {}", back.z);
    }

    #[test]
    fn orthonormal_inverse_roundtrips() {
        let m = RealMatrix4x3 {
            scale: 1.0,
            forward: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            left: RealVector3d { i: -1.0, j: 0.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            position: RealPoint3d { x: 3.0, y: 4.0, z: 5.0 },
        };
        let inv = matrix4x3_inverse_orthonormal(&m);
        let p = RealPoint3d { x: 7.0, y: -2.0, z: 0.5 };
        let w = matrix4x3_transform_point(&m, p);
        let back = matrix4x3_transform_point(&inv, w);
        assert!((back.x - p.x).abs() < 1e-4);
        assert!((back.y - p.y).abs() < 1e-4);
        assert!((back.z - p.z).abs() < 1e-4);
    }

    #[test]
    fn cross_is_right_handed() {
        let x = RealVector3d { i: 1.0, j: 0.0, k: 0.0 };
        let y = RealVector3d { i: 0.0, j: 1.0, k: 0.0 };
        let z = cross(x, y);
        assert!((z.k - 1.0).abs() < 1e-6 && z.i.abs() < 1e-6 && z.j.abs() < 1e-6);
    }
}
