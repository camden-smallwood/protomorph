//! Engine source: `Ares/source/math/matrix_math.{h,cpp}`.
//!
//! Engine-runtime matrix helpers operating on [`RealMatrix4x3`]. Currently
//! scoped to the subset Phase 7b.4c needs.
//!
//! Function name + signature from Reach (preserved symbol
//! `?matrix4x3_inverse_transform_point@@YAPEATreal_point3d@@PEBUreal_matrix4x3@@PEBT1@PEAT1@@Z`);
//! body from tool.exe (dllcache `0x1802c4ca0`).

use blam_tags::math::{RealMatrix4x3, RealPoint3d};

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

    // Engine line 15-26: scale-divide with near-zero guard.
    let mut scale = matrix.scale;
    if scale != 1.0 {
        // Engine `FLOAT_0_000099999997` / `FLOAT_N0_000099999997` = +/- 0.0001.
        const SCALE_EPS: f32 = 9.999_999_7e-5;
        if scale < 0.0 {
            if scale > -SCALE_EPS {
                scale = -SCALE_EPS;
            }
        } else if scale <= SCALE_EPS {
            scale = SCALE_EPS;
        }
        let inv = 1.0 / scale;
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
}
