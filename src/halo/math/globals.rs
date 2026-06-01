//! Engine math globals — engine-faithful constants accessed via named
//! global pointers in dllcache. Engine declares these as
//! `const real_vector3d *` / `const real_point3d *` etc. pointing into the
//! shared `.rdata` constants block; we materialize them as `const` values
//! since they never change at runtime.
//!
//! Source: dllcache `.rdata` constants block (addresses below). Verified
//! at the value level via the IDA bridge (2026-05-26).

use blam_tags::math::{RealMatrix4x3, RealPoint3d, RealQuaternion, RealVector3d};

/// `global_origin3d` @ dllcache `0x1810D50C0` → points to `(0, 0, 0)`
/// at `0x180b22d28`. The origin point in world space (same backing storage
/// as `global_zero_vector3d`).
pub const GLOBAL_ORIGIN_3D: RealPoint3d = RealPoint3d { x: 0.0, y: 0.0, z: 0.0 };

/// `global_zero_vector3d` @ dllcache `0x1810D50F8` → points to `(0, 0, 0)`
/// at `0x180b22d28`. Zero vector.
pub const GLOBAL_ZERO_VECTOR_3D: RealVector3d = RealVector3d { i: 0.0, j: 0.0, k: 0.0 };

/// `global_left3d` @ dllcache `0x1810D5108` → points to `(0, +1, 0)`
/// at `0x180b22d88`. Halo's convention: left = +Y.
pub const GLOBAL_LEFT_3D: RealVector3d = RealVector3d { i: 0.0, j: 1.0, k: 0.0 };

/// `global_up3d` @ dllcache `0x1810D5110` → points to `(0, 0, +1)`
/// at `0x180b22d94`. World up.
pub const GLOBAL_UP_3D: RealVector3d = RealVector3d { i: 0.0, j: 0.0, k: 1.0 };

/// `global_down3d` @ dllcache `0x1810D5128` → points to `(0, 0, -1)`
/// at `0x180b22dcc`. Gravity / down direction.
pub const GLOBAL_DOWN_3D: RealVector3d = RealVector3d { i: 0.0, j: 0.0, k: -1.0 };

/// `global_identity_quaternion` @ dllcache `0x1810D5148` — identity rotation
/// `(0, 0, 0, 1)`.
pub const GLOBAL_IDENTITY_QUATERNION: RealQuaternion =
    RealQuaternion { i: 0.0, j: 0.0, k: 0.0, w: 1.0 };

/// `global_identity4x3` @ dllcache `0x1810D5150` — identity transform.
/// Stored row-major as `(forward, left, up, position)` where forward = +X,
/// left = +Y, up = +Z, position = origin.
pub const GLOBAL_IDENTITY_4X3: RealMatrix4x3 = RealMatrix4x3 {
    scale: 1.0,
    forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
    left:    RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
    up:      RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
    position: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
};
