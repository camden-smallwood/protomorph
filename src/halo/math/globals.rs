//! Engine math globals — engine-faithful constants accessed via named
//! global pointers in dllcache. Engine declares these as
//! `const real_vector3d *` / `const real_point3d *` etc. pointing into the
//! shared `.rdata` constants block; we materialize them as `const` values
//! since they never change at runtime.
//!
//! Source: dllcache `.rdata` constants block (addresses below). Verified
//! at the value level via the IDA bridge (2026-05-26).

use blam_tags::math::RealVector3d;

/// `global_up3d` @ dllcache `0x1810D5110` → points to `(0, 0, +1)`
/// at `0x180b22d94`. World up.
pub const GLOBAL_UP_3D: RealVector3d = RealVector3d { i: 0.0, j: 0.0, k: 1.0 };

/// `global_down3d` @ dllcache `0x1810D5128` → points to `(0, 0, -1)`
/// at `0x180b22dcc`. Gravity / down direction.
pub const GLOBAL_DOWN_3D: RealVector3d = RealVector3d { i: 0.0, j: 0.0, k: -1.0 };
