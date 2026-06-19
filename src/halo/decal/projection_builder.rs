//! `s_decal_projection_builder` — per-surface clip volume + cull/clamp
//! angle gates. The per-decal orchestrator initializes one of these
//! from `c_decal_definition` + the placement quaternion; the BFS
//! walker folds it across each shared edge as it walks neighbor
//! surfaces.
//!
//! Anchored against dllcache:
//!   `s_decal_projection_builder::initialize @ 0x1803A10F0`
//!   `s_decal_projection_builder::build_projection @ 0x18039E510`
//!   `matrix4x3_rotation_from_axis_and_angle @ 0x1802C3EF0`
//!   `matrix4x3_multiply @ 0x1802C5A70`

use blam_tags::math::{RealPoint3d, RealVector3d};

use blam_tags::math::RealMatrix4x3;

use crate::halo::math::matrix_math::{cross, normalize3d};

use super::types::{flag, DecalProjectionBuilder, Fold};

impl DecalProjectionBuilder {
    /// Mirror of `s_decal_projection_builder::initialize @ 0x1803A10F0`.
    /// Stores the matrix, precomputes `cos(cull_angle)` /
    /// `cos(clamp_angle)`, and seeds the flag word with INITIALIZED +
    /// NEEDS_RENORMALIZE (plus LEFT_HANDED when the parent frame is
    /// left-handed).
    pub fn initialize(
        &mut self,
        desired_projection: &RealMatrix4x3,
        cull_angle_radians: f32,
        clamp_angle_radians: f32,
        left_handed: bool,
    ) {
        self.projection = *desired_projection;
        self.cull_angle_radians = cull_angle_radians;
        self.cull_angle_cos = cull_angle_radians.cos();
        self.clamp_angle_radians = clamp_angle_radians;
        self.clamp_angle_cos = clamp_angle_radians.cos();
        // Engine line: `m_flags = (m_flags & ~3) | 1` then conditionally
        // sets bit 3 (left-handed) then OR-s in bit 4
        // (needs_renormalize). The remaining bits are not touched.
        let mut flags = (self.flags & !0b11) | flag::INITIALIZED;
        if left_handed {
            flags |= flag::LEFT_HANDED;
        } else {
            flags &= !flag::LEFT_HANDED;
        }
        flags |= flag::NEEDS_RENORMALIZE;
        self.flags = flags;
    }

    /// Mirror of `s_decal_projection_builder::build_projection @ 0x18039E510`.
    ///
    /// Given an incoming `fold` from `c_collision_surface_edge_iterator
    /// ::get_opposing_surface_fold`, validates against the cull cone,
    /// rebuilds the orthonormal basis perpendicular to the fold normal,
    /// and (if the fold exceeds the clamp angle) rotates the
    /// projection around the fold axis to "wrap" the decal onto the
    /// new surface.
    ///
    /// Returns `false` when the fold is outside the cull cone (the
    /// caller drops this neighbor); `true` when the projection is
    /// built and the BFS should push the neighbor for further
    /// expansion.
    pub fn build_projection(&mut self, fold: &Fold) -> bool {
        debug_assert!(
            (self.flags & flag::INITIALIZED) != 0,
            "build_projection without initialize"
        );

        // Engine line 91-95: `cos_angle = -projection.forward · fold.normal`.
        // The `^ _xmm` in the decompile is an SSE float negate (sign bit
        // XOR), applied to the dot product result.
        let cos_angle = -(self.projection.forward.i * fold.normal.i
            + self.projection.forward.j * fold.normal.j
            + self.projection.forward.k * fold.normal.k);

        if cos_angle < self.cull_angle_cos {
            return false;
        }

        // Engine lines 100-135: rebuild basis[1] (left) from the
        // projection's CURRENT up axis (basis[2]) crossed with forward.
        // The sign depends on the LEFT_HANDED flag.
        //
        //   right_handed:  left = projection.up × forward
        //   left_handed:   left = forward × projection.up
        //
        // Then up = forward × left (or left × forward in left-handed)
        // so the basis remains orthonormal.
        //
        // **Why `projection.up` and NOT `fold.normal`:** verified against
        // engine `s_decal_projection_builder::build_projection @ 0x18039E510`
        // — engine reads `projection.n[2]` (basis[2] = current up).
        // `projection.up` is orthonormal-perpendicular to forward by
        // construction (set during initialize from the placement's
        // authored up axis); fold.normal is the surface normal, which
        // for wall-mounted decals is ≈ ±forward (we project INTO the
        // wall whose normal we're sitting on) → cross collapses to
        // zero → normalize3d returns NaN → basis_left is garbage →
        // texcoords are garbage → decals render at random rotations.
        // The re-orth block at line ~210 already uses projection.up
        // correctly — only this initial rebuild was wrong.
        let up_basis = self.projection.up;
        let forward = self.projection.forward;
        let (mut left, up);
        if (self.flags & flag::LEFT_HANDED) != 0 {
            left = cross(forward, up_basis);
        } else {
            left = cross(up_basis, forward);
        }
        normalize3d(&mut left);
        if (self.flags & flag::LEFT_HANDED) != 0 {
            up = cross(left, forward);
        } else {
            up = cross(forward, left);
        }
        self.projection.left = left;
        self.projection.up = up;

        // Engine lines 137-275: if cos_angle is BELOW the clamp angle,
        // the projection must be folded around `fold.axis` so the
        // decal wraps onto the perpendicular surface.
        if cos_angle < self.clamp_angle_cos {
            self.flags |= flag::FOLDED;

            // The math derives a rotation angle θ such that, after
            // rotation, the projected forward direction lies in the
            // half-space defined by the clamp angle. The implementation
            // first computes:
            //   t = forward · fold.axis  (signed projection of forward onto axis)
            //   projected = fold.axis * t  (component of forward along axis)
            //   perp = (forward - projected) - clamp_angle_cos * fold.normal
            //        (perpendicular component, biased toward fold.normal
            //         by the clamp angle)
            //
            // Then it solves for the rotation that aligns the perpendicular
            // part with fold.normal, using the cross-product handedness
            // gate (`sign`) and Rodrigues' formula.

            let t = forward.i * fold.axis.i
                + forward.j * fold.axis.j
                + forward.k * fold.axis.k;
            let neg_clamp_cos = -self.clamp_angle_cos;

            let proj_x = fold.axis.i * t + fold.normal.i * neg_clamp_cos;
            let proj_y = fold.axis.j * t + fold.normal.j * neg_clamp_cos;
            let proj_z = fold.axis.k * t + fold.normal.k * neg_clamp_cos;

            // Cross product (fold.axis × fold.normal) for handedness +
            // rotation-direction probing.
            let cross_x = fold.axis.k * fold.normal.j - fold.axis.j * fold.normal.k;
            let cross_y = fold.axis.i * fold.normal.k - fold.axis.k * fold.normal.i;
            let cross_z = fold.axis.j * fold.normal.i - fold.axis.i * fold.normal.j;
            let dot_cross_fwd = cross_x * forward.i + cross_y * forward.j + cross_z * forward.k;
            let sign = if dot_cross_fwd < 0.0 { -1.0 } else { 1.0 };

            // proj_sq = |proj|². Clamp to [0, 1] before sqrt(1 - proj_sq).
            let mut proj_sq = proj_x * proj_x + proj_y * proj_y + proj_z * proj_z;
            if proj_sq > 0.0 {
                if proj_sq >= 1.0 {
                    proj_sq = 1.0;
                }
            } else {
                proj_sq = 0.0;
            }
            let comp = (1.0 - proj_sq).sqrt() * sign;

            // Step the projection-space "x" component along the axis-cross by `comp`.
            let stepped_x = proj_x + comp * cross_x;
            let stepped_y = proj_y + comp * cross_y;
            let stepped_z = proj_z + comp * cross_z;

            // q1/q2/q3 = (axis × forward) per engine 0x18039E510:~248-260.
            // Verified: q1 = (axis × forward).x = axis.j*forward.k - axis.k*forward.j.
            //
            // **First-term bug fix 2026-05-13:** q1's first factor was
            // `forward.j` instead of `fold.axis.j` — typo that produced
            // wrong axis-cross magnitudes for the fold rotation,
            // visible only on folded decals (riverworld terrain).
            let q1 = fold.axis.j * self.projection.forward.k
                - fold.axis.k * self.projection.forward.j;
            let q2 = fold.axis.k * self.projection.forward.i
                - self.projection.forward.k * fold.axis.i;
            let q3 = fold.axis.i * self.projection.forward.j
                - fold.axis.j * self.projection.forward.i;

            // num = (axis × stepped) · (axis × forward), per engine
            // 0x18039E510:~268-285. Each term is a component of
            // (axis × stepped) multiplied by the matching component
            // of (axis × forward) = (q1, q2, q3).
            //
            // Engine layout:
            //   term1 = (stepped.x*axis.z - axis.x*stepped.z) * q2  // (a×s).y * (a×f).y
            //   term2 = (axis.y*stepped.z - stepped.y*axis.z) * q1  // (a×s).x * (a×f).x
            //   term3 = (axis.x*stepped.y - axis.y*stepped.x) * q3  // (a×s).z * (a×f).z
            //
            // **Term-2 bug fix 2026-05-13:** middle factor was `q1`
            // instead of `stepped_y` — typo that fed (axis×forward).x
            // back into the numerator instead of the y-component of
            // stepped, producing wildly wrong sin θ for folds.
            let num = (stepped_x * fold.axis.k - fold.axis.i * stepped_z) * q2
                + (fold.axis.j * stepped_z - stepped_y * fold.axis.k) * q1
                + (fold.axis.i * stepped_y - fold.axis.j * stepped_x) * q3;
            let denom = q1 * q1 + q2 * q2 + q3 * q3;
            let sin_theta = if denom > 0.0 { num / denom } else { 0.0 };

            let mut sin_sq = sin_theta * sin_theta;
            if sin_sq > 0.0 {
                if sin_sq >= 1.0 {
                    sin_sq = 1.0;
                }
            } else {
                sin_sq = 0.0;
            }

            // Engine 0x18039E510:~298-302 — passes:
            //   sine_arg   = -sqrt(1 - sin²θ) * sign     (= -|cos θ| * sign)
            //   cosine_arg = sin θ (the num/denom result)
            //
            // This is a complementary-angle rotation parameterization
            // (sin² + cos² = 1 still holds), NOT the obvious
            // (sin θ, cos θ) pair. Engine reads `v60 = *(double *)v53` =
            // SSE-loaded sin θ as the cosine arg, and the negated sqrt
            // as the sine arg.
            //
            // **Argument-swap bug fix 2026-05-13:** port had sine =
            // -sin θ (negated wrong value) and cosine = sqrt(1-sin²)*sign
            // (computed cosine, but engine doesn't use this — uses sin θ
            // directly). Swapping to engine convention.
            let sin_arg = -((1.0 - sin_sq).sqrt() * sign);
            let cos_arg = sin_theta;

            // Build the fold-rotation matrix R(fold.axis, θ').
            let fold_matrix = matrix4x3_rotation_from_axis_and_angle(
                &fold.axis,
                sin_arg,
                cos_arg as f64,
            );

            // Conjugate by translation to fold.origin:
            //   M_new = T(origin) * R * T(-origin) * M
            // The decompile inlines this as a manual subtract / multiply
            // / add of the position column.
            self.projection.position.x -= fold.origin.x;
            self.projection.position.y -= fold.origin.y;
            self.projection.position.z -= fold.origin.z;
            self.projection = matrix4x3_multiply(&fold_matrix, &self.projection);
            self.projection.position.x += fold.origin.x;
            self.projection.position.y += fold.origin.y;
            self.projection.position.z += fold.origin.z;

            // Engine lines 228-275: re-orthogonalize the basis when
            // NEEDS_RENORMALIZE is set. Verified via dllcache:
            // engine post-rotation always uses `left = up × forward`
            // and `up = forward × left` (no LH branch on the cross —
            // the LH negate is applied AFTER the cross). My earlier
            // port had both crosses reversed, which produced
            // negated basis vectors → flipped texcoords on
            // riverworld terrain folds.
            if (self.flags & flag::NEEDS_RENORMALIZE) != 0 {
                normalize3d(&mut self.projection.forward);
                let mut new_left =
                    cross(self.projection.up, self.projection.forward);
                self.projection.left = new_left;
                normalize3d(&mut self.projection.left);
                new_left = self.projection.left;
                let new_up = cross(self.projection.forward, new_left);
                self.projection.up = new_up;
                normalize3d(&mut self.projection.up);
                if (self.flags & flag::LEFT_HANDED) != 0 {
                    self.projection.left.i = -self.projection.left.i;
                    self.projection.left.j = -self.projection.left.j;
                    self.projection.left.k = -self.projection.left.k;
                }
            }
        }

        self.flags |= flag::BUILT;
        true
    }
}

// =============================================================================
// Inline math helpers
// =============================================================================

/// Mirror of `matrix4x3_rotation_from_axis_and_angle @ 0x1802C3EF0`.
/// Builds a rotation-only `real_matrix4x3` (scale = 1, position = 0)
/// representing rotation about `axis` by `angle` where `sine = sin θ`
/// and `cosine = cos θ` are precomputed. Equivalent to Rodrigues'
/// formula with Halo's column-major-basis storage.
fn matrix4x3_rotation_from_axis_and_angle(
    axis: &RealVector3d,
    sine: f32,
    cosine: f64,
) -> RealMatrix4x3 {
    let x = axis.i;
    let y = axis.j;
    let z = axis.k;
    let cos32 = cosine as f32;
    let one_minus_cos = 1.0 - cos32;
    let sx = x * sine;
    let sy = y * sine;
    let sz = z * sine;

    let xy_omc = x * y * one_minus_cos;
    let xz_omc = x * z * one_minus_cos;
    let yz_omc = y * z * one_minus_cos;

    RealMatrix4x3 {
        scale: 1.0,
        // basis[0] = forward = column 0 of R
        forward: RealVector3d {
            i: cos32 + x * x * one_minus_cos,
            j: xy_omc + sz,
            k: xz_omc - sy,
        },
        // basis[1] = left = column 1 of R
        left: RealVector3d {
            i: xy_omc - sz,
            j: cos32 + y * y * one_minus_cos,
            k: yz_omc + sx,
        },
        // basis[2] = up = column 2 of R
        up: RealVector3d {
            i: xz_omc + sy,
            j: yz_omc - sx,
            k: cos32 + z * z * one_minus_cos,
        },
        position: RealPoint3d {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    }
}

/// Mirror of `matrix4x3_multiply @ 0x1802C5A70`. Computes `a * b`
/// (composition: applying `b` first then `a`). Each basis column is
/// transformed by `a`, and the position is `a.position + a.scale *
/// (a.matrix * b.position)`.
fn matrix4x3_multiply(a: &RealMatrix4x3, b: &RealMatrix4x3) -> RealMatrix4x3 {
    let transform = |bx: f32, by: f32, bz: f32| RealVector3d {
        i: a.forward.i * bx + a.left.i * by + a.up.i * bz,
        j: a.forward.j * bx + a.left.j * by + a.up.j * bz,
        k: a.forward.k * bx + a.left.k * by + a.up.k * bz,
    };

    let forward = transform(b.forward.i, b.forward.j, b.forward.k);
    let left = transform(b.left.i, b.left.j, b.left.k);
    let up = transform(b.up.i, b.up.j, b.up.k);
    let pos_local = transform(b.position.x, b.position.y, b.position.z);

    RealMatrix4x3 {
        scale: a.scale * b.scale,
        forward,
        left,
        up,
        position: RealPoint3d {
            x: a.position.x + a.scale * pos_local.i,
            y: a.position.y + a.scale * pos_local.j,
            z: a.position.z + a.scale * pos_local.k,
        },
    }
}
