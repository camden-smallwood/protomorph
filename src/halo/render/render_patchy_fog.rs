//! Mirror of `Ares/source/render/render_patchy_fog.h` (62 lines).
//!
//! Volumetric "patchy fog" — Halo 3's atmospheric layered-fog system.
//! `c_patchy_fog` tracks per-camera state (last eye position +
//! orientation, accumulated roll, fog-sheet offsets) so the runtime
//! can scroll a stack of textured sheets through space and produce
//! the appearance of drifting mist as the camera moves.
//!
//! Owned per-`c_player_view`. Halo's `c_player_view::render` calls
//! `c_patchy_fog::frame_advance_all(dt)` each frame, then
//! `queue_patchy_fog()` (after render_water, before render_transparents)
//! to push fog draws into the transparency_renderer pool.
//!
//! Tag side: a `patchy_fog_texture` referenced by the scenario. The
//! actual fog sheets are textured quads anchored at distance steps
//! from the camera.

use glam::Vec3;

/// Halo: `k_fog_sheet_offset_steps`. The number of stacked offsets
/// the fog sheet attribute index can take. Used as the `[100]`
/// dimension of `m_lateral_offsets` / `m_vertical_offsets`.
pub const K_FOG_SHEET_OFFSET_STEPS: usize = 100;

/// Engine `c_patchy_fog::render_patchy_fog @ 0x1806B2160` lines
/// 268-275 / MCC `sub_18029CF98`:
///   `v48 = (m_dist < m_closest_fog_z) ? ceil((m_closest_fog_z - m_dist) / sep) : 0`
/// Used to shift the 8 drawn sheets deeper when the first sheet would
/// otherwise sit closer than `m_closest_fog_z`. Both `update_per_frame`
/// (offset accumulators) and `build_patchy_fog_params` (per-sheet UV
/// transforms) must use the SAME skip so ring slots stay in sync.
pub fn sheet_skip_count(closest_fog_z: f32, distance_to_first_sheet: f32, sep: f32) -> usize {
    if distance_to_first_sheet < closest_fog_z {
        ((closest_fog_z - distance_to_first_sheet) / sep).ceil() as usize
    } else {
        0
    }
}

/// `c_patchy_fog` (render_patchy_fog.h:24-49, 852B).
///
/// Per-`c_player_view` fog state. The tracking fields
/// (`m_last_eye_position`, `m_last_forward`, `m_last_up`, `m_roll`)
/// drive the offset accumulators each frame so the fog appears
/// consistent as the camera moves. `m_lateral_offsets` /
/// `m_vertical_offsets` are 100-element scrollers (one slot per
/// sheet attribute index).
#[derive(Debug, Clone, Copy)]
pub struct PatchyFog {
    pub last_eye_position: Vec3,                       // 0x0
    pub last_forward: Vec3,                             // 0xC
    pub last_up: Vec3,                                  // 0x18
    pub roll: f32,                                      // 0x24
    pub closest_fog_z: f32,                             // 0x28
    pub distance_to_first_sheet: f32,                   // 0x2C
    pub starting_sheet_attribute_index: i32,            // 0x30
    pub lateral_offsets: [f32; K_FOG_SHEET_OFFSET_STEPS],  // 0x34
    pub vertical_offsets: [f32; K_FOG_SHEET_OFFSET_STEPS], // 0x1C4
    /// Set to `true` on construction + by `Renderer::load_scenario`
    /// each time the scenario is reloaded. `update_per_frame` clears
    /// it after seeding `last_eye_position` so first-frame teleport
    /// doesn't register as huge camera motion. Not in the engine
    /// struct (engine has separate flow for first-frame init).
    pub is_first_frame: bool,
}

impl Default for PatchyFog {
    fn default() -> Self {
        Self {
            last_eye_position: Vec3::ZERO,
            last_forward: Vec3::X,
            last_up: Vec3::Y,
            roll: 0.0,
            closest_fog_z: 0.0,
            distance_to_first_sheet: 0.0,
            starting_sheet_attribute_index: 0,
            lateral_offsets: [0.0; K_FOG_SHEET_OFFSET_STEPS],
            vertical_offsets: [0.0; K_FOG_SHEET_OFFSET_STEPS],
            is_first_frame: true,
        }
    }
}

impl PatchyFog {
    /// Update per-frame state from the current camera + active
    /// atmosphere setting. Engine equivalent of the per-frame
    /// `get_relative_movement` + offset integration + ring-buffer
    /// ratcheting block of `c_patchy_fog::render_patchy_fog
    /// @ 0x1806B2160` lines 218-260.
    ///
    /// Engine flow:
    ///   total = (eye - last_eye) + (-wind × dt)
    ///   m_distance_to_first_sheet -= total · forward
    ///   while dist < z_near       : start += 1, dist += sep   (camera moved forward through a sheet)
    ///   while dist > z_near + sep : start -= 1, dist -= sep   (camera moved back through a sheet)
    ///   for i in [0, 8):
    ///     ring_slot = (i + start) % 100
    ///     m_lateral_offsets [ring_slot] += inv_sep × (sin_roll × proj_up + cos_roll × proj_right) + ...
    ///     m_vertical_offsets[ring_slot] += inv_sep × (cos_roll × proj_up - sin_roll × proj_right) + ...
    ///
    /// `is_first_frame` is a one-shot guard caller-side — set true
    /// before first call, will be cleared on first call here. Prevents
    /// teleport-on-load from registering as huge camera motion. We
    /// also seed `distance_to_first_sheet` mid-range so the pop-fade
    /// phase starts at 0.5 rather than producing a visible flash.
    pub fn update_per_frame(
        &mut self,
        eye: Vec3,
        forward: Vec3,
        up: Vec3,
        sheet_separation: f32,
        z_near: f32,
        wind_direction: Vec3,
        seconds_dt: f32,
    ) {
        let sep = sheet_separation.max(0.5);
        if self.is_first_frame {
            self.last_eye_position = eye;
            self.last_forward = forward;
            self.last_up = up;
            self.distance_to_first_sheet = z_near + sep * 0.5;
            self.is_first_frame = false;
            return;
        }
        let total = (eye - self.last_eye_position) + wind_direction * (-seconds_dt);

        // Engine `c_patchy_fog::render_patchy_fog @ 0x1806B2160` computes
        // `v232 = (forward × up) · total` for the lateral projection.
        // `forward × up = +right` in Halo's right-handed X=fwd / Y=left /
        // Z=up basis (and equivalently in protomorph's right-handed
        // X=right / Y=fwd / Z=up basis). Earlier we used `up × forward`
        // which gives `-right` (= left direction) — producing the
        // opposite-direction strafe/wind drift symptom.
        let right = forward.cross(up).normalize_or_zero();
        let along_forward = total.dot(forward);
        let along_right = total.dot(right);
        let along_up = total.dot(up);

        // Engine: `m_distance_to_first_sheet -= dot(total, forward)`.
        // Camera moving forward → dist decreases → first sheet
        // approaches z_near.
        self.distance_to_first_sheet -= along_forward;

        // Ratchet so dist always lies in [z_near, z_near + sep].
        // Engine uses one-shot ceil/floor; teleports + huge dt can
        // produce multi-sheet jumps so we loop.
        let n_steps = K_FOG_SHEET_OFFSET_STEPS as i32;
        while self.distance_to_first_sheet < z_near {
            self.distance_to_first_sheet += sep;
            self.starting_sheet_attribute_index =
                (self.starting_sheet_attribute_index + 1).rem_euclid(n_steps);
        }
        while self.distance_to_first_sheet > z_near + sep {
            self.distance_to_first_sheet -= sep;
            self.starting_sheet_attribute_index =
                (self.starting_sheet_attribute_index - 1).rem_euclid(n_steps);
        }

        // Engine `c_patchy_fog::get_relative_movement @ 0x1806B31B0`:
        // build the 3x3 rotation matrix mapping (last_forward, last_left,
        // last_up) → (forward, left, up) where left = up × forward
        // (Halo X=forward, Y=left, Z=up convention), then convert via
        // `matrix3x3_rotation_to_quaternion @ 0x1802c6610` +
        // `quaternion_to_euler_angles @ 0x18018b1c0`.
        //
        // Returns `(n0, n1, n2)` where for small per-frame rotations
        //   n0 ≈ yaw delta   (around UP)
        //   n1 ≈ roll delta  (around FORWARD — head twist)
        //   n2 ≈ pitch delta (around LEFT)
        //
        // Engine then:
        //   m_roll      -= n1   (accumulates ACTUAL roll, not yaw)
        //   parallax.x  = -n0   (used in sheet loop, scaled by depth)
        //   parallax.y  = -n2
        let delta_euler = relative_euler_delta(
            self.last_forward, self.last_up, forward, up,
        );
        self.roll -= delta_euler.y; // n[1] = roll, NOT yaw
        let yaw_neg   = -delta_euler.x; // -n[0]
        let pitch_neg = -delta_euler.z; // -n[2]

        // Per-sheet offset accumulation written into ring slot
        // `(i + start) % 100`. Engine accumulates TWO contributions:
        //   1) Movement, scaled by `inv_sep` (CONSTANT across sheets).
        //   2) View rotation parallax, scaled by `depth_norm`
        //      (deeper sheets sweep MORE under view rotation —
        //      foreground parallax cue).
        // Both contributions are passed through the 2D rotation by
        // m_roll: `[cos -sin; sin cos] × (proj_right, proj_up)`. Engine
        // packs the matching basis into cbuffer 0x3F0001 as
        // (cos, sin, -sin, cos); the shader's `texcoord_basis` reads
        // .xy and .zw as the two rotated axes.
        let inv_sep = 1.0 / sep;
        let cos_r = self.roll.cos();
        let sin_r = self.roll.sin();
        let move_lateral  = inv_sep * (cos_r * along_right - sin_r * along_up);
        let move_vertical = inv_sep * (sin_r * along_right + cos_r * along_up);
        let par_lateral_unit  = cos_r * yaw_neg   - sin_r * pitch_neg;
        let par_vertical_unit = sin_r * yaw_neg   + cos_r * pitch_neg;

        // Engine `c_patchy_fog::render_patchy_fog @ 0x1806B2160` lines
        // 268-275 (pre-release) / MCC `sub_18029CF98` lines 270-278:
        // if the current `m_distance_to_first_sheet` puts the first
        // sheet *closer* than `m_closest_fog_z`, the engine skips the
        // initial sheets and shifts the 8 drawn sheets deeper. Counts
        // as the `i` offset added to both the depth_norm numerator
        // and the ring-slot index. Inert when `m_closest_fog_z == 0`
        // (the constructor default — no engine setter found yet).
        let skip = sheet_skip_count(
            self.closest_fog_z, self.distance_to_first_sheet, sep,
        );
        let start = self.starting_sheet_attribute_index as usize;
        for n in 0..8 {
            let i = n + skip;
            let sheet_depth = i as f32 * sep + self.distance_to_first_sheet;
            let depth_norm = sheet_depth * inv_sep;
            let slot = (i + start) % K_FOG_SHEET_OFFSET_STEPS;
            self.lateral_offsets[slot]  += move_lateral  + depth_norm * par_lateral_unit;
            self.vertical_offsets[slot] += move_vertical + depth_norm * par_vertical_unit;
        }

        self.last_eye_position = eye;
        self.last_forward = forward;
        self.last_up = up;
    }
}

/// Build the rotation matrix mapping `(last_forward, last_left, last_up)`
/// → `(forward, left, up)` where `left = up × forward` (Halo's X=fwd,
/// Y=left, Z=up convention), convert to quaternion, then decompose to
/// Euler angles.
///
/// Mirrors `c_patchy_fog::get_relative_movement @ 0x1806B31B0`'s inlined
/// matrix construction + `matrix3x3_rotation_to_quaternion @ 0x1802c6610`
/// + `quaternion_to_euler_angles @ 0x18018b1c0`.
///
/// Returns `(yaw, roll, pitch)` per engine's index convention. For
/// small per-frame rotations:
///   .x ≈ rotation around UP (yaw delta)
///   .y ≈ rotation around FORWARD (roll delta)
///   .z ≈ rotation around LEFT (pitch delta)
fn relative_euler_delta(
    last_forward: Vec3, last_up: Vec3,
    forward: Vec3, up: Vec3,
) -> Vec3 {
    let last_left = last_up.cross(last_forward);
    // Matrix rows = new basis projected onto last basis. Engine builds
    // row 0 (forward in last) + row 2 (up in last) explicitly, then
    // derives row 1 = row 2 × row 0 (left in last).
    let m00 = last_forward.dot(forward);
    let m01 = last_left.dot(forward);
    let m02 = last_up.dot(forward);
    let m20 = last_forward.dot(up);
    let m21 = last_left.dot(up);
    let m22 = last_up.dot(up);
    let m10 = m21 * m02 - m22 * m01;
    let m11 = m22 * m00 - m20 * m02;
    let m12 = m20 * m01 - m21 * m00;
    matrix_to_quaternion_to_euler([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
}

/// `matrix3x3_rotation_to_quaternion @ 0x1802c6610` then
/// `quaternion_to_euler_angles @ 0x18018b1c0`, fused (we don't need
/// the intermediate quaternion outside this function).
///
/// Largest-diagonal algorithm with `qw ≥ 0` canonical form. Halo's
/// off-diagonal extraction has the OPPOSITE sign from textbook
/// (qw·qx uses `m12 − m21` rather than `m21 − m12`); the matrix
/// construction's index convention compensates so net Euler angles
/// match physical rotations.
fn matrix_to_quaternion_to_euler(m: [[f32; 3]; 3]) -> Vec3 {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (qx, qy, qz, qw) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt();
        let qw = s * 0.5;
        let inv = 0.25 / qw;
        let qx = (m[1][2] - m[2][1]) * inv;
        let qy = (m[2][0] - m[0][2]) * inv;
        let qz = (m[0][1] - m[1][0]) * inv;
        (qx, qy, qz, qw)
    } else {
        // Engine: find largest diagonal m[v14][v14]. v15=(v14+1)%3,
        // v16=(v14+2)%3. Then q[v14] = sqrt(m[v14][v14] - m[v15][v15]
        // - m[v16][v16] + 1) / 2.
        let mut v14 = if m[1][1] > m[0][0] { 1 } else { 0 };
        if m[2][2] > m[v14][v14] { v14 = 2; }
        let v15 = (v14 + 1) % 3;
        let v16 = (v14 + 2) % 3;
        let s = (m[v14][v14] - m[v15][v15] - m[v16][v16] + 1.0).max(0.0).sqrt();
        let qi = s * 0.5;
        let inv = if qi > 1.0e-4 { 0.25 / qi } else { 0.0 };
        let qw = (m[v15][v16] - m[v16][v15]) * inv;
        let q_v15 = (m[v14][v15] + m[v15][v14]) * inv;
        let q_v16 = (m[v14][v16] + m[v16][v14]) * inv;
        let mut q = [0.0_f32; 3];
        q[v14] = qi;
        q[v15] = q_v15;
        q[v16] = q_v16;
        (q[0], q[1], q[2], qw)
    };
    // Canonical: qw ≥ 0.
    let (qx, qy, qz, qw) = if qw < 0.0 {
        (-qx, -qy, -qz, -qw)
    } else {
        (qx, qy, qz, qw)
    };
    // Engine Euler decomposition (verbatim from 0x18018b1c0):
    //   n[0] = atan2(2(qx·qy + qz·qw), qx² − qy² − qz² + qw²)
    //   n[1] = atan2(2(qz·qy + qw·qx), qz² − qx² − qy² + qw²)
    //   n[2] = asin(clamp(-2(qz·qx − qw·qy) / |q|², -1, 1))
    let qx2 = qx * qx;
    let qy2 = qy * qy;
    let qz2 = qz * qz;
    let qw2 = qw * qw;
    let n0 = (2.0 * (qx * qy + qz * qw)).atan2(qx2 - qy2 - qz2 + qw2);
    let n1 = (2.0 * (qz * qy + qw * qx)).atan2(qz2 - qx2 - qy2 + qw2);
    let mag = qx2 + qy2 + qz2 + qw2;
    let n2 = if mag > 0.0 {
        let s = (-2.0 * (qz * qx - qw * qy) / mag).clamp(-1.0, 1.0);
        s.asin()
    } else {
        0.0
    };
    Vec3::new(n0, n1, n2)
}
