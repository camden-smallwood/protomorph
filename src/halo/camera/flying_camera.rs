//! `Ares/source/camera/flying_camera.{h,cpp}` — debug/dev fly-through
//! camera. Mirrors the engine class but as a **host-side driver**: we
//! consume WASD/mouse input directly instead of going through the
//! `s_observer_command` stream.
//!
//! Engine `c_flying_camera::update @ <virtual>` writes into an
//! `s_observer_command*`; protomorph builds an [`ObserverResult`]
//! directly via [`observer_build_result_from_point_and_vectors`]. The
//! input pipeline (WASD/mouse → orientation/velocity → result) is
//! protomorph host code with no engine analog.
//!
//! See `memory/reference_observer.md` Section 4 for the design.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};

use crate::halo::camera::observer::{
    K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW,
    ObserverResult,
    observer_build_result_from_point_and_vectors,
};

/// `c_rasterizer_globals` view+projection cbuffer payload. Engine packs
/// these into the rasterizer-globals cbuffer (slot 0x29 VS / 0x3F PS per
/// `reference_engine_cbuffers.md`). protomorph still uses this directly
/// as the bytemuck'd uniform — verify exact slot layout during Phase 1.3
/// rasterizer-globals consolidation.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
}

/// protomorph host: debug fly-through camera. Field layout loosely
/// mirrors `c_flying_camera` (Ares `camera/flying_camera.h:13-50`, 64B)
/// with host-side extras (velocity, aspect, near-clip, cached
/// view/projection) that the engine resolves elsewhere (input system,
/// `c_player_view::setup_camera`).
///
/// Engine fields preserved:
///   - `m_position` → [`position`]
///   - `m_orientation` (real_euler_angles2d yaw+pitch) → [`rotation`]
///   - `m_roll` → implicit zero (protomorph doesn't roll the flycam)
///
/// Host extras (not in engine `c_flying_camera`):
///   - `velocity` — per-frame input vector applied during `update`
///   - `field_of_view` / `aspect_ratio` / `near_clip` — engine resolves
///     these via the observer chain; we cache them here so
///     [`to_observer_result`] can stamp the result directly
///   - `forward` / `right` / `up` / `view` / `projection` — derived
///     each frame; the engine produces view+projection inside
///     `c_player_view::setup_camera` from the `ObserverResult`. Phase 3
///     will move that derivation out of here.
pub struct FlyingCamera {
    pub position: Vec3,
    /// `[0] = yaw` (rotation around Z), `[1] = pitch` (around right axis),
    /// degrees. Mirrors engine's `real_euler_angles2d m_orientation`.
    pub rotation: Vec2,
    pub velocity: Vec3,
    pub field_of_view: f32,
    pub aspect_ratio: f32,
    pub near_clip: f32,

    // Derived per-frame:
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub view: Mat4,
    pub projection: Mat4,
}

impl FlyingCamera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.7),
            rotation: Vec2::ZERO,
            velocity: Vec3::ZERO,
            field_of_view: K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW,
            aspect_ratio: 1.0,
            near_clip: 0.01,
            forward: Vec3::Y,
            right: Vec3::X,
            up: Vec3::Z,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect_ratio = width as f32 / height as f32;
        }
    }

    pub fn update(&mut self) {
        // Clamp pitch to ±89° (avoid look-at gimbal flip).
        self.rotation.y = self.rotation.y.clamp(-89.0, 89.0);

        let yaw_rad = self.rotation.x.to_radians();
        let pitch_rad = self.rotation.y.to_radians();
        let pitch_cos = pitch_rad.cos();

        // Z-up spherical → forward.
        self.forward = Vec3::new(
            yaw_rad.cos() * pitch_cos,
            yaw_rad.sin() * pitch_cos,
            pitch_rad.sin(),
        )
        .normalize();

        // right = normalize(up × forward) keeps a right-handed basis with
        // world up = (0,0,1).
        self.right = self.up.cross(self.forward).normalize();

        self.position += self.velocity;

        let target = self.position + self.forward;
        self.view = Mat4::look_at_rh(self.position, target, self.up);

        // Reverse-Z infinite-far perspective (near=1.0, far=0.0 after
        // depth-buffer remap). Engine convention stores `field_of_view`
        // as HORIZONTAL FoV (`s_observer_result::horizontal_field_of_view`
        // at offset 0x40), but glam's `perspective_infinite_reverse_rh`
        // takes VERTICAL FoV. Convert via the same formula as
        // `observer_result_compute_parameters @ 0x18035B960`:
        //   half_h_tan = tan(h_fov / 2)
        //   half_v_tan = half_h_tan / aspect
        //   v_fov      = 2 × atan2(half_v_tan, 1)
        let half_h_tan = (0.5 * self.field_of_view).tan();
        let half_v_tan = half_h_tan / self.aspect_ratio;
        let v_fov = 2.0 * half_v_tan.atan2(1.0);
        self.projection =
            Mat4::perspective_infinite_reverse_rh(v_fov, self.aspect_ratio, self.near_clip);
    }

    /// Build an `ObserverResult` from the current flycam state via
    /// `observer_build_result_from_point_and_vectors`. Engine path:
    /// `c_flying_camera::update` → `s_observer_command` →
    /// `observer_update` → `s_observer_result`. We compress that to a
    /// direct build because protomorph doesn't run the command stream.
    pub fn to_observer_result(&self) -> ObserverResult {
        let mut out = ObserverResult::default();
        observer_build_result_from_point_and_vectors(
            &mut out,
            self.position,
            self.forward,
            self.up,
        );
        // Patch the host-side overrides over the build defaults.
        out.horizontal_field_of_view = self.field_of_view;
        out.aspect_ratio = self.aspect_ratio;
        // Re-derive vertical_field_of_view + field_of_view_scale +
        // magic_crosshair_offset for the patched h-FoV/aspect.
        crate::halo::camera::observer::observer_result_compute_parameters(&mut out);
        // Velocity isn't reconstructed by observer_build_*; engine's
        // observer command stream provides it. Patch from host state.
        out.velocity = self.velocity;
        out
    }

    pub fn rotate_towards_point(&mut self, point: Vec3, amount: f32) {
        let distance = point - self.position;
        let length = distance.length().max(1e-6);
        self.rotation.x = (distance.y.atan2(distance.x)).to_degrees() * amount;
        self.rotation.y = (distance.z / length).asin().to_degrees() * amount;
    }
}

impl Default for FlyingCamera {
    fn default() -> Self {
        Self::new()
    }
}
