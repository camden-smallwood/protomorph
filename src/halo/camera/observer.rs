//! `Ares/source/camera/observer.{h,cpp}` — observer result types + the
//! protomorph-relevant subset of the runtime helpers.
//!
//! The full engine update pipeline (input → director command stream →
//! collision resolve → final `s_observer_result`) is gameplay logic that
//! protomorph doesn't run. Renderer needs:
//!   - `ObserverResult` struct (per-frame camera state read by
//!     `c_player_view::setup_camera`)
//!   - `ObserverDepthOfField` struct (DoF override sub-field)
//!   - `observer_build_result_from_point_and_vectors` — the host-friendly
//!     fast path that builds an `ObserverResult` directly from
//!     position+forward+up (bypassing the command-stream director chain)
//!   - `observer_result_compute_parameters` — derives vertical_fov + FoV
//!     scale + view-offset rotation + magic crosshair offset
//!
//! Full reference: `memory/reference_observer.md`.

use crate::halo::scenario::location::Location;
use glam::{Mat3, Vec3};

/// Engine global default horizontal field-of-view in radians (`unk_18112B018
/// @ 0x18112B018` = 1.2217305 rad = 70°). Used as the FoV-scale denominator
/// (`result.field_of_view_scale = h_fov / K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW`)
/// and as the seed value for fresh observer results.
pub const K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW: f32 = 1.2217305;

/// Minimum allowed h-FoV (`unk_18112B01C @ 0x18112B01C` = 1°).
const K_FOV_MIN_RAD: f32 = 0.017453292;

/// Maximum allowed h-FoV (`unk_18112B020 @ 0x18112B020` = 150°).
const K_FOV_MAX_RAD: f32 = 2.617994;

/// `s_observer_depth_of_field` (Ares `camera/observer.h:76-89`, 20 bytes).
///
/// Per-camera DoF override that clobbers the cfxs/scenario default when
/// `_active_bit` of `flags` is set. Read in
/// `c_player_view::render_setup` and forwarded to the postprocess
/// `c_camera_fx_values::depth_of_field` slot.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObserverDepthOfField {
    /// Bit 0 = `_active_bit` (Halo enum at observer.h:79-82).
    pub flags: u32,
    pub near_focal_plane_distance: f32,
    pub far_focal_plane_distance: f32,
    pub focal_depth: f32,
    pub blur_amount: f32,
}

impl ObserverDepthOfField {
    pub const ACTIVE_BIT: u32 = 1 << 0;

    pub fn is_active(&self) -> bool {
        self.flags & Self::ACTIVE_BIT != 0
    }
}

/// `s_observer_result` (Ares `camera/observer.h:148-163`, 112 bytes).
///
/// The per-frame camera output that `c_player_view::setup_camera @
/// 0x180689820` consumes. Field-for-field with the engine layout so
/// future ports of `observer_get_camera` (gameplay) can write directly
/// into it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObserverResult {
    pub position: Vec3,                       // 0x00
    pub location: Location,                   // 0x0C
    pub velocity: Vec3,                       // 0x10
    pub rotation: Vec3,                       // 0x1C
    pub forward: Vec3,                        // 0x28
    pub up: Vec3,                             // 0x34
    pub horizontal_field_of_view: f32,        // 0x40
    pub depth_of_field: ObserverDepthOfField, // 0x44
    pub aspect_ratio: f32,                    // 0x58
    /// `real_vector2d view_offset` — small XY screen-space offset
    /// applied to the projection center (used by splitscreen + scope
    /// effects).
    pub view_offset: [f32; 2],                // 0x5C
    /// Crosshair X offset for cinematic widescreen reframing.
    pub magic_crosshair_offset: f32,          // 0x64
    pub vertical_field_of_view: f32,          // 0x68
    /// Multiplier applied to FoV for zoom (rifle scopes etc.).
    /// `_dont_adjust_for_fov_scale` vs `_adjust_for_fov_scale` switch
    /// at `e_adjust_for_fov_scale` (observer.h:63-67) chooses whether
    /// downstream camera-build operations apply this.
    pub field_of_view_scale: f32,             // 0x6C
}

/// `e_adjust_for_fov_scale` (Ares `camera/observer.h:63-67`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdjustForFovScale {
    DontAdjust = 0,
    Adjust = 1,
}

/// `observer_result_compute_parameters @ 0x18035B960` — verbatim port.
///
/// Derives `vertical_field_of_view`, `field_of_view_scale`, applies the
/// `view_offset.y` vertical-tilt rotation to `forward`+`up`, and writes
/// `magic_crosshair_offset` (the post-tilt parallax magnitude).
///
/// Engine's first rotation block (around world-Z) is dead code — the
/// rotation angle is `atanf(0.0) = 0` in the decompile, so the
/// matrix is identity. Only the second rotation (around `forward × up`
/// by `atan(tan(vfov/2) × view_offset.y)`) does real work.
pub fn observer_result_compute_parameters(result: &mut ObserverResult) {
    let h_fov = result.horizontal_field_of_view;
    assert!(h_fov.is_finite() && h_fov >= K_FOV_MIN_RAD && h_fov <= K_FOV_MAX_RAD,
        "observer h_fov out of range: {h_fov}");

    // vertical_fov from horizontal_fov + aspect_ratio (engine `atan2` form).
    let half_h_tan = (0.5 * h_fov).tan();
    let half_v_tan = half_h_tan / result.aspect_ratio;
    let half_v = half_v_tan.atan2(1.0);
    result.vertical_field_of_view = half_v * 2.0;
    assert!(result.vertical_field_of_view < std::f32::consts::PI - 1e-4);
    assert!(result.vertical_field_of_view > 1e-4);

    // FoV scale relative to engine default (70°).
    result.field_of_view_scale = h_fov / K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW;

    // Save pre-offset forward for magic_crosshair_offset computation.
    let pre_offset_forward = result.forward;

    // View-offset vertical tilt: rotate forward+up around (forward × up) by
    // `atan(tan(vfov/2) × view_offset.y)`.
    let angle_y = ((0.5 * result.vertical_field_of_view).tan() * result.view_offset[1]).atan();
    let axis = result.forward.cross(result.up);
    let rotation = Mat3::from_axis_angle(axis.normalize_or_zero(), angle_y);
    result.forward = rotation * result.forward;
    result.up = rotation * result.up;

    // magic_crosshair_offset = length(pre_offset_forward / dot(new_forward, pre_offset_forward) - new_forward)
    // — the parallax delta in projected-screen units after view-offset tilt.
    let forward_dot = result.forward.dot(pre_offset_forward).max(1e-4);
    let inv_dot = 1.0 / forward_dot;
    let delta = pre_offset_forward * inv_dot - result.forward;
    result.magic_crosshair_offset = delta.length();
}

/// `observer_build_result_from_point_and_vectors @ 0x1803577F0` — verbatim port.
///
/// Builds an `ObserverResult` from raw position+forward+up. Asserts all
/// vectors finite (engine asserts non-null pointers; Rust ownership covers
/// that). Defaults aspect_ratio to 4:3 and h-FoV to engine default (70°);
/// callers should patch these post-build for their actual surface.
///
/// `location` is zero-initialized here — engine calls
/// `scenario_location_from_point` to fill it, but that requires the active
/// scenario which we don't pass in. Callers with scenario access should
/// fill `location` themselves before the result reaches downstream consumers
/// that read it (atmosphere cluster lookup, lightmap probe lookup).
pub fn observer_build_result_from_point_and_vectors(
    out: &mut ObserverResult,
    position: Vec3,
    forward: Vec3,
    up: Vec3,
) {
    *out = ObserverResult::default();
    out.position = position;
    out.forward = forward;
    out.up = up;
    out.aspect_ratio = 1.3333334; // engine 4:3 default
    out.horizontal_field_of_view = K_DEFAULT_HORIZONTAL_FIELD_OF_VIEW;
    // out.location stays zero — caller fills via scenario_location_from_point if needed.
    observer_result_compute_parameters(out);
}
