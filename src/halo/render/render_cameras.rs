//! Mirror of `Ares/source/render/render_cameras.h` + the dllcache
//! bodies (`?render_camera_build@@... @ 0x1806b7c80`,
//! `?render_camera_build_projection@@... @ 0x1806b7ec0`, etc.).
//!
//! Halo separates the per-view camera state from the per-frame
//! viewport / projection math. `render_camera` carries position +
//! orientation + viewport rects + z planes; `render_view_parameters`
//! is derived viewport math (frustum bounds → projection scale/offset);
//! `render_projection` is the actual matrices the shaders read.
//!
//! `c_player_view` owns two of these triples — `m_rasterizer_camera`
//! / `m_rasterizer_projection` (used for actual GPU draws) and
//! `m_render_camera` / `m_render_projection` (visibility/lighting,
//! can be frozen independently for debugging).

use crate::halo::camera::ObserverResult;
use glam::{Mat4, Vec2, Vec3};

/// `render_camera` (Ares `render_cameras.h:17-37`, 136 bytes).
#[derive(Debug, Clone, Default)]
pub struct RenderCamera {
    pub position: Vec3,                                  // 0x0
    pub forward: Vec3,                                   // 0xC
    pub up: Vec3,                                        // 0x18
    pub mirrored: bool,                                  // 0x24
    pub vertical_field_of_view: f32,                     // 0x28
    pub field_of_view_scale: f32,                        // 0x2C
    pub window_pixel_bounds: Rectangle2d,                // 0x30
    pub window_title_safe_pixel_bounds: Rectangle2d,     // 0x38
    pub window_final_location: Point2d,                  // 0x40
    pub render_pixel_bounds: Rectangle2d,                // 0x44
    pub render_title_safe_pixel_bounds: Rectangle2d,     // 0x4C
    pub display_pixel_bounds: Rectangle2d,               // 0x54
    pub z_near: f32,                                     // 0x5C
    pub z_far: f32,                                      // 0x60
    pub mirror_plane: RealPlane3d,                       // 0x64
    pub enlarge_view: bool,                              // 0x74
    pub enlarge_center: Vec2,                            // 0x78
    pub enlarge_size_x: f32,                             // 0x80
    pub enlarge_size_y: f32,                             // 0x84
}

/// `render_view_parameters` (Ares `render_cameras.h:39-48`, 64 bytes).
/// Derived viewport math used by visibility / projection setup.
#[derive(Debug, Clone, Default)]
pub struct RenderViewParameters {
    pub frustum_bounds: RealRectangle2d,    // 0x0
    pub viewport_size: Vec2,                 // 0x10
    pub projection_scale: Vec2,              // 0x18
    pub projection_offset: Vec2,             // 0x20
    pub projection_coefficients: Vec2,       // 0x28
    pub projection_bounds: RealRectangle2d,  // 0x30
}

/// `render_projection` (Ares `render_cameras.h:50-58`, 192 bytes).
///
/// Layout note: Halo's `world_to_view` / `view_to_world` are
/// `real_matrix4x3` (52 bytes each — 4×3 column-major affine). We
/// store as `Mat4` for ergonomics; the bottom row is always
/// `(0,0,0,1)`. Halo's `projection_matrix` is `float[4][4]` row-major.
#[derive(Debug, Clone, Default)]
pub struct RenderProjection {
    pub world_to_view: Mat4,                 // 0x0   (real_matrix4x3 in Halo)
    pub view_to_world: Mat4,                 // 0x34
    pub projection_bounds: RealRectangle2d,  // 0x68
    pub projection_matrix: [[f32; 4]; 4],    // 0x78
    pub world_to_screen_size: Vec2,          // 0xB8
}

/// `render_mirror` (Ares `render_cameras.h:60-68`, 32 bytes).
/// Used by `render_camera_mirror` to reflect a camera across a portal
/// for reflection-view rendering.
#[derive(Debug, Clone, Default)]
pub struct RenderMirror {
    pub plane: RealPlane3d,             // 0x0
    pub index_of_refraction: f32,       // 0x10
    pub depth: f32,                     // 0x14
    pub cluster_index: i32,             // 0x18
    pub leaf_index: i32,                // 0x1C
}

/// Integer screen-space rectangle. Mirrors Halo's `rectangle2d`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rectangle2d {
    pub top: i16,
    pub left: i16,
    pub bottom: i16,
    pub right: i16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Point2d {
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RealRectangle2d {
    pub min: Vec2,
    pub max: Vec2,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RealPlane3d {
    pub normal: Vec3,
    pub d: f32,
}

// ---------------------------------------------------------------------------
// Build helpers (free functions, mirroring Halo's namespace-style layout).
// ---------------------------------------------------------------------------

/// `render_camera_build @ 0x1806b7c80`. Populates `render_camera` from
/// an `s_observer_result`. The 0.785 multiplier is Halo's hard-coded
/// "vertical FoV from observer's value" conversion factor (the
/// observer's FoV is stored as a wider value that gets squeezed to
/// the actual render FoV here — a shipped Bungie magic number).
///
/// `z_near` / `z_far` come from `rasterizer_get_z_planes`; the caller
/// supplies them so this fn stays pure.
///
/// Asserts (parity with dllcache):
///   - `vertical_field_of_view < pi`
///   - `vertical_field_of_view > 1e-4`
///
/// Halo with `observer == nullptr` falls back to a default-FoV +
/// origin camera; our API takes `&ObserverResult` directly so the
/// fallback path doesn't apply (callers pass a default observer when
/// no live one exists).
pub fn render_camera_build(
    camera: &mut RenderCamera,
    observer: &ObserverResult,
    z_near: f32,
    z_far: f32,
) {
    use std::f32::consts::PI;

    camera.position = observer.position;
    camera.forward = observer.forward;
    camera.up = observer.up;
    camera.vertical_field_of_view = observer.vertical_field_of_view;
    camera.field_of_view_scale = observer.field_of_view_scale;

    // Halo squeezes both fields by 0.785 (`FLOAT_0_78500003` in the
    // dllcache decomp). Magic Bungie constant — it's the conversion
    // from the observer's "natural" FoV to the actual render FoV.
    camera.vertical_field_of_view *= 0.78500003;
    camera.field_of_view_scale *= 0.78500003;

    debug_assert!(
        camera.vertical_field_of_view < PI,
        "vertical_field_of_view ({}) must be < pi (Halo info_15070/15076)",
        camera.vertical_field_of_view
    );
    debug_assert!(
        camera.vertical_field_of_view > 1.0e-4,
        "vertical_field_of_view ({}) must be > 1e-4 (Halo info_15071/15077)",
        camera.vertical_field_of_view
    );

    camera.mirrored = false;
    camera.z_near = z_near;
    camera.z_far = z_far;
    camera.enlarge_view = false;
}
