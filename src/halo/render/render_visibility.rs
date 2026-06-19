//! Mirror of `Ares/source/render/render_visibility.cpp` +
//! `render_visibility_collection.cpp`. Phase G — the 4 producer entry
//! points that take a render-side camera/projection and build a
//! [`CVisibilityCollection`] for downstream consumers (Phase H/I).
//!
//! - [`render_visibility_build_projection`] (engine `0x1806950F0`) —
//!   build a [`VisibilityProjection`] from a camera + render
//!   projection + camera cluster reference.
//! - [`render_visibility_camera_collection_compute`] (engine
//!   `0x1806AB740`) — wraps the above + drives the camera collection.
//!
//! These functions return `()` (engine void) — they push their
//! results into the global [`CVisibleItems`] storage via
//! `compute()`, which collectors fill on shape-dispatch.

use blam_tags::math::{RealMatrix4x3, RealPoint2d, RealPoint3d, RealVector3d};

use crate::halo::math::matrix_math::{cross, matrix4x3_inverse_orthonormal, normalize3d};
use glam::Vec3;

use crate::halo::camera::ObserverResult;
use crate::halo::structures::clusters::ClusterReference;
use crate::halo::scenario::LoadedScenario;
use crate::halo::visibility::visibility_input::{
    visibility_collection_flags, ECollectionType,
};
use crate::halo::visibility::visibility_portal_activation::ActivePortalBitvectors;
use crate::halo::visibility::{
    visibility_volume_build, CVisibilityCollection, CVisibleItems,
    RealRectangle2d, VisibilityProjection,
};

// =============================================================================
// `render_projection` — minimal subset
//
// Engine `render_projection` carries projection_bounds (2D tangent
// rectangle) + a few other fields (aspect, fov, etc). Phase G needs
// only `projection_bounds`; promote the full struct when more
// consumers want it.
// =============================================================================

/// Subset of Ares `render_projection` consumed by visibility.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderProjection {
    /// 2D tangent-space frustum bounds (x = horizontal half-tangent,
    /// y = vertical half-tangent). Computed from the observer's FoV
    /// + aspect ratio.
    pub projection_bounds: RealRectangle2d,
}

impl RenderProjection {
    /// Build from an [`ObserverResult`]'s FoV + aspect. Engine path:
    /// horizontal/vertical half-tangents from the FoVs.
    pub fn from_observer(observer: &ObserverResult) -> Self {
        let half_h_tan = (observer.horizontal_field_of_view * 0.5).tan();
        let half_v_tan = (observer.vertical_field_of_view * 0.5).tan();
        Self {
            projection_bounds: RealRectangle2d {
                x0: -half_h_tan,
                x1: half_h_tan,
                y0: -half_v_tan,
                y1: half_v_tan,
            },
        }
    }
}

// =============================================================================
// `render_visibility_build_projection @ 0x1806950F0` (G1)
// =============================================================================

#[inline]
fn vec3_to_point(v: Vec3) -> RealPoint3d {
    RealPoint3d { x: v.x, y: v.y, z: v.z }
}

#[inline]
fn vec3_to_vector(v: Vec3) -> RealVector3d {
    RealVector3d { i: v.x, j: v.y, k: v.z }
}

/// `render_visibility_build_projection @ 0x1806950F0`. Build a
/// [`VisibilityProjection`] from camera + projection.
///
/// Halo basis convention: camera looks down `-up` axis in basis
/// space. Mapping:
/// - `basis.left = camera.up` (normalized)
/// - `basis.up = -camera.forward` (normalized)
/// - `basis.forward = left × up` (normalized)
/// - `basis.position = camera.position`
/// - `world_to_basis = inverse(basis_to_world)`
///
/// Near plane: `n = camera.forward`, `d = dot(camera.position,
/// camera.forward) + z_near`. Convex hull = 4 corners of
/// `projection.projection_bounds`. Volume built via
/// [`visibility_volume_build`].
pub fn render_visibility_build_projection(
    observer: &ObserverResult,
    z_near: f32,
    z_far: f32,
    projection: &RenderProjection,
    _camera_cluster_reference: ClusterReference,
    out: &mut VisibilityProjection,
) {
    *out = VisibilityProjection::default();
    out.parallel_projection_flag = false;

    // Build basis_to_world (Halo convention).
    let mut left = vec3_to_vector(observer.up);
    normalize3d(&mut left);
    let mut up = RealVector3d {
        i: -observer.forward.x,
        j: -observer.forward.y,
        k: -observer.forward.z,
    };
    normalize3d(&mut up);
    let mut forward = cross(left, up);
    normalize3d(&mut forward);

    out.basis_to_world = RealMatrix4x3 {
        scale: 1.0,
        forward,
        left,
        up,
        position: vec3_to_point(observer.position),
    };
    out.world_to_basis = matrix4x3_inverse_orthonormal(&out.basis_to_world);

    out.near_bounded_flag = true;
    out.near_distance = z_near;
    let cam_fwd = vec3_to_vector(observer.forward);
    let cam_pos = vec3_to_point(observer.position);
    out.near_plane = blam_tags::math::RealPlane3d {
        i: cam_fwd.i,
        j: cam_fwd.j,
        k: cam_fwd.k,
        d: cam_pos.x * cam_fwd.i + cam_pos.y * cam_fwd.j + cam_pos.z * cam_fwd.k + z_near,
    };

    out.far_bounded_flag = true;
    out.far_distance = z_far;

    // 4-corner CCW polygon from projection bounds.
    let pb = projection.projection_bounds;
    out.convex_hull_points = [
        RealPoint2d { x: pb.x0, y: pb.y0 },
        RealPoint2d { x: pb.x1, y: pb.y0 },
        RealPoint2d { x: pb.x1, y: pb.y1 },
        RealPoint2d { x: pb.x0, y: pb.y1 },
    ];
    out.convex_hull_point_count = 4;

    out.volume_bounded_flag = true;
    visibility_volume_build(
        &out.clone(),
        0,
        &projection.projection_bounds,
        &mut out.volume,
    );
}

// =============================================================================
// `render_visibility_camera_collection_compute @ 0x1806AB740` (G2)
// =============================================================================

/// Build the per-frame camera projection + drive
/// [`CVisibilityCollection::compute`] with `_camera` type. This is
/// the entry point used by `c_player_view::compute_visibility` and
/// `c_texture_camera_view::compute_visibility`.
pub fn render_visibility_camera_collection_compute(
    observer: &ObserverResult,
    z_near: f32,
    z_far: f32,
    projection: &RenderProjection,
    camera_cluster_reference: ClusterReference,
    user_index: i32,
    player_window_index: i32,
    single_cluster_only: bool,
    debug_pvs_render_all: bool,
    scenario: &LoadedScenario,
    portal_activation: &ActivePortalBitvectors,
    collection: &mut CVisibilityCollection,
    visible_items: &mut CVisibleItems,
) {
    let mut visibility_projection = VisibilityProjection::default();
    render_visibility_build_projection(
        observer,
        z_near,
        z_far,
        projection,
        camera_cluster_reference,
        &mut visibility_projection,
    );

    let mut flags = 0u32;
    if debug_pvs_render_all {
        flags |= visibility_collection_flags::USE_PVS_VISIBILITY;
    }
    if single_cluster_only {
        flags |= visibility_collection_flags::SINGLE_CLUSTER_ONLY;
    }

    collection.compute(
        ECollectionType::Camera,
        flags,
        &[visibility_projection],
        vec3_to_point(observer.position),
        1024.0, // engine constant
        0,
        camera_cluster_reference,
        user_index,
        player_window_index,
        scenario,
        portal_activation,
        visible_items,
    );
}
