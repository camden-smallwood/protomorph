//! `c_decal::build_floating_quad` — planar-fragment fast path. When
//! the BFS walker reports `m_can_be_quad` for a single-collision
//! placement, the engine REWINDS the fragment's working-buffer
//! contribution and emits 4 vertices into a per-c_decal scratch array
//! (`c_decal::m_floating_vertices[4]`) which the per-decal render
//! path drains as a triangle strip.
//!
//! Anchored against dllcache:
//!   `c_decal::build_floating_quad @ 0x18039C010`         (MCC x64)
//!   `c_decal::build_floating_quad @ 0x82a113e8`          (H3 X360 debug, 13374 — clearer math)
//!   `c_decal::build_tangent_frame @ 0x18039BD40`         (TBN packer)
//!
//! Geometry recipe (per i = 0..4):
//!   u_int = i >> 1                                       0, 0, 1, 1
//!   v_int = i & 1                                        0, 1, 0, 1
//!   texture_pos = (0,
//!                  (u - 0.5) * 2 * texture_scale.x,
//!                  (v - 0.5) * 2 * texture_scale.y)      tex-local space
//!   world_or_local = matrix4x3_transform_point(texture_to_local, texture_pos)
//!   t = (plane.d - dot(plane.n, pos)) / dot(plane.n, projection.forward)
//!   pos += t * projection.forward                        (intersect with plane)
//!   pos += X_FLOATING_Z_BIAS * plane.normal              (z-bias)
//!   if local_to_world is Some:                           (instance-geom)
//!       pos = matrix4x3_transform_point(local_to_world, pos)
//!   vertex.position = pos
//!   vertex.texcoord = (u, v)
//!   if decal_def.flags & 2:                              (bump_modulate)
//!       build_tangent_frame(vertex, local_to_world, plane.normal,
//!                           projection.up, apply_floating_z_bias=false)

use blam_tags::math::{RealPlane3d, RealPoint3d, RealVector3d};

use blam_tags::math::RealMatrix4x3;

use super::writer::{
    build_tangent_frame, matrix4x3_transform_point, RasterizerVertexWorld, X_FLOATING_Z_BIAS,
};

/// Mirror of `c_decal::build_floating_quad @ 0x18039C010`.
///
/// `collision_plane` and `texture_to_local` must be in the SAME frame
/// — world for main-BSP placements, instance-local for instance-geom
/// placements. `local_to_world` is `Some(M_inst)` for instance-geom
/// hits (lifts the final position back to world space) and `None`
/// for main-BSP hits.
///
/// `decal_definition_flags & 2` gates the tangent-frame build (bump
/// modulate); engine line `if ( (definition->m_flags & 2) != 0 )`
/// at `c_decal::build_floating_quad:62`.
///
/// `texture_scale` is the per-c_decal `m_texture_scale` (set by
/// `c_decal::choose_sprite` from the decal's bitmap aspect; in our
/// port it is the `DecalBuildContext::texture_scale_{x,y}` pair).
pub fn build_floating_quad(
    decal_definition_flags: u32,
    texture_scale: (f32, f32),
    collision_plane: &RealPlane3d,
    local_to_world: Option<&RealMatrix4x3>,
    texture_to_local: &RealMatrix4x3,
) -> [RasterizerVertexWorld; 4] {
    let plane_normal = RealVector3d {
        i: collision_plane.i,
        j: collision_plane.j,
        k: collision_plane.k,
    };
    let projection_forward = texture_to_local.forward;
    let projection_up = texture_to_local.up;

    // Engine emits an error if the decal is projecting onto the backside
    // of a surface (`dot(projection.forward, plane.normal) >= 0`). We
    // skip the log channel here — diagnostics only.

    let mut verts = [RasterizerVertexWorld::default(); 4];
    let n_dot_fwd = plane_normal.i * projection_forward.i
        + plane_normal.j * projection_forward.j
        + plane_normal.k * projection_forward.k;

    for i in 0..4u32 {
        let u_int = i >> 1;
        let v_int = i & 1;
        let u = u_int as f32;
        let v = v_int as f32;

        // Texture-local position: (0, (u-0.5)*2*tex_x, (v-0.5)*2*tex_y).
        // Engine writes texture_pos.y from u and texture_pos.z from v
        // (`y` and `z` are the left/up axes of `texture_to_local`).
        let tex_pos = RealPoint3d {
            x: 0.0,
            y: (u - 0.5) * 2.0 * texture_scale.0,
            z: (v - 0.5) * 2.0 * texture_scale.1,
        };

        // Lift into the matrix's frame (world or instance-local).
        let mut pos = matrix4x3_transform_point(texture_to_local, tex_pos);

        // Intersect with the collision plane along projection.forward.
        // Engine handles the n_dot_fwd == 0 case as `t = 0` (no shift).
        let t = if n_dot_fwd == 0.0 {
            0.0
        } else {
            // Engine: COERCE_UNSIGNED_INT((dot(plane.n, pos) - plane.d) / n_dot_fwd) ^ _xmm
            // The `^ _xmm` (= 0x80000000) flips the sign of the IEEE float.
            -(((plane_normal.i * pos.x + plane_normal.j * pos.y + plane_normal.k * pos.z)
                - collision_plane.d)
                / n_dot_fwd)
        };
        pos.x += t * projection_forward.i;
        pos.y += t * projection_forward.j;
        pos.z += t * projection_forward.k;

        // Z-bias along the (local-frame) plane normal. Engine applies
        // this PRE-local_to_world transform, so the bias direction is
        // local-space normal; the subsequent matrix multiply rotates
        // it into world space.
        pos.x += X_FLOATING_Z_BIAS * plane_normal.i;
        pos.y += X_FLOATING_Z_BIAS * plane_normal.j;
        pos.z += X_FLOATING_Z_BIAS * plane_normal.k;

        // Lift instance-local → world if applicable.
        if let Some(m) = local_to_world {
            pos = matrix4x3_transform_point(m, pos);
        }

        let vert = &mut verts[i as usize];
        vert.position = [pos.x, pos.y, pos.z];
        vert.texcoord = [u, v];

        // Tangent frame — build UNCONDITIONALLY. Engine gates this on
        // `decal_definition_flags & 2` (bump_modulate) because for the
        // !bump_modulate case it dispatches the `s_flat_world_vertex`
        // VS variant (no TBN interpolators) paired with a PS that reads
        // the world normal directly from the surface plane. Protomorph
        // doesn't have the flat-VS pipeline — every decal goes through
        // the regular VS that reads `vertex.{normal,tangent,binormal}`.
        // Skipping the build leaves the TBN at `Default::default()`
        // zeros, so the PS computes `normalize(0, 0, 0)` → degenerate
        // normal → 5a2464c's RT1 alpha-blend writes a `(0.5,0.5,0.5)`
        // sentinel into the normal G-buffer at decal centers. SL then
        // samples a near-zero `dot(N, L)` and the decal renders as a
        // dark blotch (riverworld rockblend 2026-05-23). Mirrors the
        // existing BFS-side workaround per
        // `[[feedback_decal_bfs_tbn_unconditional]]`.
        //
        // Engine passes `apply_floating_z_bias = 0` here because the
        // bias was already applied above.
        let _ = decal_definition_flags;
        build_tangent_frame(
            vert,
            local_to_world,
            plane_normal,
            projection_up,
            false,
        );
    }

    verts
}
