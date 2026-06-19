//! Instance-geometry decal raycast — D2a extension.
//!
//! Mirrors the instance branch of
//! `collision_test_vector (8-arg) @ 0x1803fa760` →
//! `collision_test_cluster @ 0x1803ff8b0` →
//! `instanced_geometry_test_vector_internal @ 0x180400170`. Instance
//! flag bit `_instanced_geometry_render_only_bit = 1` (mask 0x2) is
//! filtered per the engine line:
//!
//! ```c
//! if ((v19 & 2) != 0 || (flags & 0x400) != 0 && (v19 & 4) != 0) return 0;
//! ```
//!
//! The per-instance OBB cull (`collision_test_rectangle3d`) the engine
//! runs first reduces to `fast_vector_intersects_sphere` here — both
//! are early-out predicates and the sphere is conservative.
//!
//! ## Coordinate spaces
//!
//! The decal's projection (`params.local_to_world`) lives in world
//! space. The instance carries `instance_local_to_world`. To run the
//! BFS walker against the instance's local `collision_bsp`, we
//! inverse-compose:
//!
//! ```text
//! M_local_proj = M_inst^{-1} ∘ M_proj
//!   - basis  : R_inst^T · M_proj.basis        (rotation only — no scale-divide)
//!   - origin : M_inst.inverse_transform_point(M_proj.position)
//!   - scale  : M_proj.scale / M_inst.scale
//! ```
//!
//! See dllcache `c_decal::build_mesh_fragment @ 0x18039CCC0` lines
//! 108-172 for the engine's matrix3x4_inverse_compose path.

use blam_tags::math::RealPoint3d;
use blam_tags::structure_bsp::{Bsp3d, BspInstance, BspInstanceDefinition};

use crate::halo::structures::collision_bsp::{collision_bsp_test_vector, CollisionBspResult};
use blam_tags::math::RealMatrix4x3;

use super::mesh_builder::matrix4x3_inverse_transform_point;
use crate::halo::math::matrix_math::{
    matrix4x3_inverse_compose, matrix4x3_inverse_transform_vector,
};

/// Result of a successful instance-geometry raycast.
#[derive(Debug, Clone)]
pub struct InstanceHit {
    /// Index into `loaded_bsp.sbsp.instanced_geometry_instances`.
    pub instance_index: i32,
    /// `t` parameter along the ORIGINAL world-space ray (the
    /// parametric distance is invariant under the affine transform).
    pub t: f32,
    /// Hit result reported by the instance's local `collision_bsp`.
    /// `point` and `plane` are in **instance-local** space; the
    /// caller hands them straight to the BFS walker via
    /// `DecalCollisionResult`.
    pub local_result: CollisionBspResult,
    /// Hit point in instance-local space (== `local_origin + t *
    /// local_vector`).
    pub local_point: RealPoint3d,
    /// `M_inst` — instance local-to-world. The writer applies this
    /// to push instance-local BFS vertices out to world space.
    pub instance_local_to_world: RealMatrix4x3,
    /// `M_local_proj` — the decal projection matrix pulled into
    /// instance-local space. Pass this as `DecalParams.local_to_world`
    /// to the BFS walker.
    pub local_projection: RealMatrix4x3,
}

/// Walk every instance in `instances`, fast-sphere-reject, then for
/// the survivors raycast the instance-local `collision_bsp`. Returns
/// the closest hit with `t < max_t`, or `None` if all instances
/// missed.
///
/// `world_origin` / `world_vector` describe the projection ray in
/// world space (same parameters the main-BSP raycast used).
/// `max_t` is the upper bound on the hit's parametric distance —
/// callers pass either `1.0` or the main-BSP hit's `t` so the instance
/// search only returns CLOSER hits.
///
/// `world_projection` is the decal's world-space `RealMatrix4x3`
/// (i.e. `params.local_to_world` from the orchestrator).
pub fn try_instance_hit(
    instances: &[BspInstance],
    definitions: &[BspInstanceDefinition],
    bsp_flags: u32,
    world_origin: [f32; 3],
    world_vector: [f32; 3],
    max_t: f32,
    world_projection: &RealMatrix4x3,
) -> Option<InstanceHit> {
    let mut best: Option<InstanceHit> = None;
    let mut best_t = max_t;

    for (i, inst) in instances.iter().enumerate() {
        if let Some(hit) = try_single_instance_hit(
            i as i32,
            inst,
            definitions,
            bsp_flags,
            world_origin,
            world_vector,
            best_t,
            world_projection,
        ) {
            best_t = hit.t;
            best = Some(hit);
        }
    }
    best
}

/// Single-instance variant of [`try_instance_hit`] — runs the same
/// sphere-reject + inverse-transform + BSP raycast pipeline against
/// ONE specific instance and returns its hit (if any).
///
/// Mirrors the engine's `instanced_geometry_test_vector @ 0x180400440`
/// branch — used by `c_decal_system::collide`'s secondary
/// instance-walk loop, where each iteration tests a single
/// `instance_iterator.m_instance` against the secondary tester
/// configuration. The multi-target [`try_instance_hit`] above is the
/// `collision_test_vector(..., instanced_geometry_index = -1)` branch
/// (multi-target dispatch).
pub fn try_single_instance_hit(
    instance_index: i32,
    inst: &BspInstance,
    definitions: &[BspInstanceDefinition],
    bsp_flags: u32,
    world_origin: [f32; 3],
    world_vector: [f32; 3],
    max_t: f32,
    world_projection: &RealMatrix4x3,
) -> Option<InstanceHit> {
    if inst.flags.contains(blam_tags::structure_bsp::InstancedGeometryFlags::RenderOnly) {
        return None;
    }
    if !fast_vector_intersects_sphere(
        world_origin,
        world_vector,
        [
            inst.world_bounding_sphere_center.x,
            inst.world_bounding_sphere_center.y,
            inst.world_bounding_sphere_center.z,
        ],
        inst.world_bounding_sphere_radius,
    ) {
        return None;
    }
    let def = definitions.get(inst.definition_index as usize)?;
    let def_bsp = def.bsp.as_ref()?;
    if def_bsp.nodes.is_empty() || def_bsp.surfaces.is_empty() {
        return None;
    }

    let m_inst = instance_matrix(inst);
    let local_origin = matrix4x3_inverse_transform_point(&m_inst, point_from_arr(world_origin));
    let local_vector = matrix4x3_inverse_transform_vector(&m_inst, world_vector);
    let r = collision_bsp_test_vector(
        def_bsp,
        bsp_flags,
        [local_origin.x, local_origin.y, local_origin.z],
        local_vector,
        max_t,
    )?;

    let local_proj = matrix4x3_inverse_compose(&m_inst, world_projection);
    let local_point = RealPoint3d {
        x: local_origin.x + r.t * local_vector[0],
        y: local_origin.y + r.t * local_vector[1],
        z: local_origin.z + r.t * local_vector[2],
    };
    Some(InstanceHit {
        instance_index,
        t: r.t,
        local_point,
        local_result: r,
        instance_local_to_world: m_inst,
        local_projection: local_proj,
    })
}

/// `BspInstance` → `RealMatrix4x3`.
pub fn instance_matrix(inst: &BspInstance) -> RealMatrix4x3 {
    RealMatrix4x3 {
        scale: inst.scale,
        forward: inst.forward,
        left: inst.left,
        up: inst.up,
        position: inst.position,
    }
}

/// Mirror of `fast_vector_intersects_sphere @ 0x18018c430`. Returns
/// `true` if the ray from `point` along `vector` (length 1.0 along
/// the vector = parametric `t = 1`) approaches within `radius` of
/// `center`. Includes the "origin inside sphere" + "behind start"
/// + "discriminant" branches verbatim from the decompile.
fn fast_vector_intersects_sphere(
    point: [f32; 3],
    vector: [f32; 3],
    center: [f32; 3],
    radius: f32,
) -> bool {
    let dx = point[0] - center[0];
    let dy = point[1] - center[1];
    let dz = point[2] - center[2];
    let to_center_sq = dx * dx + dy * dy + dz * dz - radius * radius;
    if to_center_sq < 0.0 {
        return true;
    }
    let along = vector[0] * dx + vector[1] * dy + vector[2] * dz;
    if along >= 0.0 {
        return false;
    }
    let v_len_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    let disc = along * along - v_len_sq * to_center_sq;
    if disc <= 0.0 {
        return false;
    }
    let neg_along = -v_len_sq - along;
    if neg_along < 0.0 {
        return true;
    }
    disc > neg_along * neg_along
}

#[inline]
fn point_from_arr(a: [f32; 3]) -> RealPoint3d {
    RealPoint3d { x: a[0], y: a[1], z: a[2] }
}
