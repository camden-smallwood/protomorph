//! Engine source: `Ares/source/physics/collisions.{h,cpp}`.
//!
//! Orchestration layer above [`super::collision_bsp`]. `collision_test_vector`
//! is the unified entry point that callers (including
//! `c_geometry_sampler::geometry_test_vector`) invoke for ray-vs-world tests.
//! It dispatches to:
//!
//! - [`collision_bsp::test_vector`][super::collision_bsp::test_vector] for
//!   the structure BSP's collision tree (per loaded BSP)
//! - [`instanced_geometry_test_vector_internal`] for each instanced-geometry
//!   instance in each loaded BSP
//! - per-object collision (engine `object_test_vector_internal`) for each
//!   placed object (scenery, weapons, units, etc.) **— NOT ported; see below.**
//!
//! Returns the closest hit across all sources in `collision_result`.
//!
//! ## Trimmed from engine source (no behavior change for decorator bake)
//!
//! - Profiling (`collision_log_*`, `collision_start_time*`, `c_string_builder`).
//! - Debug-skip globals (`debug_collision_skip_*`) — always `false` in retail.
//! - Breakable-surface state — `collision_materials_extant_flags`/breakable
//!   set lookups are skipped when the caller passes null (decorator-bake's
//!   `c_geometry_sampler` does).
//! - Assert/event handlers (`handle_slim_assert`, `c_event`).
//!
//! ## Not ported (NEEDS follow-up Phase 5b for full object support)
//!
//! Per-object collision (engine `object_test_vector_internal`) is **not
//! ported** — placed objects are skipped. Engine body decompile at
//! `refs/engine_geometry_sampler/object_test_vector_internal.txt`. The
//! decorator-bake chain still works for placements on BSP clusters and
//! instanced geometry (the majority case) — scenery-attached decorators
//! will miss the object hit and route through the scenery_probe fallback
//! that the bake already handles.

use blam_tags::math::{RealPlane3d, RealPoint3d, RealVector3d};

use crate::halo::scenario::loader::LoadedScenario;
use crate::halo::decal::mesh_builder::matrix4x3_inverse_transform_point;

use super::collision_bsp::{self, CollisionBspTestVectorResult, CollisionBspView};
use super::collision_constants::CollisionTestFlagsPair;

// =============================================================================
// e_collision_result_type — engine `physics/collisions.h:15`
// =============================================================================

/// Mirrors:
///
/// ```cpp
/// enum e_collision_result_type
/// {
///     _collision_result_none = 0,
///     _collision_result_structure,
///     _collision_result_media,
///     _collision_result_instanced_geometry,
///     _collision_result_object,
///     k_collision_result_type_count,
/// };
/// ```
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CollisionResultType {
    #[default]
    None = 0,
    Structure = 1,
    Media = 2,
    InstancedGeometry = 3,
    Object = 4,
}

// =============================================================================
// collision_result — engine `physics/collisions.h:56` (sizeof=104)
// =============================================================================

/// Engine `collision_result` (104 B). Filled by every `collision_test_*`
/// entry point. Fields ordered to mirror the Ares struct layout exactly so
/// any callers downcasting/aliasing match engine semantics.
#[derive(Debug, Clone)]
pub struct CollisionResult {
    /// Engine `type` @ 0x0.
    pub type_: CollisionResultType,
    /// Ray parametric `t` of the hit (`0..maximum_t`). Engine `t` @ 0x4.
    pub t: f32,
    /// World-space hit position = `start + t * vector`. Engine `point` @ 0x8.
    pub point: RealPoint3d,
    /// Engine `start_location` @ 0x14 (s_location, 4 B). We expose as a
    /// packed `i32` because the inner cluster_reference / bsp_index / ...
    /// struct isn't ported yet; engine writes `-1` (all bits set) for
    /// "no valid start location".
    pub start_location: i32,
    /// Engine `location` @ 0x18. Same convention as `start_location`.
    pub location: i32,
    /// **Engine has `collision_bsp const* bsp` @ 0x20.** We store the BSP
    /// INDEX rather than the pointer for borrow simplicity; the caller can
    /// look up the bsp via `scenario.active_bsps[bsp_index].sbsp.collision_bsp`.
    pub bsp_index: i32,
    /// Engine `global_material_type` @ 0x28 (c_global_material_type, 4 B).
    /// Packed `i16` index (engine has `m_index` field).
    pub global_material_type: i16,
    /// Engine `plane` @ 0x2C (real_plane3d, 16 B). Engine builds this via
    /// `build_collision_result_from_bsp_result`, transforming the local-
    /// space plane back to world space when the hit is on an instance.
    pub plane: RealPlane3d,
    /// Engine `fog_plane_index` @ 0x3C.
    pub fog_plane_index: i32,
    /// Engine `instanced_geometry_instance_index` @ 0x40. `-1` for non-
    /// instance hits.
    pub instanced_geometry_instance_index: i32,
    /// Engine `object_index` @ 0x44. `-1` for non-object hits.
    pub object_index: i32,
    /// Engine `region_index` @ 0x48.
    pub region_index: i16,
    /// Engine `node_index` @ 0x4A.
    pub node_index: i16,
    /// Engine `bsp_reference` @ 0x4C.
    pub bsp_reference: u32,
    /// Engine `structure_bsp_index` @ 0x50.
    pub structure_bsp_index: i32,
    /// Engine `leaf_index` @ 0x54 (collision-bsp leaf).
    pub leaf_index: i32,
    /// Engine `surface_index` @ 0x58 (collision-bsp surface).
    pub surface_index: i32,
    /// Engine `plane_designator` @ 0x5C (low 31 = plane_idx, bit 31 = negate).
    pub plane_designator: i32,
    /// Engine `flags` @ 0x60 (collision-surface flags).
    pub flags: blam_tags::Flags<blam_tags::structure_bsp::CollisionSurfaceFlags, u8>,
    /// Engine `breakable_surface_index` @ 0x61.
    pub breakable_surface_index: u8,
    /// Engine `material_index` @ 0x62 (collision-material table).
    pub material_index: i16,
    /// Engine `breakable_surface_set_index` @ 0x64.
    pub breakable_surface_set_index: u8,
}

impl Default for CollisionResult {
    fn default() -> Self {
        Self {
            type_: CollisionResultType::None,
            t: 1.0,
            point: RealPoint3d::default(),
            start_location: -1,
            location: -1,
            bsp_index: -1,
            global_material_type: -1,
            plane: RealPlane3d::default(),
            fog_plane_index: -1,
            instanced_geometry_instance_index: -1,
            object_index: -1,
            region_index: -1,
            node_index: -1,
            bsp_reference: u32::MAX,
            structure_bsp_index: -1,
            leaf_index: -1,
            surface_index: -1,
            plane_designator: -1,
            flags: blam_tags::Flags::default(),
            breakable_surface_index: 0,
            material_index: -1,
            breakable_surface_set_index: 0xFF,
        }
    }
}

// =============================================================================
// initialize_collision_result @ 0x180401d50
// =============================================================================

/// Engine `initialize_collision_result @ 0x180401d50`. Resets a result struct
/// to the "no hit" sentinel state. Engine zeros explicit fields and leaves
/// the rest untouched — we use `Default::default()` to reset all fields.
pub fn initialize_collision_result(collision: &mut CollisionResult) {
    *collision = CollisionResult::default();
}

// =============================================================================
// build_bsp_flags_from_collision_flags @ 0x1804019e0
// =============================================================================

/// `build_bsp_flags_from_collision_flags` @ dllcache `0x1804019e0`.
///
/// Verbatim mapping from `c_collision_test_flags` (22-bit) to
/// `e_collision_bsp_test_flag` (6-bit). Engine bits used by the BSP test:
///
/// ```text
/// collision_test bit 11 (front_facing)     → bsp_test bit 0 (front_facing)
/// collision_test bit 12 (back_facing)      → bsp_test bit 1 (back_facing)
/// collision_test bit 13 (ignore_two_sided) → bsp_test bit 2
/// collision_test bit 14 (ignore_invisible) → bsp_test bit 3
/// collision_test bit 15 (ignore_breakable) → bsp_test bit 4
/// collision_test bit 16 (allow_early_out)  → bsp_test bit 5
/// ```
pub fn build_bsp_flags_from_collision_flags(flags: CollisionTestFlagsPair) -> u32 {
    let cf = flags.collision_flags.bits();
    let mut out: u32 = 0;
    if (cf & 0x0800) != 0 {
        out |= 0x01;
    } // front_facing
    if (cf & 0x1000) != 0 {
        out |= 0x02;
    } // back_facing
    if (cf & 0x2000) != 0 {
        out |= 0x04;
    } // ignore_two_sided
    if (cf & 0x4000) != 0 {
        out |= 0x08;
    } // ignore_invisible
    if (cf & 0x8000) != 0 {
        out |= 0x10;
    } // ignore_breakable
    if (cf & 0x10000) != 0 {
        out |= 0x20;
    } // allow_early_out
    out
}

// =============================================================================
// build_collision_result_from_bsp_result @ 0x180401a50
// =============================================================================

/// `build_collision_result_from_bsp_result` @ dllcache `0x180401a50`.
///
/// Promotes a `CollisionBspTestVectorResult` (output of `collision_bsp_test_vector`)
/// into a unified `CollisionResult`. When `transform` is `Some`, the hit plane
/// is transformed from instance-local to world space (and negated if the
/// plane_designator's negate flag is set). When `transform` is `None`, the
/// plane is copied directly (BSP cluster hits are already in world space).
pub fn build_collision_result_from_bsp_result(
    collision: &mut CollisionResult,
    bsp_planes: &[RealPlane3d],
    bsp_result: &CollisionBspTestVectorResult,
    transform: Option<&blam_tags::math::RealMatrix4x3>,
) {
    collision.t = bsp_result.t;

    let plane = match bsp_result.plane_index {
        Some(idx) => bsp_planes.get(idx as usize).copied().unwrap_or_default(),
        None => RealPlane3d::default(),
    };

    let mut transformed = if let Some(_xform) = transform {
        // Engine: matrix4x3_transform_plane(transform, plane, &dest)
        // TODO Phase 5b: port matrix4x3_transform_plane verbatim. Engine
        // body at dllcache `?matrix4x3_transform_plane@@YA...`. For now the
        // identity-transform case is correct for BSP cluster hits.
        plane
    } else {
        plane
    };

    // Engine: if (plane_designator & 0x8000) negate the plane (XOR sign bit
    // of all four components). The negate flag is in bit 31 of the canonical
    // i32 plane designator (blam-tags promotes both schemas to bit 31).
    if (bsp_result.plane_designator as u32) & 0x8000_0000 != 0 {
        transformed.i = -transformed.i;
        transformed.j = -transformed.j;
        transformed.k = -transformed.k;
        transformed.d = -transformed.d;
    }
    collision.plane = transformed;

    collision.leaf_index = bsp_result.leaf_index;
    collision.surface_index = bsp_result.surface_index;
    collision.plane_designator = bsp_result.plane_designator;
    collision.flags = bsp_result.flags.clone();
    collision.breakable_surface_set_index = bsp_result.breakable_surface_set_index;
    collision.breakable_surface_index = bsp_result.breakable_surface_index;
    collision.material_index = bsp_result.material_index;
}

// =============================================================================
// fast_vector_intersects_sphere @ 0x18018c430
// =============================================================================

/// `fast_vector_intersects_sphere` @ dllcache `0x18018c430`.
///
/// Conservative pre-test for ray-vs-sphere intersection. Returns `true` if
/// the ray segment `[point, point + vector]` MAY intersect the sphere; never
/// returns `false` when an intersection exists. Used to cull instance/object
/// tests before the expensive BSP/render walk.
///
/// Verbatim port — handles three early-out cases:
///  1. Start point inside sphere → hit
///  2. Ray points away from center (dot < 0) → no hit (engine code path)
///  3. Discriminant < 0 → no hit (line misses sphere)
///  4. Closest approach `t` outside segment range → no hit
pub fn fast_vector_intersects_sphere(
    point: RealPoint3d,
    vector: RealVector3d,
    center: RealPoint3d,
    radius: f32,
) -> bool {
    let dx = point.x - center.x;
    let dy = point.y - center.y;
    let dz = point.z - center.z;
    // `c` = |start - center|² - radius² (negative if start is inside sphere)
    let c = dx * dx + dy * dy + dz * dz - radius * radius;
    if c < 0.0 {
        return true;
    }
    // `b` = (start - center) · direction (negative if ray heads toward sphere)
    let b = vector.i * dx + vector.j * dy + vector.k * dz;
    if b >= 0.0 {
        return false;
    }
    // `a` = direction · direction (squared length)
    let a = vector.i * vector.i + vector.j * vector.j + vector.k * vector.k;
    // Discriminant of quadratic t² · a + 2t · b + c = 0
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return false;
    }
    // Engine's last step: ensure the negative root `-b - sqrt(disc)` lies
    // within `[0, a]` (i.e., the hit is within the segment). Engine writes
    // this as `(-a - b) < 0 || disc > (-a - b)²` to avoid the sqrt.
    let neg_a_minus_b = -a - b;
    if neg_a_minus_b < 0.0 {
        return true;
    }
    disc > neg_a_minus_b * neg_a_minus_b
}

// =============================================================================
// matrix4x3_inverse_transform_vector @ 0x1802c4d90
// =============================================================================

/// `matrix4x3_inverse_transform_vector` @ dllcache `0x1802c4d90`. Vector form
/// of the inverse transform (no translation subtraction). Used to bring a
/// world-space ray direction into instance-local space.
pub fn matrix4x3_inverse_transform_vector(
    m: &blam_tags::math::RealMatrix4x3,
    v: RealVector3d,
) -> RealVector3d {
    let scale = m.scale;
    let safe_scale = if scale.abs() <= 9.9999997e-5 {
        if scale < 0.0 {
            -9.9999997e-5
        } else {
            9.9999997e-5
        }
    } else {
        scale
    };
    let inv = 1.0 / safe_scale;
    let sv = RealVector3d {
        i: v.i * inv,
        j: v.j * inv,
        k: v.k * inv,
    };
    // Engine: transformed = basis_transpose * scaled_v
    // (basis rows = forward, left, up)
    RealVector3d {
        i: m.forward.i * sv.i + m.forward.j * sv.j + m.forward.k * sv.k,
        j: m.left.i * sv.i + m.left.j * sv.j + m.left.k * sv.k,
        k: m.up.i * sv.i + m.up.j * sv.j + m.up.k * sv.k,
    }
}

// =============================================================================
// test_instanced_geometry @ 0x180402140
// =============================================================================

/// `test_instanced_geometry` @ dllcache `0x180402140`. Engine filter that
/// rejects instance hits early when the instance carries no collision
/// surfaces or fails the pathfinding-ref test. The decorator-bake chain
/// passes flags with bit 6 unset → only the "has any surface" check fires.
pub fn test_instanced_geometry(
    instance_def: &blam_tags::structure_bsp::BspInstanceDefinition,
    flags_bits: u32,
) -> bool {
    let has_surfaces = instance_def
        .bsp
        .as_ref()
        .map(|b| !b.surfaces.is_empty())
        .unwrap_or(false);
    if !has_surfaces {
        return false;
    }
    // Engine bit 0x40 = pathfinding-only filter. Decorator bake never sets it.
    if (flags_bits & 0x40) == 0 {
        return true;
    }
    // TODO Phase 5b: pathfinding_data.instanced_geometry_refs lookup.
    // Engine body checks `pf_data->instanced_geometry_refs[instance_idx]
    // != 0xFFFF`. For decorator bake (flag bit 0x40 never set), always true.
    true
}

// =============================================================================
// instanced_geometry_test_vector_internal @ 0x180400170
// =============================================================================

/// `instanced_geometry_test_vector_internal` @ dllcache `0x180400170`.
///
/// Tests one instance for intersection. Returns `true` and fills `collision`
/// when the ray hits the instance's collision BSP. Verbatim port of:
///
///  1. Look up `instance` and `instance_definition` from the BSP.
///  2. Filter by instance flags (skip `flags & 2` = "disabled", and
///     `(arg.flags & 0x400) && (inst.flags & 4)` = "render-only when
///     test_render_only is unset").
///  3. Call [`test_instanced_geometry`] for the per-definition surface count
///     / pathfinding check.
///  4. Sphere-cull via [`fast_vector_intersects_sphere`] against
///     `world_bounding_sphere_{center,radius}`.
///  5. Inverse-transform ray to instance-local space.
///  6. Call `collision_bsp_test_vector` against `definition.bsp` (or
///     `render_bsp` when `arg.flags & 0x10` = test-render-only).
///  7. Populate `collision` via [`build_collision_result_from_bsp_result`]
///     + instance-specific fields (instanced_geometry_instance_index,
///     structure_bsp_index, etc.).
pub fn instanced_geometry_test_vector_internal(
    bsp: &blam_tags::structure_bsp::StructureBsp,
    bsp_index: i32,
    instanced_geometry_instance_index: i32,
    flags: CollisionTestFlagsPair,
    bsp_flags: u32,
    point: RealPoint3d,
    vector: RealVector3d,
    collision: &mut CollisionResult,
) -> bool {
    // 1. Look up instance + definition.
    let instance = match bsp
        .instanced_geometry_instances
        .get(instanced_geometry_instance_index as usize)
    {
        Some(i) => i,
        None => return false,
    };
    let definition = match bsp.instance_definitions.get(instance.definition_index as usize) {
        Some(d) => d,
        None => return false,
    };

    // 2. Engine instance-flag filter.
    //    - bit 1 (`render only`) = disabled for collision.
    //    - bit 2 (`does not block aoe damage`) gated by collision-test flag
    //      0x400 (render-only mode).
    use blam_tags::structure_bsp::InstancedGeometryFlags;
    let inst_flags = &instance.flags;
    if inst_flags.contains(InstancedGeometryFlags::RenderOnly) {
        return false;
    }
    if (flags.collision_flags.bits() & 0x0400) != 0
        && inst_flags.contains(InstancedGeometryFlags::DoesNotBlockAoeDamage)
    {
        return false;
    }

    // 3. test_instanced_geometry filter.
    if !test_instanced_geometry(definition, flags.collision_flags.bits()) {
        return false;
    }

    // 4. World-space sphere cull.
    if !fast_vector_intersects_sphere(
        point,
        vector,
        instance.world_bounding_sphere_center,
        instance.world_bounding_sphere_radius,
    ) {
        return false;
    }

    // 5. Inverse-transform ray to instance-local space. Engine constructs a
    //    `real_matrix4x3` from `forward/left/up/position/scale`; our blam-tags
    //    `BspInstance` carries the components directly. Build the matrix in
    //    flight, reusing the existing helper from the decal port.
    let xform = blam_tags::math::RealMatrix4x3 {
        scale: instance.scale,
        forward: instance.forward,
        left: instance.left,
        up: instance.up,
        position: instance.position,
    };
    let transformed_point = matrix4x3_inverse_transform_point(&xform, point);
    let transformed_vector = matrix4x3_inverse_transform_vector(&xform, vector);

    // 6. Pick which collision BSP to test. Engine flag bit 0x10 (render-only-
    //    BSPs) switches to `definition.render_bsp` when set; otherwise the
    //    standard `definition.bsp`. We currently only carry `definition.bsp`
    //    in blam-tags — render_bsp is a TODO for when needed.
    let collision_bsp = match definition.bsp.as_ref() {
        Some(b) => b,
        None => return false,
    };

    // 7. Run the BSP test.
    let mut bsp_result = CollisionBspTestVectorResult::default();
    let view = CollisionBspView::new(collision_bsp);
    let hit = collision_bsp::test_vector(
        bsp_flags,
        &view,
        transformed_point,
        transformed_vector,
        collision.t,
        &mut bsp_result,
    );
    if !hit {
        return false;
    }

    // 8. Populate result. material_index → global_material_type via the
    //    BSP's collision_materials table (engine reads
    //    `runtime_global_material_type` field at offset 16 in
    //    `BspCollisionMaterial`).
    let global_material_type = if bsp_result.material_index >= 0 {
        bsp.collision_materials
            .get(bsp_result.material_index as usize)
            .map(|m| m.runtime_global_material_index)
            .unwrap_or(-1)
    } else {
        -1
    };

    collision.type_ = CollisionResultType::InstancedGeometry;
    collision.bsp_index = bsp_index;
    collision.global_material_type = global_material_type;
    collision.fog_plane_index = -1;
    collision.instanced_geometry_instance_index = instanced_geometry_instance_index;
    collision.object_index = -1;
    collision.structure_bsp_index = bsp_index;
    build_collision_result_from_bsp_result(collision, &collision_bsp.planes, &bsp_result, Some(&xform));
    true
}

// =============================================================================
// collision_test_vector @ 0x1803fa720 (1st overload) + 0x1803fa760 (2nd, 965 LoC)
// =============================================================================

/// `collision_test_vector` (1st overload) @ dllcache `0x1803fa720`.
///
/// Verbatim wrapper that delegates to the 2nd overload with
/// `test_material=false` and `third_ignore_object_index=-1`. This is the form
/// that `c_geometry_sampler::geometry_test_vector` calls into.
pub fn collision_test_vector(
    flags: CollisionTestFlagsPair,
    scenario: &LoadedScenario,
    point: RealPoint3d,
    vector: RealVector3d,
    first_ignore_object_index: i32,
    second_ignore_object_index: i32,
    collision: &mut CollisionResult,
) -> bool {
    collision_test_vector_2(
        flags,
        false,
        scenario,
        point,
        vector,
        first_ignore_object_index,
        second_ignore_object_index,
        -1,
        collision,
    )
}

/// `collision_test_vector` (2nd overload) @ dllcache `0x1803fa760`.
///
/// **Partial port** — see module-level comment for trimmed engine paths.
///
/// Iterates loaded BSPs (engine `n0xF` = scenario.zone_set's bsp_mask, our
/// `LoadedScenario.active_bsps`). For each:
///  - BSP collision test via [`collision_bsp::test_vector`]
///  - For each instance in that BSP, [`instanced_geometry_test_vector_internal`]
///  - per-object collision (`object_test_vector_internal`) is **not ported (TODO 5b)**
/// Returns `true` if any hit found; `collision` carries the closest one.
#[allow(clippy::too_many_arguments)]
pub fn collision_test_vector_2(
    flags: CollisionTestFlagsPair,
    _test_material: bool,
    scenario: &LoadedScenario,
    point: RealPoint3d,
    vector: RealVector3d,
    first_ignore_object_index: i32,
    second_ignore_object_index: i32,
    _third_ignore_object_index: i32,
    collision: &mut CollisionResult,
) -> bool {
    // Engine asserts at least one of {STRUCTURE, MEDIA, INSTANCED_GEOMETRY,
    // OBJECTS}. We trust the caller (decorator-bake passes 0x18459 = lots of bits).
    initialize_collision_result(collision);
    collision.t = 1.0;

    // Engine inflates the BSP test-flags to force front+back testing when
    // neither bit is set. This is what `m_flags |= 0x1800` does in the
    // decompile head. Keeps the BSP walker from skipping all surfaces when
    // the caller leaves the test-direction bits unset.
    let mut effective_flags = flags;
    let cf = flags.collision_flags.bits();
    if (cf & 0x0800) == 0 && (cf & 0x1000) == 0 {
        let new_bits = cf | 0x1800;
        effective_flags = CollisionTestFlagsPair::build(new_bits, flags.object_flags.bits());
    }

    // Walk each loaded BSP. Engine uses scenario's bsp_mask × bsp_count;
    // protomorph's `active_bsps` IS the deduplicated loaded set.
    let cf_test = effective_flags.collision_flags.bits();
    let test_structure = (cf_test & 0x01) != 0;
    let test_instances = (cf_test & 0x08) != 0;
    let bsp_flags = build_bsp_flags_from_collision_flags(effective_flags);

    let mut any_hit = false;

    for (bsp_idx, bsp) in scenario.active_bsps.iter().enumerate() {
        let bsp_index_i32 = bsp_idx as i32;
        let collision_bsp = match bsp.sbsp.collision_bsp.as_ref() {
            Some(c) => c,
            None => continue,
        };

        // 1. Structure BSP test.
        if test_structure {
            let mut bsp_result = CollisionBspTestVectorResult::default();
            let view = CollisionBspView::new(collision_bsp);
            // Engine uses the collision_t-shortened max so subsequent tests
            // can't find a farther hit. Mirror by clamping with `collision.t`.
            let max_t = collision.t;
            let hit = collision_bsp::test_vector(
                bsp_flags,
                &view,
                point,
                vector,
                max_t,
                &mut bsp_result,
            );
            if hit && bsp_result.t < collision.t {
                let global_material_type = if bsp_result.material_index >= 0 {
                    bsp.sbsp
                        .collision_materials
                        .get(bsp_result.material_index as usize)
                        .map(|m| m.runtime_global_material_index)
                        .unwrap_or(-1)
                } else {
                    -1
                };
                collision.type_ = CollisionResultType::Structure;
                collision.bsp_index = bsp_index_i32;
                collision.global_material_type = global_material_type;
                collision.fog_plane_index = -1;
                collision.instanced_geometry_instance_index = -1;
                collision.object_index = -1;
                collision.structure_bsp_index = bsp_index_i32;
                build_collision_result_from_bsp_result(
                    collision,
                    &collision_bsp.planes,
                    &bsp_result,
                    None,
                );
                any_hit = true;
            }
        }

        // 2. Instanced geometry tests.
        if test_instances {
            // Divergence (authorized): for render/lighting queries (test-render-
            // only flag bit 0x10, set by the decorator/geometry-sampler bake),
            // skip instances whose render mesh is EMPTY — the authored invisible-
            // collision hulls (">invis_coll_f", def.mesh_index → render mesh with
            // 0 vertices, no lightmap entry). Their collision sits right at the
            // ground under decorators; resolving the hit to an empty render mesh
            // fails the lightmap lookup → bright unbaked fallback. Skipping them
            // lets the ray reach the lit CLUSTER the decorator actually samples.
            let skip_render_empty = (cf_test & 0x10) != 0;
            for inst_idx in 0..bsp.sbsp.instanced_geometry_instances.len() {
                if skip_render_empty {
                    let render_mesh_empty = bsp
                        .sbsp
                        .instanced_geometry_instances
                        .get(inst_idx)
                        .and_then(|i| bsp.sbsp.instance_definitions.get(i.definition_index as usize))
                        .map(|d| d.mesh_index)
                        .filter(|&mi| mi >= 0)
                        .and_then(|mi| bsp.meshes.get(mi as usize))
                        .map(|m| m.vertices.is_empty())
                        .unwrap_or(false);
                    if render_mesh_empty {
                        continue;
                    }
                }
                let mut temp_collision = collision.clone();
                let hit = instanced_geometry_test_vector_internal(
                    &bsp.sbsp,
                    bsp_index_i32,
                    inst_idx as i32,
                    effective_flags,
                    bsp_flags,
                    point,
                    vector,
                    &mut temp_collision,
                );
                if hit && temp_collision.t < collision.t {
                    *collision = temp_collision;
                    any_hit = true;
                }
            }
        }

        // 3. Object tests — TODO Phase 5b. Engine loops `object_test_vector`
        //    over object_header_data. We stub here so the chain compiles;
        //    decorator-bake on BSP+instance geometry is unaffected.
        let _ignored = (first_ignore_object_index, second_ignore_object_index);
    }

    // Engine fills `collision.point = start + t * vector` at the end.
    if any_hit {
        collision.point = RealPoint3d {
            x: point.x + collision.t * vector.i,
            y: point.y + collision.t * vector.j,
            z: point.z + collision.t * vector.k,
        };
    }

    any_hit
}
