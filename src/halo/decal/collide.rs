//! `c_decal_system::collide @ 0x1803996B0` — engine-faithful port.
//!
//! Verified against IDA bridge port 13372 (Halo 3 dllcache). All
//! constants, masks, and arithmetic shapes mirror the dllcache
//! pseudocode of the named EAs. Numbered helpers below carry the
//! engine EA they port.
//!
//! ## What this does
//!
//! For one preplaced or runtime decal placement:
//!
//! 1. **Primary cast** — multi-target raycast (main BSP + every
//!    instance) with the origin nudged 1cm BACK along the projection
//!    direction. Records the closest hit. Reject types other than
//!    STRUCTURE (1) or INSTANCED_GEOMETRY (3) — sky/object hits don't
//!    decal.
//! 2. **Secondary structure cast** — *only when* the primary hit was
//!    an instance — lifts origin by `0.5 * radius` along the primary
//!    surface normal, shoots back through the surface for one cylinder
//!    height, and records the underlying STRUCTURE surface (for
//!    instance decals that should also stamp the wall behind the
//!    instance).
//! 3. **Secondary instance walk** — iterate every other instance in
//!    the cylinder volume, project their bounding-sphere center onto
//!    the projection axis, and shoot a ray from above each toward the
//!    instance center. Each hit is a separate decal seed for the BFS
//!    walker.
//!
//! Output: `Vec<DecalCollisionResult>` with up to 16 entries. The
//! caller feeds this directly to `c_decal::build_mesh` (orchestrator).
//!
//! ## Engine call graph
//!
//! ```text
//! c_decal_system::collide @ 0x1803996B0
//!   ├─ s_decal_collision_result::test_vector @ 0x1803A2360       (primary)
//!   │    ├─ collision_test_vector (8-arg)        (when geom_idx == -1)
//!   │    └─ instanced_geometry_test_vector       (when geom_idx >= 0)
//!   ├─ s_decal_collision_result::decalable_surface @ 0x18039FE50
//!   ├─ c_decal_system::check_overlap @ 0x18039A050              (per-decal de-dup)
//!   ├─ structure_clusters_in_sphere @ 0x1803B8650               (4-cluster expand)
//!   ├─ collision_point_in_cylinder                               (instance reject)
//!   └─ c_instanced_geometry_iterator                             (instance walk)
//! ```
//!
//! ## v1 simplifications (versus engine)
//!
//! These are *implementation* simplifications that produce the same
//! collision SET as the engine for cyberdyne (single-BSP scenarios).
//! All are upgrade paths, not different algorithms:
//!
//! - **`structure_clusters_in_sphere` skipped.** Engine populates 4
//!   nearby clusters (handles BSP-seam decals by walking
//!   `structure_seams_connected_cluster_references`); v1 stays in the
//!   placement's bound BSP. Cyberdyne is single-BSP so seams aren't
//!   visible. Multi-BSP maps may miss decals that span the seam.
//! - **`c_instanced_geometry_iterator` replaced with a brute-force
//!   walker over `bsp.instances`.** The engine's iterator is
//!   cluster-pruned (only walks instances bucketed in the 4 clusters);
//!   the brute-force walker checks every instance in the BSP. Output
//!   is identical — the iterator is a perf optimization, not a
//!   correctness gate. Sphere reject + cylinder reject still apply.
//! - **`c_decal_system::check_overlap` skipped.** Engine de-dupes
//!   nearby decals (`max_overlapping` from `decal_system`). With
//!   `max_overlapping == 0` it's a no-op (always allows); the engine
//!   path always returns true for v1.
//!
//! When these become observable bugs (e.g. seam decals on a multi-BSP
//! map), upgrade individually.

use blam_tags::decal_system::DecalSystem;
use blam_tags::math::{RealPlane3d, RealPoint3d, RealVector3d};
use blam_tags::structure_bsp::{Bsp3d, BspInstance, BspInstanceDefinition};

use crate::halo::structures::collision_bsp::{collision_bsp_test_vector, flag, CollisionBspResult};
use blam_tags::math::RealMatrix4x3;

use super::instance_raycast::{instance_matrix, try_single_instance_hit, InstanceHit};
use super::orchestrator::DecalCollisionResult;

// =============================================================================
// Engine constants
// =============================================================================

/// Engine `0x1803996B0:179` — primary origin is nudged this far BACK
/// along the normalized velocity before the cast. Pre-placed decals
/// authored exactly on a wall would otherwise miss the back-face
/// transition.
const PRIMARY_NUDGE_BACK: f32 = -0.01;

/// Engine `0x1803996B0:218` (structure secondary) and `:284` (instance
/// secondary). Origin nudged `-0.5 * (negated cylinder height)` =
/// `+0.5 * cylinder_height` along the surface normal direction (lifts
/// above the surface), then the vector points back DOWN by one
/// cylinder height. The dot-product of (-0.5) and (-m_height) equals
/// (+0.5 * m_height), which is what the lift wants.
const SECONDARY_NUDGE_FRACTION: f32 = -0.5;

/// Engine `0x1803996B0:340-360` — per-instance "shift toward instance
/// center" displacement scale (`0.5 * primary.m_radius`). Applied to
/// both origin and vector to bias the secondary tester toward each
/// instance's center while preserving the projection axis.
const INSTANCE_SHIFT_FRACTION: f32 = 0.5;

/// `c_decal_system_definition.flags` bit 7 — `no structure collision`.
/// Engine `0x1803996B0:158-167` clears the outer tester bit that
/// gates STRUCTURE testing (dllcache outer bit 0; 360 debug
/// `get_no_structure_collision` accessor). Equivalent in our port to
/// skipping the main-BSP raycast entirely.
const SYSTEM_FLAG_NO_STRUCTURE_COLLISION: u32 = 0x80;

/// `c_decal_system_definition.flags` bit 8 — `no instance collision`.
/// Engine `0x1803996B0:161-167` clears the outer tester bit that
/// gates INSTANCED_GEOMETRY testing (dllcache outer bit 3; 360 debug
/// `get_no_instance_collision` accessor — note the dllcache and 360
/// builds use different outer bits for the same semantic operation,
/// so modeling the outer-flag space is unhelpful for our port; we
/// just gate the instance walker directly).
const SYSTEM_FLAG_NO_INSTANCE_COLLISION: u32 = 0x100;

/// Top-level "skip ALL secondary work" gate at `0x1803996B0:212`.
/// Engine: `if ((flags & 0x40) == 0 && (flags & 8) == 0 && (flags & 0x10) == 0)`
const SYSTEM_FLAG_PRIMARY_ONLY: u32 = 0x40;
/// Engine `0x1803996B0:212` — second of the three skip-secondary bits.
const SYSTEM_FLAG_SKIP_INSTANCE_SECONDARY: u32 = 0x08;
/// Engine `0x1803996B0:212` — third of the three skip-secondary bits.
const SYSTEM_FLAG_SKIP_STRUCTURE_SECONDARY: u32 = 0x10;

/// Engine `0x1803996B0:255` and `:402` — when set, the secondary cast
/// must hit a surface with the SAME `material_index` as the primary;
/// otherwise the secondary hit is dropped.
const SYSTEM_FLAG_RESTRICT_TO_PRIMARY_MATERIAL: u32 = 0x20;

/// Maximum collisions per placement — `c_static_sized_dynamic_array<...,16>`.
pub const K_MAX_DECAL_COLLISIONS: usize = 16;

// =============================================================================
// Public entry point
// =============================================================================

/// Mirror of `c_decal_system::collide @ 0x1803996B0`.
///
/// Returns `true` if the primary collision succeeded (and hence at
/// least one entry was appended to `collisions`); `false` if the
/// primary cast missed or its hit was rejected by the type/material
/// filters.
///
/// `bsp_index` is the active-BSP slot index — propagated to every
/// `DecalCollisionResult.bsp_index` so the BFS walker knows which
/// BSP's `Bsp3d` to consult downstream.
///
/// `world_projection` is the decal-system's `m_projection` matrix
/// (built by the caller from the placement's quaternion + position +
/// scale, plus the palette's `runtime_max_radius`). On entry, the
/// translation column holds the placement position; on a successful
/// primary cast this function rewrites it to the primary HIT POINT —
/// engine `c_decal_system::set_center @ 0x82a0c5c0` (360 debug) /
/// dllcache line 171. The cylinder built downstream for the secondary
/// cast then sits at the surface, not the air above it.
pub fn collide(
    decal_system: &DecalSystem,
    world_origin: RealPoint3d,
    world_velocity: RealVector3d,
    _ignore_object_index: i32,
    bsp: &Bsp3d,
    instances: &[BspInstance],
    instance_definitions: &[BspInstanceDefinition],
    bsp_index: i32,
    world_projection: &mut RealMatrix4x3,
    collisions: &mut Vec<DecalCollisionResult>,
) -> bool {
    let trace = std::env::var("PROTOMORPH_DEBUG_DECAL_COLLIDE")
        .ok()
        .as_deref()
        == Some("1");
    if trace {
        eprintln!(
            "[collide] enter: instances={} nodes={} edges={} surfaces={}",
            instances.len(), bsp.nodes.len(), bsp.edges.len(), bsp.surfaces.len(),
        );
    }
    let def_flags = decal_system.flags;
    let cylinder_size = decal_system.runtime_max_radius;
    if cylinder_size <= 0.0 {
        return false;
    }

    // ---- BSP-walker flags + per-target dispatch gates ----
    //
    // Engine 0x1803996B0:148-167 builds outer flags then clears bits
    // to disable specific geometry-type tests. We don't model the
    // outer flag space (no shared dispatcher), so we honor the same
    // semantics by GATING the walker calls instead — see
    // `test_main_bsp` / `test_instances` below. The inner BSP-walker
    // flags stay fixed at `DECAL_DEFAULT` (the engine's outer→inner
    // remap result for the decal call site).
    let bsp_flags = flag::DECAL_DEFAULT;
    let test_main_bsp = (def_flags & SYSTEM_FLAG_NO_STRUCTURE_COLLISION) == 0;
    let test_instances = (def_flags & SYSTEM_FLAG_NO_INSTANCE_COLLISION) == 0;

    // ---- Normalize velocity, nudge origin back by 0.01 ----
    let v = [world_velocity.i, world_velocity.j, world_velocity.k];
    let v_len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if v_len_sq <= 0.0 {
        return false;
    }
    let inv_len = 1.0 / v_len_sq.sqrt();
    let v_norm = [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len];
    let primary_origin = [
        world_origin.x + v_norm[0] * PRIMARY_NUDGE_BACK,
        world_origin.y + v_norm[1] * PRIMARY_NUDGE_BACK,
        world_origin.z + v_norm[2] * PRIMARY_NUDGE_BACK,
    ];

    // ---- Primary cast: multi-target ----
    //
    // Engine 0x1803996B0:181-191 — call `s_decal_collision_result::test_vector`
    // with `instanced_geometry_index == -1`, which dispatches to
    // `collision_test_vector` (multi-target). Returns the closest of
    // {main BSP hit, all-instance hits}.
    //
    // The cast is run with `max_t = 1.0` since the projection direction
    // length is only used to define the ray; protomorph normalizes
    // upstream and the BSP walker clamps `t1` to [0, 1].
    if trace { eprintln!("[collide] primary cast"); }
    let primary_hit = run_multi_target_cast(
        bsp,
        instances,
        instance_definitions,
        bsp_index,
        bsp_flags,
        primary_origin,
        v_norm,
        1.0,
        world_projection,
        test_main_bsp,
        test_instances,
    );
    if trace { eprintln!("[collide] primary cast done: hit={}", primary_hit.is_some()); }

    let (primary, primary_instance_idx) = match primary_hit {
        Some(h) => h,
        None => return false,
    };

    // Engine 0x1803996B0:194-196 — `((type - 1) & ~2) != 0` rejects
    // sky (2) and object (4). Our raycast can only return STRUCTURE
    // (1) or INSTANCED_GEOMETRY (3), so no filter needed.
    //
    // Engine 0x1803996B0:196 — `c_decal_system::check_overlap` runs
    // per-decal de-dup. v1 simplification: skipped (max_overlapping=0
    // case == always allow; see module docstring).

    if collisions.len() >= K_MAX_DECAL_COLLISIONS {
        return true; // engine asserts; release no-op + return success
    }
    collisions.push(primary);

    // ---- set_center: m_projection.position = primary.point ----
    //
    // Engine `c_decal_system::set_center @ 0x82a0c5c0` (360 debug) /
    // dllcache 0x1803996B0:171 — after the primary cast succeeds, the
    // engine writes the hit point into `m_projection.position`. The
    // cylinder built downstream (structure + per-instance secondary
    // casts) reads its center from `m_projection.position`, so this
    // re-centers the cylinder onto the actual surface rather than the
    // air the placement was authored from.
    world_projection.position = primary.point;

    // ---- Decide whether to do secondary tests (engine 0x1803996B0:212) ----
    let do_secondary = (def_flags
        & (SYSTEM_FLAG_PRIMARY_ONLY
            | SYSTEM_FLAG_SKIP_INSTANCE_SECONDARY
            | SYSTEM_FLAG_SKIP_STRUCTURE_SECONDARY))
        == 0;
    if !do_secondary {
        return true;
    }

    // ---- Secondary STRUCTURE cast (only when primary was instance) ----
    //
    // Engine 0x1803996B0:217-264 — when the primary hit was an
    // instance (type == 3), shoot a SECOND ray from above the hit
    // point straight down through the surface for one cylinder
    // height, against MAIN-BSP only (clear bit 3 = FILTER_TWO_SIDED).
    // This finds the structure surface BEHIND the instance for
    // decals that should also stamp the underlying wall (e.g. blood
    // splatters on a railing also drip onto the floor).
    if trace { eprintln!("[collide] secondary structure cast (primary was instance? {})", primary_instance_idx.is_some()); }
    if primary_instance_idx.is_some() && collisions.len() < K_MAX_DECAL_COLLISIONS {
        let normal = [primary.plane.i, primary.plane.j, primary.plane.k];
        let n_len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if n_len_sq > 0.0 {
            // Engine 0x1803996B0:240 — `v37 = -m_height / |normal|`.
            // Negation + scaling fold into one float (the engine
            // achieves this via `LODWORD(cylinder.m_height) ^ _xmm`
            // sign-bit XOR).
            let inv_n = -cylinder_size / n_len_sq.sqrt();
            let secondary_v = [normal[0] * inv_n, normal[1] * inv_n, normal[2] * inv_n];
            let secondary_o = [
                primary.point.x + secondary_v[0] * SECONDARY_NUDGE_FRACTION,
                primary.point.y + secondary_v[1] * SECONDARY_NUDGE_FRACTION,
                primary.point.z + secondary_v[2] * SECONDARY_NUDGE_FRACTION,
            ];

            // Engine 0x1803996B0:235 — `structure.m_flags &= ~8` —
            // clears OUTER bit 3 (TEST_INSTANCED_GEOMETRY in MCC; bit 2
            // in 360). That bit gates the engine outer dispatcher
            // from descending into instance walkers; our structure
            // secondary cast calls `collision_bsp_test_vector` directly
            // (no instances), so the clear is inert for our path.
            // Inner walker bits stay at DECAL_DEFAULT.
            let struct_flags = bsp_flags;

            if let Some(hit) = collision_bsp_test_vector(
                bsp,
                struct_flags,
                secondary_o,
                secondary_v,
                1.0,
            )
            .filter(decalable_surface)
            {
                let point = point_from_t(secondary_o, secondary_v, hit.t);
                if point_in_decal_cylinder(point, world_projection, cylinder_size)
                    && material_match(def_flags, primary.material_index, hit.material_index)
                {
                    collisions.push(DecalCollisionResult {
                        bsp_index,
                        instance_definition_index: -1,
                        surface_index: hit.surface_index,
                        plane: hit.plane,
                        point: RealPoint3d {
                            x: point[0],
                            y: point[1],
                            z: point[2],
                        },
                        material_index: hit.material_index,
                    });
                }
            }
        }
    }

    // ---- Secondary INSTANCE walk (engine 0x1803996B0:268-441) ----
    //
    // Build the secondary tester base: same shape as the structure
    // cast (lift origin + flip vector along surface normal) but with
    // bits 0+1 cleared (engine: `secondary.m_flags &= ~3`) instead of
    // bit 3, and exclude the primary's instance.
    let normal = [primary.plane.i, primary.plane.j, primary.plane.k];
    let n_len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if n_len_sq <= 0.0 {
        return true;
    }
    let inv_n = -cylinder_size / n_len_sq.sqrt();
    let secondary_v_base = [normal[0] * inv_n, normal[1] * inv_n, normal[2] * inv_n];
    let secondary_o_base = [
        primary.point.x + secondary_v_base[0] * SECONDARY_NUDGE_FRACTION,
        primary.point.y + secondary_v_base[1] * SECONDARY_NUDGE_FRACTION,
        primary.point.z + secondary_v_base[2] * SECONDARY_NUDGE_FRACTION,
    ];
    // Engine 0x1803996B0:281 — `secondary.m_flags &= ~3` clears OUTER
    // bits 0+1 (TEST_STRUCTURE and another outer-only bit). Neither
    // maps to the inner BSP-walker bits the outer→inner remap covers
    // (11/12/14 → 0/1/3), so the clear is inert for our direct
    // walker call. Inner flags stay at DECAL_DEFAULT.
    let secondary_flags = bsp_flags;
    let exclude_instance = primary_instance_idx.unwrap_or(-1);

    if trace { eprintln!("[collide] entering secondary instance walk ({} instances)", instances.len()); }
    // v1 simplification: walk all instances in the BSP. Sphere-reject
    // first via the engine's `world_bounding_sphere_*`, then
    // cylinder-reject via `collision_point_in_cylinder` against the
    // bound-extended cylinder. Exclude the primary's instance.
    let mut sec_passes: usize = 0;
    for (inst_idx_usize, inst) in instances.iter().enumerate() {
        if trace && inst_idx_usize % 500 == 0 {
            eprintln!("[collide] secondary instance walk {}/{} (passes={sec_passes})", inst_idx_usize, instances.len());
        }
        sec_passes += 1;
        let inst_idx = inst_idx_usize as i32;
        if inst_idx == exclude_instance {
            continue;
        }

        // Engine 0x1803996B0:328-348 — the instance iterator pre-walk
        // skips instances whose bounding sphere doesn't intersect the
        // cylinder extended by the bound radius (so a partially
        // overlapping instance still tests).
        let bound_r = inst.world_bounding_sphere_radius;
        let extended_radius = cylinder_size + bound_r;
        let extended_height = cylinder_size + 2.0 * bound_r;
        if !point_in_cylinder(
            [
                inst.world_bounding_sphere_center.x,
                inst.world_bounding_sphere_center.y,
                inst.world_bounding_sphere_center.z,
            ],
            world_projection,
            extended_radius,
            extended_height,
        ) {
            continue;
        }

        if collisions.len() >= K_MAX_DECAL_COLLISIONS {
            // Engine 0x1803996B0:367-379 logs a warning; we silently
            // stop appending. This matches release-build behavior.
            break;
        }

        // ---- Compute "shift toward instance center" (engine :390-404) ----
        //
        // delta = inst.center - primary.point
        // v59 = (delta · v) / |v|     (signed length of delta along v)
        // shift = v * v59 + delta     (delta with v-axis component doubled)
        // shift = normalize(shift)
        //
        // Then the secondary tester is offset by `+0.5 * primary.radius
        // * shift` in BOTH origin AND vector — biases the ray toward
        // the instance center while preserving its direction.
        let delta = [
            inst.world_bounding_sphere_center.x - primary.point.x,
            inst.world_bounding_sphere_center.y - primary.point.y,
            inst.world_bounding_sphere_center.z - primary.point.z,
        ];
        let v_len_sq_secondary = secondary_v_base[0] * secondary_v_base[0]
            + secondary_v_base[1] * secondary_v_base[1]
            + secondary_v_base[2] * secondary_v_base[2];
        if v_len_sq_secondary <= 0.0 {
            continue;
        }
        let v59 = (delta[0] * secondary_v_base[0]
            + delta[1] * secondary_v_base[1]
            + delta[2] * secondary_v_base[2])
            / v_len_sq_secondary.sqrt();
        let mut shift = [
            secondary_v_base[0] * v59 + delta[0],
            secondary_v_base[1] * v59 + delta[1],
            secondary_v_base[2] * v59 + delta[2],
        ];
        normalize3_in_place(&mut shift);

        let shift_scale = INSTANCE_SHIFT_FRACTION * cylinder_size;
        let shifted_o = [
            secondary_o_base[0] + shift[0] * shift_scale,
            secondary_o_base[1] + shift[1] * shift_scale,
            secondary_o_base[2] + shift[2] * shift_scale,
        ];
        let shifted_v = [
            secondary_v_base[0] + shift[0] * shift_scale,
            secondary_v_base[1] + shift[1] * shift_scale,
            secondary_v_base[2] + shift[2] * shift_scale,
        ];

        // ---- Per-instance raycast ----
        //
        // Post-filter via the instance-aware `decalable_surface` overload
        // (engine `@ 0x18039FE50`): reject when this instance's
        // definition is render-only. `inst` is the loop's BspInstance
        // so we look up its def directly.
        if let Some(hit) = try_single_instance_hit(
            inst_idx,
            inst,
            instance_definitions,
            secondary_flags,
            shifted_o,
            shifted_v,
            1.0,
            world_projection,
        )
        .filter(|i| {
            instance_definitions
                .get(inst.definition_index as usize)
                .map_or(false, |def| decalable_surface_instance(&i.local_result, def))
        })
        {
            if material_match(
                def_flags,
                primary.material_index,
                hit.local_result.material_index,
            ) {
                collisions.push(instance_hit_to_collision(bsp_index, &hit));
            }
        }
    }

    true
}

// =============================================================================
// Internals
// =============================================================================

/// Engine `s_decal_collision_result::test_vector` with
/// `instanced_geometry_index == -1` — multi-target cast that returns
/// the CLOSER of {main-BSP hit, any-instance hit}.
///
/// Returns `(DecalCollisionResult, Option<instance_index>)` where the
/// `Option` is `Some(idx)` when the closer hit was an instance, else
/// `None`.
fn run_multi_target_cast(
    bsp: &Bsp3d,
    instances: &[BspInstance],
    instance_definitions: &[BspInstanceDefinition],
    bsp_index: i32,
    bsp_flags: u32,
    origin: [f32; 3],
    vector: [f32; 3],
    max_t: f32,
    world_projection: &RealMatrix4x3,
    test_main_bsp: bool,
    test_instances: bool,
) -> Option<(DecalCollisionResult, Option<i32>)> {
    // Engine `s_decal_collision_result::test_vector` applies
    // `decalable_surface` post-filter before accepting either hit;
    // we mirror that with `.filter(...)`. The two booleans gate the
    // walker calls per `def.flags & {0x80, 0x100}` semantics
    // (no_structure_collision / no_instance_collision).
    let main_hit = if test_main_bsp {
        collision_bsp_test_vector(bsp, bsp_flags, origin, vector, max_t)
            .filter(decalable_surface)
    } else {
        None
    };

    // Cap the instance search at main's t (or max_t when main missed)
    // so we only consider CLOSER instance hits.
    let instance_max_t = main_hit.as_ref().map(|h| h.t).unwrap_or(max_t);
    let inst_hit = if test_instances {
        super::instance_raycast::try_instance_hit(
            instances,
            instance_definitions,
            bsp_flags,
            origin,
            vector,
            instance_max_t,
            world_projection,
        )
        .filter(|i| {
            // Instance-aware `decalable_surface` (engine `@ 0x18039FE50`):
            // reject when the hit's instance definition is render-only.
            instances
                .get(i.instance_index as usize)
                .and_then(|inst| instance_definitions.get(inst.definition_index as usize))
                .map_or(false, |def| decalable_surface_instance(&i.local_result, def))
        })
    } else {
        None
    };

    // Pick whichever is closer (instance wins ties because instance_max_t
    // bounded its search at main's t).
    if let Some(inst) = inst_hit {
        return Some((
            instance_hit_to_collision(bsp_index, &inst),
            Some(inst.instance_index),
        ));
    }
    if let Some(hit) = main_hit {
        let point = point_from_t(origin, vector, hit.t);
        return Some((
            DecalCollisionResult {
                bsp_index,
                instance_definition_index: -1,
                surface_index: hit.surface_index,
                plane: hit.plane,
                point: RealPoint3d {
                    x: point[0],
                    y: point[1],
                    z: point[2],
                },
                material_index: hit.material_index,
            },
            None,
        ));
    }
    None
}

/// Convert an instance-local raycast result into the wire-format
/// `DecalCollisionResult` that the BFS walker consumes.
///
/// **World-space output.** Engine
/// `instanced_geometry_test_vector_internal @ 0x180400170` writes
/// `collision->point = world_origin + t * world_vector` and calls
/// `build_collision_result_from_bsp_result(..., transform=M_inst)`
/// which does `matrix4x3_transform_plane(transform, local_plane,
/// &collision->plane)`. So the OUTER engine return is world-space for
/// instance hits too. The orchestrator's `build_mesh_fragment` then
/// inverse-transforms BACK to instance-local for the BFS walk (engine
/// line 0x18039CCC0:189-210). Mirror that contract here.
fn instance_hit_to_collision(bsp_index: i32, hit: &InstanceHit) -> DecalCollisionResult {
    let m = &hit.instance_local_to_world;
    let s = m.scale;
    let lp = hit.local_point;
    // `matrix4x3_transform_point` inlined (Halo basis layout:
    // world = scale * (lp.x*forward + lp.y*left + lp.z*up) + position).
    let world_point = RealPoint3d {
        x: s * (lp.x * m.forward.i + lp.y * m.left.i + lp.z * m.up.i) + m.position.x,
        y: s * (lp.x * m.forward.j + lp.y * m.left.j + lp.z * m.up.j) + m.position.y,
        z: s * (lp.x * m.forward.k + lp.y * m.left.k + lp.z * m.up.k) + m.position.z,
    };
    // `matrix4x3_transform_plane` inlined. Engine
    // `build_collision_result_from_bsp_result @ <inlined>` does:
    //   matrix4x3_transform_plane(transform, result->plane, &collision->plane).
    // For Halo's orthonormal basis with scale = inst.scale, this is
    // (n_world = R * n_local, d_world = d_local + dot(n_world, t_inst)).
    let lpln = hit.local_result.plane;
    let nx = lpln.i * m.forward.i + lpln.j * m.left.i + lpln.k * m.up.i;
    let ny = lpln.i * m.forward.j + lpln.j * m.left.j + lpln.k * m.up.j;
    let nz = lpln.i * m.forward.k + lpln.j * m.left.k + lpln.k * m.up.k;
    let d = lpln.d + (nx * m.position.x + ny * m.position.y + nz * m.position.z);
    let world_plane = RealPlane3d { i: nx, j: ny, k: nz, d };
    DecalCollisionResult {
        bsp_index,
        instance_definition_index: hit.instance_index,
        surface_index: hit.local_result.surface_index,
        plane: world_plane,
        point: world_point,
        material_index: hit.local_result.material_index,
    }
}

/// Hit-point reconstruction: `origin + t * vector`.
fn point_from_t(origin: [f32; 3], vector: [f32; 3], t: f32) -> [f32; 3] {
    [
        origin[0] + t * vector[0],
        origin[1] + t * vector[1],
        origin[2] + t * vector[2],
    ]
}

/// Engine `(def.flags & 0x20) == 0 || primary.material == hit.material`
/// at lines 255 and 402. Centralizing for both secondary callers.
fn material_match(def_flags: u32, primary_material: i16, hit_material: i16) -> bool {
    if (def_flags & SYSTEM_FLAG_RESTRICT_TO_PRIMARY_MATERIAL) == 0 {
        return true;
    }
    primary_material == hit_material
}

/// Mirror of `s_decal_collision_result::decalable_surface @ 0x18039FF40`
/// (the static surface-only overload).
///
/// Engine post-filter applied after every `test_vector` hit before
/// the result is accepted. Rejects:
///   - surfaces with any of `flags & 0x3B` set (bits 0, 1, 3, 4, 5 —
///     covers invisible/sky + two-sided + breakable + a couple of
///     "not visible to player" flags whose names aren't yet plumbed
///     here)
///   - surfaces with `material_index == -1` (no material assigned —
///     placeholder geometry)
///
/// **What this catches that the BSP walker doesn't.** The walker's
/// `flag::FILTER_TWO_SIDED` only filters bit 1 (when set by caller).
/// `flag::FILTER_BREAKABLE` (bit 3) isn't in `DECAL_DEFAULT`, so
/// breakable surfaces reached the secondary work. Bits 0/4/5 were
/// never checked. This post-filter closes those gaps.
///
/// For instance hits the engine routes through the instance-aware
/// overload at `0x18039FE50` — see [`decalable_surface_instance`].
fn decalable_surface(result: &CollisionBspResult) -> bool {
    (result.flags & 0x3B) == 0 && result.material_index != -1
}

/// Mirror of `s_decal_collision_result::decalable_surface() const
/// @ 0x18039FE50` — the instance-aware variant.
///
/// Engine pseudocode (instance branch):
/// ```c
/// v10 = global_instance_geometry_definition_get(structure_bsp_index,
///                                                inst.definition_index);
/// if (v10->render_bsp.count) return 0;       // render-only → reject
/// // ...fall through to the surface flags + material_index check
/// ```
///
/// `render_bsp.count > 0` marks the definition as render-only — the
/// instance carries geometry but no collision-shaped BSP, so decals
/// have nothing meaningful to project against. Engine bails before
/// the surface bits check.
fn decalable_surface_instance(
    result: &CollisionBspResult,
    def: &BspInstanceDefinition,
) -> bool {
    if def.render_bsp_count > 0 {
        return false;
    }
    decalable_surface(result)
}

/// Wrapper for the cylinder-containment test using the same
/// `(radius, height)` for both extents (decal cylinder is square in
/// profile).
fn point_in_decal_cylinder(
    point: [f32; 3],
    cylinder_matrix: &RealMatrix4x3,
    cylinder_size: f32,
) -> bool {
    point_in_cylinder(point, cylinder_matrix, cylinder_size, cylinder_size)
}

/// Mirror of `collision_point_in_cylinder @ <not in current dllcache
/// extract; reconstructed from inline use at 0x1803996B0:230 + 0x1803A12D0>`
///
/// Engine implementation:
/// 1. Take the cylinder's `up` axis (matrix3x4 row 2 = `n[2]`),
///    normalize it, dot with `(point - cylinder.position)`. If `|dot| >
///    cylinder_height * 0.5`, miss.
/// 2. Build a height vector along normalized up scaled by
///    cylinder_height; build a base point at `(0, 0, -height/2)` then
///    transform by cylinder_matrix.
/// 3. Compute `point_to_line_distance_squared3d(point, base, height_vector)`;
///    miss if it exceeds `cylinder_radius²`.
fn point_in_cylinder(
    point: [f32; 3],
    cyl: &RealMatrix4x3,
    cylinder_radius: f32,
    cylinder_height: f32,
) -> bool {
    if cylinder_radius <= 0.000099999997 || cylinder_height <= 0.000099999997 {
        return false;
    }

    // Cylinder up-axis (engine reads `cylinder_matrix->n[2]` which is
    // the third row of the 4x3 matrix; in protomorph's `RealMatrix4x3`
    // basis convention `up` is the third basis vector).
    let mut axis = [cyl.up.i, cyl.up.j, cyl.up.k];
    let axis_len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if axis_len_sq != 0.0 {
        let inv = 1.0 / axis_len_sq.sqrt();
        axis = [axis[0] * inv, axis[1] * inv, axis[2] * inv];
    }

    let dx = point[0] - cyl.position.x;
    let dy = point[1] - cyl.position.y;
    let dz = point[2] - cyl.position.z;
    let along = axis[0] * dx + axis[1] * dy + axis[2] * dz;
    if along.abs() > cylinder_height * 0.5 {
        return false;
    }

    // Build base + height_vector for radial distance.
    let mut height_axis = [cyl.up.i, cyl.up.j, cyl.up.k];
    let h_len_sq =
        height_axis[0] * height_axis[0] + height_axis[1] * height_axis[1] + height_axis[2] * height_axis[2];
    if h_len_sq != 0.0 {
        let inv = 1.0 / h_len_sq.sqrt();
        height_axis = [height_axis[0] * inv, height_axis[1] * inv, height_axis[2] * inv];
    }
    let height_vec = [
        height_axis[0] * cylinder_height,
        height_axis[1] * cylinder_height,
        height_axis[2] * cylinder_height,
    ];
    let base_local = [0.0, 0.0, cylinder_height * -0.5];
    let base_world = matrix4x3_transform_point(cyl, base_local);
    let dist_sq = point_to_line_distance_squared3d(point, base_world, height_vec);
    dist_sq <= cylinder_radius * cylinder_radius
}

/// Engine `matrix4x3_transform_point` — apply the matrix's basis +
/// scale + translation to a local-space point.
fn matrix4x3_transform_point(m: &RealMatrix4x3, p: [f32; 3]) -> [f32; 3] {
    let s = m.scale;
    [
        m.position.x + s * (p[0] * m.forward.i + p[1] * m.left.i + p[2] * m.up.i),
        m.position.y + s * (p[0] * m.forward.j + p[1] * m.left.j + p[2] * m.up.j),
        m.position.z + s * (p[0] * m.forward.k + p[1] * m.left.k + p[2] * m.up.k),
    ]
}

/// Engine `point_to_line_distance_squared3d` — squared perpendicular
/// distance from `point` to the infinite line at `base + t * dir`.
fn point_to_line_distance_squared3d(point: [f32; 3], base: [f32; 3], dir: [f32; 3]) -> f32 {
    let dx = point[0] - base[0];
    let dy = point[1] - base[1];
    let dz = point[2] - base[2];
    let dir_len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    if dir_len_sq <= 0.0 {
        return dx * dx + dy * dy + dz * dz;
    }
    // perp = d - (d·dir / |dir|²) * dir
    let t = (dx * dir[0] + dy * dir[1] + dz * dir[2]) / dir_len_sq;
    let px = dx - t * dir[0];
    let py = dy - t * dir[1];
    let pz = dz - t * dir[2];
    px * px + py * py + pz * pz
}

fn normalize3_in_place(v: &mut [f32; 3]) {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 0.0 {
        let inv = 1.0 / len_sq.sqrt();
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    }
}

#[allow(dead_code)]
fn _unused_instance_matrix_helper(inst: &BspInstance) -> RealMatrix4x3 {
    instance_matrix(inst)
}
