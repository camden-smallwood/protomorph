//! `c_decal::build_mesh` + `c_decal::build_mesh_fragment` — the
//! per-decal orchestrator that feeds D2a collision seeds into the
//! BFS walker.
//!
//! Anchored against dllcache:
//!   `c_decal::build_mesh @ 0x18039B820`
//!   `c_decal::build_mesh_fragment @ 0x18039CCC0`
//!
//! ### Scope of this commit
//!
//! Ports the orchestrator skeleton + per-seed setup. The
//! instance-geometry transform path (engine lines 108-172) is
//! stubbed — cyberdyne's 228 working raycasts are all main-BSP
//! placements (matching the D2a hit pattern). When Phase C
//! (instance-geometry collision) ports, this stub turns into the
//! `matrix4x3_inverse_transform_{point,normal,plane}` block that
//! brings the seed into instance-local space.
//!
//! Also deferred: `c_decal::smooth_mesh_fragment @ 0x18039D220`
//! (seam normal averaging between fragments) and
//! `c_decal::build_floating_quad` (planar-fragment fast path). The
//! orchestrator skips both and produces the same output as the BFS
//! walker plus the per-fragment working-interval bookkeeping.

use blam_tags::math::{RealPlane3d, RealPoint3d, RealVector3d};
use blam_tags::structure_bsp::{Bsp3d, BspCollisionMaterial, BspInstance, BspInstanceDefinition};

use blam_tags::math::RealMatrix4x3;

use crate::halo::math::matrix_math::{
    matrix4x3_inverse_compose, matrix4x3_inverse_transform_normal,
    matrix4x3_inverse_transform_plane, matrix4x3_inverse_transform_point_value,
    normalize3d_value,
};

use super::instance_raycast::instance_matrix;
use super::mesh_builder::{build_mesh_fragment_recursive, DecalBuildContext};
use super::types::{
    flag, DecalFragment, DecalMeshBuilder, DecalMeshFragmentBuilder, FragmentBufferIntervals,
    NeighborSurface,
};

/// Per-decal collision seed — one of up to 16 hits the engine
/// collects from `c_decal_system::collide` (our D2a). The Rust port
/// only carries the fields the BFS walker needs.
#[derive(Debug, Clone, Copy)]
pub struct DecalCollisionResult {
    /// Active-BSP slot index (zone-set local). For instance-geom
    /// hits this is the bsp_slot of the parent BSP — same as a
    /// main-BSP hit; the BFS only consults
    /// `instance_definition_index` for routing decisions.
    pub bsp_index: i32,
    /// `instanced_geometry_definitions[]` index when the hit was an
    /// instance, else `-1`. Threaded through every `Fold` so the
    /// edge-iterator skips its cross-BSP seam logic and stays in the
    /// instance's local collision_bsp (`edge_iterator.rs:179`).
    pub instance_definition_index: i32,
    /// Surface index within the BSP's collision_bsp.
    pub surface_index: i32,
    /// Plane the hit landed on. World-space for main-BSP hits,
    /// **instance-local space** when `instance_definition_index !=
    /// -1`.
    pub plane: RealPlane3d,
    /// Hit point — becomes the seed's `Fold::origin`. World-space
    /// for main-BSP hits, instance-local for instance-geom hits.
    pub point: RealPoint3d,
    /// `collision_surface.material_index` of the hit surface. Engine
    /// reads this at offset 98 of `s_decal_collision_result` to gate
    /// the secondary cast's "same material only" filter
    /// (`c_decal_system_definition.flags & 0x20`).
    pub material_index: i16,
}

/// Per-decal-placement transform + flags. Caller computes from the
/// scenario decal placement's `rotation`, `position`, `scale`, plus
/// the `.decal_system` tag's first definition (cull_angle,
/// clamp_angle, texture_scale, flags) and the parent system's
/// handedness bit.
#[derive(Debug, Clone)]
pub struct DecalParams {
    /// Decal-system local-to-world matrix — the engine's
    /// `c_decal_system` carries this at offset 24 of the runtime
    /// struct. Forward axis is the projection "right" direction;
    /// `up` is the projection "down" direction; etc. (Halo's basis
    /// names don't match the engine's "projection" semantics —
    /// scroll back through dllcache for the orientation; for our
    /// purposes the matrix is opaque and just passes through into
    /// the projection_builder.)
    pub local_to_world: RealMatrix4x3,
    /// Per-definition values consumed by the walker.
    pub context: DecalBuildContext,
}

/// Mirror of `c_decal::build_mesh @ 0x18039B820`. Loops over the
/// per-decal collision hits, calling `build_mesh_fragment` per seed.
/// Returns the count of fragments that built successfully (0..=16),
/// with `false` indicating the walker hit its vertex/index buffer
/// caps mid-build (the engine returns 0 in that case after logging
/// `"Exceeded maximum decal *_count."`).
///
/// `fragments_out` must have one slot per collision in
/// `collisions`. Slots beyond `collisions.len()` are left unchanged.
pub fn build_mesh(
    bsp: &Bsp3d,
    collision_materials: &[BspCollisionMaterial],
    instances: &[BspInstance],
    instance_definitions: &[BspInstanceDefinition],
    params: &DecalParams,
    mesh_builder: &mut DecalMeshBuilder,
    collisions: &[DecalCollisionResult],
    fragments_out: &mut [DecalFragment],
) -> bool {
    // Reset mesh_builder counts (engine memsets the relevant fields
    // at the top of build_mesh).
    mesh_builder.working_vertex_count = 0;
    mesh_builder.working_index_count = 0;
    mesh_builder.output_vertex_count = 0;
    mesh_builder.output_index_count = 0;

    let count = collisions.len().min(fragments_out.len());
    if count == 0 {
        return true;
    }
    // PROTOMORPH_FORCE_QUAD=1 — force every placement through the
    // floating-quad path even for multi-collision. Engine never does
    // this for non-force_quad palettes, but it's an isolation tool:
    // if visible artifacts disappear under this gate, the BFS-mesh
    // renderer is the smoking gun.
    let force_quad = std::env::var("PROTOMORPH_FORCE_QUAD").ok().as_deref() == Some("1");
    let can_use_quad = count == 1 || force_quad;
    for i in 0..count {
        if !build_mesh_fragment(
            bsp,
            collision_materials,
            instances,
            instance_definitions,
            params,
            mesh_builder,
            &mut fragments_out[i],
            &collisions[i],
            can_use_quad,
        ) {
            return false;
        }
        super::smoother::smooth_mesh_fragment(mesh_builder, &mut fragments_out[i]);
    }
    true
}

/// Mirror of `c_decal::build_mesh_fragment @ 0x18039CCC0`. Seeds the
/// BFS state from a single `DecalCollisionResult`, builds the seed
/// projection, then calls `build_mesh_fragment_recursive`. Writes
/// the resulting working-buffer ranges into `fragment`.
///
/// Engine contract — collisions arrive in **world space** for both
/// main-BSP and instance-geom hits (engine
/// `build_collision_result_from_bsp_result @ <inlined>` transforms
/// instance plane to world via `matrix4x3_transform_plane`, and
/// `instanced_geometry_test_vector @ 0x180400440` writes
/// `collision->point = world_origin + t * world_vector`). When the seed
/// is an instance hit (`collision.instance_definition_index >= 0`),
/// this function resolves the instance's `M_inst` + local BSP, then
/// inverse-transforms the seed fold (origin, normal, plane) and the
/// projection basis into instance-local space before invoking the BFS
/// walker against the instance's local collision BSP. The fragment
/// records `instance_local_to_world` so the writer can push BFS output
/// back to world space. Engine lines: `c_decal::build_mesh_fragment @
/// 0x18039CCC0:184-218`.
pub fn build_mesh_fragment(
    bsp: &Bsp3d,
    collision_materials: &[BspCollisionMaterial],
    instances: &[BspInstance],
    instance_definitions: &[BspInstanceDefinition],
    params: &DecalParams,
    mesh_builder: &mut DecalMeshBuilder,
    fragment: &mut DecalFragment,
    collision: &DecalCollisionResult,
    can_use_quad: bool,
) -> bool {
    // Snapshot working-buffer cursors before the walker mutates them.
    fragment.working_intervals.starting_vertex = mesh_builder.working_vertex_count;
    fragment.working_intervals.starting_index = mesh_builder.working_index_count;
    fragment.local_to_world = 0;
    fragment.requires_floating_z_bias = false;
    fragment.instance_local_to_world = None;
    fragment.floating_quad = None;

    // === Resolve target BSP + instance transform per seed ===
    //
    // For main-BSP seeds: walk `bsp` (the active BSP's collision
    // tree), no transform needed.
    //
    // For instance-geom seeds (engine line 0x18039CCC0:184): look up
    // the instance + its definition, get M_inst, walk
    // `instance_definitions[inst.definition_index].bsp` in
    // instance-local space, store M_inst on the fragment.
    let mut seed_normal = RealVector3d {
        i: collision.plane.i,
        j: collision.plane.j,
        k: collision.plane.k,
    };
    let mut seed_origin = collision.point;
    let mut seed_plane = collision.plane;
    let mut effective_proj = params.local_to_world;
    let walker_bsp: &Bsp3d;
    let mut effective_instance_definition_index = collision.instance_definition_index;

    if collision.instance_definition_index >= 0 {
        let inst_idx = collision.instance_definition_index as usize;
        let Some(inst) = instances.get(inst_idx) else {
            fragment.working_intervals.vertex_count = 0;
            fragment.working_intervals.index_count = 0;
            return true;
        };
        let inst_matrix = instance_matrix(inst);
        let Some(inst_def) = instance_definitions.get(inst.definition_index as usize) else {
            fragment.working_intervals.vertex_count = 0;
            fragment.working_intervals.index_count = 0;
            return true;
        };
        let Some(inst_bsp) = inst_def.bsp.as_ref() else {
            fragment.working_intervals.vertex_count = 0;
            fragment.working_intervals.index_count = 0;
            return true;
        };

        // Mirror engine lines 0x18039CCC0:195-211 — invert M_inst on
        // the projection basis, the fold (origin, normal), and the
        // plane. Normalize the resulting fold normal.
        effective_proj = matrix4x3_inverse_compose(&inst_matrix, &params.local_to_world);
        seed_origin = matrix4x3_inverse_transform_point_value(&inst_matrix, seed_origin);
        let local_normal = matrix4x3_inverse_transform_normal(&inst_matrix, seed_normal);
        seed_normal = normalize3d_value(local_normal);
        seed_plane = matrix4x3_inverse_transform_plane(&inst_matrix, seed_plane);

        fragment.instance_local_to_world = Some(inst_matrix);
        effective_instance_definition_index = inst.definition_index as i32;
        walker_bsp = inst_bsp;
    } else {
        walker_bsp = bsp;
    }

    // Build a fresh BFS state on the stack (heap-Box'd in the Rust
    // port for the 15 KB queue).
    let mut fragment_builder = DecalMeshFragmentBuilder::new();
    fragment_builder.instanced_geometry_index = effective_instance_definition_index;
    fragment_builder.can_be_quad = true;
    fragment_builder.neighbor_surface_count = 1;

    // Seed neighbor[0] from the (possibly-transformed) collision data.
    fragment_builder.neighbor_surfaces[0] = NeighborSurface {
        parent_index: -1,
        fold: super::types::Fold {
            bsp_index: collision.bsp_index,
            instance_definition_index: effective_instance_definition_index,
            surface_index: collision.surface_index,
            normal: seed_normal,
            origin: seed_origin,
            axis: RealVector3d::default(), // filled below
        },
        projection_builder: Default::default(),
    };
    let _ = seed_plane; // currently unused (projection_builder rebuilds plane from fold); kept for engine-shape parity.

    // === Compute fold axis ===
    // Engine `c_decal::build_mesh_fragment @ 0x18039CCC0` body:
    //   axis = cross(projection.forward, fold.normal);
    //   normalize3d(&axis);              // <-- normalize FIRST
    //   v34 = |axis|² - 1.0;             // <-- then check
    //   if (NaN || |v34| >= 0.001) {     // <-- only fall back when normalize FAILED
    //     axis = cross(projection.up, fold.normal);
    //     normalize3d(&axis);
    //   }
    //
    // **2026-05-17 bug fix:** prior port computed `|axis|² - 1` on the
    // UN-NORMALIZED cross. cross(unit_a, unit_b) has magnitude
    // `sin(angle)`, so `|axis|² = sin²(angle)`. The `|v - 1| < 0.001`
    // test only passed when forward ⟂ normal exactly. For typical
    // decal placements where forward is nearly ANTI-parallel to normal
    // (projecting INTO the surface), `sin²(angle)` is small → check
    // failed → fallback path fired → fold.axis = up × normal instead
    // of forward × normal. Wrong fold axis → wrong projection
    // rotation in `build_projection` → mesh fragments wrapped in the
    // wrong direction → over-expanded polygons (visible as huge brown
    // patches in deadlock + dome-shaped fragments on riverworld
    // curved cliffs) AND missing decals on walls (hardhat-style).
    // Per `c_decal::build_mesh_fragment @ 0x18039CCC0`.
    let fwd = effective_proj.forward;
    let axis_primary = RealVector3d {
        i: fwd.j * seed_normal.k - fwd.k * seed_normal.j,
        j: fwd.k * seed_normal.i - fwd.i * seed_normal.k,
        k: fwd.i * seed_normal.j - fwd.j * seed_normal.i,
    };
    let axis_primary_normalized = normalize3d_value(axis_primary);
    let len_sq_post_normalize = axis_primary_normalized.i * axis_primary_normalized.i
        + axis_primary_normalized.j * axis_primary_normalized.j
        + axis_primary_normalized.k * axis_primary_normalized.k;
    let post_diff = len_sq_post_normalize - 1.0;
    let primary_ok = post_diff.is_finite() && post_diff.abs() < 0.001;
    let axis = if primary_ok {
        axis_primary_normalized
    } else {
        let up = effective_proj.up;
        let axis_fallback = RealVector3d {
            i: up.j * seed_normal.k - up.k * seed_normal.j,
            j: up.k * seed_normal.i - up.i * seed_normal.k,
            k: up.i * seed_normal.j - up.j * seed_normal.i,
        };
        normalize3d_value(axis_fallback)
    };
    fragment_builder.neighbor_surfaces[0].fold.axis = axis;

    // === Initialize seed projection builder ===
    let cull_rad = params.context.cull_angle_radians;
    let clamp_rad = params.context.clamp_angle_radians;
    let mut flags = flag::INITIALIZED | flag::NEEDS_RENORMALIZE;
    if params.context.left_handed {
        flags |= flag::LEFT_HANDED;
    }
    fragment_builder.neighbor_surfaces[0].projection_builder =
        super::types::DecalProjectionBuilder {
            projection: effective_proj,
            cull_angle_radians: cull_rad,
            cull_angle_cos: cull_rad.cos(),
            clamp_angle_radians: clamp_rad,
            clamp_angle_cos: clamp_rad.cos(),
            flags,
        };

    // === Build seed projection + run BFS ===
    let fold0 = fragment_builder.neighbor_surfaces[0].fold;
    let built = fragment_builder.neighbor_surfaces[0]
        .projection_builder
        .build_projection(&fold0);
    if !built {
        // Seed cull-cone rejected — fragment is empty.
        fragment.working_intervals.vertex_count = 0;
        fragment.working_intervals.index_count = 0;
        return true;
    }

    let walker_ok = build_mesh_fragment_recursive(
        walker_bsp,
        collision_materials,
        mesh_builder,
        &mut fragment_builder,
        &params.context,
    );

    // ---- Floating-quad fast path — engine `build_mesh_fragment:226-239` ----
    //
    // When the BFS walker succeeds and reports `m_can_be_quad` for a
    // single-collision placement, the engine REWINDS the working-buffer
    // cursors (zeroing this fragment's BFS contribution) and emits 4
    // vertices into `c_decal::m_floating_vertices[4]`. The per-decal
    // render path (`c_decal::render @ 0x18039B100`) draws those via
    // `draw_primitive_up(triangle_strip, 2, m_floating_vertices, 0x2Cu)`.
    //
    // Protomorph mirrors the rewind here and stashes the 4 verts on
    // `fragment.floating_quad`. The loader (after `write_mesh_fragment`,
    // which sees `output_intervals.vertex_count == 0` and writes nothing)
    // appends those 4 verts + 4 strip indices to the per-placement
    // packed buffer. Net rendered geometry matches the engine's quad.
    //
    // Engine `c_decal::build_mesh_fragment @ 0x18039CCC0:226-239`
    // ALWAYS takes the floating-quad path when `can_be_quad &&
    // can_use_quad`. **2026-05-17:** flipped to default-on after MCC
    // A/B confirmed our BFS-mesh-for-quadable-decals produces
    // visible artifacts (white wings / over-large fragments) where
    // engine renders a clean 4-vertex flat quad. Disable with
    // `PROTOMORPH_DISABLE_DECAL_FLOATING_QUAD=1` for bisecting.
    let floating_quad_enabled = std::env::var("PROTOMORPH_DISABLE_DECAL_FLOATING_QUAD")
        .map(|v| v != "1")
        .unwrap_or(true);
    // PROTOMORPH_FORCE_QUAD=1 forces the floating-quad emit regardless
    // of can_be_quad (which the BFS walker clears whenever an edge
    // crosses the unit tile). Diagnostic-only — non-engine-faithful.
    let force_quad_take = std::env::var("PROTOMORPH_FORCE_QUAD").ok().as_deref() == Some("1");
    let take_quad = walker_ok
        && floating_quad_enabled
        && (force_quad_take || (can_use_quad && fragment_builder.can_be_quad));
    if take_quad {
        // Use the projection as it stands AFTER `build_projection` ran
        // (engine reads `v50.m_neighbor_surfaces[0].m_projection_builder
        // .m_projection`, the post-build copy).
        let projection_after = fragment_builder.neighbor_surfaces[0]
            .projection_builder
            .projection;
        let local_to_world_ref = fragment.instance_local_to_world.as_ref();
        let quad = super::floating_quad::build_floating_quad(
            params.context.decal_definition_flags,
            (
                params.context.texture_scale_x,
                params.context.texture_scale_y,
            ),
            &seed_plane,
            local_to_world_ref,
            &projection_after,
        );
        fragment.floating_quad = Some(quad);

        // Engine REWINDS the working-buffer cursors — this fragment's
        // BFS triangles are discarded.
        mesh_builder.working_vertex_count = fragment.working_intervals.starting_vertex;
        mesh_builder.working_index_count = fragment.working_intervals.starting_index;
    }
    let _ = collision;

    // Record the fragment's working-buffer span (zero for floating-quad).
    fragment.working_intervals.vertex_count =
        mesh_builder.working_vertex_count - fragment.working_intervals.starting_vertex;
    fragment.working_intervals.index_count =
        mesh_builder.working_index_count - fragment.working_intervals.starting_index;

    walker_ok
}

// Inverse-transform helpers now live in `crate::halo::math::matrix_math`
// (canonical, sign-preserving scale clamp). The decal instance-seed
// branch above imports them directly. Mirror engine
// `matrix4x3_inverse_transform_*` (`@ 0x1802c4ca0` family); pull a
// world-space `DecalCollisionResult` into instance-local space (engine
// `c_decal::build_mesh_fragment @ 0x18039CCC0:195-211`).
