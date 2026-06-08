//! `c_decal::build_mesh_fragment_recursive` — the per-fragment BFS
//! walker that turns a seed surface into a triangulated decal mesh.
//!
//! Anchored against dllcache:
//!   `c_decal::build_mesh_fragment_recursive @ 0x18039C2D0`
//!   `line_segment_intersects_unit_texture_tile @ 0x18039EB80`
//!   `matrix4x3_inverse_transform_point @ 0x1802C4CA0`
//!
//! ### Algorithm
//!
//! The fragment builder enters with one or more `NeighborSurface`
//! entries already pushed (the seed from D2a + any prior expansions).
//! For each queued neighbor in FIFO order:
//!
//! 1. **Emit surface vertices** — walk the surface's edge ring via
//!    `c_collision_surface_edge_iterator`. For each edge, copy the
//!    "trailing" vertex (start_vertex when this surface is on left,
//!    end_vertex when on right) into `DecalMeshBuilder::work_vertex_buffer`.
//!    Compute UVs by inverse-transforming the world position through
//!    `neighbor.projection_builder.projection` and biasing by the
//!    decal's `texture_scale`.
//! 2. **Emit triangle strip** — the engine packs the polygon as a
//!    zigzag triangle strip: `[v0, v0, v1, vN-1, v2, vN-2, ...]` plus
//!    a trailing duplicate. Length is `N + 2` indices for an
//!    N-vertex polygon. Degenerate-padded strips let the strip be
//!    appended to the global index buffer without breaking previous
//!    surfaces' triangulation.
//! 3. **Find neighbors** — re-walk the same edges. For each edge that
//!    crosses the unit texture tile (the decal's projected footprint),
//!    call `get_opposing_surface_fold` and try to push the neighbor.
//!    The new neighbor inherits the parent's projection matrix +
//!    re-initializes cull/clamp angles from the decal definition,
//!    then runs `build_projection` against the fold. If the
//!    projection fails the cull-cone test, the push is undone.
//! 4. **Advance** — increment the queue cursor; loop until the queue
//!    is drained or the BFS hits the work-buffer caps (1024
//!    vertices, 1024 indices) → return false.

use blam_tags::decal_system::DecalSystemFlags;
use blam_tags::math::{RealPoint2d, RealPoint3d};
use blam_tags::structure_bsp::{Bsp3d, BspCollisionMaterial};

use blam_tags::math::RealMatrix4x3;

use super::edge_iterator::*;
use super::types::{
    flag, CollisionSurfaceEdgeIterator, DecalMeshBuilder, DecalMeshFragmentBuilder,
    DecalProjectionBuilder, Fold, NeighborSurface, WorkingVertex, K_MAX_NEIGHBOR_SURFACES,
    K_MAX_WORK_INDICES, K_MAX_WORK_VERTICES,
};

/// Per-decal inputs the walker needs from the orchestrator. Mirrors
/// the values the engine reads from `c_decal_definition` + the parent
/// `c_decal_system_definition` at the start of every BFS step.
#[derive(Debug, Clone)]
pub struct DecalBuildContext {
    /// `c_decal_definition::cull_angle` in radians (offset 100 in the
    /// engine's c_decal_definition). **Engine stores degrees on disk
    /// and converts via `* 0.017453292` (π/180) at call site (`@
    /// 0x18039C2D0` lines 453-456). Caller must do the same.**
    pub cull_angle_radians: f32,
    /// `c_decal_definition::clamp_angle` in radians (offset 96).
    /// **Same deg→rad note as cull_angle_radians.**
    pub clamp_angle_radians: f32,
    /// `c_decal_definition::texture_scale` — UV bias divisor.
    pub texture_scale_x: f32,
    pub texture_scale_y: f32,
    /// `c_decal_system_definition::m_flags` — bit 4 = same_plane_only,
    /// bit 5 = same_material_only. Read from the SYSTEM tag (parent of
    /// c_decal_definition), not from c_decal_definition itself —
    /// matches engine `c_decal::build_mesh_fragment_recursive @
    /// 0x18039C2D0` line 414 which reads `v106->m_flags` where
    /// `v106 = c_decal_system_definition`.
    pub system_definition_flags: blam_tags::Flags<DecalSystemFlags, u32>,
    /// `((parent_system->m_flags >> 1) ^ (parent_system->m_flags >> 2))
    /// & 1` precomputed by the caller — controls the LEFT_HANDED bit
    /// on each new neighbor's projection.
    pub left_handed: bool,
    /// `c_decal_definition::m_flags` — bit 1 = bump_modulate (gates
    /// the build_tangent_frame call in the floating-quad fast path,
    /// engine `c_decal::build_floating_quad @ 0x18039C010:62`), bit 0
    /// = additive blend (consumed by the writer's TBN gate). Caller
    /// pulls this from `decal_def.flags`.
    pub decal_definition_flags: u32,
}

/// Mirror of `c_decal::build_mesh_fragment_recursive @ 0x18039C2D0`.
///
/// Returns `false` when the vertex (1024) or index (1024) work
/// buffer cap is exceeded mid-walk; the engine logs `"Exceeded
/// maximum decal vertex count."` / `"Exceeded maximum decal index
/// count."` in those cases. Returns `true` on a clean drain.
///
/// `bsp` is the BSP that the seed's `Fold` references; the cross-BSP
/// seam-traversal path inside `get_opposing_surface_fold` is stubbed
/// to "no connection" by that function (matches engine behavior on
/// single-BSP scenarios such as cyberdyne).
pub fn build_mesh_fragment_recursive(
    bsp: &Bsp3d,
    collision_materials: &[BspCollisionMaterial],
    mesh_builder: &mut DecalMeshBuilder,
    fragment_builder: &mut DecalMeshFragmentBuilder,
    ctx: &DecalBuildContext,
) -> bool {
    // Engine `c_decal::build_mesh_fragment_recursive @ 0x18039C2D0`
    // opens with:
    //
    //   v14 = c_decal_system_definition->m_flags >> 3;
    //   if ( (v14 & 1) != 0 ) return 1;
    //
    // Bit 3 of `decal_system_flags` is `_force_quad_bit` (per Ares
    // `enum { _random_rotation_bit = 0, _random_u_mirror_bit,
    //         _random_v_mirror_bit, _force_quad_bit, ... }`). When the
    // authoring tag forces quad rendering, BFS does no work — the
    // orchestrator then either takes the floating-quad path (when
    // `can_use_quad` is true for single-collision placements) or
    // produces an empty fragment for multi-collision placements. The
    // empty-fragment case is what we were missing: previously we ran
    // the full BFS walker for multi-collision force-quad placements
    // and emitted geometry the engine never produces.
    if ctx.system_definition_flags.contains(DecalSystemFlags::ForceQuad) {
        return true;
    }
    if fragment_builder.neighbor_surface_count <= 0 {
        return true;
    }

    let mut current_neighbor_index: i32 = 0;
    loop {
        let neighbor_idx = current_neighbor_index as usize;
        let surface_vertex_start = mesh_builder.working_vertex_count;

        // Snapshot the projection (and fold) we need before the
        // walker mutates the queue — pushing new neighbors via
        // get_opposing_surface_fold may grow `neighbor_surfaces` and
        // invalidate any held borrow.
        let projection = fragment_builder.neighbor_surfaces[neighbor_idx]
            .projection_builder
            .projection;
        let parent_flags = fragment_builder.neighbor_surfaces[neighbor_idx]
            .projection_builder
            .flags;
        let fold_normal = fragment_builder.neighbor_surfaces[neighbor_idx].fold.normal;
        let surface_index = fragment_builder.neighbor_surfaces[neighbor_idx]
            .fold
            .surface_index;
        let surface_bsp_index = fragment_builder.neighbor_surfaces[neighbor_idx]
            .fold
            .bsp_index;
        let instance_def_index = fragment_builder.instanced_geometry_index;

        debug_assert!(
            (fragment_builder.neighbor_surfaces[neighbor_idx]
                .projection_builder
                .flags
                & flag::BUILT)
                != 0,
            "neighbor projection_builder not BUILT before walker entry"
        );

        // === Pass 1: collect surface vertices ===
        //
        // Walk the surface's edge ring and gather each trailing vertex
        // into a temp polygon (world position + UV). The polygon is
        // then clipped to the [0,1]² UV unit tile (Sutherland-Hodgman)
        // BEFORE being emitted to the working buffer, so the resulting
        // BFS-mesh geometry is bounded to the projection footprint and
        // doesn't include parts of the BSP polygon that lie outside
        // the decal.
        //
        // Why this DEVIATES from engine: engine's
        // `c_decal::build_mesh_fragment_recursive @ 0x18039C2D0`
        // emits ALL surface vertices and relies on the PS `clip()` to
        // discard out-of-range UVs. Our shader does the same discard,
        // but pre-clipping produces tighter geometry (fragments don't
        // have to be discarded en masse), matches the floating-quad
        // path's size, and is necessary to make the visual output of
        // multi-collision BFS-mesh decals look like engine MCC's
        // (where BSP collision granularity differs from ours in ways
        // that make engine's whole-polygon-emit look acceptable).
        //
        // Pass 2 (neighbor walking) still uses the ORIGINAL BSP edges
        // — it reads start/end positions directly from `surface_poly`
        // (the pre-clip polygon) so the BFS expansion logic is
        // unchanged.
        let basis_up = projection.up;
        let mut surface_poly: Vec<(RealPoint3d, RealPoint2d)> = Vec::with_capacity(8);
        let mut iter = CollisionSurfaceEdgeIterator::new(
            bsp,
            surface_bsp_index,
            instance_def_index,
            surface_index,
        );
        while iter.surface_edge_index != -1 {
            let Some(edge) = bsp.edges.get(iter.surface_edge_index as usize).copied() else {
                break;
            };
            let on_right = edge.right_surface as i32 == surface_index;
            // Engine: trailing vertex = on_right ? end_vertex : start_vertex.
            let vertex_idx = if on_right {
                edge.end_vertex
            } else {
                edge.start_vertex
            };
            let Some(vertex) = bsp.vertices.get(vertex_idx as usize) else {
                break;
            };

            let uv = compute_texcoord(
                &projection,
                vertex.point,
                ctx.texture_scale_x,
                ctx.texture_scale_y,
            );
            surface_poly.push((vertex.point, uv));

            // Advance iterator: next edge in the ring.
            let next_edge = if on_right {
                edge.reverse_edge
            } else {
                edge.forward_edge
            };
            let first_edge = bsp
                .surfaces
                .get(surface_index as usize)
                .map(|s| s.first_edge as i32)
                .unwrap_or(-1);
            iter.surface_edge_index = if next_edge as i32 == first_edge {
                -1
            } else {
                next_edge as i32
            };
        }

        // === Clip surface_poly to [0,1]² UV ===
        let clipped = clip_polygon_to_unit_tile(&surface_poly);
        let n_verts_clip = clipped.len();

        // === Emit clipped vertices to working buffer ===
        if mesh_builder.working_vertex_count as usize + n_verts_clip > K_MAX_WORK_VERTICES {
            return false;
        }
        for (world_pos, uv) in clipped.iter() {
            let slot = mesh_builder.working_vertex_count as usize;
            mesh_builder.working_vertex_count += 1;
            mesh_builder.work_vertex_buffer[slot] = WorkingVertex {
                // -1 marks a clipped vertex with no BSP-index reference.
                // The writer reads `world_position_override` for these
                // and skips the BSP vertex lookup.
                position: -1,
                texcoord: *uv,
                normal: fold_normal,
                binormal: basis_up,
                world_position_override: Some(*world_pos),
            };
        }

        let n_verts = n_verts_clip as u32;
        if n_verts < 3 {
            // Clipped polygon is empty or degenerate (surface lies
            // entirely outside the unit tile, OR has only 1-2 vertices
            // after clipping which can't form triangles). Skip
            // geometry emission for this surface — but still run pass
            // 2 below to expand BFS to neighbors that may be in
            // range.
            //
            // Note: this rewinds the just-emitted (degenerate) verts
            // so the smoother doesn't see them.
            mesh_builder.working_vertex_count -= n_verts_clip as u32;
        } else {
            // === Triangle-strip emission ===
            //
            // Same zigzag pattern as engine for an N-vertex polygon:
            //   index[0] = ss   (degenerate-start duplicate)
            //   index[1] = ss
            //   index[2] = ss + 1
            //   index[3] = ss + N-1
            //   index[4] = ss + 2
            //   index[5] = ss + N-2
            //   ...
            //   index[N+1] = (last)  (degenerate-end duplicate)
            let new_idx_count = mesh_builder.working_index_count + n_verts + 2;
            if new_idx_count as usize > K_MAX_WORK_INDICES {
                return false;
            }
            let wic = mesh_builder.working_index_count as usize;
            let ss = surface_vertex_start as u16;
            mesh_builder.work_index_buffer[wic] = ss;
            mesh_builder.work_index_buffer[wic + 1] = ss;
            let n = n_verts as i32;
            for k in 1..n {
                let offset: i32 = if k & 1 != 0 {
                    (k + 1) >> 1
                } else {
                    n - (k >> 1)
                };
                mesh_builder.work_index_buffer[wic + 1 + k as usize] = ss + offset as u16;
            }
            // Trailing duplicate for degenerate end.
            mesh_builder.work_index_buffer[wic + n as usize + 1] =
                mesh_builder.work_index_buffer[wic + n as usize];
            mesh_builder.working_index_count = (wic + n as usize + 2) as u32;
        }

        // === Pass 2: find neighbors across each edge ===
        //
        // Uses the pre-clip `surface_poly` for start/end positions and
        // UVs. The ring index `i` is the source-polygon vertex index;
        // the edge between `surface_poly[i]` and `surface_poly[i+1]`
        // (wrapping) is the BSP edge being considered.
        let mut iter2 = CollisionSurfaceEdgeIterator::new(
            bsp,
            surface_bsp_index,
            instance_def_index,
            surface_index,
        );
        let mut ring_i: usize = 0;
        while iter2.surface_edge_index != -1 {
            let Some(edge) = bsp.edges.get(iter2.surface_edge_index as usize).copied() else {
                break;
            };
            let on_right = edge.right_surface as i32 == surface_index;

            if ring_i < surface_poly.len() {
                let start_idx = ring_i;
                let end_idx = (ring_i + 1) % surface_poly.len();
                let (start_pos, start_texcoord) = surface_poly[start_idx];
                let (end_pos, end_texcoord) = surface_poly[end_idx];

                if line_segment_intersects_unit_texture_tile(&start_texcoord, &end_texcoord) {
                    fragment_builder.can_be_quad = false;

                    let mut fold = Fold::default();
                    let same_plane_only = ctx.system_definition_flags.contains(DecalSystemFlags::ForcePlanar);
                    let same_material_only = ctx.system_definition_flags.contains(DecalSystemFlags::RestrictToSingleMaterial);
                    if iter2.get_opposing_surface_fold(
                        bsp,
                        collision_materials,
                        start_pos,
                        end_pos,
                        &mut fold,
                        same_plane_only,
                        same_material_only,
                    ) {
                        try_push_neighbor(
                            fragment_builder,
                            &projection,
                            parent_flags,
                            &fold,
                            current_neighbor_index,
                            ctx,
                        );
                    }
                }
            }

            // Advance iterator (same logic as pass 1 + ring_cursor).
            let next_edge = if on_right {
                edge.reverse_edge
            } else {
                edge.forward_edge
            };
            let first_edge = bsp
                .surfaces
                .get(surface_index as usize)
                .map(|s| s.first_edge as i32)
                .unwrap_or(-1);
            iter2.surface_edge_index = if next_edge as i32 == first_edge {
                -1
            } else {
                next_edge as i32
            };
            ring_i += 1;
        }

        current_neighbor_index += 1;
        if current_neighbor_index >= fragment_builder.neighbor_surface_count {
            return true;
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Sutherland-Hodgman polygon clipping against the `[0,1]²` UV unit
/// tile. Input is a convex N-gon of `(world_position, uv)` pairs;
/// output is the clipped polygon (also convex, ≤ N+4 vertices).
///
/// At each clip-line crossing the function emits a NEW vertex with
/// the world position linearly interpolated from the source edge's
/// endpoints. The UV at the new vertex lands exactly on the clip
/// line (e.g., u=0 or u=1).
///
/// Engine `c_decal::build_mesh_fragment_recursive` does NOT clip
/// here — it emits whole BSP polygons and relies on the PS `clip()`
/// to discard out-of-range UVs. Our deviation produces tighter
/// geometry so multi-collision BFS-mesh decals have a footprint
/// similar to floating-quad decals (no huge over-emitted triangles
/// that are mostly discarded).
fn clip_polygon_to_unit_tile(
    poly: &[(RealPoint3d, RealPoint2d)],
) -> Vec<(RealPoint3d, RealPoint2d)> {
    // 4 clip lines for the unit tile, each with (axis_index, value, inside_predicate):
    //   axis 0 = uv.x, axis 1 = uv.y
    //   value = 0.0 or 1.0
    //   inside = true when uv[axis] is on the "keep" side of value
    fn lerp_pt(a: RealPoint3d, b: RealPoint3d, t: f32) -> RealPoint3d {
        RealPoint3d {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }
    fn lerp_uv(a: RealPoint2d, b: RealPoint2d, t: f32) -> RealPoint2d {
        RealPoint2d {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
        }
    }

    let mut input: Vec<(RealPoint3d, RealPoint2d)> = poly.to_vec();
    let mut output: Vec<(RealPoint3d, RealPoint2d)> = Vec::with_capacity(input.len() + 4);

    // 4 clip stages — order matters but result is the same convex region.
    //   stage 0: uv.x >= 0
    //   stage 1: uv.x <= 1
    //   stage 2: uv.y >= 0
    //   stage 3: uv.y <= 1
    let stages: [(u8, f32, bool); 4] = [
        (0, 0.0, true),  // uv.x >= 0
        (0, 1.0, false), // uv.x <= 1
        (1, 0.0, true),  // uv.y >= 0
        (1, 1.0, false), // uv.y <= 1
    ];

    for (axis, value, keep_above) in stages.iter().copied() {
        output.clear();
        if input.is_empty() {
            break;
        }
        let inside = |uv: RealPoint2d| -> bool {
            let v = if axis == 0 { uv.x } else { uv.y };
            if keep_above {
                v >= value
            } else {
                v <= value
            }
        };
        let intersect_t = |a: RealPoint2d, b: RealPoint2d| -> f32 {
            let va = if axis == 0 { a.x } else { a.y };
            let vb = if axis == 0 { b.x } else { b.y };
            let denom = vb - va;
            if denom.abs() < 1e-9 {
                0.0
            } else {
                (value - va) / denom
            }
        };

        let n = input.len();
        let mut prev = input[n - 1];
        let mut prev_in = inside(prev.1);
        for &curr in input.iter() {
            let curr_in = inside(curr.1);
            if curr_in {
                if !prev_in {
                    // Edge crosses INTO the keep side — emit intersection then current.
                    let t = intersect_t(prev.1, curr.1);
                    let new_pt = lerp_pt(prev.0, curr.0, t);
                    let mut new_uv = lerp_uv(prev.1, curr.1, t);
                    // Snap the clipped axis to the exact clip line so
                    // downstream consumers see uv exactly at 0 or 1.
                    if axis == 0 {
                        new_uv.x = value;
                    } else {
                        new_uv.y = value;
                    }
                    output.push((new_pt, new_uv));
                }
                output.push(curr);
            } else if prev_in {
                // Edge crosses OUT — emit intersection only.
                let t = intersect_t(prev.1, curr.1);
                let new_pt = lerp_pt(prev.0, curr.0, t);
                let mut new_uv = lerp_uv(prev.1, curr.1, t);
                if axis == 0 {
                    new_uv.x = value;
                } else {
                    new_uv.y = value;
                }
                output.push((new_pt, new_uv));
            }
            prev = curr;
            prev_in = curr_in;
        }

        // Swap output → input for the next clip stage.
        std::mem::swap(&mut input, &mut output);
    }

    input
}

/// Push a new neighbor onto the BFS queue, dedupe against existing
/// entries, inherit the parent's projection matrix + re-initialize
/// cull/clamp angles from the decal definition, then build the
/// neighbor's projection. If the cull-cone test rejects the fold,
/// undo the push.
fn try_push_neighbor(
    fragment_builder: &mut DecalMeshFragmentBuilder,
    parent_projection: &RealMatrix4x3,
    parent_flags: u32,
    fold: &Fold,
    parent_index: i32,
    ctx: &DecalBuildContext,
) {
    let count = fragment_builder.neighbor_surface_count;
    if count < 0 || count as usize >= K_MAX_NEIGHBOR_SURFACES {
        return;
    }
    // Dedupe: skip if (fold.bsp_index, fold.surface_index) already in queue.
    for existing in fragment_builder.neighbor_surfaces[..count as usize].iter() {
        if existing.fold.surface_index == fold.surface_index
            && existing.fold.bsp_index == fold.bsp_index
        {
            return;
        }
    }
    let new_idx = count as usize;
    // Engine flag computation (verbatim from `c_decal::build_mesh_fragment_recursive
    // @ 0x18039C2D0` lines 457-465):
    //   flags = (parent.flags & ~3) | 1                  // clear bits 0,1; set INITIALIZED
    //   flags = handed ? (flags | 8) : (flags & ~8)      // set/clear LEFT_HANDED
    //   flags |= 0x10                                    // set NEEDS_RENORMALIZE
    // Net mask: clear bits 0,1,3; preserve 2 (FOLDED), 4+; force 0+3+4.
    let mut new_flags = (parent_flags & !(flag::INITIALIZED | flag::BUILT)) | flag::INITIALIZED;
    new_flags = if ctx.left_handed {
        new_flags | flag::LEFT_HANDED
    } else {
        new_flags & !flag::LEFT_HANDED
    };
    new_flags |= flag::NEEDS_RENORMALIZE;
    let new_neighbor = NeighborSurface {
        parent_index,
        fold: *fold,
        projection_builder: DecalProjectionBuilder {
            projection: *parent_projection,
            cull_angle_radians: ctx.cull_angle_radians,
            cull_angle_cos: ctx.cull_angle_radians.cos(),
            clamp_angle_radians: ctx.clamp_angle_radians,
            clamp_angle_cos: ctx.clamp_angle_radians.cos(),
            flags: new_flags,
        },
    };
    fragment_builder.neighbor_surfaces[new_idx] = new_neighbor;
    fragment_builder.neighbor_surface_count += 1;

    // Build the projection in place. If cull-cone rejects, undo the push.
    let built = fragment_builder.neighbor_surfaces[new_idx]
        .projection_builder
        .build_projection(fold);
    if !built {
        fragment_builder.neighbor_surface_count -= 1;
    }
}

/// UV computation: project world position through inverse of the
/// projection matrix, then bias by texture_scale + 0.5 offset.
/// Engine: `mesh_builder.work_vertex_buffer[slot].texcoord.{x,y} =
///   (texture_position.{y,z} / (2 * texture_scale.{x,y})) + 0.5`.
fn compute_texcoord(
    projection: &RealMatrix4x3,
    point: RealPoint3d,
    texture_scale_x: f32,
    texture_scale_y: f32,
) -> RealPoint2d {
    let texture_position = matrix4x3_inverse_transform_point(projection, point);
    RealPoint2d {
        x: texture_position.y / (2.0 * texture_scale_x) + 0.5,
        y: texture_position.z / (2.0 * texture_scale_y) + 0.5,
    }
}

/// Mirror of `matrix4x3_inverse_transform_point @ 0x1802C4CA0`.
/// Subtracts translation, scale-corrects with a 1e-4 epsilon guard,
/// then transforms by the basis transpose. Equivalent to applying
/// the inverse of an orthonormal-scaled affine transform.
pub(crate) fn matrix4x3_inverse_transform_point(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    let mut dx = p.x - m.position.x;
    let mut dy = p.y - m.position.y;
    let mut dz = p.z - m.position.z;
    let scale = m.scale;
    if scale != 1.0 {
        // Engine clamps |scale| >= 1e-4 to avoid division blow-up.
        const EPS: f32 = 0.000099999997;
        let safe_scale = if scale < 0.0 {
            if scale > -EPS {
                -EPS
            } else {
                scale
            }
        } else {
            if scale <= EPS {
                EPS
            } else {
                scale
            }
        };
        let inv = 1.0 / safe_scale;
        dx *= inv;
        dy *= inv;
        dz *= inv;
    }
    RealPoint3d {
        x: dy * m.forward.j + dx * m.forward.i + dz * m.forward.k,
        y: dy * m.left.j + dx * m.left.i + dz * m.left.k,
        z: dy * m.up.j + dx * m.up.i + dz * m.up.k,
    }
}

/// Mirror of `line_segment_intersects_unit_texture_tile @ 0x18039EB80`.
/// True when the line segment from `start` to `end` (in projection
/// UV space) passes within 0.5 of the texture-tile origin (0.5, 0.5).
/// This is the gate the engine uses to decide whether a polygon edge
/// has "escaped" the decal's projected footprint, which means the
/// neighbor surface across that edge needs to be considered for the
/// BFS expansion.
fn line_segment_intersects_unit_texture_tile(
    start: &RealPoint2d,
    end: &RealPoint2d,
) -> bool {
    // Recenter on tile origin (0.5, 0.5) → segment endpoints in [-0.5, 0.5] coords.
    let sx = start.x - 0.5;
    let sy = start.y - 0.5;
    let ex = end.x - 0.5;
    let ey = end.y - 0.5;
    let dx = ex - sx;
    let dy = ey - sy;
    let len_sq = dx * dx + dy * dy;
    let mut t = if len_sq <= 0.0 {
        0.0
    } else {
        // Engine uses a sign-of-product gate to pick the bisector
        // direction for projection: when dx*dy < 0 the perpendicular
        // foot uses (s.y - s.x), (dy - dx); otherwise (s.y + s.x),
        // (dy + dx).
        let (num, denom) = if dy * dx < 0.0 {
            (sy - sx, dy - dx)
        } else {
            (sy + sx, dy + dx)
        };
        if denom != 0.0 {
            -(num / denom)
        } else {
            0.0
        }
    };
    if t <= 0.0 {
        t = 0.0;
    } else if t >= 1.0 {
        t = 1.0;
    }
    let one_minus_t = 1.0 - t;
    let x = (one_minus_t * sx + t * ex).abs();
    let y = (one_minus_t * sy + t * ey).abs();
    let max_axis = if x > y { x } else { y };
    max_axis < 0.5
}
