//! Mirror of the portal-hull / 2D convex polygon math from
//! `Ares/source/visibility/visibility_portal_traversal.cpp:65-94` +
//! `Ares/source/math/convex_hulls.h` consumed by the projections
//! walker (Phase E).
//!
//! Sub-pieces:
//! - [`PortalHull`] (32B) — basis-space portal polygon (bounds +
//!   point list).
//! - [`TransformedPortal`] (48B) — cached portal projection: the
//!   transformed polygon + facing classification + nearest_z.
//! - [`portal_hulls_intersect`] @ 0x180509EF0 — convex-hull
//!   intersection wrapper.
//! - [`convex_hull2d_intersect_new`] @ 0x1804E0DC0 — Sutherland-
//!   Hodgman edge-by-edge clipping.
//! - [`convex_polygon2d_clip_to_plane`] @ 0x1804E4820 — single-edge
//!   half-space clip (the inner workhorse).

use blam_tags::math::{RealPlane2d, RealPlane3d, RealPoint2d, RealPoint3d};

use crate::halo::visibility::RealRectangle2d;

// =============================================================================
// Constants — Ares `visibility_portal_traversal.cpp:46-51`
// =============================================================================

pub const MAXIMUM_POINTS_PER_PORTAL_HULL: usize = 128;
pub const MAXIMUM_WORKING_PORTALS: usize = 256;
pub const AVERAGE_POINTS_PER_PORTAL_HULL: usize = 20;
pub const MAXIMUM_WORKING_PORTAL_STACK_POINTS: usize = 5120;

/// `e_portal_facing` (Ares `visibility_portal_traversal.cpp:54-61`).
/// Result of `compute_portal_facing` — which side of the projection
/// plane a portal lies on (or whether it's degenerate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EPortalFacing {
    None = 0,
    Below = 1,
    Above = 2,
    Degenerate = 3,
}

impl Default for EPortalFacing {
    fn default() -> Self {
        Self::None
    }
}

// =============================================================================
// `s_portal_hull` (32B)
//
// In the engine, `points` is a raw `real_point2d *` into a stack
// allocator (`s_working_portal_stack::point_stack`). For our port we
// keep it as a heap `Vec<RealPoint2d>` for simplicity — the per-frame
// arena allocator pattern can be added later if perf demands it.
// Engine size includes a stable 32B layout for the C struct; we
// preserve via an inline `_pad` to keep cross-references readable.
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct PortalHull {
    pub bounds: RealRectangle2d,    // 0x0  (16B)
    pub point_count: i16,           // 0x10
    // Engine: real_point2d *points @ 0x18. We store inline because
    // we don't own the engine's stack allocator pattern — heap
    // allocation per hull is fine for the per-frame walker (each
    // working portal owns one).
    pub points: Vec<RealPoint2d>,
}

impl PortalHull {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bounds: RealRectangle2d::default(),
            point_count: 0,
            points: Vec::with_capacity(cap),
        }
    }

    /// Refresh `bounds` from the current points list.
    pub fn recompute_bounds(&mut self) {
        if self.points.is_empty() {
            self.bounds = RealRectangle2d::default();
            return;
        }
        let mut bounds = RealRectangle2d {
            x0: f32::INFINITY,
            x1: f32::NEG_INFINITY,
            y0: f32::INFINITY,
            y1: f32::NEG_INFINITY,
        };
        for p in &self.points {
            if p.x < bounds.x0 { bounds.x0 = p.x; }
            if p.x > bounds.x1 { bounds.x1 = p.x; }
            if p.y < bounds.y0 { bounds.y0 = p.y; }
            if p.y > bounds.y1 { bounds.y1 = p.y; }
        }
        self.bounds = bounds;
    }
}

// =============================================================================
// `s_transformed_portal` (48B)
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct TransformedPortal {
    /// Copied from the source `BspClusterPortal.flags` (one-way / door /
    /// no-way / AI sound occlusion). Engine stores it as an int @ 0x0.
    pub flags: blam_tags::Flags<blam_tags::structure_bsp::StructureBspClusterPortalFlags, u32>, // 0x0
    pub facing: EPortalFacing,        // 0x4 (1B)
    // 1B padding @ 0x5
    pub cluster_indices: [i16; 2],    // 0x6 (back, front)
    // 4B padding @ 0xA
    pub nearest_z: f32,               // 0xC
    pub hull: PortalHull,             // 0x10
}

// =============================================================================
// `convex_polygon2d_clip_to_plane_cpp @ 0x1804E4820`
//
// Single-edge Sutherland-Hodgman clip. Classify each input point as
// IN / OUT / BOUNDARY relative to `plane`; walk edges and emit IN
// points + intersection points where edges cross. Returns the new
// vertex count.
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InOutFlag {
    In,       // point on positive side (inside half-space)
    Out,      // point on negative side (outside)
    Boundary, // point on the plane
}

/// `convex_polygon2d_clip_to_plane_cpp @ 0x1804E4820`. Clip a convex
/// polygon to one half-space. Returns the output vertex count
/// (writes into `outputs`). Engine convention: positive-side points
/// (`signed_dist > +eps`) are kept.
pub fn convex_polygon2d_clip_to_plane(
    points: &[RealPoint2d],
    plane: &RealPlane2d,
    outputs: &mut [RealPoint2d],
    epsilon: f32,
) -> usize {
    let count = points.len();
    if count == 0 {
        return 0;
    }
    debug_assert!(
        outputs.len() >= count + 1,
        "clip_to_plane: outputs must have room for count+1"
    );

    // Classify each point.
    let mut flags: [InOutFlag; MAXIMUM_POINTS_PER_PORTAL_HULL] =
        [InOutFlag::Boundary; MAXIMUM_POINTS_PER_PORTAL_HULL];
    let mut in_count = 0;
    let mut out_count = 0;
    for i in 0..count {
        let p = points[i];
        let d = plane.i * p.x + plane.j * p.y - plane.d;
        if d > epsilon {
            flags[i] = InOutFlag::In;
            in_count += 1;
        } else if d < -epsilon {
            flags[i] = InOutFlag::Out;
            out_count += 1;
        } else {
            flags[i] = InOutFlag::Boundary;
        }
    }

    // No OUT points → polygon entirely on inside; copy verbatim.
    if out_count == 0 {
        outputs[..count].copy_from_slice(points);
        return count;
    }
    // No IN points → entirely outside (engine returns 0 here too;
    // the loop body produces no output).
    if in_count == 0 {
        return 0;
    }

    // Walk edges, emit IN points + intersection points.
    let max_outputs = outputs.len();
    let mut out_n = 0usize;
    for i in 0..count {
        let prev = if i == 0 { count - 1 } else { i - 1 };
        let cur = points[i];
        let prev_p = points[prev];
        let f_cur = flags[i];
        let f_prev = flags[prev];

        // Transition into the IN region: emit intersection point.
        if f_cur == InOutFlag::Out && f_prev == InOutFlag::In {
            // Edge from prev (IN) to cur (OUT) — emit intersection.
            let dx = cur.x - prev_p.x;
            let dy = cur.y - prev_p.y;
            let denom = plane.i * dx + plane.j * dy;
            if denom.abs() > epsilon && out_n < max_outputs {
                let t = (plane.i * cur.x + plane.j * cur.y - plane.d) / denom;
                outputs[out_n] = RealPoint2d {
                    x: cur.x - dx * t,
                    y: cur.y - dy * t,
                };
                out_n += 1;
            }
        } else if f_cur == InOutFlag::In && f_prev == InOutFlag::Out {
            // Edge from prev (OUT) to cur (IN) — emit intersection
            // before the IN point itself.
            let dx = prev_p.x - cur.x;
            let dy = prev_p.y - cur.y;
            let denom = plane.i * dx + plane.j * dy;
            if denom.abs() > epsilon && out_n < max_outputs {
                let t = (plane.i * prev_p.x + plane.j * prev_p.y - plane.d) / denom;
                outputs[out_n] = RealPoint2d {
                    x: prev_p.x - dx * t,
                    y: prev_p.y - dy * t,
                };
                out_n += 1;
            }
        }

        // Always emit IN and BOUNDARY points; skip OUT points.
        if f_cur != InOutFlag::Out && out_n < max_outputs {
            outputs[out_n] = cur;
            out_n += 1;
        }
    }

    out_n
}

// =============================================================================
// `convex_hull2d_intersect_new @ 0x1804E0DC0`
//
// Clip polygon Q against each edge of polygon P (Sutherland-Hodgman).
// Ping-pongs between two scratch buffers; final clip writes to
// `result`.
// =============================================================================

/// `convex_hull2d_intersect_new`. Intersect two convex polygons.
/// Writes the result into `result` (must have room for
/// `MAXIMUM_POINTS_PER_PORTAL_HULL`). Returns the result vertex
/// count, or `None` to mean "P entirely contains Q" (engine returns
/// `-1` in that case).
pub fn convex_hull2d_intersect_new(
    p: &[RealPoint2d],
    q: &[RealPoint2d],
    result: &mut Vec<RealPoint2d>,
    epsilon: f32,
) -> Option<i16> {
    let p_count = p.len();
    if p_count == 0 || q.is_empty() {
        result.clear();
        return Some(0);
    }
    let mut storage: [Vec<RealPoint2d>; 2] = [
        vec![RealPoint2d::default(); MAXIMUM_POINTS_PER_PORTAL_HULL],
        vec![RealPoint2d::default(); MAXIMUM_POINTS_PER_PORTAL_HULL],
    ];
    let mut current_q: Vec<RealPoint2d> = q.to_vec();
    let mut q_count = current_q.len();

    for edge_idx in 0..p_count {
        if q_count == 0 {
            break;
        }
        let prev_idx = if edge_idx == 0 { p_count - 1 } else { edge_idx - 1 };
        let p_cur = p[edge_idx];
        let p_prev = p[prev_idx];

        // Edge vector + perpendicular. Engine uses CCW convention:
        // inward normal = (-edge.y, edge.x) / length.
        let edge_dx = p_cur.x - p_prev.x;
        let edge_neg_dy = p_prev.y - p_cur.y; // = -edge.y
        let length_sq = edge_dx * edge_dx + edge_neg_dy * edge_neg_dy;
        let length = length_sq.sqrt();

        // Output buffer: ping-pong, with last edge writing into result's storage.
        let is_last = edge_idx == p_count - 1;
        let outputs_slot = if is_last {
            // Final iteration: write into result via temp then transfer.
            None
        } else {
            Some(edge_idx & 1)
        };

        if length < 0.000099999997 {
            // Degenerate edge — pass-through.
            match outputs_slot {
                Some(slot) => {
                    storage[slot][..q_count].copy_from_slice(&current_q[..q_count]);
                    current_q = storage[slot].clone();
                }
                None => {
                    result.clear();
                    result.extend_from_slice(&current_q[..q_count]);
                }
            }
        } else {
            let inv_len = 1.0 / length;
            let plane = RealPlane2d {
                i: edge_neg_dy * inv_len, // n.x
                j: edge_dx * inv_len,     // n.y
                d: p_cur.x * (edge_neg_dy * inv_len) + p_cur.y * (edge_dx * inv_len),
            };

            match outputs_slot {
                Some(slot) => {
                    let new_count = convex_polygon2d_clip_to_plane(
                        &current_q[..q_count],
                        &plane,
                        &mut storage[slot],
                        epsilon,
                    );
                    q_count = new_count;
                    current_q = storage[slot].clone();
                }
                None => {
                    // Last edge — write to a scratch and copy to result.
                    let mut final_buf =
                        vec![RealPoint2d::default(); MAXIMUM_POINTS_PER_PORTAL_HULL];
                    let new_count = convex_polygon2d_clip_to_plane(
                        &current_q[..q_count],
                        &plane,
                        &mut final_buf,
                        epsilon,
                    );
                    result.clear();
                    result.extend_from_slice(&final_buf[..new_count]);
                    q_count = new_count;
                }
            }
        }
    }

    // Engine has a defensive `q_count == 0xFFFF` check that triggers
    // "P entirely contains Q" — preserved as `None` even though no
    // path in the engine appears to produce it.
    Some(q_count as i16)
}

// =============================================================================
// `plane3d_clip_polygon @ 0x180193100` — single half-space clip on a
// 3D polygon. Used by `transform_portal` to clip portal vertices
// against the projection's near plane.
// =============================================================================

/// Clip a 3D convex polygon to one half-space. `clip_above=true`
/// keeps points with `signed_dist >= +epsilon`; `clip_above=false`
/// flips the epsilon so points with `signed_dist >= -epsilon` are
/// kept. Writes into `clipped_points`; returns the new vertex count.
pub fn plane3d_clip_polygon(
    plane: &RealPlane3d,
    clip_above: bool,
    epsilon: f32,
    points: &[RealPoint3d],
    clipped_points: &mut [RealPoint3d],
) -> usize {
    if points.is_empty() {
        return 0;
    }
    let eps = if clip_above { epsilon } else { -epsilon };
    let max_count = clipped_points.len();
    let mut out_n = 0usize;

    let signed_dist = |p: &RealPoint3d| -> f32 {
        plane.i * p.x + plane.j * p.y + plane.k * p.z - plane.d
    };

    let mut prev = points.last().unwrap();
    let mut prev_dist = signed_dist(prev) - eps;
    let mut prev_kept = prev_dist >= 0.0;

    for cur in points {
        let cur_dist = signed_dist(cur) - eps;
        let cur_kept = cur_dist >= 0.0;
        if cur_kept != prev_kept {
            if out_n >= max_count {
                return out_n;
            }
            let t = prev_dist / (prev_dist - cur_dist);
            clipped_points[out_n] = RealPoint3d {
                x: prev.x + t * (cur.x - prev.x),
                y: prev.y + t * (cur.y - prev.y),
                z: prev.z + t * (cur.z - prev.z),
            };
            out_n += 1;
        }
        if cur_kept == clip_above {
            if out_n >= max_count {
                return out_n;
            }
            clipped_points[out_n] = *cur;
            out_n += 1;
        }
        prev = cur;
        prev_dist = cur_dist;
        prev_kept = cur_kept;
    }
    out_n
}

// =============================================================================
// `polygon2d_from_bounds @ 0x1801921B0`
// =============================================================================

/// Emit a 4-corner CCW polygon (bottom-left, bottom-right, top-right,
/// top-left) from a 2D AABB. Engine writes 4 vertices to a `points`
/// array; we push into a `Vec` for caller convenience.
pub fn polygon2d_from_bounds(bounds: &RealRectangle2d, points: &mut Vec<RealPoint2d>) -> usize {
    points.clear();
    points.push(RealPoint2d { x: bounds.x0, y: bounds.y0 });
    points.push(RealPoint2d { x: bounds.x1, y: bounds.y0 });
    points.push(RealPoint2d { x: bounds.x1, y: bounds.y1 });
    points.push(RealPoint2d { x: bounds.x0, y: bounds.y1 });
    4
}

// =============================================================================
// `portal_hulls_intersect @ 0x180509EF0`
// =============================================================================

/// Intersect two portal hulls. Returns `true` when the result is a
/// non-degenerate polygon (≥ 3 vertices); engine `bool` return.
pub fn portal_hulls_intersect(
    hull0: &PortalHull,
    hull1: &PortalHull,
    result: &mut PortalHull,
) -> bool {
    let v5 = convex_hull2d_intersect_new(
        &hull0.points[..hull0.point_count as usize],
        &hull1.points[..hull1.point_count as usize],
        &mut result.points,
        0.000099999997,
    );
    match v5 {
        Some(count) => {
            result.point_count = count;
        }
        None => {
            // Engine fallback: P entirely contains Q → use hull0 verbatim.
            result.points.clear();
            result.points.extend_from_slice(&hull0.points[..hull0.point_count as usize]);
            result.point_count = hull0.point_count;
        }
    }
    result.recompute_bounds();
    result.point_count >= 3
}
