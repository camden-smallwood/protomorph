//! Verbatim port of `source/structures/structures.{h,cpp}`'s cluster
//! marker + sphere-walk family. Atmosphere Phase 3a per
//! [[project_atmosphere_engine_faithful_port_plan_2026_05_19]].
//!
//! Provides the portal-flood BFS used by `compute_cluster_weights`
//! (Phase 3c) to find all clusters whose authored atmosphere settings
//! contribute to the eye-point's weighted blend.
//!
//! Engine references (Halo 3 MCC dllcache, port 13372):
//! - `structure_cluster_marker_begin @ 0x1803B9A40`
//! - `structure_cluster_marker_end   @ 0x1803B9C40`
//! - `structure_cluster_mark         @ 0x1803B9B80`
//! - `structure_cluster_unmarked     @ 0x1803B9AC0`
//! - `sphere_intersects_cluster_portal_internal @ 0x1803BA3C0`
//! - `convex_hull2d_test_circle      @ 0x1804E0830`
//! - `structure_clusters_in_sphere   @ 0x1803B8650`
//! - `structure_seams_connected_cluster_references_get @ 0x180336EA0`
//!
//! Ares mirror: `source/structures/structures.h:28-38`.

use blam_tags::math::{RealPlane3d, RealPoint2d, RealPoint3d};
use blam_tags::structure_bsp::{BspClusterPortal, StructureBsp};

use crate::halo::structures::clusters::ClusterReference;

// `short[3][2][3]` — projection-axis lookup. Indexed by
// `(projection_axis, sign_of_dominant_component, kind)`.
// kind=0 → 3D component that becomes 2D X.
// kind=1 → 3D component that becomes 2D Y.
// kind=2 → dropped axis (= projection_axis itself).
//
// Verified against `global_projection3d_mappings @ 0x180B255B0` via
// IDA read_bytes (see [`crate::halo::structures::collision_bsp`] for
// the original mirror). Re-declared here to keep the cluster walker
// independent of the collision module.
const PROJECTION_3D_MAPPINGS: [[[usize; 3]; 2]; 3] = [
    [[2, 1, 0], [1, 2, 0]], // axis=0 (drop X)
    [[0, 2, 1], [2, 0, 1]], // axis=1 (drop Y)
    [[1, 0, 2], [0, 1, 2]], // axis=2 (drop Z)
];

// =============================================================================
// `ClusterMarker` — 4096-entry epoch bitset
// =============================================================================
//
// Engine uses a per-thread global `thread_structure_globals[thread_idx]`
// holding `cluster_marker: i32` (the current epoch) + a 4096-entry
// `i32[]` array. Each call to `marker_begin` increments the epoch; a
// slot is "marked" iff `array[idx] == current_epoch`. This makes
// "clearing" O(1) per BFS run.
//
// We translate to a stack-allocated struct that the caller threads
// through the BFS. The semantics match the engine — same wraparound
// behavior, same 4096-entry capacity (16 BSPs × 256 clusters/BSP) —
// but Rust's borrow checker enforces the begin/end invariant for free
// (you can't share a `&mut ClusterMarker` across nested walkers).

/// 4096-entry epoch table mirroring `cluster_marker_array` in
/// `s_structure_globals`. Indexed by `(bsp_index << 8) | cluster_index`.
pub struct ClusterMarker {
    /// Epoch values; a slot is marked iff `slots[idx] == epoch`.
    slots: Box<[i32; 4096]>,
    /// Current epoch; bumped by [`begin`](Self::begin).
    epoch: i32,
}

impl Default for ClusterMarker {
    fn default() -> Self {
        Self {
            slots: Box::new([0; 4096]),
            // Engine starts the counter at 0 and bumps to 1 on the
            // first begin. Initial slots are also 0 so without the
            // initial bump every cluster would appear "marked" on
            // first call — match the engine ordering exactly.
            epoch: 0,
        }
    }
}

impl ClusterMarker {
    /// `structure_cluster_marker_begin @ 0x1803B9A40`. Increments the
    /// epoch; subsequent [`mark`](Self::mark) / [`unmarked`](Self::unmarked)
    /// operate against this new epoch.
    pub fn begin(&mut self) {
        // Engine `++v1->cluster_marker`. Wraps naturally on i32 overflow
        // — would require ~2³¹ BFS runs in one process, not a real risk.
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Compute the flat array index from a cluster reference per engine
    /// `v3 = cluster_index + (bsp_index << 8)`. Returns `None` if the
    /// reference is out of the 4096-slot range (engine asserts).
    #[inline]
    fn slot_index(r: ClusterReference) -> Option<usize> {
        if r.bsp_index < 0 || r.cluster_index < 0 {
            return None;
        }
        let idx = ((r.bsp_index as u16) << 8) | (r.cluster_index as u8 as u16);
        if (idx as usize) >= 4096 {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// `structure_cluster_mark @ 0x1803B9B80`. Writes the current epoch
    /// into the slot. Returns `true` iff the slot was previously
    /// unmarked (engine returns `prev != epoch`).
    pub fn mark(&mut self, r: ClusterReference) -> bool {
        let Some(idx) = Self::slot_index(r) else { return false };
        let prev = self.slots[idx];
        self.slots[idx] = self.epoch;
        prev != self.epoch
    }

    /// `structure_cluster_unmarked @ 0x1803B9AC0`. Returns `true` iff
    /// the slot is not equal to the current epoch.
    pub fn unmarked(&self, r: ClusterReference) -> bool {
        let Some(idx) = Self::slot_index(r) else { return true };
        self.slots[idx] != self.epoch
    }
}

// =============================================================================
// `convex_hull2d_test_circle @ 0x1804E0830`
// =============================================================================

/// Returns `true` iff a circle at `center` with `radius` intersects or
/// is contained by the convex polygon `points`. Engine verbatim port.
///
/// Algorithm: for each edge, compute the signed perpendicular distance
/// from the circle center to the edge. If center is on the "outside"
/// (cross > 0) AND the squared perpendicular distance exceeds
/// `radius²`, the circle is fully outside the hull on that edge —
/// return false. Otherwise after walking all edges, return true.
///
/// Empty polygons (`points.is_empty()`) return true (engine: early
/// `if (count <= 0) return 1`).
fn convex_hull2d_test_circle(points: &[RealPoint2d], center: RealPoint2d, radius: f32) -> bool {
    let count = points.len();
    if count == 0 {
        return true;
    }
    let radius_sq = radius * radius;
    for i in 0..count {
        let p0 = points[i];
        let p1 = points[(i + 1) % count];
        // Vector from edge start to circle center.
        let to_c_x = center.x - p0.x;
        let to_c_y = center.y - p0.y;
        // Edge vector.
        let e_x = p1.x - p0.x;
        let e_y = p1.y - p0.y;
        let edge_len_sq = e_x * e_x + e_y * e_y;
        // 2D cross product (to_c × edge). Positive when center is on
        // the right of the edge if edges wind CCW (matches engine
        // `(v10 * v13) - (v11 * v12) > 0`).
        let cross = to_c_x * e_y - to_c_y * e_x;
        // Engine guards against degenerate edges via `edge_len != 0`.
        // Without that guard a zero-length edge would falsely reject.
        if edge_len_sq != 0.0 && cross > 0.0 && cross * cross > edge_len_sq * radius_sq {
            return false;
        }
    }
    true
}

// =============================================================================
// `sphere_intersects_cluster_portal_internal @ 0x1803BA3C0`
// =============================================================================

/// Sphere-vs-cluster-portal intersection. Returns `true` iff the sphere
/// at `point` with `radius` intersects the portal polygon clipped to
/// its supporting plane.
///
/// 4-stage cull, verbatim from engine:
/// 1. **Plane-slab test.** `|distance(point, portal.plane)| < radius`.
/// 2. **Centroid-sphere bound.** `dist(point, portal.centroid)² <
///    (radius + portal.bounding_radius)²`.
/// 3. **Project to 2D** using the dominant axis of the plane normal
///    (`PROJECTION_3D_MAPPINGS`), with sign-based orientation handling
///    so winding stays consistent across both plane orientations.
/// 4. **Hull-circle test.** [`convex_hull2d_test_circle`] on the
///    projected polygon + the projected sphere center, with the
///    circle radius reduced to `sqrt(radius² - plane_dist²)` — the
///    radius of the sphere/plane intersection circle.
pub fn sphere_intersects_cluster_portal_internal(
    portal: &BspClusterPortal,
    bsp_planes: &[RealPlane3d],
    point: RealPoint3d,
    radius: f32,
) -> bool {
    let plane_index = portal.plane_index;
    if plane_index < 0 || (plane_index as usize) >= bsp_planes.len() {
        return false;
    }
    let plane = bsp_planes[plane_index as usize];

    // Stage 1 — plane slab.
    let plane_dist = plane.i * point.x + plane.j * point.y + plane.k * point.z - plane.d;
    if plane_dist.abs() >= radius {
        return false;
    }

    // Stage 2 — centroid sphere bound.
    let dx = portal.centroid.x - point.x;
    let dy = portal.centroid.y - point.y;
    let dz = portal.centroid.z - point.z;
    let centroid_dist_sq = dx * dx + dy * dy + dz * dz;
    let combined_radius = radius + portal.bounding_radius;
    if centroid_dist_sq >= combined_radius * combined_radius {
        return false;
    }

    // Stage 3 — pick projection axis = the dominant component of the
    // plane normal. Engine emits `|nz| < |ny| || |nz| < |nx|` to
    // detect "Z is not the largest" then picks between X and Y; if Z
    // is largest it falls to axis=2.
    let nx_abs = plane.i.abs();
    let ny_abs = plane.j.abs();
    let nz_abs = plane.k.abs();
    let axis: usize = if nz_abs < ny_abs || nz_abs < nx_abs {
        // Z is not the largest; pick between X (0) and Y (1).
        if ny_abs >= nx_abs {
            1
        } else {
            0
        }
    } else {
        2
    };

    // Sign of the dominant component decides projection orientation
    // (so 2D winding stays consistent regardless of which side of the
    // plane the normal faces).
    let dominant = match axis {
        0 => plane.i,
        1 => plane.j,
        _ => plane.k,
    };
    let sign: usize = if dominant > 0.0 { 1 } else { 0 };

    let mapping = &PROJECTION_3D_MAPPINGS[axis][sign];
    let map_x = mapping[0];
    let map_y = mapping[1];

    // Project the sphere center (after shifting it onto the plane) to
    // 2D. Engine computes:
    //   projected_3d = point + plane.normal × (-plane_dist)
    // The sign-flip `^_xmm` in the decompile is `-` applied to plane_dist.
    let proj_x_3d = plane.i * -plane_dist + point.x;
    let proj_y_3d = plane.j * -plane_dist + point.y;
    let proj_z_3d = plane.k * -plane_dist + point.z;
    let projected_3d = [proj_x_3d, proj_y_3d, proj_z_3d];
    let center_2d = RealPoint2d {
        x: projected_3d[map_x],
        y: projected_3d[map_y],
    };

    // Project all polygon vertices to 2D using the same mapping.
    let mut polygon_2d: Vec<RealPoint2d> = Vec::with_capacity(portal.vertices.len());
    for v in &portal.vertices {
        let coords = [v.x, v.y, v.z];
        polygon_2d.push(RealPoint2d {
            x: coords[map_x],
            y: coords[map_y],
        });
    }

    // Stage 4 — circle of intersection between sphere & plane has
    // radius `sqrt(radius² - plane_dist²)`.
    let effective_radius_sq = radius * radius - plane_dist * plane_dist;
    let effective_radius = if effective_radius_sq > 0.0 {
        effective_radius_sq.sqrt()
    } else {
        0.0
    };

    convex_hull2d_test_circle(&polygon_2d, center_2d, effective_radius)
}

// =============================================================================
// `structure_seams_connected_cluster_references_get @ 0x180336EA0` — STUB
// =============================================================================
//
// Engine reads `g_structure_seam_globals.seam_mappings[]` (populated
// from `scenario.structure_seams` — a separate tag class) and walks
// the seam graph to find clusters reachable across BSP boundaries.
//
// Our blam-tags parser doesn't yet read `cluster.seam_indices` or the
// `.structure_seams` tag, so for single-BSP scenarios (every MP map
// we ship today: guardian, riverworld, chill, etc.) the engine path
// returns just `[initial]` (when `include_initial` is true) or empty.
//
// Multi-BSP campaign maps require the full seam graph — port deferred
// to Phase 3b per the master plan.

/// Stub: returns just the initial reference when `include_initial` is
/// true. **Single-BSP scenarios only.** Multi-BSP cross-seam
/// connectivity deferred until `scenario.structure_seams` parsing +
/// `cluster.seam_indices` extraction land.
///
/// Engine signature: returns `true` iff `out` is non-empty after the
/// call.
fn structure_seams_connected_cluster_references_get(
    initial: ClusterReference,
    include_initial: bool,
    out: &mut Vec<ClusterReference>,
) -> bool {
    if include_initial {
        out.push(initial);
    }
    !out.is_empty()
}

// =============================================================================
// `structure_clusters_in_sphere @ 0x1803B8650`
// =============================================================================

/// Engine `structure_clusters_in_sphere`. Returns the count of unique
/// `ClusterReference`s within `radius` of `position` via portal-flood
/// BFS starting from `initial`, written into `out` (capped at
/// `max_count`).
///
/// `bsps_by_scenario_index[i]` is the `StructureBsp` whose
/// `scenario_bsp_index == i`, or `None` for sparse slots. Engine
/// equivalent: `global_structure_bsp_get(bsp_index)`.
///
/// Special cases (mirroring engine):
/// - `initial.bsp_index < 0` → returns 0.
/// - `radius <= 0` → returns just the seam-connected initial set
///   (after our stub, that's `[initial]`).
/// - `out` will be appended to, not cleared — callers must clear first
///   if they want fresh results.
///
/// Return value is `min(actual_count, max_count)` per engine's clamp
/// at the bottom of the function. The full unclamped count is also
/// pushed into `out` only up to the cap (engine writes `intersected_references[i]`
/// inside the BFS loop but only when `i < max_count`).
pub fn structure_clusters_in_sphere(
    bsps_by_scenario_index: &[Option<&StructureBsp>],
    initial: ClusterReference,
    position: RealPoint3d,
    radius: f32,
    max_count: usize,
    out: &mut Vec<ClusterReference>,
) -> usize {
    if initial.bsp_index < 0 {
        return 0;
    }

    // Engine: c_static_stack<s_cluster_reference, 32>. Our seam stub
    // pushes at most one entry, so 32 capacity is plenty.
    let mut initial_refs: Vec<ClusterReference> = Vec::with_capacity(32);
    structure_seams_connected_cluster_references_get(initial, true, &mut initial_refs);

    // Radius<=0 fast path — engine drops the BFS entirely and just
    // copies initial_refs into out (up to max_count).
    if radius <= 0.0 {
        let mut count = 0usize;
        for r in &initial_refs {
            if count >= max_count {
                break;
            }
            out.push(*r);
            count += 1;
        }
        return count;
    }

    // ── BFS portal flood ─────────────────────────────────────────
    let mut marker = ClusterMarker::default();
    marker.begin();

    // Engine: c_static_stack<s_cluster_reference, 4096>. 4096 ==
    // maximum total clusters across all BSPs in a scenario, so the
    // stack can never overflow in well-formed maps.
    let mut stack: Vec<ClusterReference> = Vec::with_capacity(4096);

    // Push every initial-set ref (after marking).
    for r in &initial_refs {
        if marker.mark(*r) {
            stack.push(*r);
        }
    }

    // Engine tracks the visit count separately from the output write
    // count — `out_actual_cluster_count` is bumped per visit even
    // when the output is full. Mirror that so callers can detect
    // overflow via the return-vs-max comparison.
    let mut actual_count = 0usize;
    let mut emitted = 0usize;

    while let Some(here) = stack.pop() {
        // Look up BSP + cluster. Engine: `global_structure_bsp_get`
        // + indexed read into the bsp's `clusters` block.
        let Some(bsp) = bsps_by_scenario_index
            .get(here.bsp_index as usize)
            .and_then(|o| *o)
        else {
            // BSP not active — skip but still consume the visit so
            // emit/cap accounting matches engine.
            continue;
        };
        let Some(cluster) = bsp.clusters.get(here.cluster_index as usize) else {
            continue;
        };
        // Cluster portal planes live in the shared bsp3d plane table
        // (engine: extracted from `structure_bsp->resource_interface`).
        // Our blam-tags representation is `StructureBsp::collision_bsp.planes`.
        let bsp_planes: &[RealPlane3d] = bsp
            .collision_bsp
            .as_ref()
            .map(|c| c.planes.as_slice())
            .unwrap_or(&[]);

        // Engine writes `intersected_references[v69] = v24` BEFORE
        // checking the output cap. We cap on emit but always count.
        if emitted < max_count {
            out.push(here);
            emitted += 1;
        }
        actual_count += 1;

        // ── Seam fanout (engine: `if (*((int *)v31 + 55) > 0) { ...
        // seams_connected_cluster_references_get(scenario, here, 0,
        // &cluster_refs); ... }`). Our seam stub returns empty when
        // include_initial=false, so this loop is a no-op for now —
        // kept for shape parity with the engine when seams land.
        let mut seam_refs: Vec<ClusterReference> = Vec::with_capacity(32);
        if structure_seams_connected_cluster_references_get(here, false, &mut seam_refs) {
            for sr in seam_refs {
                if marker.mark(sr) {
                    stack.push(sr);
                }
            }
        }

        // ── Portal fanout ──────────────────────────────────────
        for &portal_idx_signed in &cluster.portals {
            let portal_idx = portal_idx_signed as i32;
            if portal_idx < 0 || (portal_idx as usize) >= bsp.cluster_portals.len() {
                continue;
            }
            let portal = &bsp.cluster_portals[portal_idx as usize];

            // Engine: `v50 = portal->cluster_indices[0]; if (v50 == here.cluster_index) v50 = portal->cluster_indices[1];`
            // — picks the cluster on the OTHER side of the portal.
            let portal_back = portal.back_cluster as i32;
            let portal_front = portal.front_cluster as i32;
            let other_cluster = if portal_back == here.cluster_index as i32 {
                portal_front
            } else {
                portal_back
            };

            let next_ref = ClusterReference {
                bsp_index: here.bsp_index,
                cluster_index: other_cluster as i16,
            };

            if !marker.unmarked(next_ref) {
                continue;
            }
            if !sphere_intersects_cluster_portal_internal(
                portal,
                bsp_planes,
                position,
                radius,
            ) {
                continue;
            }
            marker.mark(next_ref);
            stack.push(next_ref);
        }
    }

    // Engine returns `min(actual_count, max_count)`. Note `actual_count`
    // can exceed `max_count` when the BFS visits more clusters than
    // the caller provided room for; callers compare to detect this.
    actual_count.min(max_count)
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::math::RealBounds;
    use blam_tags::structure_bsp::{Bsp3d, BspCluster};

    // Helper to build a minimal cluster (only the fields the walker
    // reads — bounds + portals).
    fn make_cluster(portals: Vec<i16>) -> BspCluster {
        BspCluster {
            bounds_x: RealBounds::default(),
            bounds_y: RealBounds::default(),
            bounds_z: RealBounds::default(),
            scenario_sky_index: -1,
            atmosphere_index: -1,
            camera_fx_index: -1,
            mesh_index: -1,
            flags: blam_tags::Flags::default(),
            portals,
        }
    }

    // Build a portal whose supporting plane is x = plane_x, normal +X.
    // Polygon is the unit square in the YZ plane at x=plane_x.
    fn make_portal_yz_square(plane_index: i32, back: i16, front: i16) -> BspClusterPortal {
        BspClusterPortal {
            back_cluster: back,
            front_cluster: front,
            plane_index,
            centroid: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 }, // set per scenario below
            bounding_radius: std::f32::consts::SQRT_2 * 0.5 + 0.001,
            flags: blam_tags::Flags::default(),
            vertices: vec![
                RealPoint3d { x: 0.0, y: -0.5, z: -0.5 },
                RealPoint3d { x: 0.0, y: 0.5, z: -0.5 },
                RealPoint3d { x: 0.0, y: 0.5, z: 0.5 },
                RealPoint3d { x: 0.0, y: -0.5, z: 0.5 },
            ],
        }
    }

    /// 4 clusters in a row connected by 3 portals at x=1, x=2, x=3.
    ///   C0 [0..1] ──p01──> C1 [1..2] ──p12──> C2 [2..3] ──p23──> C3 [3..4]
    fn build_4cluster_strip() -> StructureBsp {
        let mut bsp = StructureBsp::default();
        // Cluster-portal planes share the bsp3d plane table (engine
        // pulls them from `structure_bsp->resource_interface →
        // bsp3d->planes`). Mirror that by parking the planes inside a
        // synthetic `collision_bsp`.
        let mut collision = Bsp3d::default();
        collision.planes.push(RealPlane3d { i: 1.0, j: 0.0, k: 0.0, d: 1.0 });
        collision.planes.push(RealPlane3d { i: 1.0, j: 0.0, k: 0.0, d: 2.0 });
        collision.planes.push(RealPlane3d { i: 1.0, j: 0.0, k: 0.0, d: 3.0 });
        bsp.collision_bsp = Some(collision);

        let mut p01 = make_portal_yz_square(0, 0, 1);
        p01.vertices = vec![
            RealPoint3d { x: 1.0, y: -0.5, z: -0.5 },
            RealPoint3d { x: 1.0, y: 0.5, z: -0.5 },
            RealPoint3d { x: 1.0, y: 0.5, z: 0.5 },
            RealPoint3d { x: 1.0, y: -0.5, z: 0.5 },
        ];
        p01.centroid = RealPoint3d { x: 1.0, y: 0.0, z: 0.0 };

        let mut p12 = make_portal_yz_square(1, 1, 2);
        p12.vertices = vec![
            RealPoint3d { x: 2.0, y: -0.5, z: -0.5 },
            RealPoint3d { x: 2.0, y: 0.5, z: -0.5 },
            RealPoint3d { x: 2.0, y: 0.5, z: 0.5 },
            RealPoint3d { x: 2.0, y: -0.5, z: 0.5 },
        ];
        p12.centroid = RealPoint3d { x: 2.0, y: 0.0, z: 0.0 };

        let mut p23 = make_portal_yz_square(2, 2, 3);
        p23.vertices = vec![
            RealPoint3d { x: 3.0, y: -0.5, z: -0.5 },
            RealPoint3d { x: 3.0, y: 0.5, z: -0.5 },
            RealPoint3d { x: 3.0, y: 0.5, z: 0.5 },
            RealPoint3d { x: 3.0, y: -0.5, z: 0.5 },
        ];
        p23.centroid = RealPoint3d { x: 3.0, y: 0.0, z: 0.0 };

        bsp.cluster_portals.push(p01);
        bsp.cluster_portals.push(p12);
        bsp.cluster_portals.push(p23);

        bsp.clusters.push(make_cluster(vec![0])); // C0 → portal 0
        bsp.clusters.push(make_cluster(vec![0, 1])); // C1 → 0, 1
        bsp.clusters.push(make_cluster(vec![1, 2])); // C2 → 1, 2
        bsp.clusters.push(make_cluster(vec![2])); // C3 → portal 2
        bsp
    }

    fn cluster_ref(bsp: i16, cluster: i16) -> ClusterReference {
        ClusterReference { bsp_index: bsp, cluster_index: cluster }
    }

    #[test]
    fn cluster_marker_mark_returns_true_first_time_only() {
        let mut m = ClusterMarker::default();
        m.begin();
        let r = cluster_ref(0, 5);
        assert!(m.unmarked(r));
        assert!(m.mark(r), "first mark returns true (was unmarked)");
        assert!(!m.unmarked(r));
        assert!(!m.mark(r), "second mark returns false (already marked)");
    }

    #[test]
    fn cluster_marker_begin_resets_state() {
        let mut m = ClusterMarker::default();
        m.begin();
        m.mark(cluster_ref(0, 0));
        assert!(!m.unmarked(cluster_ref(0, 0)));
        m.begin(); // new walk → all slots logically unmarked again
        assert!(m.unmarked(cluster_ref(0, 0)));
    }

    #[test]
    fn cluster_marker_indexes_by_bsp_and_cluster() {
        let mut m = ClusterMarker::default();
        m.begin();
        // (bsp=0, cluster=5) and (bsp=1, cluster=5) must hash distinctly.
        m.mark(cluster_ref(0, 5));
        assert!(m.unmarked(cluster_ref(1, 5)));
        assert!(!m.unmarked(cluster_ref(0, 5)));
    }

    #[test]
    fn sphere_misses_distant_portal() {
        let bsp = build_4cluster_strip();
        let point = RealPoint3d { x: 0.5, y: 0.0, z: 0.0 };
        // Portal at x=1, sphere center at x=0.5 with radius 0.3 doesn't
        // reach the plane.
        let hit = sphere_intersects_cluster_portal_internal(
            &bsp.cluster_portals[0],
            &bsp.collision_bsp.as_ref().unwrap().planes,
            point,
            0.3,
        );
        assert!(!hit);
    }

    #[test]
    fn sphere_hits_nearby_portal() {
        let bsp = build_4cluster_strip();
        let point = RealPoint3d { x: 0.5, y: 0.0, z: 0.0 };
        // Radius 0.6 reaches the portal plane at x=1.
        let hit = sphere_intersects_cluster_portal_internal(
            &bsp.cluster_portals[0],
            &bsp.collision_bsp.as_ref().unwrap().planes,
            point,
            0.6,
        );
        assert!(hit);
    }

    #[test]
    fn walker_radius_zero_returns_initial_only() {
        let bsp = build_4cluster_strip();
        let bsps: Vec<Option<&StructureBsp>> = vec![Some(&bsp)];
        let mut out = Vec::new();
        let count = structure_clusters_in_sphere(
            &bsps,
            cluster_ref(0, 0),
            RealPoint3d { x: 0.5, y: 0.0, z: 0.0 },
            0.0,
            16,
            &mut out,
        );
        assert_eq!(count, 1);
        assert_eq!(out, vec![cluster_ref(0, 0)]);
    }

    #[test]
    fn walker_radius_just_reaches_first_portal() {
        let bsp = build_4cluster_strip();
        let bsps: Vec<Option<&StructureBsp>> = vec![Some(&bsp)];
        let mut out = Vec::new();
        // Camera in C0 at x=0.5; radius 0.6 hits portal at x=1 → reach C1.
        // Does NOT reach portal at x=2 (would need radius > 1.5).
        let count = structure_clusters_in_sphere(
            &bsps,
            cluster_ref(0, 0),
            RealPoint3d { x: 0.5, y: 0.0, z: 0.0 },
            0.6,
            16,
            &mut out,
        );
        assert_eq!(count, 2);
        assert!(out.contains(&cluster_ref(0, 0)));
        assert!(out.contains(&cluster_ref(0, 1)));
    }

    #[test]
    fn walker_large_radius_visits_all_clusters() {
        let bsp = build_4cluster_strip();
        let bsps: Vec<Option<&StructureBsp>> = vec![Some(&bsp)];
        let mut out = Vec::new();
        let count = structure_clusters_in_sphere(
            &bsps,
            cluster_ref(0, 0),
            RealPoint3d { x: 0.5, y: 0.0, z: 0.0 },
            10.0,
            16,
            &mut out,
        );
        assert_eq!(count, 4);
        for c in 0..4 {
            assert!(out.contains(&cluster_ref(0, c)));
        }
    }

    #[test]
    fn walker_respects_max_count_cap() {
        let bsp = build_4cluster_strip();
        let bsps: Vec<Option<&StructureBsp>> = vec![Some(&bsp)];
        let mut out = Vec::new();
        // Big radius visits all 4, but cap at 2 → return 2 written
        // entries (engine returns min(actual, cap)).
        let count = structure_clusters_in_sphere(
            &bsps,
            cluster_ref(0, 0),
            RealPoint3d { x: 0.5, y: 0.0, z: 0.0 },
            10.0,
            2,
            &mut out,
        );
        assert_eq!(count, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn walker_initial_invalid_bsp_returns_zero() {
        let bsps: Vec<Option<&StructureBsp>> = vec![];
        let mut out = Vec::new();
        let count = structure_clusters_in_sphere(
            &bsps,
            cluster_ref(-1, -1),
            RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            1.0,
            16,
            &mut out,
        );
        assert_eq!(count, 0);
        assert!(out.is_empty());
    }
}
