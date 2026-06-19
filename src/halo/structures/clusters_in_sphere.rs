//! Engine-faithful port of the structure cluster-sphere + cluster-distance
//! subsystem consumed by `c_atmosphere_fog_interface::compute_cluster_weights`
//! (the SP multi-cluster atmosphere blend).
//!
//! Ports (decompiled from `halo3_dllcache_play.dll`):
//! - `structure_clusters_in_sphere @ 0x1803B8650` — a portal-BFS that gathers
//!   every cluster reachable from the starting cluster through portals the
//!   sphere `(position, radius)` intersects.
//! - `sphere_intersects_cluster_portal_internal @ 0x1803BA3C0` +
//!   `convex_hull2d_test_circle @ 0x1804E0830` — sphere-vs-portal test.
//! - `structure_compute_cluster_distances @ 0x1803B9750` →
//!   `structure_find_closest_cluster_distance_by_seams @ 0x1803B94D0` →
//!   `structure_distance_to_closest_portal @ 0x1803B9380` →
//!   `structure_distance_to_portal @ 0x1803BA1F0` — per-cluster distance.
//!
//! ## Cross-BSP seams
//! The engine seeds/expands the walk via
//! `structure_seams_connected_cluster_references_get`, which reads the
//! (currently undecoded) `structure_seams` tag. On a **single-BSP** map a
//! cluster has no seam mappings, so that call returns just the input cluster
//! and the seam expansion is the identity — making this within-BSP port
//! byte-faithful for every single-BSP scenario (all shipping MP maps + most
//! SP). Multi-BSP seam traversal (the `structure_seams` decoder) is a
//! documented follow-up; until then the gather stays within `start.bsp_index`.

use blam_tags::math::RealPlane3d;
use blam_tags::structure_bsp::{BspCluster, BspClusterPortal};

/// Resolve a cluster portal's plane equation. Cluster portals store a
/// non-negative `plane_index` directly into `collision_bsp.planes` (the
/// sign-bit-negation convention is a BSP3D-leaf thing, not used here — the
/// engine's `sphere_intersects_cluster_portal_internal` /
/// `structure_distance_to_portal` index with `>= 0 && < count` and no negate).
#[inline]
fn portal_plane(portal: &BspClusterPortal, planes: &[RealPlane3d]) -> Option<RealPlane3d> {
    let idx = portal.plane_index;
    if idx < 0 {
        return None;
    }
    planes.get(idx as usize).copied()
}

/// `convex_hull2d_test_circle @ 0x1804E0830` — does a circle `(center, radius)`
/// intersect the convex polygon `poly`? The engine assumes CCW winding and
/// reports "separated" when the center lies strictly outside an edge by more
/// than `radius`. We make it **winding-robust** (multiply the edge cross by the
/// polygon's signed-area sign) so the predicate is identical regardless of the
/// projection handedness — this replaces the engine's
/// `global_projection3d_mappings` sign table with an equivalent result.
fn circle_intersects_convex_polygon(poly: &[[f32; 2]], center: [f32; 2], radius: f32) -> bool {
    let n = poly.len();
    if n == 0 {
        return true; // engine returns 1 for count <= 0
    }
    // Winding sign from the signed area (CCW > 0).
    let mut area2 = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    let w = if area2 >= 0.0 { 1.0 } else { -1.0 };
    let r2 = radius * radius;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let dpx = center[0] - a[0];
        let dpy = center[1] - a[1];
        let ex = b[0] - a[0];
        let ey = b[1] - a[1];
        let len2 = ex * ex + ey * ey;
        // Engine cross = (p - a) × edge = dpx*ey - dpy*ex; *w for winding.
        let cross = (dpx * ey - dpy * ex) * w;
        if len2 != 0.0 && cross > 0.0 && cross * cross > len2 * r2 {
            return false; // separating edge → circle entirely outside
        }
    }
    true
}

/// `sphere_intersects_cluster_portal_internal @ 0x1803BA3C0`. Plane-reach test
/// → centroid bounding-sphere test → project onto the plane (dropping the
/// dominant normal axis) → circle-vs-portal-polygon test with the intersection
/// circle radius `sqrt(radius² − planeDist²)`.
fn sphere_intersects_cluster_portal(
    portal: &BspClusterPortal,
    planes: &[RealPlane3d],
    point: [f32; 3],
    radius: f32,
) -> bool {
    let plane = match portal_plane(portal, planes) {
        Some(p) => p,
        None => return false,
    };
    let n = [plane.i, plane.j, plane.k];
    // Signed distance point → plane (v19).
    let pd = n[0] * point[0] + n[1] * point[1] + n[2] * point[2] - plane.d;
    if pd.abs() >= radius {
        return false;
    }
    // Centroid bounding-sphere reject.
    let c = [portal.centroid.x, portal.centroid.y, portal.centroid.z];
    let dc = [c[0] - point[0], c[1] - point[1], c[2] - point[2]];
    let dc2 = dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2];
    let rr = radius + portal.bounding_radius;
    if dc2 >= rr * rr {
        return false;
    }
    // Dominant axis = argmax(|n.x|,|n.y|,|n.z|) with the engine's tie rule.
    let an = [n[0].abs(), n[1].abs(), n[2].abs()];
    let axis = if an[2] >= an[0] && an[2] >= an[1] {
        2usize
    } else if an[1] >= an[0] {
        1
    } else {
        0
    };
    let drop = |v: [f32; 3]| -> [f32; 2] {
        match axis {
            0 => [v[1], v[2]],
            1 => [v[0], v[2]],
            _ => [v[0], v[1]],
        }
    };
    // Project the point onto the plane along the normal (proj = point − n·pd).
    let proj = [point[0] - n[0] * pd, point[1] - n[1] * pd, point[2] - n[2] * pd];
    let center2 = drop(proj);
    let poly: Vec<[f32; 2]> = portal
        .vertices
        .iter()
        .map(|p| drop([p.x, p.y, p.z]))
        .collect();
    // Radius of the sphere's intersection circle with the plane.
    let r2 = (radius * radius - pd * pd).max(0.0).sqrt();
    circle_intersects_convex_polygon(&poly, center2, r2)
}

/// `structure_distance_to_portal @ 0x1803BA1F0`. Distance from `point` to the
/// portal's bounding disk (centroid + `bounding_radius` in the portal plane).
fn distance_to_portal(portal: &BspClusterPortal, planes: &[RealPlane3d], point: [f32; 3]) -> f32 {
    let plane = match portal_plane(portal, planes) {
        Some(p) => p,
        None => return f32::MAX,
    };
    let n = [plane.i, plane.j, plane.k];
    let d = [
        point[0] - portal.centroid.x,
        point[1] - portal.centroid.y,
        point[2] - portal.centroid.z,
    ];
    let nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    let dn = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    let t = if nn > 0.0 { dn / nn } else { 0.0 };
    // parallel = n·t, perpendicular = d − parallel; `along` = n·parallel = dn.
    let par = [n[0] * t, n[1] * t, n[2] * t];
    let perp = [d[0] - par[0], d[1] - par[1], d[2] - par[2]];
    let perp2 = perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2];
    let along = dn;
    let br = portal.bounding_radius;
    if perp2 > br * br {
        let pl = perp2.sqrt();
        ((pl - br) * (pl - br) + along * along).sqrt()
    } else {
        along.abs()
    }
}

/// `structure_distance_to_closest_portal @ 0x1803B9380` — min portal distance
/// over the cluster's portals.
fn distance_to_closest_portal(
    cluster: &BspCluster,
    portals: &[BspClusterPortal],
    planes: &[RealPlane3d],
    point: [f32; 3],
) -> f32 {
    let mut best = f32::MAX;
    for &pi in &cluster.portals {
        if pi < 0 {
            continue;
        }
        if let Some(portal) = portals.get(pi as usize) {
            best = best.min(distance_to_portal(portal, planes, point));
        }
    }
    best
}

/// `structure_clusters_in_sphere @ 0x1803B8650` (within-BSP). Portal-BFS from
/// `start_cluster`, crossing a portal only when the sphere `(point, radius)`
/// intersects it. Returns the gathered cluster indices (within the BSP),
/// capped at `max_count`, in BFS pop order (start cluster first).
pub fn clusters_in_sphere(
    clusters: &[BspCluster],
    portals: &[BspClusterPortal],
    planes: &[RealPlane3d],
    start_cluster: usize,
    point: [f32; 3],
    radius: f32,
    max_count: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    if start_cluster >= clusters.len() || radius <= 0.0 {
        if start_cluster < clusters.len() {
            out.push(start_cluster);
        }
        return out;
    }
    let mut marked = vec![false; clusters.len()];
    // Seed = seam-connected-with-self → just the start cluster (single BSP).
    let mut stack = vec![start_cluster];
    marked[start_cluster] = true;
    while let Some(c) = stack.pop() {
        if out.len() >= max_count {
            break;
        }
        out.push(c);
        let cluster = match clusters.get(c) {
            Some(cl) => cl,
            None => continue,
        };
        for &pi in &cluster.portals {
            if pi < 0 {
                continue;
            }
            let portal = match portals.get(pi as usize) {
                Some(p) => p,
                None => continue,
            };
            // Neighbour across the portal: cluster_indices = [back, front].
            let back = portal.back_cluster as i64;
            let front = portal.front_cluster as i64;
            let neighbour = if back == c as i64 { front } else { back };
            if neighbour < 0 || neighbour as usize >= clusters.len() {
                continue;
            }
            let nb = neighbour as usize;
            if marked[nb] {
                continue;
            }
            if sphere_intersects_cluster_portal(portal, planes, point, radius) {
                marked[nb] = true;
                stack.push(nb);
            }
        }
    }
    out
}

/// `structure_compute_cluster_distances @ 0x1803B9750` (within-BSP, no seams).
/// Distance for the starting cluster is 0; every other gathered cluster gets
/// `structure_distance_to_closest_portal`.
pub fn compute_cluster_distances(
    clusters: &[BspCluster],
    portals: &[BspClusterPortal],
    planes: &[RealPlane3d],
    start_cluster: usize,
    gathered: &[usize],
    point: [f32; 3],
) -> Vec<f32> {
    gathered
        .iter()
        .map(|&c| {
            if c == start_cluster {
                0.0
            } else if let Some(cluster) = clusters.get(c) {
                distance_to_closest_portal(cluster, portals, planes, point)
            } else {
                f32::MAX
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::math::RealPoint3d;

    fn pt(x: f32, y: f32, z: f32) -> RealPoint3d {
        RealPoint3d { x, y, z }
    }

    #[test]
    fn circle_inside_square_intersects() {
        // Unit square CCW.
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(circle_intersects_convex_polygon(&sq, [0.5, 0.5], 0.1));
        // Far outside, small radius → no intersection.
        assert!(!circle_intersects_convex_polygon(&sq, [5.0, 0.5], 0.1));
        // Just outside but radius reaches the edge → intersects.
        assert!(circle_intersects_convex_polygon(&sq, [1.05, 0.5], 0.1));
    }

    #[test]
    fn circle_test_is_winding_robust() {
        let ccw = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        for c in [[0.5_f32, 0.5], [5.0, 0.5], [1.05, 0.5]] {
            assert_eq!(
                circle_intersects_convex_polygon(&ccw, c, 0.1),
                circle_intersects_convex_polygon(&cw, c, 0.1),
                "winding must not change the result at {c:?}"
            );
        }
    }

    #[test]
    fn distance_to_portal_within_disk_is_plane_distance() {
        // Portal in the z=0 plane (normal +z), centroid at origin, radius 2.
        let planes = vec![RealPlane3d { i: 0.0, j: 0.0, k: 1.0, d: 0.0 }];
        let portal = BspClusterPortal {
            back_cluster: 0,
            front_cluster: 1,
            plane_index: 0,
            centroid: pt(0.0, 0.0, 0.0),
            bounding_radius: 2.0,
            flags: Default::default(),
            vertices: vec![pt(-1.0, -1.0, 0.0), pt(1.0, -1.0, 0.0), pt(1.0, 1.0, 0.0), pt(-1.0, 1.0, 0.0)],
        };
        // Point straight above within the disk → distance = |z|.
        let d = distance_to_portal(&portal, &planes, [0.0, 0.0, 3.0]);
        assert!((d - 3.0).abs() < 1e-4, "got {d}");
        // Point off to the side beyond the disk → sqrt((perp-br)² + along²).
        let d2 = distance_to_portal(&portal, &planes, [5.0, 0.0, 3.0]);
        let expected = (((5.0_f32 - 2.0).powi(2)) + 9.0).sqrt();
        assert!((d2 - expected).abs() < 1e-4, "got {d2} want {expected}");
    }
}
