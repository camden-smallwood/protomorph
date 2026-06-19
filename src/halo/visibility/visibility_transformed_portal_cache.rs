//! Mirror of `s_transformed_portal_cache` (Ares
//! `visibility_portal_traversal.cpp:121-134`, 82,992B in the engine).
//!
//! Per-projection cache that lazily transforms portals into the
//! projection's basis space + caches the result. Each
//! `transform_portal` call:
//!
//! 1. Looks up the portal in the cache (per-BSP × per-portal flag).
//! 2. If cached: returns the previously computed
//!    [`TransformedPortal`].
//! 3. Else: pulls the portal's polygon from the BSP, transforms
//!    vertices into basis space, clips against the projection's
//!    near plane, computes the facing classification + perspective-
//!    projects to 2D, and stores the result in the cache.
//!
//! This is the heart of E2 — Phase E4's recursive walker uses the
//! cache to avoid retransforming a portal each time the BFS visits
//! it from a different parent cluster.

use blam_tags::math::{RealPlane3d, RealPoint2d, RealPoint3d};
use blam_tags::structure_bsp::BspClusterPortal;

use crate::halo::scenario::{LoadedScenario, MAX_BSPS};
use crate::halo::visibility::visibility_portal_activation::{MAX_TRANSFORMED_PORTALS, PortalReference};
use crate::halo::visibility::visibility_portal_hulls::{
    plane3d_clip_polygon, polygon2d_from_bounds, EPortalFacing, PortalHull, TransformedPortal,
    MAXIMUM_POINTS_PER_PORTAL_HULL,
};
use crate::halo::visibility::visibility_projections_and_volumes::matrix4x3_transform_point_pub;
use crate::halo::visibility::{MAX_PORTALS_PER_BSP, PORTAL_FLAG_WORDS_PER_BSP, VisibilityProjection};

// =============================================================================
// `s_transformed_portal_cache` — per-projection portal cache
//
// Engine-faithful flag layout (16 BSPs × 512 portal-bits) for fast
// "already transformed?" check. The cached entries live in a flat
// `actual_transformed_portals` Vec; `transformed_portal_indices`
// maps (bsp, portal) → index into that Vec.
// =============================================================================

/// `s_transformed_portal_cache` (engine 82,992B; ours larger due to
/// `Vec<RealPoint2d>` per cache entry rather than the engine's
/// shared `c_static_aligned_stack_allocator<real_point2d, 5120, 2>`).
/// Behaviorally equivalent.
pub struct STransformedPortalCache {
    /// `clip_plane` — the half-space the input portal polygon is
    /// clipped against before perspective projection. Engine
    /// initializes this to `(0, 0, -1, 0)` (cull anything behind
    /// camera in basis space) at the top of `from_projections`.
    pub clip_plane: RealPlane3d,
    /// Per-BSP × per-portal "already transformed" bit-flags (engine
    /// `c_static_array<c_static_flags<512>, 16>`, 1024B).
    pub transformed_portal_flags: Box<[[u32; PORTAL_FLAG_WORDS_PER_BSP]; MAX_BSPS]>,
    /// `(bsp, portal) → cache slot` lookup (engine
    /// `c_static_array<c_static_array<short, 512>, 16>`, 16384B).
    pub transformed_portal_indices: Box<[[i16; MAX_PORTALS_PER_BSP]; MAX_BSPS]>,
    /// Stack of cached transformed portals — append-only per
    /// projection, cleared on next projection.
    pub actual_transformed_portals: Vec<TransformedPortal>,
}

impl STransformedPortalCache {
    pub fn new() -> Self {
        Self {
            clip_plane: RealPlane3d {
                i: 0.0,
                j: 0.0,
                k: -1.0,
                d: 0.0,
            },
            transformed_portal_flags: Box::new([[0; PORTAL_FLAG_WORDS_PER_BSP]; MAX_BSPS]),
            transformed_portal_indices: Box::new([[0; MAX_PORTALS_PER_BSP]; MAX_BSPS]),
            actual_transformed_portals: Vec::with_capacity(MAX_TRANSFORMED_PORTALS),
        }
    }

    pub fn clear(&mut self) {
        for bsp in self.transformed_portal_flags.iter_mut() {
            *bsp = [0; PORTAL_FLAG_WORDS_PER_BSP];
        }
        self.actual_transformed_portals.clear();
    }

    fn cache_test(&self, portal_ref: PortalReference) -> Option<i16> {
        let bsp = portal_ref.bsp_index();
        let p = portal_ref.portal_index();
        if bsp < 0 || (bsp as usize) >= MAX_BSPS || p < 0 || (p as usize) >= MAX_PORTALS_PER_BSP {
            return None;
        }
        let word = self.transformed_portal_flags[bsp as usize][p as usize >> 5];
        if (word >> (p & 31)) & 1 != 0 {
            Some(self.transformed_portal_indices[bsp as usize][p as usize])
        } else {
            None
        }
    }

    fn cache_set(&mut self, portal_ref: PortalReference, slot: i16) {
        let bsp = portal_ref.bsp_index() as usize;
        let p = portal_ref.portal_index() as usize;
        self.transformed_portal_flags[bsp][p >> 5] |= 1u32 << (p & 31);
        self.transformed_portal_indices[bsp][p] = slot;
    }
}

impl Default for STransformedPortalCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// `transform_portal @ 0x180508FB0`
// =============================================================================

/// Lazy-transform a portal into basis space + clip against the
/// projection's near plane + perspective-project to 2D. Cached on
/// repeat calls. Returns `Some(slot)` for success, `None` if the
/// cache or point pool is full (matches engine "return 0" overflow).
pub fn transform_portal(
    cache: &mut STransformedPortalCache,
    projection: &VisibilityProjection,
    scenario: &LoadedScenario,
    portal_ref: PortalReference,
) -> Option<usize> {
    // Cache hit?
    if let Some(slot) = cache.cache_test(portal_ref) {
        return Some(slot as usize);
    }

    // Overflow guard.
    if cache.actual_transformed_portals.len() >= MAX_TRANSFORMED_PORTALS {
        return None;
    }

    let bsp_index = portal_ref.bsp_index();
    let portal_index = portal_ref.portal_index();
    let bsp = scenario
        .active_bsps
        .iter()
        .find(|b| b.scenario_bsp_index as i16 == bsp_index)?;
    let cluster_portal: &BspClusterPortal =
        bsp.sbsp.cluster_portals.get(portal_index as usize)?;
    // `plane_index` carries a plane_designator-style direction sign bit (bit 31):
    // low 31 bits = index into the planes block, high bit = use the NEGATED plane.
    // Indexing it raw makes any sign-bit-set portal resolve to `None` →
    // zero plane → classified degenerate → never traversed → clusters behind it
    // dropped from the PVS. Mask the index and negate the plane when the bit is
    // set. (-1, the "no plane" default, masks to 0x7FFFFFFF → out of range →
    // zero plane, preserving the genuine no-plane case.)
    let raw_plane_index = cluster_portal.plane_index;
    let mut plane = bsp
        .sbsp
        .collision_bsp
        .as_ref()
        .and_then(|c| c.planes.get((raw_plane_index & 0x7FFF_FFFF) as usize).copied())
        .unwrap_or_default();
    if raw_plane_index < 0 {
        plane.i = -plane.i;
        plane.j = -plane.j;
        plane.k = -plane.k;
        plane.d = -plane.d;
    }

    let slot = cache.actual_transformed_portals.len();

    // Step 1: world-space portal vertices → basis-space.
    let mut basis_points = vec![RealPoint3d::default(); MAXIMUM_POINTS_PER_PORTAL_HULL];
    let n_in = cluster_portal.vertices.len().min(MAXIMUM_POINTS_PER_PORTAL_HULL);
    for i in 0..n_in {
        basis_points[i] =
            matrix4x3_transform_point_pub(&projection.world_to_basis, cluster_portal.vertices[i]);
    }

    // Step 2: clip against `cache.clip_plane` (engine: clip_above=1,
    // epsilon=0.0078125).
    let mut clipped_points = vec![RealPoint3d::default(); MAXIMUM_POINTS_PER_PORTAL_HULL];
    let clipped_count = plane3d_clip_polygon(
        &cache.clip_plane,
        true,
        0.0078125,
        &basis_points[..n_in],
        &mut clipped_points,
    );

    // Step 3: facing classification + origin-in-portal test.
    let mut origin_in_portal_flag = false;
    let facing = compute_portal_facing(
        cache,
        projection,
        cluster_portal,
        &plane,
        clipped_count as i32,
        &mut origin_in_portal_flag,
    );

    // Step 4: build the TransformedPortal entry.
    let mut tp = TransformedPortal::default();
    tp.cluster_indices = [cluster_portal.back_cluster, cluster_portal.front_cluster];
    tp.nearest_z = f32::INFINITY;
    tp.flags = cluster_portal.flags.clone();
    tp.facing = facing;
    tp.hull = PortalHull::with_capacity(MAXIMUM_POINTS_PER_PORTAL_HULL);

    if facing != EPortalFacing::None {
        // Step 5: perspective-project each clipped 3D point to 2D.
        // Engine iterates in different orders depending on facing
        // (Below = reverse, Above = forward). We mirror that to
        // preserve CCW winding for the downstream poly clipper.
        let direction: i32 = if facing == EPortalFacing::Below { -1 } else { 1 };
        let start: i32 = if facing == EPortalFacing::Below {
            (clipped_count as i32) - 1
        } else {
            0
        };

        let mut bounds_x0 = f32::INFINITY;
        let mut bounds_x1 = f32::NEG_INFINITY;
        let mut bounds_y0 = f32::INFINITY;
        let mut bounds_y1 = f32::NEG_INFINITY;
        let mut nearest_z = f32::INFINITY;

        for i in 0..(clipped_count as i32) {
            let src_idx = (start + i * direction) as usize;
            let p = clipped_points[src_idx];
            // Engine: v39 = -1.0 / p.z; emit (x*v39, y*v39).
            // Asserts p.z is sufficiently negative (we silently
            // skip degenerate projections).
            if p.z >= -0.000099999997 {
                continue;
            }
            let inv_neg_z = -1.0 / p.z;
            let x2 = p.x * inv_neg_z;
            let y2 = p.y * inv_neg_z;
            tp.hull.points.push(RealPoint2d { x: x2, y: y2 });
            bounds_x0 = bounds_x0.min(x2);
            bounds_x1 = bounds_x1.max(x2);
            bounds_y0 = bounds_y0.min(y2);
            bounds_y1 = bounds_y1.max(y2);
            // nearest_z = min(-p.z) — engine uses XOR with sign mask
            // (negate float).
            let neg_z = -p.z;
            if neg_z < nearest_z {
                nearest_z = neg_z;
            }
        }
        tp.hull.point_count = tp.hull.points.len() as i16;
        tp.hull.bounds.x0 = bounds_x0;
        tp.hull.bounds.x1 = bounds_x1;
        tp.hull.bounds.y0 = bounds_y0;
        tp.hull.bounds.y1 = bounds_y1;
        tp.nearest_z = nearest_z;

        // Step 6: degenerate-facing case (facing == 3).
        if facing == EPortalFacing::Degenerate {
            if origin_in_portal_flag {
                // Portal contains the projection origin → use the
                // full projection's hull as the portal hull.
                let nz = tp.nearest_z.min(0.015625);
                tp.nearest_z = nz;
                tp.hull.bounds = projection.volume.frustum_bounds;
                tp.hull.point_count = projection.convex_hull_point_count as i16;
                tp.hull.points.clear();
                for i in 0..(projection.convex_hull_point_count as usize) {
                    tp.hull.points.push(projection.convex_hull_points[i]);
                }
            } else {
                // Inflate bounds + emit polygon from inflated bounds.
                tp.hull.bounds.x0 -= 0.00390625;
                tp.hull.bounds.x1 += 0.00390625;
                tp.hull.bounds.y0 -= 0.00390625;
                tp.hull.bounds.y1 += 0.00390625;
                let n = polygon2d_from_bounds(&tp.hull.bounds, &mut tp.hull.points);
                tp.hull.point_count = n as i16;
            }
        }

        // Step 7: far-plane reject — if portal is entirely beyond
        // the projection's far distance, treat as not facing.
        if projection.far_bounded_flag && projection.far_distance < tp.nearest_z {
            tp.facing = EPortalFacing::None;
        }
    }

    cache.actual_transformed_portals.push(tp);
    cache.cache_set(portal_ref, slot as i16);
    Some(slot)
}

// =============================================================================
// `compute_portal_facing @ 0x1805096A0`
// =============================================================================

/// Classify how the portal sits relative to the projection's plane
/// — above (camera in front), below (camera behind), or degenerate
/// (very close / origin inside portal). `None` means the portal
/// shouldn't be traversed.
pub fn compute_portal_facing(
    cache: &STransformedPortalCache,
    projection: &VisibilityProjection,
    portal: &BspClusterPortal,
    plane: &RealPlane3d,
    clipped_point_count: i32,
    projection_origin_in_portal_flag: &mut bool,
) -> EPortalFacing {
    let _ = cache; // engine reads cache.projection here; we already have projection
    let origin = projection.basis_to_world.position;
    let signed_dist = origin.x * plane.i + origin.y * plane.j + origin.z * plane.k - plane.d;
    if signed_dist.abs() > 0.015625 {
        *projection_origin_in_portal_flag = false;
        if clipped_point_count > 0 {
            if clipped_point_count >= 3 {
                if signed_dist >= 0.0 {
                    EPortalFacing::Above
                } else {
                    EPortalFacing::Below
                }
            } else {
                EPortalFacing::Degenerate
            }
        } else {
            EPortalFacing::None
        }
    } else {
        // Plane is almost coincident with origin — test if origin
        // projects inside the portal polygon.
        let in_portal = projection_origin_in_portal(projection, portal, plane);
        *projection_origin_in_portal_flag = in_portal;
        if !in_portal && clipped_point_count <= 0 {
            EPortalFacing::None
        } else {
            EPortalFacing::Degenerate
        }
    }
}

// =============================================================================
// `projection_origin_in_portal @ 0x180509750`
// =============================================================================

/// Test whether the projection's origin (camera position) projects
/// to a point inside the portal polygon (when treated as 2D against
/// the dominant axis).
pub fn projection_origin_in_portal(
    projection: &VisibilityProjection,
    portal: &BspClusterPortal,
    plane: &RealPlane3d,
) -> bool {
    // Pick dominant axis of plane normal.
    let abs_i = plane.i.abs();
    let abs_j = plane.j.abs();
    let abs_k = plane.k.abs();
    let axis = if abs_k >= abs_j && abs_k >= abs_i {
        2
    } else if abs_j >= abs_i {
        1
    } else {
        0
    };
    // Project origin onto plane.
    let origin = projection.basis_to_world.position;
    let signed_dist = origin.x * plane.i + origin.y * plane.j + origin.z * plane.k - plane.d;
    let neg = -signed_dist;
    let origin_on_plane = RealPoint3d {
        x: plane.i * neg + origin.x,
        y: plane.j * neg + origin.y,
        z: plane.k * neg + origin.z,
    };

    // Project portal vertices + origin to 2D using the OTHER two
    // axes (skip dominant axis). Then point-in-polygon test.
    let project_2d = |p: RealPoint3d| -> RealPoint2d {
        match axis {
            0 => RealPoint2d { x: p.y, y: p.z },
            1 => RealPoint2d { x: p.x, y: p.z },
            _ => RealPoint2d { x: p.x, y: p.y },
        }
    };
    let origin2d = project_2d(origin_on_plane);

    let n = portal.vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut prev = project_2d(portal.vertices[n - 1]);
    for v in &portal.vertices {
        let cur = project_2d(*v);
        // Standard point-in-polygon ray cast.
        let crosses = (cur.y > origin2d.y) != (prev.y > origin2d.y);
        if crosses {
            let slope = (origin2d.y - cur.y) / (prev.y - cur.y);
            let x_intersect = cur.x + slope * (prev.x - cur.x);
            if origin2d.x < x_intersect {
                inside = !inside;
            }
        }
        prev = cur;
    }
    inside
}
