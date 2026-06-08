//! Mirror of `physics/collision_bsp.cpp` — ray-vs-BSP3D + ray-vs-leaf
//! surface test.
//!
//! Anchored against dllcache:
//!   `collision_bsp_test_vector @ 0x180512c70` (outer wrapper)
//!   `collision_bsp_test_vector_recursive @ 0x180513f80` (BSP3D ray walk)
//!   `collision_leaf_test_vector @ 0x180514460` (per-leaf surface lookup)
//!   `collision_surface_test_point @ 0x180514870` (edge-loop containment)
//!   `project_point3d @ 0x180195090` + `global_projection3d_mappings` data
//!
//! Drives the decal projection bake (D2a precondition for D2b/c). The
//! decal projection volume needs a seed surface (`bsp_index`,
//! `surface_index`, `plane`, `point`) from a 1cm-back ray cast along
//! `projection_forward`; this module provides that single-BSP raycast.
//!
//! Iterating across active BSPs (and instance-geometry) is the caller's
//! responsibility — engine equivalent is the 8-arg
//! `collision_test_vector` overload, which we'll add when D2c needs it.

use blam_tags::math::RealPlane3d;
use blam_tags::structure_bsp::{Bsp3d, Bsp3dNode};

// =============================================================================
// `global_projection3d_mappings @ 0x180B255B0` — verified via IDA read_bytes
// =============================================================================
//
// `short[3][2][3]` indexed by `(projection_axis, projection_sign, kind)`.
// kind=0 → which 3D component becomes 2D X.
// kind=1 → which 3D component becomes 2D Y.
// kind=2 → which 3D component was dropped (= projection_axis; unused by
//          collision testing, kept for parity with the engine table).
//
// projection_axis selects the component with the largest |plane normal|;
// projection_sign captures the orientation of that normal so the 2D
// edge-cross-product winding stays consistent with 3D surface winding.
const PROJECTION_3D_MAPPINGS: [[[usize; 3]; 2]; 3] = [
    [[2, 1, 0], [1, 2, 0]], // drop X
    [[0, 2, 1], [2, 0, 1]], // drop Y
    [[1, 0, 2], [0, 1, 2]], // drop Z
];

// =============================================================================
// Collision test flag bits — matches engine `s_collision_test_flags::collision_flags`
// =============================================================================

/// Inner-walker flag bits read by `collision_bsp_test_vector_recursive`.
/// Produced from the outer `s_collision_test_flags::collision_flags` via
/// `build_bsp_flags_from_collision_flags @ 0x1804019E0` which remaps
/// outer bits 11-16 to inner bits 0-5. Instance-geometry handling lives
/// entirely in the OUTER 8-arg `collision_test_vector` — the inner
/// walker only knows about a single `collision_bsp` and these 5 bits.
pub mod flag {
    /// Bit 0 — test surfaces on the empty/double→solid transition
    /// (records the previous leaf's surface_index).
    pub const TEST_ENTRY: u32 = 0x1;
    /// Bit 1 — test surfaces on the solid→empty/double transition
    /// (records the current leaf's surface_index).
    pub const TEST_EXIT: u32 = 0x2;
    /// Bit 2 — when set, skip the "both leaves are double-sided-solid"
    /// interior-surface test.
    pub const SKIP_INTERIOR: u32 = 0x4;
    /// Bit 3 — REJECT surfaces whose `flags & 2` (two-sided) is set.
    /// (Engine name: `filter_two_sided`. Despite the suggestive value,
    /// this is NOT a "test instances" bit — instances are handled by
    /// the outer ctv.)
    pub const FILTER_TWO_SIDED: u32 = 0x8;
    /// Bit 4 — REJECT surfaces whose `flags & 8` (breakable) is set.
    pub const FILTER_BREAKABLE: u32 = 0x10;
    /// Decal-collide effective `bsp_flags`. The OUTER
    /// `c_decal_system::collide` builds `m_flags = environment |
    /// objects_all_types | 0x4000`; the 8-arg `collision_test_vector`
    /// remaps outer bits 11/12/14 → inner bits 0/1/3, yielding
    /// `0xB = ENTRY | EXIT | FILTER_TWO_SIDED`. The EXIT bit is what
    /// makes preplaced decals work — authoring bakes `position`
    /// slightly inside the wall, the 1cm back-shift keeps origin in
    /// the slab, and the ray's EXIT transition on the back face is
    /// what registers.
    pub const DECAL_DEFAULT: u32 = TEST_ENTRY | TEST_EXIT | FILTER_TWO_SIDED;
}

// =============================================================================
// Result of `collision_bsp_test_vector`
// =============================================================================

/// Output of a BSP-scoped ray cast. `t` is the parametric distance
/// along the ray (0..1 of the input `vector` parameter when the cast
/// uses normalized vectors; otherwise world units depending on caller
/// convention). `surface_index` indexes `Bsp3d::surfaces`.
#[derive(Debug, Clone, Default)]
pub struct CollisionBspResult {
    pub t: f32,
    pub leaf_index: i32,
    pub surface_index: i32,
    pub plane: RealPlane3d,
    /// Canonical bit-31 designator (parser-normalized from i16 for
    /// SMALL collision_bsp schemas).
    pub plane_designator: i32,
    pub flags: blam_tags::Flags<blam_tags::structure_bsp::CollisionSurfaceFlags, u8>,
    pub material_index: i16,
}

// =============================================================================
// `collision_bsp_test_vector @ 0x180512c70`
// =============================================================================

/// Cast a ray against one BSP's collision tree. Returns `Some(result)`
/// on hit, `None` otherwise.
///
/// `vector` is the ray's direction-and-length; the parametric `t` of
/// the hit is in [0, max_t]. The engine clamps `max_t` to [0, 1] —
/// callers normalize the projection volume so a unit step covers the
/// search range.
pub fn collision_bsp_test_vector(
    bsp: &Bsp3d,
    flags: u32,
    point: [f32; 3],
    vector: [f32; 3],
    max_t: f32,
) -> Option<CollisionBspResult> {
    let mut data = TestData {
        bsp,
        flags,
        point,
        vector,
        result: CollisionBspResult {
            t: max_t.max(0.0),
            ..CollisionBspResult::default()
        },
        last_leaf_index: -1,
        last_contents: 0,
        last_plane_index: -1,
        hit: false,
    };
    // Engine: `if (maximum_t <= 0) t1 = 0; else if (>= 1) t1 = 1; else t1 = maximum_t`.
    let t1 = if max_t <= 0.0 {
        0.0
    } else if max_t >= 1.0 {
        1.0
    } else {
        max_t as f64
    };
    test_recursive(&mut data, 0, 0.0, t1);
    if data.hit { Some(data.result) } else { None }
}

struct TestData<'a> {
    bsp: &'a Bsp3d,
    flags: u32,
    point: [f32; 3],
    vector: [f32; 3],
    result: CollisionBspResult,
    last_leaf_index: i32,
    last_contents: u8,
    last_plane_index: i32,
    hit: bool,
}

// -----------------------------------------------------------------------------
// `collision_bsp_test_vector_recursive @ 0x180513f80`
// -----------------------------------------------------------------------------

fn test_recursive(data: &mut TestData, mut child_index: i32, mut t0: f64, mut t1: f64) -> bool {
    // Engine `collision_bsp_test_vector_recursive @ 0x180513f80`.
    // Descend through interior nodes until we land in a leaf or hit
    // the sentinel `-1` exit. Engine iterative-then-recursive shape
    // preserved.
    while !Bsp3dNode::child_is_leaf(child_index) {
        let node_idx = child_index as usize;
        let Some(&node) = data.bsp.nodes.get(node_idx) else {
            return false;
        };
        let plane = match data.bsp.planes.get(node.plane_index as usize) {
            Some(p) => *p,
            None => RealPlane3d::default(),
        };
        let pn_d = plane.i * data.point[0]
            + plane.j * data.point[1]
            + plane.k * data.point[2]
            - plane.d;
        let dot = plane.i * data.vector[0]
            + plane.j * data.vector[1]
            + plane.k * data.vector[2];
        let front_at_t0 = (t0 * dot as f64 + pn_d as f64) >= 0.0;
        let front_at_t1 = (t1 * dot as f64 + pn_d as f64) >= 0.0;

        if front_at_t0 == front_at_t1 {
            // Ray segment is entirely on one side of the plane.
            child_index = if front_at_t0 {
                node.above_child
            } else {
                node.below_child
            };
            continue;
        }

        // Ray crosses the plane somewhere in [t0, t1] — split.
        let t_split = -(pn_d as f64) / dot as f64;
        // `first_child` is the side at t0; we descend it with [t0,
        // t_split] first so result.t gets the nearest hit.
        let (first_child, second_child) = if front_at_t0 {
            (node.above_child, node.below_child)
        } else {
            (node.below_child, node.above_child)
        };
        if test_recursive(data, first_child, t0, t_split) {
            return true;
        }
        if t_split as f32 >= data.result.t {
            return false;
        }
        data.last_plane_index = node.plane_index;
        t0 = t_split;
        child_index = second_child;
    }

    // Reached a leaf (or sentinel). v32 = leaf_index, v33 = contents.
    let (leaf_index, contents): (i32, u8) = if child_index == -1 {
        (-1, 3u8) // sentinel: outside-tree counts as "interior" for transition logic
    } else {
        let li = Bsp3dNode::child_leaf_index(child_index);
        let li_usize = li as usize;
        let flags_bit = data
            .bsp
            .leaves
            .get(li_usize)
            .map(|l| {
                l.flags.contains(
                    blam_tags::structure_bsp::CollisionLeafFlags::ContainsDoubleSidedSurfaces,
                ) as u8
            })
            .unwrap_or(0u8);
        // contents = (flags & 1) + 1; engine encodes contents as 1, 2, or 3.
        (li, flags_bit + 1)
    };
    // Decide which leaf's surfaces to test based on the
    // last_contents → contents transition. Mirrors the engine
    // pseudocode lines 184-201 in `collision_bsp_test_vector_recursive`.
    let mut test_surface_flag = false;
    let chosen_leaf: i32;

    if (data.flags & flag::TEST_ENTRY) != 0
        && data.last_contents.wrapping_sub(1) <= 1
        && contents == 3
    {
        // Transition empty → solid: test prior leaf's surfaces (the
        // surface lies on the boundary, in the leaf we just left).
        chosen_leaf = data.last_leaf_index;
    } else if (data.flags & flag::TEST_EXIT) != 0
        && data.last_contents == 3
        && contents.wrapping_sub(1) <= 1
    {
        // Transition solid → empty.
        chosen_leaf = leaf_index;
    } else if (data.flags & flag::SKIP_INTERIOR) == 0
        && data.last_contents == 2
        && contents == 2
    {
        // Special "interior" case: both leaves are solid-double-sided.
        chosen_leaf = if (data.flags & flag::TEST_ENTRY) != 0 {
            data.last_leaf_index
        } else {
            leaf_index
        };
        test_surface_flag = true;
    } else {
        // No transition of interest. Record leaf for visibility
        // walkers (the engine populates result.leaf_indices[]) and
        // continue. We don't need leaf_indices for decals — the
        // caller only consumes `hit` + ray result.
        data.last_leaf_index = leaf_index;
        data.last_contents = contents;
        return false;
    }

    if chosen_leaf == -1 {
        data.last_leaf_index = leaf_index;
        data.last_contents = contents;
        return false;
    }

    let surface_idx = match leaf_test_vector(
        data.bsp,
        data.point,
        data.vector,
        chosen_leaf,
        data.last_plane_index,
        t0 as f32,
        test_surface_flag,
    ) {
        Some(s) => s,
        None => {
            data.last_leaf_index = leaf_index;
            data.last_contents = contents;
            return false;
        }
    };

    let Some(surface) = data.bsp.surfaces.get(surface_idx as usize) else {
        data.last_leaf_index = leaf_index;
        data.last_contents = contents;
        return false;
    };
    // Surface-flag filter — engine rejects a hit on the `Invisible`
    // surface bit (bit 1) when the caller set the matching decal-query
    // flag, or on the `Breakable` bit (bit 3) when FILTER_BREAKABLE is
    // set. (The `flag::*` query bits are the runtime decal namespace, not
    // tag surface flags — left raw.)
    use blam_tags::structure_bsp::CollisionSurfaceFlags;
    if (surface.flags.contains(CollisionSurfaceFlags::Invisible)
        && (data.flags & flag::FILTER_TWO_SIDED) != 0)
        || (surface.flags.contains(CollisionSurfaceFlags::Breakable)
            && (data.flags & flag::FILTER_BREAKABLE) != 0)
    {
        data.last_leaf_index = leaf_index;
        data.last_contents = contents;
        return false;
    }

    // HIT — populate result and bail. The `last_plane_index` is the
    // last split plane the walker crossed before landing here, which
    // IS the surface plane at the entry/exit transition.
    data.result.t = t0 as f32;
    if data.last_plane_index >= 0 {
        if let Some(p) = data.bsp.planes.get(data.last_plane_index as usize) {
            // Apply plane_designator high-bit flip — the surface lives
            // on the negated half-space of the raw BSP plane, so its
            // outward normal is the negated plane. Engine mirror:
            // `collision_test_vector @ 0x1803fa760` does this when
            // copying the recursive result into the outer
            // collision_result struct. Without this, EXIT-side hits
            // (decal preplaced on the back of a wall slab) return a
            // normal pointing AWAY from the projection origin, which
            // the decal cull-cone test rejects.
            let mut plane = *p;
            if surface.plane_designator < 0 {
                plane.i = -plane.i;
                plane.j = -plane.j;
                plane.k = -plane.k;
                plane.d = -plane.d;
            }
            data.result.plane = plane;
        }
    }
    data.result.leaf_index = chosen_leaf;
    data.result.surface_index = surface_idx;
    data.result.plane_designator = surface.plane_designator;
    data.result.flags = surface.flags.clone();
    data.result.material_index = surface.material;
    data.hit = true;
    true
}

// -----------------------------------------------------------------------------
// `collision_leaf_test_vector @ 0x180514460`
// -----------------------------------------------------------------------------

fn leaf_test_vector(
    bsp: &Bsp3d,
    point: [f32; 3],
    vector: [f32; 3],
    leaf_index: i32,
    plane_index: i32,
    t: f32,
    test_surface_flag: bool,
) -> Option<i32> {
    let leaf = bsp.leaves.get(leaf_index as usize)?;
    let first = leaf.first_bsp2d_reference;
    let count = leaf.bsp2d_reference_count as i32;
    let end = first + count;
    if first >= end {
        return None;
    }

    for ref_idx in first..end {
        let bsp2d_ref = match bsp.bsp2d_references.get(ref_idx as usize) {
            Some(r) => r,
            None => continue,
        };
        // Only consider bsp2d_references whose plane MATCHES the last
        // crossed BSP3D plane (the surface lies ON that plane).
        // Engine: `if ((bsp2d_ref.plane & 0x7FFF) != plane_index) continue;`
        // We canonicalize plane_designator to bit-31 form at parse
        // time, so the index mask is 31-bit here too.
        if (bsp2d_ref.plane_designator & 0x7FFF_FFFF) != plane_index {
            continue;
        }
        let plane = match bsp.planes.get(plane_index as usize) {
            Some(p) => *p,
            None => continue,
        };
        // Pick projection axis = the component with the largest
        // |plane normal|. Engine `collision_leaf_test_vector` form:
        //   if abs_k < abs_j || abs_k < abs_i: axis = (abs_j >= abs_i) ? 1 : 0
        //   else:                              axis = 2          (Z-biased ties)
        let abs_x = plane.i.abs();
        let abs_y = plane.j.abs();
        let abs_z = plane.k.abs();
        let axis: usize = if abs_z < abs_y || abs_z < abs_x {
            if abs_y >= abs_x { 1 } else { 0 }
        } else {
            2
        };
        // Sign: plane normal points in the +axis direction XOR the
        // designator's high bit (the surface lives on the negated
        // half-space of the plane).
        let plane_normal_axis = [plane.i, plane.j, plane.k][axis];
        // bit 31 (canonical) means "use the negated plane"
        let negated = bsp2d_ref.plane_designator < 0;
        let sign = (plane_normal_axis > 0.0) != negated;
        let sign_idx = if sign { 1 } else { 0 };

        // Project the hit point to 2D.
        let hit3 = [
            point[0] + t * vector[0],
            point[1] + t * vector[1],
            point[2] + t * vector[2],
        ];
        let map = &PROJECTION_3D_MAPPINGS[axis][sign_idx];
        let p2d = [hit3[map[0]], hit3[map[1]]];

        // Descend the BSP2D tree for this surface.
        let bsp2d_root = bsp2d_ref.bsp2d_node as i32;
        let surface_idx = bsp2d_test_point(bsp, p2d, bsp2d_root);

        if let Some(si) = surface_idx {
            // Optional: enforce point lies inside the surface polygon
            // (engine uses this for the "double-sided interior"
            // transition case).
            if !test_surface_flag || surface_test_point(bsp, si, axis as u16, sign, p2d) {
                return Some(si);
            }
        }
    }
    None
}

// -----------------------------------------------------------------------------
// `bsp2d_test_point` — BSP2D recursive descent
// -----------------------------------------------------------------------------

fn bsp2d_test_point(bsp: &Bsp3d, point: [f32; 2], mut child: i32) -> Option<i32> {
    // Engine `bsp2d_test_point @ 0x180576A50` (SMALL variant) reads
    // children as i16 with bit 15 as the leaf flag. The LARGE variant
    // stores them as i32 with bit 31. Our parser normalizes BOTH to
    // canonical bit-31 form (negative i32 = leaf, low 31 bits = leaf
    // payload), so this walker uses a single sign-bit check + 31-bit
    // mask regardless of source format.
    while child >= 0 {
        let node = bsp.bsp2d_nodes.get(child as usize)?;
        let dist = node.plane.i * point[0] + node.plane.j * point[1] - node.plane.d;
        child = if dist >= 0.0 { node.right_child } else { node.left_child };
    }
    if child == -1 {
        None
    } else {
        Some(child & 0x7FFF_FFFF)
    }
}

// -----------------------------------------------------------------------------
// `collision_surface_test_point @ 0x180514870`
// -----------------------------------------------------------------------------

fn surface_test_point(
    bsp: &Bsp3d,
    surface_idx: i32,
    projection_axis: u16,
    projection_sign: bool,
    point: [f32; 2],
) -> bool {
    let Some(surface) = bsp.surfaces.get(surface_idx as usize) else {
        return false;
    };
    if projection_axis > 2 {
        return false;
    }
    let axis = projection_axis as usize;
    let sign_idx = if projection_sign { 1 } else { 0 };
    let map = &PROJECTION_3D_MAPPINGS[axis][sign_idx];

    // Walk the edge ring of `surface` via `forward_edge` (or
    // `reverse_edge` when this surface is the edge's right side).
    let first_edge = surface.first_edge;
    if first_edge < 0 {
        return false;
    }
    let mut edge_idx = first_edge as i32;
    loop {
        let edge = match bsp.edges.get(edge_idx as usize) {
            Some(e) => e,
            None => return false,
        };
        let on_right = edge.right_surface as i32 == surface_idx;
        // from_vert = edge.end_vertex when right, edge.start_vertex when left.
        // to_vert   = opposite.
        let from_vert_idx = if on_right { edge.end_vertex } else { edge.start_vertex };
        let to_vert_idx = if on_right { edge.start_vertex } else { edge.end_vertex };

        let from_v = bsp.vertices.get(from_vert_idx as usize);
        let to_v = bsp.vertices.get(to_vert_idx as usize);
        let (Some(from_v), Some(to_v)) = (from_v, to_v) else {
            return false;
        };

        let from = [
            [from_v.point.x, from_v.point.y, from_v.point.z][map[0]],
            [from_v.point.x, from_v.point.y, from_v.point.z][map[1]],
        ];
        let to = [
            [to_v.point.x, to_v.point.y, to_v.point.z][map[0]],
            [to_v.point.x, to_v.point.y, to_v.point.z][map[1]],
        ];

        // Engine `collision_surface_test_point @ 0x180514870` form:
        //   reject when ((to.y - from.y) * (point.x - from.x))
        //             - ((to.x - from.x) * (point.y - from.y)) > 0
        // This is the NEGATIVE of the conventional 2D cross product
        // `(to-from) × (point-from)`. Equivalently: reject when the
        // conventional cross is < 0 (point is to the RIGHT of the
        // from→to edge in projected 2D).
        let cross_neg = (to[1] - from[1]) * (point[0] - from[0])
            - (to[0] - from[0]) * (point[1] - from[1]);
        if cross_neg > 0.0 {
            return false;
        }

        let next = if on_right { edge.reverse_edge } else { edge.forward_edge };
        if next as i32 == first_edge as i32 {
            return true;
        }
        edge_idx = next as i32;
    }
}
