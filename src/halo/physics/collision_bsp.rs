//! Engine source: `Ares/source/physics/collision_bsp.{h,cpp}`.
//!
//! Ray-vs-BSP-collision-tree walker. Three functions, in dependency order:
//!
//! 1. [`test_vector`] — top-level entry. Sets up scratch state and dispatches
//!    into the recursive walker. Mirrors `collision_bsp_test_vector @ 0x180512c70`.
//! 2. [`test_vector_recursive`] — descends the BSP3D kd-tree, splitting the
//!    ray segment `[t0..t1]` at each plane intersection and recursing into the
//!    near child first. Mirrors `collision_bsp_test_vector_recursive @ 0x180513f80`.
//! 3. [`leaf_test_vector`] — at each visited leaf, walks the leaf's
//!    `bsp2d_references` to find which 2D surface tree contains the ray's
//!    plane-intersection point, returning the hit surface index. Mirrors
//!    `collision_leaf_test_vector @ 0x180514460`.
//!
//! Engine uses 24-bit child indices (`& 0x7FFFFF`, leaf bit 23 = `0x800000`);
//! blam-tags promotes both schemas to canonical i32 with bit 31 = leaf. Walker
//! uses `< 0` for leaf detection accordingly.
//!
//! Trimmed from engine source (no behavior change): profiling/logging
//! (`collision_log_*`, `c_string_builder`, `handle_slim_assert`), debug globals
//! (`global_collision_log_enable`), and the breakable-surface path
//! (`collision_surface_test_point` — only reached when caller's `test_surface_flag`
//! is true, which the decorator-bake chain never sets).

use blam_tags::math::{RealPoint3d, RealVector3d};
use blam_tags::structure_bsp::Bsp3d;

use super::bsp2d;
use super::bsp3d::plane3d_distance_to_point_safe;

// =============================================================================
// collision_bsp_test_vector_result — engine output struct (Ares header
// `physics/collision_bsp.h:88`, sizeof=1064)
// =============================================================================

pub const MAXIMUM_COLLISION_LEAVES_PER_TEST: usize = 256;

/// Engine `collision_bsp_test_vector_result` (1064 B). Heap allocations would
/// be expensive at engine call rates; engine stack-allocates this. We mirror
/// — caller owns the struct. `Default` zeros every field.
///
/// `plane` is a raw reference back into the BSP's plane table. We use an
/// `Option<usize>` plane index instead of a `*const real_plane3d` so the
/// struct is `Send`-friendly and doesn't carry pointers across module
/// boundaries. Engine's `result->plane = &bsp.planes[i]` is equivalent.
#[derive(Debug, Clone)]
pub struct CollisionBspTestVectorResult {
    /// Ray parametric t of the hit (`0..maximum_t`). Engine `t` @ 0x0.
    pub t: f32,
    /// Index into `bsp.planes` of the hit surface's plane. `None` for misses.
    /// Engine `plane: real_plane3d const*` @ 0x8.
    pub plane_index: Option<u32>,
    /// Index into `bsp.leaves` of the hit leaf. `-1` for misses.
    /// Engine `leaf_index` @ 0x10.
    pub leaf_index: i32,
    /// Index into `bsp.surfaces` of the hit surface. `-1` for misses.
    /// Engine `surface_index` @ 0x14.
    pub surface_index: i32,
    /// Plane-designator (plane index + bit-31 negate flag). Engine
    /// `plane_designator` @ 0x18.
    pub plane_designator: i32,
    /// Surface flags (bit 1 = `Invisible`, bit 3 = `Breakable`), copied
    /// from the hit surface. Engine `flags` @ 0x1C.
    pub flags: blam_tags::Flags<blam_tags::structure_bsp::CollisionSurfaceFlags, u8>,
    /// Engine `breakable_surface_index` @ 0x1D.
    pub breakable_surface_index: u8,
    /// Engine `material_index` @ 0x1E (collision-material table).
    pub material_index: i16,
    /// Number of leaves the walker visited but didn't find a hit in.
    /// Engine `leaf_count` @ 0x20.
    pub leaf_count: i32,
    /// First `leaf_count` entries valid. Engine
    /// `leaf_indices[256]` @ 0x24.
    pub leaf_indices: [i32; MAXIMUM_COLLISION_LEAVES_PER_TEST],
    /// Engine `breakable_surface_set_index` @ 0x424.
    pub breakable_surface_set_index: u8,
}

impl Default for CollisionBspTestVectorResult {
    fn default() -> Self {
        Self {
            t: 0.0,
            plane_index: None,
            leaf_index: -1,
            surface_index: -1,
            plane_designator: -1,
            flags: blam_tags::Flags::default(),
            breakable_surface_index: 0,
            material_index: -1,
            leaf_count: 0,
            leaf_indices: [-1; MAXIMUM_COLLISION_LEAVES_PER_TEST],
            breakable_surface_set_index: 0,
        }
    }
}

// =============================================================================
// e_collision_bsp_test_flag — engine flags from Ares header
// =============================================================================
//
// ```cpp
// enum {
//     _collision_bsp_test_front_facing_surfaces_bit = 0,
//     _collision_bsp_test_back_facing_surfaces_bit,
//     _collision_bsp_test_ignore_two_sided_surfaces_bit,
//     _collision_bsp_test_ignore_invisible_surfaces_bit,
//     _collision_bsp_test_ignore_breakable_surfaces_bit,
//     _collision_bsp_test_allow_early_out_bit,
// };
// ```

pub const FRONT_FACING_SURFACES: u32 = 1 << 0;
pub const BACK_FACING_SURFACES: u32 = 1 << 1;
pub const IGNORE_TWO_SIDED_SURFACES: u32 = 1 << 2;
pub const IGNORE_INVISIBLE_SURFACES: u32 = 1 << 3;
pub const IGNORE_BREAKABLE_SURFACES: u32 = 1 << 4;

// =============================================================================
// test_vector_data — engine local struct that threads recursion state
// =============================================================================
//
// Defined locally in `physics/collision_bsp.cpp`; not in any header. Engine
// passes a `test_vector_data*` through every `test_vector_recursive` frame
// so each recursive call can read both the ray (constant per query) and
// the running scratch state (`last_leaf_index`, `last_contents`,
// `last_plane_index`) without re-threading 9 arguments.

struct TestVectorData<'a> {
    bsp: &'a CollisionBspView<'a>,
    point: RealPoint3d,
    vector: RealVector3d,
    flags: u32,
    /// Scratch: leaf index visited on the previous segment-side, used by the
    /// contents-transition logic in `test_vector_recursive`'s tail.
    last_leaf_index: i32,
    /// Scratch: "contents" byte of that previous leaf. 3 = solid leaf, 2 =
    /// transparent, 1 = ?. Engine encodes from `leaf.flags & 1`.
    last_contents: u8,
    /// Scratch: plane index that split off the previous segment-side. Engine
    /// uses this to record which plane was crossed at the hit point.
    last_plane_index: i32,
    result: &'a mut CollisionBspTestVectorResult,
}

// =============================================================================
// CollisionBspView — borrow-friendly wrapper around the parsed blam-tags
// `collision_bsp`. Engine passes a raw `collision_bsp const*`; we pass slices
// for each sub-block to avoid lifetime contortions in the recursive walker.
// =============================================================================

/// Borrowed view of a `blam_tags::structure_bsp::Bsp3d` for the walkers in
/// this module. Engine accesses are via `bsp->bsp3d.{nodes,planes}` and
/// `bsp->{leaves, bsp2d_references, surfaces}` — same data, different
/// borrow form so callers don't need to share `&Bsp3d` everywhere.
pub struct CollisionBspView<'a> {
    pub bsp3d: &'a Bsp3d,
}

impl<'a> CollisionBspView<'a> {
    pub fn new(bsp: &'a Bsp3d) -> Self {
        Self { bsp3d: bsp }
    }
}

// =============================================================================
// collision_bsp_test_vector — entry (`@ 0x180512c70`)
// =============================================================================

/// `collision_bsp_test_vector` @ dllcache `0x180512c70`.
///
/// Set up scratch + dispatch into the recursive walker. The engine's full
/// signature includes `collision_materials_extant_flags`,
/// `breakable_surface_set_count`, and `breakable_surface_sets` — we omit them
/// here because the decorator-bake chain always passes `nullptr/0/nullptr`
/// (collision_test_vector → c_geometry_sampler::sample never passes a
/// material filter or breakable surface set).
///
/// Returns `true` when a hit was found; `result` is populated. `maximum_t`
/// is the ray-parametric end (typically 1.0 for full segment, smaller for
/// truncated rays).
pub fn test_vector(
    flags: u32,
    bsp: &CollisionBspView<'_>,
    point: RealPoint3d,
    vector: RealVector3d,
    maximum_t: f32,
    result: &mut CollisionBspTestVectorResult,
) -> bool {
    // Engine: clamp `result->t` to `max(0, maximum_t)`. Engine then sets the
    // `maximum_t_1` local to `clamp(maximum_t, 0.0, 1.0)` for the recursion
    // (the recursion expects t1 in `[0, 1]`).
    let t_init = maximum_t.max(0.0);
    result.t = t_init;
    result.leaf_count = 0;

    let mut data = TestVectorData {
        bsp,
        point,
        vector,
        flags,
        last_leaf_index: -1,
        last_contents: 0,
        last_plane_index: -1,
        result,
    };

    let t1 = if maximum_t <= 0.0 {
        0.0
    } else if maximum_t >= 1.0 {
        1.0
    } else {
        maximum_t
    };

    test_vector_recursive(&mut data, 0, 0.0, t1)
}

// =============================================================================
// collision_bsp_test_vector_recursive — main walker (`@ 0x180513f80`)
// =============================================================================

/// `collision_bsp_test_vector_recursive` @ dllcache `0x180513f80`.
///
/// Descends `data.bsp.bsp3d.nodes` starting at `child_index` over the ray
/// segment `[t0, t1]`. At each interior node:
///
///  1. Look up the node's split plane.
///  2. Compute signed distance from the ray's endpoints (at `t0` and `t1`)
///     to the plane.
///  3. If both endpoints are on the same side, recurse into that single child.
///  4. Otherwise compute the split-t (where the ray crosses the plane),
///     recurse into the NEAR child over `[t0, t_split]`. If that returns a
///     hit, propagate up immediately. Else recurse into the FAR child over
///     `[t_split, t1]` (loop tail).
///
/// At a leaf (`child_index < 0`): consult the leaf's contents-transition
/// flags to decide whether to call [`leaf_test_vector`] (only invoked at
/// transitions between solid/air leaves — see `flags` bits 0/1/2). On a real
/// hit, populate `data.result` and return `true`.
///
/// Returns `true` when a hit was found AND the walker should stop. Otherwise
/// loop tail continues to the far child.
fn test_vector_recursive(
    data: &mut TestVectorData<'_>,
    mut child_index: i32,
    mut t0: f32,
    t1: f32,
) -> bool {
    // Engine loop guard: `(child_index & 0x800000) == 0`. blam-tags promoted
    // leaf bit to bit 31 → `>= 0` means "interior node".
    loop {
        if child_index < 0 {
            break;
        }
        let node_idx = (child_index & 0x7FFF_FFFF) as usize;
        let node = match data.bsp.bsp3d.nodes.get(node_idx) {
            Some(n) => *n,
            None => break, // bogus index — engine reads through null → all-zero
        };

        let plane = if node.plane_index >= 0 {
            data.bsp.bsp3d.planes.get(node.plane_index as usize).copied()
        } else {
            None
        };

        // Plane test at t0. Engine computes both `signed_distance(plane, point)`
        // and the dir·normal once, then derives the t1 distance via
        // `dist + t1 * dir·n`.
        let d_point = plane3d_distance_to_point_safe(plane.as_ref(), data.point);
        let n_dot_dir = match plane.as_ref() {
            Some(p) => p.i * data.vector.i + p.j * data.vector.j + p.k * data.vector.k,
            None => 0.0,
        };
        let d_at_t0 = t0 * n_dot_dir + d_point;
        let d_at_t1 = t1 * n_dot_dir + d_point;

        let t0_side_positive = d_at_t0 >= 0.0;
        let t1_side_positive = d_at_t1 >= 0.0;

        if t0_side_positive == t1_side_positive {
            // Same side — descend into that child only (loop continues).
            let next = if t0_side_positive {
                node.above_child
            } else {
                node.below_child
            };
            child_index = next;
            continue;
        }

        // Endpoints on different sides — compute split t.
        // Engine: `t_split = -d_point / n_dot_dir`. Note the XOR-with-sign-bit
        // in the decompile (`COERCE_UNSIGNED_INT(...) ^ _xmm`) negates the f32
        // by flipping the sign bit; equivalent to `-x` for finite values.
        let t_split = -d_point / n_dot_dir;

        // Near child is the side `t0` is on; recurse into it over [t0, t_split].
        // Engine accesses `*(int *)((char *)&v17[v26 <= 0.0] + (v26 <= 0.0) + 1) >> 8`
        // which is a packed-bit-trick to pick `below_child` vs `above_child`
        // based on `n_dot_dir <= 0`. After unpacking, the logic is: if the
        // segment's `t0` side is positive (`t0_side_positive`), the near child
        // is `above_child`; else `below_child`. (Equivalent to `n_dot_dir > 0
        // → above is near`, but the implementation tests t0_side directly.)
        let near = if t0_side_positive {
            node.above_child
        } else {
            node.below_child
        };
        let far = if t0_side_positive {
            node.below_child
        } else {
            node.above_child
        };

        if test_vector_recursive(data, near, t0, t_split) {
            return true;
        }
        // Early-out if the near-child traversal already filled `result.t`
        // less than `t_split` (we can't beat it on the far side).
        if t_split >= data.result.t {
            return false;
        }
        // Record which plane we crossed and continue down the far child.
        data.last_plane_index = node.plane_index;
        t0 = t_split;
        child_index = far;
    }

    // ============================================================
    // Leaf handling.
    // ============================================================
    //
    // `child_index` is a leaf designator. Decode the index, fetch the leaf
    // (engine packs leaf flags in byte 0), and run the contents-transition
    // logic to decide whether to test surfaces.

    let leaf_index_1 = if child_index == -1 {
        -1
    } else {
        (child_index & 0x7FFF_FFFF) as i32
    };

    // `n3` is engine's "contents code" for this leaf:
    //   3 = "solid" or "out-of-bounds" (no leaf, or leaf with bit 0 unset)
    //   2 = leaf with bit 0 set ("transparent" / air-like)
    let n3: u8 = if leaf_index_1 < 0 {
        3
    } else {
        match data.bsp.bsp3d_leaf_flags(leaf_index_1 as u32) {
            // bit 0 = ContainsDoubleSidedSurfaces → contents 2, else 1.
            Some(flags) => {
                (flags.contains(blam_tags::structure_bsp::CollisionLeafFlags::ContainsDoubleSidedSurfaces)
                    as u8)
                    + 1
            }
            None => 3,
        }
    };

    // ============================================================
    // Contents-transition logic.
    //
    // Engine `data.flags` bits:
    //   bit 0 = FRONT_FACING_SURFACES (test air→solid transitions)
    //   bit 1 = BACK_FACING_SURFACES (test solid→air transitions)
    //   bit 2 = IGNORE_TWO_SIDED_SURFACES (skip air↔air, requires test_surface_flag)
    //
    // The transition fires when the previous contents and current contents
    // form one of the configured pairs. `leaf_index` (engine local; reused
    // as the leaf we'll test against) is set to whichever side's leaf we
    // actually want to consult.
    // ============================================================
    let mut test_surface_flag = false;
    let mut leaf_index_to_test: i32 = -1;
    let mut should_test = false;

    if (data.flags & FRONT_FACING_SURFACES) != 0
        && (data.last_contents.wrapping_sub(1)) <= 1
        && n3 == 3
    {
        // Air-side → solid-side: test against the previous (air) leaf.
        leaf_index_to_test = data.last_leaf_index;
        should_test = true;
    } else if (data.flags & BACK_FACING_SURFACES) != 0
        && data.last_contents == 3
        && (n3.wrapping_sub(1)) <= 1
    {
        // Solid-side → air-side: test against this (air) leaf.
        leaf_index_to_test = leaf_index_1;
        should_test = true;
    } else if (data.flags & IGNORE_TWO_SIDED_SURFACES) == 0
        && data.last_contents == 2
        && n3 == 2
    {
        // Air↔air transition with two-sided surfaces allowed.
        if (data.flags & FRONT_FACING_SURFACES) != 0 {
            leaf_index_to_test = data.last_leaf_index;
        } else {
            leaf_index_to_test = leaf_index_1;
        }
        test_surface_flag = true;
        should_test = true;
    }

    if should_test && leaf_index_to_test != -1 {
        let surface_index = leaf_test_vector(
            data.bsp,
            data.point,
            data.vector,
            leaf_index_to_test,
            data.last_plane_index,
            t0,
            test_surface_flag,
        );
        if let Some(surface_idx) = surface_index {
            // Surface hit. Engine filters by surface flags vs `data.flags`:
            //   surface `Invisible` (bit 1) gated by IGNORE_INVISIBLE_SURFACES
            //   surface `Breakable` (bit 3) gated by IGNORE_BREAKABLE_SURFACES
            // Cloned out of the BSP so the immutable borrow releases before
            // we write `data.result` below.
            let surface = data
                .bsp
                .surfaces()
                .get(surface_idx as usize)
                .cloned();
            if let Some(s) = surface {
                use blam_tags::structure_bsp::CollisionSurfaceFlags;
                let invisible_skip = s.flags.contains(CollisionSurfaceFlags::Invisible)
                    && (data.flags & IGNORE_INVISIBLE_SURFACES) != 0;
                let breakable_skip = s.flags.contains(CollisionSurfaceFlags::Breakable)
                    && (data.flags & IGNORE_BREAKABLE_SURFACES) != 0;
                if !invisible_skip && !breakable_skip {
                    // Populate result and signal hit.
                    data.result.t = t0;
                    let last_pi = data.last_plane_index;
                    data.result.plane_index = if last_pi >= 0 {
                        Some(last_pi as u32)
                    } else {
                        None
                    };
                    data.result.leaf_index = leaf_index_to_test;
                    data.result.surface_index = surface_idx as i32;
                    data.result.plane_designator = s.plane_designator;
                    data.result.flags = s.flags;
                    data.result.breakable_surface_index =
                        s.breakable_surface as i8 as u8;
                    data.result.breakable_surface_set_index =
                        s.breakable_surface_set as i8 as u8;
                    data.result.material_index = s.material;
                    return true;
                }
            }
        }
    }

    // No hit at this leaf — record it as visited (capped at 256, mirrors
    // engine's `leaf_indices[255] = leaf_index_1` clamp).
    if leaf_index_1 != -1 {
        let count = data.result.leaf_count as usize;
        if count < MAXIMUM_COLLISION_LEAVES_PER_TEST {
            data.result.leaf_indices[count] = leaf_index_1;
            data.result.leaf_count += 1;
        } else {
            data.result.leaf_indices[MAXIMUM_COLLISION_LEAVES_PER_TEST - 1] = leaf_index_1;
        }
    }
    data.last_leaf_index = leaf_index_1;
    data.last_contents = n3;
    false
}

// =============================================================================
// collision_leaf_test_vector — leaf-level surface lookup (`@ 0x180514460`)
// =============================================================================

/// Engine `global_projection3d_mappings[3][2][3]` @ dllcache `0x180b255b0`.
///
/// SINGLE SOURCE OF TRUTH for the 2D-projection axis table shared by every
/// collision/geometry path (this module, `structures::collision_bsp`, and
/// `geometry::geometry_sampling`). Both the 3-wide and 2-wide forms below
/// derive from this constant so they can never re-diverge (a prior bug had
/// the two homes carrying inconsistent transcriptions — see git history).
///
/// `short[3][2][3]` indexed by `(projection_axis, projection_sign, kind)`:
///   kind 0 → which 3D component becomes 2D X.
///   kind 1 → which 3D component becomes 2D Y.
///   kind 2 → the dropped component (= `projection_axis`; carried only for
///            parity with the engine table, unused by 2D testing).
///
/// `projection_axis` is the dominant plane-normal component (0=X, 1=Y, 2=Z);
/// `projection_sign` controls the 2D winding order. A projection MUST drop the
/// dominant axis and keep the OTHER two — neither of kinds 0/1 may equal
/// `projection_axis` (that would collapse the surface to a line). The guard
/// test below enforces this invariant.
pub const PROJECTION3D_MAPPINGS_FULL: [[[u8; 3]; 2]; 3] = [
    [[2, 1, 0], [1, 2, 0]], // axis 0 (X dropped): project onto {Z,Y}/{Y,Z}
    [[0, 2, 1], [2, 0, 1]], // axis 1 (Y dropped): project onto {X,Z}/{Z,X}
    [[1, 0, 2], [0, 1, 2]], // axis 2 (Z dropped): project onto {Y,X}/{X,Y}
];

/// 2-wide `[axis][sign] -> (2D-X, 2D-Y)` view, derived from
/// [`PROJECTION3D_MAPPINGS_FULL`] by dropping the parity (kind-2) column.
/// This is the form consumed by the 2D bsp2d / point-in-triangle tests.
pub const PROJECTION3D_MAPPINGS: [[[u8; 2]; 2]; 3] = projection3d_mappings_2wide();

const fn projection3d_mappings_2wide() -> [[[u8; 2]; 2]; 3] {
    let f = PROJECTION3D_MAPPINGS_FULL;
    [
        [[f[0][0][0], f[0][0][1]], [f[0][1][0], f[0][1][1]]],
        [[f[1][0][0], f[1][0][1]], [f[1][1][0], f[1][1][1]]],
        [[f[2][0][0], f[2][0][1]], [f[2][1][0], f[2][1][1]]],
    ]
}

#[cfg(test)]
mod projection_mapping_tests {
    use super::{PROJECTION3D_MAPPINGS, PROJECTION3D_MAPPINGS_FULL};

    /// Every (axis, sign) entry must project onto the two axes OTHER than the
    /// dropped/dominant `axis` — never onto `axis` itself (that would collapse
    /// the triangle to a line). Guards against transcription drift.
    #[test]
    fn no_entry_projects_onto_dropped_axis() {
        for axis in 0..3u8 {
            for sign in 0..2 {
                let pair = PROJECTION3D_MAPPINGS[axis as usize][sign];
                assert_ne!(pair[0], axis, "axis {axis} sign {sign} projects onto dropped axis");
                assert_ne!(pair[1], axis, "axis {axis} sign {sign} projects onto dropped axis");
                assert_ne!(pair[0], pair[1], "axis {axis} sign {sign} has duplicate axis");
                // kind-2 parity column must equal the dropped/dominant axis.
                assert_eq!(
                    PROJECTION3D_MAPPINGS_FULL[axis as usize][sign][2], axis,
                    "axis {axis} sign {sign} parity column != dropped axis"
                );
            }
        }
    }

    /// The 2-wide view must stay byte-identical to the first two columns of
    /// the full table — this is what prevents the two homes from re-diverging.
    #[test]
    fn two_wide_matches_full() {
        for axis in 0..3 {
            for sign in 0..2 {
                assert_eq!(PROJECTION3D_MAPPINGS[axis][sign][0], PROJECTION3D_MAPPINGS_FULL[axis][sign][0]);
                assert_eq!(PROJECTION3D_MAPPINGS[axis][sign][1], PROJECTION3D_MAPPINGS_FULL[axis][sign][1]);
            }
        }
    }
}

/// `collision_leaf_test_vector` @ dllcache `0x180514460`.
///
/// For the named leaf, iterates over its `bsp2d_references` range
/// (`leaf.first_bsp2d_reference .. + bsp2d_reference_count`). Each reference
/// pairs a plane index with a per-plane bsp2d tree root. The walker:
///
///  1. Filters references to the one whose plane matches the recursion's
///     `plane_index` (the plane the ray crossed entering this leaf).
///  2. Computes the ray's 3D position at parametric `t`.
///  3. Picks the projection axis based on the plane's dominant normal
///     component (so the projected 2D test isn't degenerate).
///  4. Projects the 3D point to 2D via [`PROJECTION3D_MAPPINGS`].
///  5. Walks the bsp2d tree via [`bsp2d::test_point`] → surface index.
///  6. Returns the surface index. (Material/breakable filters in the engine
///     are stripped here for the decorator-bake chain that doesn't supply
///     them.)
///
/// `test_surface_flag` is engine's gate for calling
/// `collision_surface_test_point` (a more refined per-polygon check) —
/// always `false` from the decorator-bake call sites we care about. We
/// honour the parameter for completeness but never branch on it (engine's
/// extra check is omitted with the rest of the breakable path).
#[allow(clippy::too_many_arguments)]
fn leaf_test_vector(
    bsp: &CollisionBspView<'_>,
    point: RealPoint3d,
    vector: RealVector3d,
    leaf_index: i32,
    plane_index: i32,
    t: f32,
    _test_surface_flag: bool,
) -> Option<u32> {
    let leaf = bsp.leaves().get(leaf_index as usize)?;
    let first_ref = leaf.first_bsp2d_reference;
    let ref_count = leaf.bsp2d_reference_count as u32;
    if ref_count == 0 {
        return None;
    }

    let plane = bsp.bsp3d.planes.get(plane_index as usize).copied()?;

    // Pick projection axis based on dominant normal component (engine math).
    let abs_i = plane.i.abs();
    let abs_j = plane.j.abs();
    let abs_k = plane.k.abs();
    let projection_axis: usize = if abs_k < abs_j || abs_k < abs_i {
        if abs_j >= abs_i {
            1
        } else {
            0
        }
    } else {
        2
    };
    // Engine: `projection_sign = (plane.normal[axis] > 0) != (designator-negate-flag)`.
    // We don't carry the designator's negate flag through this entry, so use
    // the raw sign of the normal component. The bsp2d test only depends on
    // projection_sign for orientation tie-breaking; the surface index
    // returned is the same.
    let proj_n = match projection_axis {
        0 => plane.i,
        1 => plane.j,
        _ => plane.k,
    };
    let projection_sign: usize = (proj_n > 0.0) as usize;

    // Ray position at parametric `t`.
    let p3d = RealPoint3d {
        x: point.x + t * vector.i,
        y: point.y + t * vector.j,
        z: point.z + t * vector.k,
    };
    let p3d_components = [p3d.x, p3d.y, p3d.z];

    let axes = PROJECTION3D_MAPPINGS[projection_axis][projection_sign];
    let p2d = glam::Vec2::new(
        p3d_components[axes[0] as usize],
        p3d_components[axes[1] as usize],
    );

    // Walk the bsp2d_references range looking for a matching plane.
    for offset in 0..ref_count {
        let ref_idx = first_ref as u32 + offset;
        let bsp2d_ref = match bsp.bsp2d_references().get(ref_idx as usize) {
            Some(r) => r,
            None => break,
        };
        // Engine `(v22->flags & 0x7FFF) != plane_index` — compares lower 15
        // bits of the plane_designator to the recursion's plane_index. Both
        // sides use the SAME-bits-as-designator encoding (low 15/31 = index,
        // high = negate flag); the recursion's `last_plane_index` is from
        // `bsp3d_node.plane_index` which is just an index, no flag.
        if (bsp2d_ref.plane_designator & 0x7FFF_FFFF) != plane_index {
            continue;
        }
        let surface = bsp2d::test_point(&bsp.bsp2d_nodes(), p2d, bsp2d_ref.bsp2d_node);
        if surface != bsp2d::NONE {
            // Engine also checks `collision_materials_extant_flags_test` and
            // `collision_surface_test_point` when their respective inputs are
            // supplied; both are absent in our call paths.
            return Some(surface);
        }
    }
    None
}

// =============================================================================
// Helpers that bridge engine's nested-pointer access patterns to our flat
// blam-tags slices.
// =============================================================================

impl CollisionBspView<'_> {
    fn leaves(&self) -> &[blam_tags::structure_bsp::CollisionLeaf] {
        // Engine `bsp->leaves` is the BSP-collision leaf table (per
        // `collision_bsp` struct); blam-tags stores it on `Bsp3d`.
        &self.bsp3d.leaves
    }
    fn bsp2d_references(&self) -> &[blam_tags::structure_bsp::CollisionBsp2dReference] {
        &self.bsp3d.bsp2d_references
    }
    fn bsp2d_nodes(&self) -> &[blam_tags::structure_bsp::CollisionBsp2dNode] {
        &self.bsp3d.bsp2d_nodes
    }
    fn surfaces(&self) -> &[blam_tags::structure_bsp::CollisionSurface] {
        &self.bsp3d.surfaces
    }
    fn bsp3d_leaf_flags(
        &self,
        leaf_index: u32,
    ) -> Option<&blam_tags::Flags<blam_tags::structure_bsp::CollisionLeafFlags, u16>> {
        self.bsp3d.leaves.get(leaf_index as usize).map(|l| &l.flags)
    }
}
