//! `c_collision_surface_edge_iterator` — walks the edge ring of a
//! collision surface to find opposing surfaces for the decal BFS.
//!
//! Anchored against dllcache:
//!   `c_collision_surface_edge_iterator::ctor   @ 0x18039F270`
//!   `c_collision_surface_edge_iterator::next   @ 0x1803A14A0`
//!   `c_collision_surface_edge_iterator::get_datum @ 0x1803A0660`
//!   `c_collision_surface_edge_iterator::get_opposing_surface_fold @ 0x18039DF30`
//!
//! The engine caches a `collision_bsp const*` on the iterator (resolved
//! at construction time via `global_collision_bsp_get` /
//! `global_instance_geometry_definition_get`). The Rust port stores
//! only the indices and takes a `&Bsp3d` per method — the caller
//! already owns the resolved BSP and the borrow keeps lifetimes clean.

use blam_tags::math::{RealPoint3d, RealVector3d};
use blam_tags::structure_bsp::{Bsp3d, BspCollisionMaterial, CollisionEdge};

use crate::halo::structures::bsp3d_get_plane_from_designator;

use super::types::{CollisionSurfaceEdgeIterator, Fold};

impl CollisionSurfaceEdgeIterator {
    /// Mirror of `c_collision_surface_edge_iterator::c_collision_surface_edge_iterator
    /// @ 0x18039F270`.
    ///
    /// Engine resolves the collision BSP from
    /// `global_collision_bsp_get(bsp_index)` when `instance_definition_index
    /// == -1`, else from `instanced_geometry_definitions[def].bsp`. The
    /// Rust port expects the caller to have done that lookup and to
    /// pass the resolved `&Bsp3d` in here for the plane fetch.
    pub fn new(
        bsp: &Bsp3d,
        bsp_index: i32,
        instance_definition_index: i32,
        surface_index: i32,
    ) -> Self {
        let plane;
        let surface_edge_index;
        if let Some(surface) = bsp.surfaces.get(surface_index as usize) {
            plane = bsp3d_get_plane_from_designator(bsp, surface.plane_designator);
            surface_edge_index = surface.first_edge as i32;
        } else {
            plane = Default::default();
            surface_edge_index = -1;
        }
        Self {
            bsp_index,
            instance_definition_index,
            surface_index,
            surface_edge_index,
            plane,
        }
    }

    /// Mirror of `c_collision_surface_edge_iterator::next @ 0x1803A14A0`.
    ///
    /// Advances `surface_edge_index` to the next edge in the surface's
    /// edge ring. When we walk back to the surface's `first_edge`, the
    /// iterator marks itself done by setting `surface_edge_index =
    /// -1`. The on-left vs on-right choice mirrors the engine pun
    /// `*((i16*)edge + 5) == surface_index`: when this surface IS the
    /// edge's `right_surface`, follow `reverse_edge`; otherwise follow
    /// `forward_edge`.
    pub fn next(&mut self, bsp: &Bsp3d) {
        if self.surface_edge_index == -1 {
            return;
        }
        let Some(edge) = bsp.edges.get(self.surface_edge_index as usize).copied() else {
            self.surface_edge_index = -1;
            return;
        };
        let on_right = edge.right_surface as i32 == self.surface_index;
        let next_edge = if on_right {
            edge.reverse_edge
        } else {
            edge.forward_edge
        };
        let first_edge = bsp
            .surfaces
            .get(self.surface_index as usize)
            .map(|s| s.first_edge as i32)
            .unwrap_or(-1);
        self.surface_edge_index = if next_edge as i32 == first_edge {
            -1
        } else {
            next_edge as i32
        };
    }

    /// Mirror of `c_collision_surface_edge_iterator::get_datum @ 0x1803A0660`.
    /// Returns `None` when the iterator has been advanced past the
    /// last edge.
    pub fn get_datum<'a>(&self, bsp: &'a Bsp3d) -> Option<&'a CollisionEdge> {
        if self.surface_edge_index == -1 {
            return None;
        }
        bsp.edges.get(self.surface_edge_index as usize)
    }

    /// Mirror of `c_collision_surface_edge_iterator::get_opposing_surface_fold
    /// @ 0x18039DF30`.
    ///
    /// Fills `fold` with the surface on the OTHER side of the iterated
    /// edge (the one that isn't `self.surface_index`), computing its
    /// plane normal + the shared-edge origin/axis. Engine algorithm:
    ///
    /// 1. Identify the opposing surface (the edge's left or right
    ///    surface, whichever isn't `self.surface_index`).
    /// 2. If its material is `-1`, no opposing surface → return false.
    /// 3. If this is an instance-geometry iteration (`instance_definition_index
    ///    != -1`) OR this surface's `collision_materials[material].
    ///    seam_mapping_block_index == -1` (no cross-BSP seam), accept
    ///    the opposing surface in the same BSP.
    /// 4. Otherwise try cross-BSP seam traversal via
    ///    `structure_seams_connected_edge_get`. **The Rust port stubs
    ///    this to always return false** — multi-BSP seam traversal is
    ///    a separate engine track (`structure_seams.cpp`) we have not
    ///    ported. Cyberdyne (Epitaph) is single-BSP so this path is
    ///    inert there; other scenarios may lose a small number of
    ///    decal-seam-crossing folds until ported.
    /// 5. Filter the accepted opposing surface through the
    ///    `decalable_surface` mask (flags & 0x3B != 0 → reject — bits
    ///    0/1/3/4/5 = invisible / two-sided / breakable / sky /
    ///    climbable), material != -1, optional `same_material_only`
    ///    (compare `runtime_global_material_type`), optional
    ///    `same_plane_only` (compare plane normal components within
    ///    1e-4).
    /// 6. Write fold: `normal` = opposing plane normal, `origin` =
    ///    `start`, `axis` = `normalize(end - start)`.
    ///
    /// `collision_materials` is `this.m_structure_bsp.collision_materials`
    /// (24-byte per entry in the engine; engine offset 0x14
    /// `seam_mapping_block_index` and offset 0x10
    /// `runtime_global_material_type` are the consumed fields).
    pub fn get_opposing_surface_fold(
        &self,
        bsp: &Bsp3d,
        collision_materials: &[BspCollisionMaterial],
        start: RealPoint3d,
        end: RealPoint3d,
        fold: &mut Fold,
        same_plane_only: bool,
        same_material_only: bool,
    ) -> bool {
        // Sentinel init (engine lines 76-87): -1 indices + identity-ish
        // normal/axis. We only need to track the surface_index hit
        // path; downstream filters early-return when sentinel is
        // still present at LABEL_66.
        fold.bsp_index = -1;
        fold.instance_definition_index = -1;
        fold.surface_index = -1;
        fold.normal = RealVector3d::default();
        fold.origin = RealPoint3d::default();
        fold.axis = RealVector3d::default();

        // --- Step 1: locate the opposing surface across the current edge.
        if self.surface_edge_index == -1 {
            return false;
        }
        let Some(edge) = bsp.edges.get(self.surface_edge_index as usize).copied() else {
            return false;
        };
        let on_right = edge.right_surface as i32 == self.surface_index;
        let opposing_surface_idx = if on_right {
            edge.left_surface as i32
        } else {
            edge.right_surface as i32
        };
        let Some(opposing_surface) = bsp.surfaces.get(opposing_surface_idx as usize)
        else {
            return false;
        };

        // --- Step 2-4: decide which BSP the opposing surface lives in.
        if opposing_surface.material != -1 {
            // The engine's "is this an instance iteration?" fast path —
            // instances never participate in cross-BSP seams.
            let take_same_bsp = if self.instance_definition_index != -1 {
                true
            } else {
                // Check the THIS surface's material's seam mapping. If
                // the material has no seam, the opposing surface is
                // accepted in the same BSP. (Engine line 156: `*((_WORD
                // *)v26 + 10) == 0xFFFF` reads byte 20 = the engine
                // field `structure_collision_material::seam_mapping_block_index`.
                // blam-tags schema names this `seam_mapping_index`.)
                let this_surface = bsp.surfaces.get(self.surface_index as usize);
                let seam_mapping = this_surface
                    .and_then(|s| collision_materials.get(s.material as usize))
                    .map(|m| m.seam_mapping_index)
                    .unwrap_or(-1);
                seam_mapping == -1
            };

            if take_same_bsp {
                fold.bsp_index = self.bsp_index;
                fold.instance_definition_index = self.instance_definition_index;
                fold.surface_index = opposing_surface_idx;
            }
            // The cross-BSP seam-traversal path is intentionally
            // unported. Engine line 167-275 calls
            // `structure_seams_connected_edge_get` + iterates the two
            // surfaces of the seam-connected edge in the OTHER BSP,
            // picking the first one whose collision_material has
            // `seam_mapping_block_index == -1`. We stub this as "no
            // connection found" — cyberdyne is single-BSP so this
            // change is inert there.
        }

        // --- Step 5 (LABEL_66): if we still have no surface, give up.
        if fold.surface_index == -1 {
            return false;
        }

        // We resolved the opposing surface in THE SAME bsp as `self`
        // (cross-BSP path is stubbed, so this is always true). Re-load
        // it for the post-filter pass.
        let Some(opposing) = bsp.surfaces.get(fold.surface_index as usize) else {
            return false;
        };

        // `decalable_surface` filter — engine line 317. Reject the 0x3B
        // set: TwoSided / Invisible / Breakable / Invalid / Conveyor.
        use blam_tags::structure_bsp::CollisionSurfaceFlags::*;
        if opposing.flags.test_any(&[TwoSided, Invisible, Breakable, Invalid, Conveyor]) {
            return false;
        }
        if opposing.material == -1 {
            return false;
        }

        // `same_material_only` — engine line 322: compare per-surface
        // `runtime_global_material_type` (engine field at material
        // struct offset 0x10).
        if same_material_only {
            let this_material_type = bsp
                .surfaces
                .get(self.surface_index as usize)
                .and_then(|s| collision_materials.get(s.material as usize))
                .map(|m| m.runtime_global_material_index)
                .unwrap_or(-1);
            let opposing_material_type = collision_materials
                .get(opposing.material as usize)
                .map(|m| m.runtime_global_material_index)
                .unwrap_or(-1);
            if this_material_type != opposing_material_type {
                return false;
            }
        }

        // Plane lookup — engine line 359.
        let plane = bsp3d_get_plane_from_designator(bsp, opposing.plane_designator);

        // `same_plane_only` — engine line 360-366: |this_plane.n -
        // opposing_plane.n|_inf < ~1e-4.
        if same_plane_only {
            const EPS: f32 = 0.000099999997;
            if (plane.i - self.plane.i).abs() >= EPS
                || (plane.j - self.plane.j).abs() >= EPS
                || (plane.k - self.plane.k).abs() >= EPS
            {
                return false;
            }
        }

        // --- Step 6: write the fold. Normal = opposing plane (i,j,k).
        // Origin = `start`. Axis = normalize(end - start).
        fold.normal = RealVector3d {
            i: plane.i,
            j: plane.j,
            k: plane.k,
        };
        fold.origin = start;
        let mut axis = RealVector3d {
            i: end.x - start.x,
            j: end.y - start.y,
            k: end.z - start.z,
        };
        let len_sq = axis.i * axis.i + axis.j * axis.j + axis.k * axis.k;
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            axis.i *= inv_len;
            axis.j *= inv_len;
            axis.k *= inv_len;
        }
        fold.axis = axis;
        true
    }
}
