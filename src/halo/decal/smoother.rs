//! `c_decal::smooth_mesh_fragment` — vertex sort + dedupe pass that
//! runs per fragment between the BFS walker (`build_mesh_fragment`)
//! and the final pack (`write_mesh_fragment`).
//!
//! Despite the engine name, the function does NOT average normals.
//! It sorts the fragment's work vertices, collapses runs of identical
//! position + near-identical texcoord into a single output vertex,
//! and records the run intervals in `mesh_builder.grouper` so that
//! the later `write_mesh_fragment` step can average tangent-frame
//! normals across the merged inputs.
//!
//! Anchored against dllcache:
//!   `c_decal::smooth_mesh_fragment @ 0x18039D220`
//!   `c_decal::compare_vertices    @ 0x18039BC90`
//!   `c_decal_sorter::fill / sort` (inlined into smooth_mesh_fragment)
//!
//! Algorithm:
//!   1. Configure the sorter for `[working.starting_vertex,
//!      working.starting_vertex + working.vertex_count)`.
//!   2. Fill `sorter_order[start..end]` with the identity range.
//!   3. Sort `sorter_order` by `(position, texcoord.x, texcoord.y)`
//!      ascending (position compared as `u32`, matching the engine's
//!      `u64` pointer-bit compare on its `*real_point3d` field).
//!   4. Walk the sorted order; for each run of vertices with
//!      identical `position` (index/pointer equality) AND
//!      `|texcoord| < 0.02` in both axes, collapse to one output
//!      vertex and record the run as a `grouper` entry.
//!   5. Index count is preserved as-is (no triangle merging); the
//!      remap from work-index to output-index lives in
//!      `mesh_builder.collapser` and is consumed by
//!      `write_mesh_fragment`.

use super::types::{DecalFragment, DecalMeshBuilder, FragmentBufferIntervals};

/// Mirror of `c_decal::smooth_mesh_fragment @ 0x18039D220`.
///
/// Mutates `mesh_builder.sorter_order`, `collapser`, `grouper`, and
/// the running `output_*_count` totals; writes the fragment's
/// `output_intervals`.
pub fn smooth_mesh_fragment(mesh_builder: &mut DecalMeshBuilder, fragment: &mut DecalFragment) {
    let working_start = fragment.working_intervals.starting_vertex;
    let working_count = fragment.working_intervals.vertex_count;

    // Snapshot output cursors before the collapse loop.
    fragment.output_intervals.starting_vertex = mesh_builder.output_vertex_count;
    fragment.output_intervals.starting_index = mesh_builder.output_index_count;
    fragment.output_intervals.vertex_count = 0;
    fragment.output_intervals.index_count = 0;

    debug_assert!(working_start <= 0x400, "set_start beyond K_MAX_WORK_VERTICES");
    debug_assert!(working_count <= 0x400, "set_count beyond K_MAX_WORK_VERTICES");

    mesh_builder.sorter_start = working_start as u16;
    mesh_builder.sorter_count = working_count as u16;

    // === Step 1+2 (fill) ===
    // Engine `c_decal_sorter::fill` populates the order array with
    // identity indices in the active range. Subsequent sorting
    // permutes only this range.
    let start = working_start as usize;
    let count = working_count as usize;
    let end = start + count;
    for (i, slot) in (start..end).enumerate() {
        mesh_builder.sorter_order[slot] = (start + i) as u16;
    }

    // === Step 3 (sort) ===
    // Engine calls `qsort_2byte` with `compare_vertices` as the
    // predicate. The predicate's `data` arg is the mesh_builder, used
    // to dereference each u16 index back into the work_vertex_buffer.
    {
        // Borrow-split: take an immutable view of the work buffer for
        // the comparator, then sort the order range.
        let work = mesh_builder.work_vertex_buffer.as_ref();
        let slice = &mut mesh_builder.sorter_order[start..end];
        slice.sort_by(|&a, &b| compare_vertices(a, b, work));
    }

    // === Step 4 (collapse) ===
    // Walk the sorted order. Each run of consecutive identical-key
    // vertices collapses to a single output vertex. The `collapser`
    // table maps every work-vertex index in the run to that output
    // index; `grouper` records the run's position in the sorter for
    // the later write step to average normals.
    let mut cursor = start;
    while cursor < end {
        let run_start = cursor;
        let lead_work_idx = mesh_builder.sorter_order[run_start] as usize;
        let lead = mesh_builder.work_vertex_buffer[lead_work_idx];
        let output_idx = (fragment.output_intervals.starting_vertex
            + fragment.output_intervals.vertex_count) as u16;
        mesh_builder.collapser[lead_work_idx] = output_idx;

        cursor += 1;
        while cursor < end {
            let cand_work_idx = mesh_builder.sorter_order[cursor] as usize;
            let cand = mesh_builder.work_vertex_buffer[cand_work_idx];
            // Position match is integer equality (engine: pointer
            // equality, mirrored as collision_bsp.vertices[] index).
            if cand.position != lead.position {
                break;
            }
            // Texcoord match within 0.02 in each axis.
            if (cand.texcoord.x - lead.texcoord.x).abs() >= 0.02 {
                break;
            }
            if (cand.texcoord.y - lead.texcoord.y).abs() >= 0.02 {
                break;
            }
            mesh_builder.collapser[cand_work_idx] = output_idx;
            cursor += 1;
        }

        mesh_builder.grouper[output_idx as usize] = FragmentBufferIntervals {
            starting_vertex: run_start as u32,
            vertex_count: (cursor - run_start) as u32,
            starting_index: 0,
            index_count: 0,
        };
        fragment.output_intervals.vertex_count += 1;
    }

    // Index count is preserved — write_mesh_fragment remaps the
    // work-buffer indices via `collapser` when it packs.
    fragment.output_intervals.index_count = fragment.working_intervals.index_count;
    mesh_builder.output_vertex_count += fragment.output_intervals.vertex_count;
    mesh_builder.output_index_count += fragment.output_intervals.index_count;
}

/// Mirror of `c_decal::compare_vertices @ 0x18039BC90`. Strict ordering
/// `a < b` on `(position, texcoord.x, texcoord.y)`. Engine compares
/// the first 8 bytes (a `*const real_point3d` pointer) as `u64`;
/// our port stores `position` as `i32` (collision_bsp.vertices index)
/// and compares as `u32` to preserve the unsigned-bit-pattern order
/// the engine relies on.
fn compare_vertices(
    a: u16,
    b: u16,
    work: &[super::types::WorkingVertex],
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let va = work[a as usize];
    let vb = work[b as usize];
    let pa = va.position as u32;
    let pb = vb.position as u32;
    match pa.cmp(&pb) {
        Ordering::Equal => {}
        ord => return ord,
    }
    if va.texcoord.x < vb.texcoord.x {
        return Ordering::Less;
    }
    if va.texcoord.x > vb.texcoord.x {
        return Ordering::Greater;
    }
    va.texcoord.y.partial_cmp(&vb.texcoord.y).unwrap_or(Ordering::Equal)
}
