//! Mirror of `Ares/source/geometry/triangle_strip_utilities.h`
//! (24 lines).
//!
//! Triangle-strip → triangle-list conversion + degenerate-strip
//! detection. blam-tags already has the canonical
//! `strip_to_list` helper that the BSP+model loaders use; the
//! signatures here mirror Halo's API for callers that might want
//! the count + iteration helpers.

/// `count_strip_triangles @ h:13`. Walks a strip-indices array,
/// counts the valid (non-degenerate) and degenerate triangles.
/// Halo's strip format inserts degenerate triangles (two-equal-
/// indices entries) as separators between sub-strips — those are
/// considered "degenerate" here.
pub fn count_strip_triangles(strip_indices: &[u16]) -> (usize, usize) {
    if strip_indices.len() < 3 {
        return (0, 0);
    }
    let mut valid = 0;
    let mut degenerate = 0;
    for triangle_start in 0..(strip_indices.len() - 2) {
        let a = strip_indices[triangle_start];
        let b = strip_indices[triangle_start + 1];
        let c = strip_indices[triangle_start + 2];
        if a == b || b == c || a == c {
            degenerate += 1;
        } else {
            valid += 1;
        }
    }
    (valid, degenerate)
}

/// `find_next_valid_strip_triangle @ h:14`. Walks forward from
/// `previous_triangle_start_index` until a non-degenerate triangle
/// is found; returns its starting index, or -1 if none.
pub fn find_next_valid_strip_triangle(
    strip_indices: &[u16],
    previous_triangle_start_index: i64,
) -> i64 {
    let start = (previous_triangle_start_index + 1).max(0) as usize;
    if strip_indices.len() < 3 {
        return -1;
    }
    for triangle_start in start..(strip_indices.len() - 2) {
        let a = strip_indices[triangle_start];
        let b = strip_indices[triangle_start + 1];
        let c = strip_indices[triangle_start + 2];
        if a != b && b != c && a != c {
            return triangle_start as i64;
        }
    }
    -1
}

/// `find_triangle @ h:15`. Searches `indices` for the triangle
/// (A, B, C) considering all 3 rotations. Returns
/// (triangle_index, rotation) or (-1, _) if not found.
/// `strips=true` walks as a strip (sliding window of 3); else
/// walks 3-at-a-time.
pub fn find_triangle(
    vertex_a: u16,
    vertex_b: u16,
    vertex_c: u16,
    indices: &[u16],
    strips: bool,
) -> (i64, i64) {
    let stride = if strips { 1 } else { 3 };
    let mut i = 0;
    while i + 2 < indices.len() {
        let a = indices[i];
        let b = indices[i + 1];
        let c = indices[i + 2];
        if a == vertex_a && b == vertex_b && c == vertex_c {
            return (i as i64, 0);
        }
        if a == vertex_b && b == vertex_c && c == vertex_a {
            return (i as i64, 1);
        }
        if a == vertex_c && b == vertex_a && c == vertex_b {
            return (i as i64, 2);
        }
        i += stride;
    }
    (-1, 0)
}
