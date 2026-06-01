//! Mirror of `Ares/source/rasterizer/rasterizer_occlusion_queries.h`
//! (42 lines).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OcclusionQueryType {
    LensFlare = 0,
    Distortion = 1,
}

pub mod occlusion_query_rendering_flags {
    pub const OCCLUSION: u32 = 1 << 0;
    pub const CONDITIONAL: u32 = 1 << 1;
    pub const DONT_CULL_GEOMETRY: u32 = 1 << 2;
    pub const ALL: u32 = OCCLUSION | CONDITIONAL;
}

/// `rasterizer_occlusion_submit @ h:32`. Returns the query slot
/// index, or -1 if the queue is full.
pub fn rasterizer_occlusion_submit(
    _query_type: OcclusionQueryType,
    _flags: u32,
    _user_index: i32,
    _player_window_index: i32,
    _render_callback: Option<fn(i32)>,
) -> i32 {
    -1
}

/// `rasterizer_occlusions_retrieve @ h:33`.
pub fn rasterizer_occlusions_retrieve(_player_window_index: i32) {}

/// `rasterizer_occlusions_get_result @ h:34`. Returns true if the
/// query result is ready; writes pixels_visible +
/// conditional_rendering_index out-pointers.
pub fn rasterizer_occlusions_get_result(
    _query_type: OcclusionQueryType,
    _user_index: i32,
    _player_window_index: i32,
) -> Option<(i32, i32)> {
    None
}
