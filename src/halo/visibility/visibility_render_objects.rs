//! Mirror of `Ares/source/visibility/visibility_render_objects.h`.

/// `s_render_object_info` (visibility_render_objects.h:16-29, 76B).
///
/// Per-object render data populated by
/// `visibility_render_objects_get_model_info`. Carries the runtime
/// references the object renderer needs: render_model index,
/// region/mesh selection, skinning context, lightmap object index,
/// shader extern info.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct RenderObjectInfo {
    pub region_count: i32,                          // 0x0
    pub render_model_index: i32,                    // 0x4
    /// Memory-region designator into the per-frame skinning matrix
    /// pool (Halo packs all visible objects' skinning matrices into
    /// one contiguous buffer; this is the offset within it).
    pub skinning_memory_designator: i32,            // 0x8
    pub skinning_matrix_count: u8,                  // 0xC
    pub region_mesh_indices: [u8; 16],              // 0xD — selected mesh per region
    _pad0: u8,                                      // 0x1D — align u16 array to 0x1E
    pub region_z_sort_offset_enum_index: [u16; 16], // 0x1E
    pub lod_index: i16,                             // 0x3E
    pub lightmap_object_index: i16,                 // 0x40
    _pad1: [u8; 2],                                 // 0x42 — align u32 to 0x44
    pub clip_plane_masks: u32,                      // 0x44
    /// Halo: `ptr32_t<s_shader_extern_info> render_info`. We keep as
    /// an i32 designator into a parallel render-info table.
    pub render_info_index: i32,                     // 0x48 (was 32-bit pointer)
}

const _: () = assert!(std::mem::size_of::<RenderObjectInfo>() == 76);
