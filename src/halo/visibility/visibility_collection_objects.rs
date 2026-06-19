//! Mirror of `Ares/source/visibility/visibility_collection_objects.h`.

use crate::halo::visibility::{LodTransparency, RenderObjectInfo};

/// `s_visible_object_hierarchy` (visibility_collection_objects.h:25-32,
/// 16B). One per root object — groups its visibility entries +
/// region cluster bitvector start index.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VisibleObjectHierarchy {
    pub flags: i16,                                // 0x0
    pub region_cluster_bitvector_start_index: u16, // 0x2
    pub first_visibility_object_index: i32,        // 0x4
    pub visibility_object_index_count: u16,        // 0x8
    pub lod_transparency: LodTransparency,         // 0xA  (4B u8 fields)
    _pad0: [u8; 2],                                // 0xE — align to 16B
}

const _: () = assert!(std::mem::size_of::<VisibleObjectHierarchy>() == 16);

/// `s_visible_object_render_visibility` (96B).
///
/// One per visible object instance (root + children). Pairs the
/// `RenderObjectInfo` (76B) the renderer reads with the visibility-
/// side fields (object_index, subpart bitvector, flags).
///
/// `subpart_bitvector` is engine-side `unsigned long *` — a 64-bit
/// pointer into `c_visible_items::m_items.visible_subpart_bitvector`.
/// We store the raw 8-byte slot opaquely (size parity); accessor on
/// `c_visible_items` resolves to a `&[u32]` slice when needed.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct VisibleObjectRenderVisibility {
    pub info: RenderObjectInfo,         // 0x0  (76B)
    pub object_index: i32,              // 0x4C
    pub subpart_bitvector: u64,         // 0x50 — opaque pointer slot
    pub flags: u8,                      // 0x58
    _pad0: [u8; 7],                     // 0x59 — align to 96B
}

const _: () = assert!(std::mem::size_of::<VisibleObjectRenderVisibility>() == 96);

