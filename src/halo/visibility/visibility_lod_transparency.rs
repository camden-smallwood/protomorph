//! Mirror of `Ares/source/visibility/visibility_lod_transparency.h`.

/// `s_lod_transparency` (visibility_lod_transparency.h:11-17, 4B).
///
/// LOD-fade alpha values. `model_alpha` fades the entire model out
/// for distance LOD; `instance_alpha` is the per-instance alpha
/// (used by BSP instanced geometry); `shadow_alpha` controls the
/// shadow-caster's fade independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LodTransparency {
    pub model_alpha: u8,
    pub instance_alpha: u8,
    pub shadow_alpha: u8,
    pub unused_alpha: u8,
}

const _: () = assert!(std::mem::size_of::<LodTransparency>() == 4);

impl LodTransparency {
    /// All-opaque (default rendering — no LOD fade).
    pub const OPAQUE: Self = Self {
        model_alpha: 0xFF,
        instance_alpha: 0xFF,
        shadow_alpha: 0xFF,
        unused_alpha: 0,
    };
}
