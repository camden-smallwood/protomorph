//! Mirror of `Ares/source/rasterizer/rasterizer_clear.h`.
//!
//! `e_clear` flag bits — passed to `c_rasterizer::clear` to choose
//! which color targets / depth / stencil to clear.

pub mod clear_flags {
    pub const TARGET_0: u32 = 1 << 0;
    pub const TARGET_1: u32 = 1 << 1;
    pub const TARGET_2: u32 = 1 << 2;
    pub const TARGET_3: u32 = 1 << 3;
    pub const ALL_TARGETS: u32 = TARGET_0 | TARGET_1 | TARGET_2 | TARGET_3;
    pub const Z_BUFFER: u32 = 1 << 4;
    pub const STENCIL: u32 = 1 << 5;
}
