//! Mirror of `Ares/source/rasterizer/dx11/rasterizer_dx11_input_layout_cache.{h,cpp}`.
//! Caches D3D11 input layouts — wgpu equivalent is the
//! `vertex.buffers[]` arrays in pipeline descriptors.

#[derive(Default)]
pub struct InputLayoutCache {
    _placeholder: (),
}

impl InputLayoutCache {
    pub fn new() -> Self { Self::default() }
}
