//! Mirror of `Ares/source/rasterizer/dx11/rasterizer_dx11_depth_stencil_state_cache.{h,cpp}`.
//!
//! Engine `set_z_buffer_mode @ 0x18066EC60` reads
//! `m_use_floating_point_z_buffer` (PC = true → reverse-Z) to pick the
//! comparison func, then applies per-mode depth bias values. All
//! decoded in `reference_engine_depth_stencil_modes.md`.
//!
//! StencilMode bit-packing is deferred — protomorph hasn't ported
//! the volume/tron/decals stencil flows yet. Implemented modes:
//! Off (no stencil) only. P10 fills in the rest.

#[derive(Default)]
pub struct DepthStencilStateCache {
    // wgpu materializes DepthStencilState at pipeline construction
    // (it's part of RenderPipelineDescriptor). The cache exists for
    // API parity — actual lookups happen in pipeline_cache via the
    // (z_buffer_mode, stencil_mode, ...) tuple being part of the key.
    _placeholder: (),
}

impl DepthStencilStateCache {
    pub fn new() -> Self {
        Self::default()
    }
}
