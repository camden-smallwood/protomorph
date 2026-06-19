//! Mirror of `Ares/source/rasterizer/dx11/rasterizer_dx11_rasterizer_state_cache.{h,cpp}`.
//! "Rasterizer state" in D3D11 = cull/fill/scissor/depth-bias.
//!
//! Engine `e_cull_mode { Off=1, Cw=2, Ccw=3 }` matches D3D11
//! D3D11_CULL_NONE/FRONT/BACK enumeration in a different order; engine
//! `set_cull_mode` calls D3D11RSSetState with the cull-mode mapped to
//! D3D11. We mirror to wgpu's `Face` model (Cw means cull-clockwise,
//! Ccw means cull-counterclockwise) with front_face derived from the
//! winding the engine treats as front.
//!
//! Engine convention (D3D11 default): winding=CW → front. So
//! `CullMode::Cw` means "cull back-facing (CCW) triangles" in our
//! protomorph terminology, BUT engine names it by what gets culled.
//! Verify direction by sampling engine `set_cull_mode @ 0x18066ECF0`.

#[derive(Default)]
pub struct RasterizerStateCache {
    // wgpu's PrimitiveState is materialized at pipeline construction
    // (RenderPipelineDescriptor.primitive). The cache exists for API
    // parity; per-pipeline lookups happen in pipeline_cache.
    _placeholder: (),
}

impl RasterizerStateCache {
    pub fn new() -> Self {
        Self::default()
    }
}
