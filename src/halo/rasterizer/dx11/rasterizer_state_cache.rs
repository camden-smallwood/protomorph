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

use crate::halo::rasterizer::{CullMode, FillMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RasterizerStateKey {
    pub cull_mode: CullMode,
    pub fill_mode: FillMode,
    pub scissor_enabled: bool,
}

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

    /// Resolve to wgpu's `PrimitiveState`. Mirrors
    /// `rasterizer_dx11_rasterizer_state_cache::get` / engine
    /// `set_cull_mode @ 0x18066ECF0`.
    ///
    /// `topology` defaults to `TriangleList` since engine almost
    /// always draws triangles; callers needing strips override after
    /// the call. `polygon_mode` derives from FillMode.
    pub fn get(
        &mut self,
        cull_mode: CullMode,
        fill_mode: FillMode,
        _scissor_enabled: bool,
    ) -> wgpu::PrimitiveState {
        // Engine front_face is CCW (D3D11 default `FrontCounterClockwise = FALSE`
        // would make CW the front, but Halo's authoring tools export
        // CCW-wound triangles — verified via tag mesh tooling. The
        // engine's `set_cull_mode` sets D3D11 `FrontCounterClockwise =
        // FALSE` and cull = back for the default rendering pass).
        //
        // For PC builds — TBD if engine flips this. Until verified,
        // mirror what protomorph's existing pipelines already use.
        let (cull, front_face) = match cull_mode {
            CullMode::Off => (None, wgpu::FrontFace::Ccw),
            CullMode::Cw => (Some(wgpu::Face::Back), wgpu::FrontFace::Ccw),
            CullMode::Ccw => (Some(wgpu::Face::Front), wgpu::FrontFace::Ccw),
        };

        let polygon_mode = match fill_mode {
            FillMode::Point => wgpu::PolygonMode::Point,
            FillMode::Wireframe => wgpu::PolygonMode::Line,
            FillMode::Solid => wgpu::PolygonMode::Fill,
        };

        wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face,
            cull_mode: cull,
            unclipped_depth: false,
            polygon_mode,
            conservative: false,
        }
    }
}
