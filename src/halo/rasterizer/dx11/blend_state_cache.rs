//! Mirror of `Ares/source/rasterizer/dx11/rasterizer_dx11_blend_state_cache.{h,cpp}`.
//!
//! Engine `set_alpha_blend_mode_no_cache @ 0x18066E930` walks
//! `gc_d3d11_render_target_blend_state_disabled = 0x3C844422` as the
//! BlendEnable=FALSE sentinel and emits a packed-int blend state per
//! mode. Decoded values in `reference_engine_blend_modes_decoded.md`.

#[derive(Default)]
pub struct BlendStateCache {
    // For wgpu, the state is materialized at pipeline construction
    // time (since `wgpu::BlendState` is part of `ColorTargetState`).
    // We don't actually need a runtime hashmap — the cache exists in
    // the pipeline cache via the blend mode being part of the
    // pipeline key. Kept here for API parity with engine.
    _placeholder: (),
}

impl BlendStateCache {
    pub fn new() -> Self {
        Self::default()
    }
}
