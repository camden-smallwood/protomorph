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

use crate::halo::rasterizer::{StencilMode, ZBufferMode};

/// Engine PC `_surface_depth_stencil` uses reverse-Z (m_use_floating_point_z_buffer = true).
/// Comparison flips from LessEqual to GreaterEqual; depth-bias values
/// also flip sign (engine multiplies stored value by -1.0 before
/// scaling on PC).
pub const REVERSE_Z: bool = true;

/// Default depth-bias values from `g_*_depth_bias` / `g_*_slope_depth_bias`
/// globals (engine dump at EAs in reference doc).
mod depth_bias {
    pub const SHADOW_GENERATE: (f32, f32) = (0.003, 1.414);
    pub const SHADOW_GENERATE_DL: (f32, f32) = (0.0008, 1.414);
    pub const SHADOW_APPLY: (f32, f32) = (-2.0e-7, -0.1);
    pub const DECALS: (f32, f32) = (-5.0e-6, -0.5);
    pub const DEBUG_GEOMETRY: (f32, f32) = (-1.0e-4, 0.0); // slope NaN in engine
}

/// Engine multiplies `stored_value * sign(reverse_z_factor) * 2^24` to
/// convert from fractional depth units to D3D11 integer DepthBias.
const DEPTH_BIAS_SCALE: f32 = 16777216.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilStateKey {
    pub z_buffer_mode: ZBufferMode,
    pub stencil_mode: StencilMode,
    pub stencil_value: u8,
    pub stencil_write_mask: u8,
}

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

    /// Resolve `(ZBufferMode, StencilMode)` to wgpu's
    /// `DepthStencilState`. Mirrors
    /// `rasterizer_dx11_depth_stencil_state_cache::get`.
    ///
    /// Returns `None` for `ZBufferMode::Off` to signal the pipeline
    /// should set `depth_stencil: None`. Caller must supply the
    /// expected `wgpu::TextureFormat` matching the bound depth target.
    pub fn get(
        &mut self,
        z_buffer_mode: ZBufferMode,
        _stencil_mode: StencilMode,
        _stencil_value: u8,
        _stencil_write_mask: u8,
        depth_format: wgpu::TextureFormat,
    ) -> Option<wgpu::DepthStencilState> {
        let (test, write, bias) = match z_buffer_mode {
            ZBufferMode::Write => (true, true, None),
            ZBufferMode::Read => (true, false, None),
            ZBufferMode::Off => return None,
            ZBufferMode::ShadowGenerate => (true, true, Some(depth_bias::SHADOW_GENERATE)),
            ZBufferMode::ShadowGenerateDynamicLights => {
                (true, true, Some(depth_bias::SHADOW_GENERATE_DL))
            }
            ZBufferMode::ShadowApply => (true, false, Some(depth_bias::SHADOW_APPLY)),
            ZBufferMode::Decals => (true, false, Some(depth_bias::DECALS)),
            ZBufferMode::DebugGeometry => (true, false, Some(depth_bias::DEBUG_GEOMETRY)),
        };

        let compare = if !test {
            wgpu::CompareFunction::Always
        } else if REVERSE_Z {
            wgpu::CompareFunction::GreaterEqual
        } else {
            wgpu::CompareFunction::LessEqual
        };

        let bias_state = match bias {
            Some((db, sb)) => {
                let sign = if REVERSE_Z { -1.0_f32 } else { 1.0_f32 };
                wgpu::DepthBiasState {
                    constant: (db * sign * DEPTH_BIAS_SCALE) as i32,
                    slope_scale: sb * sign,
                    clamp: 0.0,
                }
            }
            None => wgpu::DepthBiasState::default(),
        };

        // Stencil deferred — P10 fills in when volume/tron/decals
        // stencil flows land.
        Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: write,
            depth_compare: compare,
            stencil: wgpu::StencilState::default(),
            bias: bias_state,
        })
    }
}
