//! Mirror of `Ares/source/rasterizer/dx11/rasterizer_dx11_blend_state_cache.{h,cpp}`.
//!
//! Engine `set_alpha_blend_mode_no_cache @ 0x18066E930` walks
//! `gc_d3d11_render_target_blend_state_disabled = 0x3C844422` as the
//! BlendEnable=FALSE sentinel and emits a packed-int blend state per
//! mode. Decoded values in `reference_engine_blend_modes_decoded.md`.

use crate::halo::rasterizer::{AlphaBlendMode, SeparateAlphaBlendMode};

/// Cache key — exact mirror of `rasterizer_dx11_blend_state_cache`'s
/// hash key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendStateKey {
    pub alpha_blend_mode: AlphaBlendMode,
    pub separate_alpha_blend_mode: SeparateAlphaBlendMode,
    pub color_write_enable: [u32; 4],
    pub alpha_to_coverage: bool,
}

/// Resolved blend state for a single render-target slot. Engine's
/// "Opaque" maps to `BlendEnable=FALSE` which is `None` in wgpu's
/// `ColorTargetState::blend`.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedBlendState {
    pub blend: Option<wgpu::BlendState>,
    pub write_mask: wgpu::ColorWrites,
    /// True for MotionBlurStatic / MotionBlurInhibit — caller must
    /// `rpass.set_blend_constant([0,0,0,0])` resp. `[1,1,1,1]` after
    /// pipeline bind.
    pub needs_blend_constant: Option<[f32; 4]>,
}

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

    /// Resolve `(AlphaBlendMode, SeparateAlphaBlendMode)` to the
    /// wgpu blend state. Mirrors
    /// `rasterizer_dx11_blend_state_cache::get` / engine's
    /// `set_alpha_blend_mode_no_cache @ 0x18066E930`.
    ///
    /// `separate_alpha_blend_mode` overrides the alpha-channel pair
    /// when not `Off`. Engine uses this for the "color is opaque,
    /// alpha is additive"-style splits.
    pub fn get(
        &mut self,
        alpha_blend_mode: AlphaBlendMode,
        separate_alpha_blend_mode: SeparateAlphaBlendMode,
        color_write_enable: [u32; 4],
        _alpha_to_coverage: bool,
    ) -> ResolvedBlendState {
        let mut state = resolve_alpha_blend_mode(alpha_blend_mode);

        if let Some(blend) = state.blend.as_mut() {
            if separate_alpha_blend_mode != SeparateAlphaBlendMode::Off {
                blend.alpha = resolve_separate_alpha(separate_alpha_blend_mode);
            }
        }

        // Slot 0's write mask drives `ColorTargetState::write_mask`.
        // Engine has per-slot masks (color_write_enable[4]); we expose
        // slot 0 here. Multi-RT callers query per-slot via repeated calls.
        state.write_mask = color_writes_from_u32(color_write_enable[0]);
        state
    }
}

/// Map `e_alpha_blend_mode` to `(wgpu::BlendState, optional blend
/// constant)`. Engine source: `set_alpha_blend_mode_no_cache @
/// 0x18066E930` + `reference_engine_blend_modes_decoded.md`.
fn resolve_alpha_blend_mode(mode: AlphaBlendMode) -> ResolvedBlendState {
    use wgpu::{BlendComponent, BlendFactor as BF, BlendOperation as OP, BlendState};

    let blend = match mode {
        AlphaBlendMode::Opaque => None,
        AlphaBlendMode::Additive => Some(BlendState {
            color: BlendComponent { src_factor: BF::One, dst_factor: BF::One, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::One, dst_factor: BF::One, operation: OP::Add },
        }),
        AlphaBlendMode::Multiply => Some(BlendState {
            color: BlendComponent { src_factor: BF::Dst, dst_factor: BF::Zero, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::DstAlpha, dst_factor: BF::Zero, operation: OP::Add },
        }),
        AlphaBlendMode::AlphaBlend => Some(BlendState::ALPHA_BLENDING),
        AlphaBlendMode::DoubleMultiply => Some(BlendState {
            color: BlendComponent { src_factor: BF::Dst, dst_factor: BF::Src, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::DstAlpha, dst_factor: BF::SrcAlpha, operation: OP::Add },
        }),
        AlphaBlendMode::PreMultipliedAlpha | AlphaBlendMode::MultiplyAdd => {
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING)
        }
        AlphaBlendMode::Maximum => Some(BlendState {
            color: BlendComponent { src_factor: BF::One, dst_factor: BF::One, operation: OP::Max },
            alpha: BlendComponent { src_factor: BF::One, dst_factor: BF::One, operation: OP::Max },
        }),
        AlphaBlendMode::AddSrcTimesDstAlpha => Some(BlendState {
            color: BlendComponent { src_factor: BF::DstAlpha, dst_factor: BF::One, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::DstAlpha, dst_factor: BF::One, operation: OP::Add },
        }),
        AlphaBlendMode::AddSrcTimesSrcAlpha => Some(BlendState {
            color: BlendComponent { src_factor: BF::SrcAlpha, dst_factor: BF::One, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::SrcAlpha, dst_factor: BF::One, operation: OP::Add },
        }),
        AlphaBlendMode::InvAlphaBlend => Some(BlendState {
            color: BlendComponent { src_factor: BF::OneMinusSrcAlpha, dst_factor: BF::SrcAlpha, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::OneMinusSrcAlpha, dst_factor: BF::SrcAlpha, operation: OP::Add },
        }),
        AlphaBlendMode::MotionBlurStatic | AlphaBlendMode::MotionBlurInhibit => Some(BlendState {
            color: BlendComponent { src_factor: BF::One, dst_factor: BF::Zero, operation: OP::Add },
            alpha: BlendComponent { src_factor: BF::Constant, dst_factor: BF::Zero, operation: OP::Add },
        }),
    };

    // Engine calls `set_blend_factor(0)` for MotionBlurStatic and
    // `set_blend_factor(0xFFFFFFFF)` for MotionBlurInhibit — caller
    // applies via `rpass.set_blend_constant(...)`.
    let needs_blend_constant = match mode {
        AlphaBlendMode::MotionBlurStatic => Some([0.0; 4]),
        AlphaBlendMode::MotionBlurInhibit => Some([1.0; 4]),
        _ => None,
    };

    ResolvedBlendState {
        blend,
        write_mask: wgpu::ColorWrites::ALL,
        needs_blend_constant,
    }
}

/// Override the alpha-channel blend pair when `separate_alpha_blend_mode != Off`.
fn resolve_separate_alpha(mode: SeparateAlphaBlendMode) -> wgpu::BlendComponent {
    use wgpu::{BlendComponent, BlendFactor as BF, BlendOperation as OP};
    match mode {
        SeparateAlphaBlendMode::Off => BlendComponent::REPLACE,
        SeparateAlphaBlendMode::Opaque => BlendComponent {
            src_factor: BF::One, dst_factor: BF::Zero, operation: OP::Add,
        },
        SeparateAlphaBlendMode::Additive => BlendComponent {
            src_factor: BF::One, dst_factor: BF::One, operation: OP::Add,
        },
        SeparateAlphaBlendMode::Multiply => BlendComponent {
            src_factor: BF::DstAlpha, dst_factor: BF::Zero, operation: OP::Add,
        },
        SeparateAlphaBlendMode::ToConstant => BlendComponent {
            src_factor: BF::Constant, dst_factor: BF::Zero, operation: OP::Add,
        },
    }
}

/// Decode `color_write_enable` u32 mask to wgpu's `ColorWrites`. Engine
/// uses the same bit layout as D3D11 (`color_write_enable::RED|GREEN|BLUE|ALPHA`).
fn color_writes_from_u32(mask: u32) -> wgpu::ColorWrites {
    use crate::halo::rasterizer::color_write_enable as cw;
    let mut out = wgpu::ColorWrites::empty();
    if mask & cw::RED != 0 { out |= wgpu::ColorWrites::RED; }
    if mask & cw::GREEN != 0 { out |= wgpu::ColorWrites::GREEN; }
    if mask & cw::BLUE != 0 { out |= wgpu::ColorWrites::BLUE; }
    if mask & cw::ALPHA != 0 { out |= wgpu::ColorWrites::ALPHA; }
    out
}
