//! `d3d11_sampler_state_cache` mirror — engine `c_rasterizer` sampler cache.
//!
//! Engine: `d3d11_sampler_state_cache::get @ 0x1806F1C70` (dllcache).
//! Keyed by a packed `u32` sampler-state ID with the following bit
//! layout (audit-verified 2026-05-13):
//!
//! ```text
//! bits 0-3   filter_mode    → indexes `c_d3d11_filter_modes[8]` → (D3D11_FILTER, max_aniso)
//! bits 4-6   address_u      → indexes `c_d3d11_texture_address_modes[4]` (WRAP/CLAMP/MIRROR/BORDER)
//! bits 7-9   address_v      → same table
//! bits 10-12 address_w      → same table
//! bits 13-16 mip_bias_idx   → bias = 2.0 - (idx * 0.25)
//! bit  17    min_lod_active
//! bits 18-21 min_lod_value
//! bit  22    max_lod_active
//! bits 23-26 max_lod_value
//! ```
//!
//! Protomorph stores the cache key as a typed struct (better Rust
//! ergonomics than the u32 packing). mip_bias / min_lod / max_lod are
//! omitted from the key for now — blam-tags doesn't expose these per
//! parameter today; if a rmt2 needs them, extend the key.
//!
//! ## Engine → wgpu filter mapping (verified against `c_d3d11_filter_modes`):
//! | Engine `BitmapFilterMode`    | wgpu (mag, min, mip)              | aniso |
//! |---|---|---|
//! | `Trilinear`                  | Linear/Linear/Linear              | 1     |
//! | `Point`                      | Nearest/Nearest/Nearest           | 1     |
//! | `Bilinear`                   | Linear/Linear/Nearest             | 1     |
//! | `Anisotropic1`               | Linear/Linear/Linear              | 1     |
//! | `Anisotropic2Expensive`      | Linear/Linear/Linear              | 2     |
//! | `Anisotropic3Expensive`      | Linear/Linear/Linear              | **4** (wgpu requires power-of-2; engine 3 rounds up) |
//! | `Anisotropic4Expensive`      | Linear/Linear/Linear              | 4     |
//! | `LightprobeTextureArray`     | Linear/Linear/Linear              | 1     |
//! | `ComparisonPoint`            | Nearest/Nearest/Nearest           | 1     |
//! | `ComparisonBilinear`         | Linear/Linear/Nearest             | 1     |
//!
//! ## Address mode mapping
//! | `BitmapAddressMode` | wgpu |
//! |---|---|
//! | `Wrap` | `Repeat` |
//! | `Clamp` | `ClampToEdge` |
//! | `Mirror` | `MirrorRepeat` |
//! | `BlackBorder` | `ClampToEdge` (wgpu `ClampToBorder` requires a feature; degrades safely) |

use std::collections::HashMap;

use blam_tags::render_method::{BitmapAddressMode, BitmapFilterMode};

/// Cache key — derived from blam-tags' per-bitmap-parameter sampler
/// config (`BitmapBinding`). Hashable so the cache can dedupe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerKey {
    pub filter: BitmapFilterMode,
    pub address_u: BitmapAddressMode,
    pub address_v: BitmapAddressMode,
    pub address_w: BitmapAddressMode,
}

impl SamplerKey {
    /// Protomorph fallback / "filtering_sampler" baseline — what the
    /// existing `shared.filtering_sampler` does today. Linear/Repeat
    /// without anisotropy. Used when a material declares no bitmap
    /// bindings whose filter mode can drive sampler selection.
    pub const FILTERING_FALLBACK: Self = Self {
        filter: BitmapFilterMode::Trilinear,
        address_u: BitmapAddressMode::Wrap,
        address_v: BitmapAddressMode::Wrap,
        address_w: BitmapAddressMode::Wrap,
    };

    /// Same as `for_binding` but takes the resolved render method
    /// directly. The WGSL assembler doesn't carry a full `MaterialData`
    /// (only `ResolvedRenderMethod`), so this is the lower-level
    /// variant the per-binding sampler map calls.
    pub fn for_binding_in_resolved(
        rm: &blam_tags::render_method::ResolvedRenderMethod,
        name: &str,
    ) -> Option<Self> {
        use blam_tags::render_method::{ParameterSource, ResolvedValue};
        for p in &rm.parameters {
            if p.name == name {
                if let ParameterSource::Inline(ResolvedValue::Bitmap(b)) = &p.source {
                    return Some(Self {
                        filter: b.filter_mode,
                        // Per-axis address modes mirror the engine's
                        // `s_render_method_postprocess_texture` layout
                        // (`address_mode_x : 4` / `address_mode_y : 4`).
                        // address_w gets the address_mode_x broadcast
                        // per engine `c_rasterizer::set_sampler_address_mode(reg, x, y, x)`
                        // at `render_method_submit_volatile_per_shader @ 0x180685cf0`.
                        // 2D textures ignore W; cubemaps use this for the
                        // third axis and need the broadcast to match the
                        // engine's runtime behavior.
                        address_u: b.address_mode_x,
                        address_v: b.address_mode_y,
                        address_w: b.address_mode_x,
                    });
                }
            }
        }
        None
    }
}

/// Per-rasterizer sampler cache. Mirrors `d3d11_sampler_state_cache`'s
/// pattern — wgpu::Sampler is internally Arc'd, so storing by value and
/// cheap-cloning out matches the engine's pointer-handing-out behavior.
#[derive(Debug, Default)]
pub struct SamplerStateCache {
    cache: HashMap<SamplerKey, wgpu::Sampler>,
}

impl SamplerStateCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-create the sampler matching `key`. Allocation hits only on
    /// the first call per unique key (handful of samplers across an
    /// entire scenario).
    pub fn get(&mut self, device: &wgpu::Device, key: SamplerKey) -> wgpu::Sampler {
        if let Some(s) = self.cache.get(&key) {
            return s.clone();
        }
        let sampler = build_sampler(device, key);
        self.cache.insert(key, sampler.clone());
        sampler
    }
}

fn build_sampler(device: &wgpu::Device, key: SamplerKey) -> wgpu::Sampler {
    let (mag, min, mip, aniso) = filter_to_wgpu(key.filter);
    let label = format!(
        "sampler_{:?}_{:?}_{:?}_{:?}_aniso{}",
        key.filter, key.address_u, key.address_v, key.address_w, aniso,
    );
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(&label),
        address_mode_u: address_to_wgpu(key.address_u),
        address_mode_v: address_to_wgpu(key.address_v),
        address_mode_w: address_to_wgpu(key.address_w),
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mip,
        anisotropy_clamp: aniso,
        ..Default::default()
    })
}

/// Engine `BitmapFilterMode` → wgpu `(mag, min, mip, anisotropy_clamp)`.
/// Engine table caps at aniso=4; wgpu requires power-of-2 (1/2/4/8/16),
/// so the engine's aniso=3 entry rounds up to wgpu's 4.
fn filter_to_wgpu(
    mode: BitmapFilterMode,
) -> (wgpu::FilterMode, wgpu::FilterMode, wgpu::MipmapFilterMode, u16) {
    use BitmapFilterMode as F;
    use wgpu::FilterMode as M;
    use wgpu::MipmapFilterMode as Mp;
    match mode {
        F::Trilinear              => (M::Linear,  M::Linear,  Mp::Linear,  1),
        F::Point                  => (M::Nearest, M::Nearest, Mp::Nearest, 1),
        F::Bilinear               => (M::Linear,  M::Linear,  Mp::Nearest, 1),
        F::Anisotropic1           => (M::Linear,  M::Linear,  Mp::Linear,  1),
        F::Anisotropic2Expensive  => (M::Linear,  M::Linear,  Mp::Linear,  2),
        F::Anisotropic3Expensive  => (M::Linear,  M::Linear,  Mp::Linear,  4),
        F::Anisotropic4Expensive  => (M::Linear,  M::Linear,  Mp::Linear,  4),
        F::LightprobeTextureArray => (M::Linear,  M::Linear,  Mp::Linear,  1),
        F::ComparisonPoint        => (M::Nearest, M::Nearest, Mp::Nearest, 1),
        F::ComparisonBilinear     => (M::Linear,  M::Linear,  Mp::Nearest, 1),
    }
}

fn address_to_wgpu(mode: BitmapAddressMode) -> wgpu::AddressMode {
    match mode {
        BitmapAddressMode::Wrap        => wgpu::AddressMode::Repeat,
        BitmapAddressMode::Clamp       => wgpu::AddressMode::ClampToEdge,
        BitmapAddressMode::Mirror      => wgpu::AddressMode::MirrorRepeat,
        // wgpu::AddressMode::ClampToBorder requires the
        // `ADDRESS_MODE_CLAMP_TO_BORDER` feature; not enabled in
        // protomorph today. ClampToEdge is the closest safe fallback —
        // visible only at extreme grazing UVs where the cubemap/border
        // sample would have shown the border color.
        BitmapAddressMode::BlackBorder => wgpu::AddressMode::ClampToEdge,
    }
}
