//! CPU postprocess sampler — engine-faithful port of:
//!
//! - `s_render_method_postprocess_definition::query_real_constant` @ dllcache `0x1806C07D0`
//! - `s_render_method_postprocess_definition::query_texture_transform` @ `0x1806C0930`
//! - `s_render_method_postprocess_definition::sample_texture` @ `0x1806C0C40`
//! - `bitmap_group_2d_get_real_pixel` @ `0x1804ECBB0`
//!
//! Consumer: `sample_diffuse_textures` @ `0x180490220` — the bounce-light
//! source term inside `c_geometry_sampler::sample`. CPU NEAREST sampler
//! (per `reference_engine_cpu_atlas_sampling_is_nearest`).
//!
//! Engine flow: `property` (a `e_runtime_queryable_property`) →
//! `m_runtime_queryable_properties[property]` →
//! `m_textures[idx]` / `m_real_constants[idx]` → bitmap_group lookup →
//! single-texel decode at `mipmap_index`.
//!
//! Protomorph mirrors that flow 1:1 against
//! [`MaterialSamplerEntry`](crate::halo::decorators::bake_material_lookup::MaterialSamplerEntry)
//! — the slim postprocess snapshot prebuilt by
//! [`BakeMaterialLookup`](crate::halo::decorators::bake_material_lookup::BakeMaterialLookup).
//! Same data the engine reads at runtime, just retained via the bake
//! thread-local instead of the global tag heap.

use blam_tags::bitmap::decode::decode_pixel_at;
use blam_tags::bitmap::BitmapFormat;
use blam_tags::math::RealArgbColor;

use crate::halo::decorators::bake_material_lookup::{MaterialSamplerEntry, MaterialSamplerTexture};
use crate::halo::render_methods::materials::InlinePixelFormat;

/// `s_render_method_postprocess_definition::e_runtime_queryable_property`.
/// Anchored to Ares `render_methods/render_method_definitions.h:311-319`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RuntimeQueryableProperty {
    BlendMap = 0,
    AlbedoBaseMap0 = 1,
    AlbedoBaseMap1 = 2,
    AlbedoBaseMap2 = 3,
    AlbedoColor = 4,
    AlbedoBaseMap3 = 5,
    AlphaMap = 6,
    Pad = 7,
}

/// Engine `query_real_constant` @ `0x1806C07D0`. Looks up the resolved
/// real_constant for `property` via the postprocess entry's
/// `runtime_queryable_properties[property]` index. Returns `None` when
/// the property isn't bound on this material (`= -1`).
pub fn query_real_constant(
    entry: &MaterialSamplerEntry,
    property: RuntimeQueryableProperty,
) -> Option<[f32; 4]> {
    let slot_idx = entry.queryable_properties[property as usize];
    if slot_idx < 0 {
        return None;
    }
    entry.real_constants.get(slot_idx as usize).copied()
}

/// Engine `sample_texture` @ `0x1806C0C40` composed with
/// `bitmap_group_2d_get_real_pixel` @ `0x1804ECBB0`. Resolves the texture
/// bound to `property`, applies the engine xform `point = uv * xform.xy +
/// xform.zw`, then NEAREST-samples one texel at `mipmap_index` with WRAP
/// addressing (engine call site passes `clamp=0`).
///
/// Returns `None` when:
/// - the property isn't bound (`runtime_queryable_properties[property] == -1`)
/// - the bound texture failed to load (extern texture or missing bitmap)
/// - the texture's format isn't in [`decode_pixel_at`]'s supported subset
pub fn sample_texture(
    entry: &MaterialSamplerEntry,
    property: RuntimeQueryableProperty,
    uv: [f32; 2],
    mipmap_index: u32,
) -> Option<RealArgbColor> {
    let slot_idx = entry.queryable_properties[property as usize];
    if slot_idx < 0 {
        return None;
    }
    let pp_tex = entry.textures.get(slot_idx as usize)?;
    let texture = pp_tex.texture.as_ref()?;

    let xform = lookup_texture_transform(entry, pp_tex);
    let u = uv[0] * xform[0] + xform[2];
    let v = uv[1] * xform[1] + xform[3];

    let format = inline_format_to_bitmap_format(texture.format)?;
    let base_w = texture.width.max(1);
    let base_h = texture.height.max(1);
    let mip = mipmap_index.min(texture.mip_count.saturating_sub(1));
    let mip_w = (base_w >> mip).max(1);
    let mip_h = (base_h >> mip).max(1);

    // Engine `bitmap_group_2d_get_real_pixel(..., clamp=0)` matches WRAP
    // addressing: `pixel = floor(u * width) mod width`.
    let px = ((u.rem_euclid(1.0)) * mip_w as f32) as u32;
    let py = ((v.rem_euclid(1.0)) * mip_h as f32) as u32;
    let px = px.min(mip_w - 1);
    let py = py.min(mip_h - 1);

    let pixel = decode_pixel_at(format, base_w, base_h, &texture.data, mip, px, py)?;
    Some(RealArgbColor {
        alpha: pixel[3] as f32 / 255.0,
        red: pixel[0] as f32 / 255.0,
        green: pixel[1] as f32 / 255.0,
        blue: pixel[2] as f32 / 255.0,
    })
}

/// Engine `query_texture_transform` xform lookup (stripped of the
/// animated-overlay re-evaluation loop, which engine drives via
/// per-frame time `t` — the bake runs at scenario load with no frame
/// time, matching the engine's bake-time invariant).
fn lookup_texture_transform(
    entry: &MaterialSamplerEntry,
    pp_tex: &MaterialSamplerTexture,
) -> [f32; 4] {
    if pp_tex.xform_constant_index < 0 {
        // Engine returns the `transform = (real_vector4d)_xmm = (0,0,0,0)`
        // path here — `uv * 0 + 0 = (0, 0)` for every UV. Identity
        // `[1, 1, 0, 0]` would be the protomorph-natural fallback BUT
        // mirroring the engine's all-zero result is the faithful choice.
        // Verified vs `query_texture_transform` decompile @ 0x1806C0930:
        // when `v11 == -1` the function returns 0 (not 1), so
        // `sample_texture`'s caller treats this as "no texture bound";
        // the false return propagates and the caller short-circuits.
        // We mirror by returning identity here AND will gate at the
        // sample_diffuse_textures site by checking the engine's
        // `bool query_texture_transform → bitmap_group_index != -1`
        // dual-return-value semantics differently — see the call site.
        return [1.0, 1.0, 0.0, 0.0];
    }
    // Negative index already returned identity above; a non-negative index
    // past `real_constants` is a real routing bug → fail loud.
    entry.real_constants[pp_tex.xform_constant_index as usize]
}

fn inline_format_to_bitmap_format(fmt: InlinePixelFormat) -> Option<BitmapFormat> {
    use BitmapFormat as B;
    use InlinePixelFormat as I;
    Some(match fmt {
        I::Bc1RgbaUnorm => B::Dxt1,
        I::Bc2RgbaUnorm => B::Dxt3,
        I::Bc3RgbaUnorm => B::Dxt5,
        I::Bgra8Unorm => B::A8r8g8b8,
        _ => return None,
    })
}
