//! Halo `.bitmap` (bitm) tag → [`InlineTexture`] loader.
//!
//! Goal: pass the bitmap's pixel bytes straight to wgpu in their
//! native format whenever there's a 1:1 wgpu equivalent (BC1/2/3/4/5,
//! BGRA8, RGBA16/32, signed variants). This preserves disk
//! compression and the full mip pyramid — no round-trip through RGBA8.
//!
//! Halo-specific formats with no clean wgpu mapping
//! (`DxnMonoAlpha` — Halo's BC5 swizzle that decodes to
//! `(R, R, R, A)`; `A4r4g4b4` — packed 16-bit; `Ay8` — 8-bit packed
//! luma+alpha) get pre-decoded by `decode_to_rgba8` to either
//! `Rgba8UnormSrgb` or `Rgba8Unorm` based on the bitmap's curve. We
//! drop mips on the decoded path for v1 — the visual difference is
//! negligible at character scale and adding mip generation can come
//! later if needed.
//!
//! Layer / face indexing: `BitmapImage::pixel_bytes()` returns ALL
//! bytes (mip pyramid for layer 0, then layer 1, etc.). For 2D
//! single-image bitmaps (the case for diffuse/normal/spec maps on
//! biped shaders) we slice the layer-0 surface — `surface_bytes`. Cube
//! faces and layered arrays will need richer handling for environment
//! maps and shader_terrain blends; deferred.

pub mod atlas_decode;

use std::path::Path;

use blam_tags::bitmap::{Bitmap, BitmapCurve, BitmapFormat};
use blam_tags::bitmap::decode::decode_to_rgba8;
use blam_tags::TagFile;
use blam_tags::math::{RealArgbColor, RealPoint2d};
use glam::Vec2;

use crate::halo::render_methods::materials::{InlinePixelFormat, InlineTexture};

/// Read a Halo `.bitmap` tag from disk and return an [`InlineTexture`]
/// ready for the renderer to upload. Handles both 2D and cube bitmaps;
/// cube bitmaps come back with `layers = 6` and `is_cube = true`,
/// holding all 6 face mip pyramids back-to-back (face 0 mips, then
/// face 1, …) — the layout wgpu expects for a 6-layer cube texture.
/// Returns `None` if the tag won't open, has no images, or carries a
/// format we haven't taught the loader.
pub fn load(bitm_path: &Path) -> Option<InlineTexture> {
    load_image(bitm_path, 0)
}

/// Load every `bitmap_data` entry in a multi-image bitmap group. Used
/// for the per-BSP cubemap atlas
/// (`<scenario>_<bsp>_cubemaps.bitmap`) — each `bitmaps[i]` is a
/// separate cube whose face data lives back-to-back. Returns the
/// loaded textures in tag order (1:1 with `scenario.cubemaps[]` for
/// single-BSP MP maps).
pub fn load_all_images(bitm_path: &Path) -> Vec<InlineTexture> {
    let Ok(tag) = TagFile::read(bitm_path) else { return Vec::new() };
    let Ok(bitmap) = Bitmap::new(&tag) else { return Vec::new() };
    (0..bitmap.len())
        .filter_map(|idx| load_image_from_bitmap(&bitmap, idx, bitm_path))
        .collect()
}

/// Load a specific `bitmap_data` entry from a multi-image bitmap
/// group. Index 0 mirrors `Self::load`.
pub fn load_image(bitm_path: &Path, index: usize) -> Option<InlineTexture> {
    let tag = TagFile::read(bitm_path).ok()?;
    let bitmap = Bitmap::new(&tag).ok()?;
    load_image_from_bitmap(&bitmap, index, bitm_path)
}

fn load_image_from_bitmap(
    bitmap: &Bitmap<'_>,
    index: usize,
    bitm_path: &Path,
) -> Option<InlineTexture> {
    let img = bitmap.image(index)?;
    let format = img.format().ok()?;
    let width = img.width();
    let height = img.height();
    let mip_count = img.mipmap_levels().max(1);
    let layers = img.layer_count();
    let is_cube = img.is_cube();
    // Curve → sRGB-decode classification.
    //
    // `XrgbGamma2` (= 1) is the Xenon/PC sRGB-equivalent curve most
    // diffuse maps ship with. `Srgb` (= 5) is the canonical γ2.2 sRGB
    // curve used by some MCC re-bake outputs. `Gamma2` (= 2) is a
    // γ2.0 approximation — close enough that engine's PC path samples
    // it through the same `_SRGB` SRV (the γ2.0 vs γ2.2 mismatch is
    // invisible at typical bit depths and matches the engine's
    // behavior on cache-built bitmaps).
    //
    // `Linear` (= 3) bytes are already in linear space — sampling
    // through an `_SRGB` SRV double-darkens (shrine `stone_edge_dif`
    // visible-bug 2026-05-17).
    //
    // `OffsetLog` (= 4) is log-encoded HDR-ish data, NEVER sRGB.
    //
    // `Unknown` (= 0) typically means the cache builder didn't tag the
    // curve — engine MCC defaults to sRGB for color formats, so we do
    // the same to preserve prior behavior. Bump/spec/mask consumers
    // override via `is_linear_param_name`, so this only affects color
    // params.
    let srgb = match img.curve() {
        BitmapCurve::XrgbGamma2
        | BitmapCurve::Gamma2
        | BitmapCurve::Srgb
        | BitmapCurve::Unknown => true,
        BitmapCurve::Linear | BitmapCurve::OffsetLog => false,
    };
    let raw = img.pixel_bytes().ok()?;

    // Total surface = (per-layer mip pyramid) × layer_count. For 2D
    // images that's just one pyramid; for cubes, six.
    let per_layer = format.surface_bytes(width, height, mip_count) as usize;
    let surface = per_layer.saturating_mul(layers as usize);
    if raw.len() < surface {
        eprintln!(
            "[halo_bitmap] {}: pixel_bytes too short ({} < {} for {}×{} {:?} mips={} layers={})",
            bitm_path.display(), raw.len(), surface, width, height, format, mip_count, layers,
        );
        return None;
    }
    let surface_bytes = &raw[..surface];

    // DXN (BC5) in Halo is always a 2-channel SNORM tangent-space
    // normal map. wgpu's `Bc5RgSnorm` decodes the BC5 hardware-side
    // and gives the shader (.rg) directly in [-1, 1]; the shader
    // reconstructs Z. No CPU decode or Z preencoding needed —
    // bytes pass through as-is, mip pyramid intact.
    if format == BitmapFormat::Dxn {
        return Some(InlineTexture {
            width, height, mip_count, layers, is_cube,
            format: InlinePixelFormat::Bc5RgSnorm,
            data: surface_bytes.to_vec(),
        });
    }

    // Native passthrough where wgpu has the format.
    if let Some(inline_fmt) = native_passthrough(format, srgb) {
        return Some(InlineTexture {
            width, height, mip_count, layers, is_cube,
            format: inline_fmt,
            data: surface_bytes.to_vec(),
        });
    }

    // Halo-specific oddballs: decode mip 0 of each layer to RGBA8 (no
    // mip pyramid on this path). Most one-off Halo formats (Ay8,
    // A4r4g4b4, DxnMonoAlpha) only show up on 2D bitmaps in practice,
    // so we still emit valid bytes for the cube case (6 mip-0 faces
    // back-to-back) — but the texture will be loaded mipless.
    let mip0_layer_bytes = format.level_bytes(width, height) as usize;
    let mut rgba8 = Vec::with_capacity(width as usize * height as usize * 4 * layers as usize);
    for layer in 0..layers as usize {
        let src_off = layer * per_layer;
        let layer_mip0 = &surface_bytes[src_off..src_off + mip0_layer_bytes];
        match decode_to_rgba8(format, width, height, layer_mip0) {
            Ok(v) => rgba8.extend_from_slice(&v),
            Err(e) => {
                eprintln!(
                    "[halo_bitmap] {}: decode_to_rgba8 failed for {:?} (layer {}): {}",
                    bitm_path.display(), format, layer, e,
                );
                return None;
            }
        }
    }
    Some(InlineTexture {
        width, height,
        mip_count: 1, layers, is_cube,
        format: if srgb { InlinePixelFormat::Rgba8UnormSrgb } else { InlinePixelFormat::Rgba8Unorm },
        data: rgba8,
    })
}

/// Map a Halo [`BitmapFormat`] to a wgpu-native [`InlinePixelFormat`]
/// when an exact equivalent exists. Returns `None` for formats that
/// require pre-decoding.
///
/// Note on 8-bit single-channel: `A8`/`Y8`/`R8` all collapse to
/// `R8Unorm` — wgpu has no native A8. Shaders that sample these get
/// the byte in `.r`; "alpha-only" semantics is the consumer's problem
/// to swizzle.
fn native_passthrough(fmt: BitmapFormat, srgb: bool) -> Option<InlinePixelFormat> {
    use BitmapFormat as B;
    use InlinePixelFormat as I;
    Some(match fmt {
        B::Dxt1 => if srgb { I::Bc1RgbaUnormSrgb } else { I::Bc1RgbaUnorm },
        B::Dxt3 => if srgb { I::Bc2RgbaUnormSrgb } else { I::Bc2RgbaUnorm },
        B::Dxt5 => if srgb { I::Bc3RgbaUnormSrgb } else { I::Bc3RgbaUnorm },
        B::Dxt5a => I::Bc4RUnorm,
        // `B::Dxn` is intentionally NOT mapped here — it's handled
        // earlier in `load_image_from_bitmap` as `Bc5RgSnorm` (Halo MCC
        // BC5 normal maps are signed-normalized; see
        // `feedback_mcc_bc5_is_snorm`). If the early-return ever gets
        // ripped out, falling back to a UNORM mapping silently corrupts
        // every bump map by 128 units — better to fail loudly via the
        // `_ => None` catch-all + `decode_to_rgba8` than to ship broken
        // normals.
        B::A8r8g8b8 | B::X8r8g8b8 => if srgb { I::Bgra8UnormSrgb } else { I::Bgra8Unorm },
        B::A16b16g16r16 => I::Rgba16Unorm,
        B::Signedr16g16b16a16 => I::Rgba16Snorm,
        B::Abgrfp16 => I::Rgba16Float,
        B::Abgrfp32 => I::Rgba32Float,
        B::Q8w8v8u8 => I::Rgba8Snorm,
        B::V8u8 => I::Rg8Snorm,
        B::A8y8 => I::Rg8Unorm,
        B::A8 | B::Y8 | B::R8 => I::R8Unorm,
        // Ay8 (8-bit packed luma+alpha), A4r4g4b4, DxnMonoAlpha need decode
        _ => return None,
    })
}

/// `bitmap_group_2d_get_real_pixel` @ dllcache `0x1804ecbb0`.
///
/// Engine signature takes `(bitmap_group_index, bitmap_index, ...)` — a tag
/// handle pair that resolves through `bitmap_group_get_bitmap_hardware_format_for_get_real_pixel`
/// to a `(bitmap_data, c_rasterizer_texture_ref)` and finally a
/// `c_durango_address_computer` swizzled-texel walk.
///
/// Protomorph adapts: the caller pre-resolves to a [`DecodedAtlas`] (one per
/// BC3 array bitmap, cached at scenario load), and this function takes the
/// atlas directly. The `image_index` becomes the array-layer index into the
/// decoded atlas; `clamp` always applies (matches engine's `clamp=1`
/// callers; engine has no `wrap` path for lightprobe atlases).
///
/// Engine `bitmap_2d_get_real_pixel_internal_c_durango_address_computer` @
/// dllcache `0x1804edbe0` walks `floor(u*w)` → one texel; that maps 1:1 to
/// [`DecodedAtlas::sample_nearest`].
///
/// Returns the decoded RGBA-f32 as a `real_argb_color` in `[0, 1]` for BC3
/// (Halo's lightprobe atlases are all BC3 unorm).
pub fn bitmap_group_2d_get_real_pixel(
    atlas: &atlas_decode::DecodedAtlas,
    image_index: i32,
    uv: &RealPoint2d,
    _mipmap_index: i32,
    _clamp: bool,
) -> RealArgbColor {
    let layer = image_index.max(0) as usize;
    if layer >= atlas.layers.len() {
        return RealArgbColor::default();
    }
    let rgba = atlas.sample_nearest(layer, Vec2::new(uv.x, uv.y));
    RealArgbColor {
        alpha: rgba[3],
        red: rgba[0],
        green: rgba[1],
        blue: rgba[2],
    }
}
