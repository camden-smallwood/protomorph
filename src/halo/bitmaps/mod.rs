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

use blam_tags::bitmap::{Bitmap, BitmapFormat};
use blam_tags::bitmap::decode::decode_to_rgba8;
use blam_tags::TagFile;
use blam_tags::math::{RealArgbColor, RealPoint2d};
use glam::Vec2;

use crate::halo::render_methods::materials::{InlinePixelFormat, InlineTexture, TexelEncoding};

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

/// Read the sprite-sheet `sequences[]` from a `.bitmap` tag (per-frame
/// atlas sub-rects). Empty for plain (non-animated) textures. Used to
/// bake particle sprite-sheet frame UVs at batch registration.
pub fn load_sequences(bitm_path: &Path) -> Vec<blam_tags::bitmap::BitmapSequence> {
    let Ok(tag) = TagFile::read(bitm_path) else { return Vec::new() };
    let Ok(bitmap) = Bitmap::new(&tag) else { return Vec::new() };
    bitmap.sequences()
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
    // Record the authored encoding as data (curve → Gamma/Linear is in
    // `TexelEncoding::from_curve`). The sRGB-vs-linear *sampling* decision
    // is deferred to `resolve_wgpu_format` (encoding + call-site intent) so
    // it's made in exactly one place; bump/spec/mask consumers request
    // `ForceLinear` there via `is_linear_param_name`.
    let encoding = TexelEncoding::from_curve(img.curve());
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
            encoding: TexelEncoding::Linear,
            data: surface_bytes.to_vec(),
        });
    }

    // `A8` → DX11 `DXGI_FORMAT_A8_UNORM` (verified: engine
    // `c_rasterizer::bitmap_format_to_hardware_format_pc_unchecked`
    // table_16[0] = 0x41), which samples `(0, 0, 0, a)`. wgpu/Metal has
    // no native A8 and no stable view swizzle, so decode to RGBA8 with
    // rgb zeroed. CRUCIALLY decode the FULL MIP PYRAMID — the engine's A8
    // masks are mipped (e.g. guardian glass `specular_mask` is 2048² ×12
    // mips); dropping mips makes the mask's dark texels alias into dark,
    // unreflective patches at distance instead of averaging to the engine's
    // smooth value. blam-tags `decode_a8` is white-with-alpha (a preview
    // convention), so we zero rgb to match DXGI. Linear (A8 has no sRGB
    // variant; alpha is the mask). Non-cube only (cube A8 is unobserved;
    // it falls through to the mip-0 decode below).
    if format == BitmapFormat::A8 && !is_cube {
        if std::env::var("PROTOMORPH_LOG_A8").is_ok() {
            eprintln!("[a8] {} ({}x{} {} mips)", bitm_path.display(), width, height, mip_count);
        }
        let mut data = Vec::with_capacity(width as usize * height as usize * 4 * 2);
        for layer in 0..layers as usize {
            let layer_base = layer * per_layer;
            let mut lvl_off = 0usize;
            for mip in 0..mip_count {
                let mw = (width >> mip).max(1);
                let mh = (height >> mip).max(1);
                let lvl_bytes = format.level_bytes(mw, mh) as usize;
                let src = &surface_bytes[layer_base + lvl_off..layer_base + lvl_off + lvl_bytes];
                match decode_to_rgba8(format, mw, mh, src) {
                    Ok(mut v) => {
                        for px in v.chunks_mut(4) {
                            px[0] = 0;
                            px[1] = 0;
                            px[2] = 0;
                        }
                        data.extend_from_slice(&v);
                    }
                    Err(e) => {
                        eprintln!(
                            "[halo_bitmap] {}: A8 decode failed (layer {layer} mip {mip}): {e}",
                            bitm_path.display(),
                        );
                        return None;
                    }
                }
                lvl_off += lvl_bytes;
            }
        }
        return Some(InlineTexture {
            width, height, mip_count, layers, is_cube: false,
            format: InlinePixelFormat::Rgba8Unorm,
            encoding: TexelEncoding::Linear,
            data,
        });
    }

    // Native passthrough where wgpu has the format.
    if let Some(inline_fmt) = native_passthrough(format) {
        return Some(InlineTexture {
            width, height, mip_count, layers, is_cube,
            format: inline_fmt,
            encoding,
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
    // DX11 `DXGI_FORMAT_A8_UNORM` samples as `(0, 0, 0, a)`. blam-tags
    // `decode_a8` expands to white-with-alpha (its documented "useful
    // preview" convention), so zero the rgb here to give shaders the
    // engine's sample (e.g. `decal_fx::sample_diffuse` reads `.rgb`,
    // which must be 0 for an alpha_blend A8 decal to darken rather than
    // paint colour).
    if format == BitmapFormat::A8 {
        if std::env::var("PROTOMORPH_LOG_A8").is_ok() {
            eprintln!("[a8] {}", bitm_path.display());
        }
        for px in rgba8.chunks_mut(4) {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        }
    }
    Some(InlineTexture {
        width, height,
        mip_count: 1, layers, is_cube,
        format: InlinePixelFormat::Rgba8Unorm,
        encoding,
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
fn native_passthrough(fmt: BitmapFormat) -> Option<InlinePixelFormat> {
    use BitmapFormat as B;
    use InlinePixelFormat as I;
    Some(match fmt {
        B::Dxt1 => I::Bc1RgbaUnorm,
        B::Dxt3 => I::Bc2RgbaUnorm,
        B::Dxt5 => I::Bc3RgbaUnorm,
        B::Dxt5a => I::Bc4RUnorm,
        // `B::Dxn` is intentionally NOT mapped here — it's handled
        // earlier in `load_image_from_bitmap` as `Bc5RgSnorm` (Halo MCC
        // BC5 normal maps are signed-normalized; see
        // `feedback_mcc_bc5_is_snorm`). If the early-return ever gets
        // ripped out, falling back to a UNORM mapping silently corrupts
        // every bump map by 128 units — better to fail loudly via the
        // `_ => None` catch-all + `decode_to_rgba8` than to ship broken
        // normals.
        B::A8r8g8b8 | B::X8r8g8b8 => I::Bgra8Unorm,
        B::A16b16g16r16 => I::Rgba16Unorm,
        B::Signedr16g16b16a16 => I::Rgba16Snorm,
        B::Abgrfp16 => I::Rgba16Float,
        B::Abgrfp32 => I::Rgba32Float,
        B::Q8w8v8u8 => I::Rgba8Snorm,
        B::V8u8 => I::Rg8Snorm,
        B::A8y8 => I::Rg8Unorm,
        // `A8` is intentionally NOT here — DX11 `DXGI_FORMAT_A8_UNORM`
        // samples as `(0, 0, 0, a)`, but wgpu/Metal has no native A8 and
        // no stable texture-view swizzle, so an `R8Unorm` upload would
        // sample as `(a, 0, 0, 1)` — leaking the mask into red and
        // pinning alpha to 1 (the snow scorpion's `scorpion_decal` then
        // alpha-blended an opaque red box instead of darkening). A8 falls
        // through to the decode path below, which zeroes rgb to match
        // DXGI. `Y8`/`R8` keep R8Unorm: their value genuinely belongs in
        // a colour channel (luma / red), not alpha.
        B::Y8 | B::R8 => I::R8Unorm,
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
