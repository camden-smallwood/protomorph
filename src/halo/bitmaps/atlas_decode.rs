//! CPU-side lightprobe-atlas decoder for the decorator bake. Mirrors
//! the GPU path in `assets/halo_shaders/lightmap_sampling_fx.wgsl`:
//! 8 BC3 array slices encoding 4 SH coefs as (L, U/V/W) half-pairs;
//! `decode_bpp16_luvw` reconstructs each coef.
//!
//! BC3 (DXT5) decoding is hand-rolled — keeps the dependency graph
//! flat. Full-atlas decode at scenario load (~2 MB per riverworld-size
//! atlas) avoids per-query block decoding for the 260K rays the bake
//! fires. CPU output is bilinear-sampleable at any UV.

use blam_tags::bitmap::{BitmapFormat, BitmapImage};
use glam::{Vec2, Vec3};

/// One BC3 atlas decoded to per-layer RGBA-f32 pixel grids. `layers`
/// is `layer_count` long, each entry `(width × height × 4)` floats in
/// 0..1. Stored row-major (`y * width + x`).
pub struct DecodedAtlas {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<Vec<[f32; 4]>>,
}

impl DecodedAtlas {
    /// Decode every layer of a DXT5 (BC3) array bitmap into
    /// pre-decoded RGBA-f32 pixel grids. Returns `None` on format
    /// mismatch or pixel-slice unavailability — matches the existing
    /// `upload_bc3_array_bitmap` gate.
    pub fn from_bitmap(img: &BitmapImage<'_>) -> Option<Self> {
        let format = img.format().ok()?;
        if !matches!(format, BitmapFormat::Dxt5) {
            return None;
        }
        let width = img.width();
        let height = img.height();
        let layers = img.layer_count();
        let mips = img.mipmap_levels().max(1);
        if width == 0 || height == 0 || layers == 0 {
            return None;
        }
        let pixels = img.pixel_bytes().ok()?;

        // Per-layer chain length includes all mip levels; we only
        // decode mip 0 for atlas sampling (atlases don't use mips).
        let bytes_per_layer = format.surface_bytes(width, height, mips) as usize;
        let block_w = ((width + 3) / 4).max(1) as usize;
        let block_h = ((height + 3) / 4).max(1) as usize;
        let base_mip_bytes = block_w * block_h * 16; // BC3 = 16 B/block

        let mut decoded_layers: Vec<Vec<[f32; 4]>> = Vec::with_capacity(layers as usize);
        for layer in 0..(layers as usize) {
            let layer_offset = layer * bytes_per_layer;
            if layer_offset + base_mip_bytes > pixels.len() {
                return None;
            }
            let layer_bc3 = &pixels[layer_offset..layer_offset + base_mip_bytes];
            decoded_layers.push(decode_bc3_layer(layer_bc3, width, height));
        }
        Some(Self { width, height, layers: decoded_layers })
    }

    /// Bilinear-sample one layer at a UV in [0, 1]. Returns RGBA in
    /// 0..1. Out-of-bounds UVs are clamped. Mirrors the GPU sampler's
    /// `MIN_MAG_LINEAR_MIP_POINT` behavior at LOD 0.
    ///
    /// Used by the decorator atlas-color bake. The lightprobe SH +
    /// intensity readers use [`Self::sample_nearest`] to match engine's
    /// CPU-side `bitmap_2d_get_real_pixel_internal` semantics — engine
    /// does NOT bilinear-blend on CPU lightprobe reads; it grabs the
    /// single nearest texel. Bilinear at chart-gap UVs averages 4
    /// neighbors (most alpha=0) and produces dim values vs engine's
    /// "pick one corner, take its color".
    pub fn sample(&self, layer: usize, uv: Vec2) -> [f32; 4] {
        let layer_data = &self.layers[layer];
        let w = self.width as f32;
        let h = self.height as f32;
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);
        let fx = u * w - 0.5;
        let fy = v * h - 0.5;
        let x0 = fx.floor().max(0.0) as u32;
        let y0 = fy.floor().max(0.0) as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = (fx - fx.floor()).clamp(0.0, 1.0);
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);

        let p00 = layer_data[(y0 * self.width + x0) as usize];
        let p10 = layer_data[(y0 * self.width + x1) as usize];
        let p01 = layer_data[(y1 * self.width + x0) as usize];
        let p11 = layer_data[(y1 * self.width + x1) as usize];
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let a = p00[c] * (1.0 - tx) + p10[c] * tx;
            let b = p01[c] * (1.0 - tx) + p11[c] * tx;
            out[c] = a * (1.0 - ty) + b * ty;
        }
        out
    }

    /// NEAREST-sample one layer at a UV in [0, 1]. Mirrors engine
    /// `bitmap_2d_get_real_pixel_internal_c_durango_address_computer @
    /// 0x1804edbe0` — `pixel_x = clamp(floor(u * width), 0, width-1)`,
    /// returns the single texel value. Engine uses this for CPU-side
    /// lightprobe sampling in `sample_lightprobe_textures @ 0x18048fc60`
    /// (called from `c_geometry_sampler::sample`'s atlas branch).
    ///
    /// Bilinear vs nearest divergence is silent on dense-coverage UVs
    /// but becomes a multi-x brightness gap at chart-packing seams
    /// (BC3 alpha-0 gaps surrounding an authored chart): bilinear
    /// weights 0-alpha neighbors into the average, nearest picks one
    /// texel and takes its bright value as-is.
    pub fn sample_nearest(&self, layer: usize, uv: Vec2) -> [f32; 4] {
        let layer_data = &self.layers[layer];
        let w = self.width as f32;
        let h = self.height as f32;
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);
        let x = ((u * w) as i32).clamp(0, self.width as i32 - 1) as u32;
        let y = ((v * h) as i32).clamp(0, self.height as i32 - 1) as u32;
        layer_data[(y * self.width + x) as usize]
    }
}

/// Sample the dominant-light-intensity atlas at `uv` — engine-faithful
/// mirror of the `lightprobe_intensity_atlas` half of
/// `lightmap_sampling_fx.wgsl::sample_lightprobe_texture` (line 48-49,
/// 64). The intensity bitmap is exactly 2 BC3 layers; the pair encodes
/// one Vec3 via the same `decode_bpp16_luvw` primitive used for SH
/// coefs, scaled by `compress[1][1]` (= `compression_vectors[8].x`).
///
/// Engine call site: `sample_lightprobe_textures @ ~0x180XXXXX` (inside
/// the per-pixel atlas branch of `c_geometry_sampler::sample`). Feeds
/// the `dom_intensity` argument of
/// `reconstruct_quadratic_lightprobe_from_linear_and_intensity`.
pub fn sample_lightprobe_intensity(
    atlas: &DecodedAtlas,
    uv: Vec2,
    compress: &[[f32; 4]; 3],
) -> Vec3 {
    if atlas.layers.len() < 2 {
        return Vec3::ZERO;
    }
    // Engine `bitmap_2d_get_real_pixel_internal` uses nearest, not bilinear.
    let i0 = atlas.sample_nearest(0, uv);
    let i1 = atlas.sample_nearest(1, uv);
    decode_bpp16_luvw(i0, i1, compress[1][1])
}

/// CPU mirror of `lightmap_sampling_fx.wgsl::decode_bpp16_luvw`. Two
/// BC3 atlas samples encode (L, U/V/W) as a half-pair; this
/// reconstructs one SH coefficient (vec3) from them.
pub fn decode_bpp16_luvw(s0: [f32; 4], s1: [f32; 4], l_range: f32) -> Vec3 {
    let l = s0[3] * s1[3] * l_range;
    let uvw = Vec3::new(s0[0] + s1[0], s0[1] + s1[1], s0[2] + s1[2]);
    (uvw * 2.0 - Vec3::splat(2.0)) * l
}

/// Sample the per-cluster SH atlas at `uv` and decode 4 SH coefs
/// (DC + 3 L1 bands), each as RGB. Mirror of
/// `sample_lightprobe_texture` (lightmap_sampling_fx.wgsl).
///
/// `compress` is the lightmap_compress cbuffer's three vec4s as
/// `[constant_0, constant_1, constant_2]`. Same memory layout used at
/// `render::mod::upload_lightmap_atlases` (see comments there for the
/// per-coef range mapping).
pub fn sample_lightprobe_sh(
    atlas: &DecodedAtlas,
    uv: Vec2,
    compress: &[[f32; 4]; 3],
) -> [Vec3; 4] {
    if atlas.layers.len() < 8 {
        return [Vec3::ZERO; 4];
    }
    // Engine `sample_lightprobe_textures` — uses the raw raycast UV and
    // NEAREST sampling. Engine `bitmap_2d_get_real_pixel_internal_c_durango_address_computer
    // @ 0x1804edbe0` resolves `floor(u*w)` → single texel read. Bilinear
    // dilutes chart-gap regions (most neighbors alpha=0) into dim values;
    // nearest grabs one texel as-is, often landing on the authored bright
    // chart pixel. That's the difference between MCC's bright object
    // lighting and our previously-dim sample at chart-packing seams.
    let s0 = atlas.sample_nearest(0, uv);
    let s1 = atlas.sample_nearest(1, uv);
    let s2 = atlas.sample_nearest(2, uv);
    let s3 = atlas.sample_nearest(3, uv);
    let s4 = atlas.sample_nearest(4, uv);
    let s5 = atlas.sample_nearest(5, uv);
    let s6 = atlas.sample_nearest(6, uv);
    let s7 = atlas.sample_nearest(7, uv);
    [
        decode_bpp16_luvw(s0, s1, compress[0][0]), // SH coef 0 (DC) ← const_0.x
        decode_bpp16_luvw(s2, s3, compress[0][1]), // SH coef 1 (L1 m=-1) ← const_0.y
        decode_bpp16_luvw(s4, s5, compress[0][2]), // SH coef 2 (L1 m=0)  ← const_0.z
        decode_bpp16_luvw(s6, s7, compress[1][0]), // SH coef 3 (L1 m=+1) ← const_1.x
    ]
}


// ---------------------------------------------------------------------------
// BC3 (DXT5) block decode — RGB via DXT1-style endpoints, alpha via 8-stop
// gradient. Each 4×4 block is 16 bytes:
//   [0..1]   alpha endpoints (a0, a1)
//   [2..7]   16 × 3-bit alpha indices (48 bits packed LSB-first)
//   [8..9]   color endpoint 0 (RGB565)
//   [10..11] color endpoint 1 (RGB565)
//   [12..15] 16 × 2-bit color indices (32 bits packed LSB-first, row-major)
// ---------------------------------------------------------------------------

fn decode_bc3_layer(layer_bc3: &[u8], width: u32, height: u32) -> Vec<[f32; 4]> {
    let block_w = ((width + 3) / 4) as usize;
    let block_h = ((height + 3) / 4) as usize;
    let mut out = vec![[0.0f32; 4]; (width as usize) * (height as usize)];
    let inv = 1.0 / 255.0;
    for by in 0..block_h {
        for bx in 0..block_w {
            let block_offset = (by * block_w + bx) * 16;
            let block: [u8; 16] = layer_bc3[block_offset..block_offset + 16]
                .try_into()
                .expect("block slice");
            let pixels = decode_bc3_block(block);
            for py in 0..4 {
                for px in 0..4 {
                    let x = bx * 4 + px;
                    let y = by * 4 + py;
                    if x >= width as usize || y >= height as usize {
                        continue;
                    }
                    let p = pixels[py * 4 + px];
                    let dst = &mut out[y * (width as usize) + x];
                    dst[0] = p[0] as f32 * inv;
                    dst[1] = p[1] as f32 * inv;
                    dst[2] = p[2] as f32 * inv;
                    dst[3] = p[3] as f32 * inv;
                }
            }
        }
    }
    out
}

fn decode_bc3_block(block: [u8; 16]) -> [[u8; 4]; 16] {
    // ---- Alpha ----
    let a0 = block[0];
    let a1 = block[1];
    let mut alpha = [0u8; 8];
    alpha[0] = a0;
    alpha[1] = a1;
    if a0 > a1 {
        for i in 1..7 {
            alpha[i + 1] = (((7 - i) as u32 * a0 as u32 + i as u32 * a1 as u32) / 7) as u8;
        }
    } else {
        for i in 1..5 {
            alpha[i + 1] = (((5 - i) as u32 * a0 as u32 + i as u32 * a1 as u32) / 5) as u8;
        }
        alpha[6] = 0;
        alpha[7] = 255;
    }
    // 16 × 3-bit alpha indices — 48 bits packed in bytes 2..7 LSB-first.
    let bits_lo = u32::from_le_bytes([block[2], block[3], block[4], block[5]]) as u64;
    let bits_hi = u16::from_le_bytes([block[6], block[7]]) as u64;
    let alpha_bits = bits_lo | (bits_hi << 32);
    let mut alpha_idx = [0u8; 16];
    for i in 0..16 {
        alpha_idx[i] = ((alpha_bits >> (i * 3)) & 0x7) as u8;
    }

    // ---- Color ----
    let c0_565 = u16::from_le_bytes([block[8], block[9]]);
    let c1_565 = u16::from_le_bytes([block[10], block[11]]);
    let (c0_r, c0_g, c0_b) = unpack_565(c0_565);
    let (c1_r, c1_g, c1_b) = unpack_565(c1_565);
    let mut palette = [[0u8; 3]; 4];
    palette[0] = [c0_r, c0_g, c0_b];
    palette[1] = [c1_r, c1_g, c1_b];
    // BC3 is alpha-separated → ALWAYS the BC1 "c0 > c1" code interpretation:
    palette[2] = [
        ((2 * c0_r as u32 + c1_r as u32) / 3) as u8,
        ((2 * c0_g as u32 + c1_g as u32) / 3) as u8,
        ((2 * c0_b as u32 + c1_b as u32) / 3) as u8,
    ];
    palette[3] = [
        ((c0_r as u32 + 2 * c1_r as u32) / 3) as u8,
        ((c0_g as u32 + 2 * c1_g as u32) / 3) as u8,
        ((c0_b as u32 + 2 * c1_b as u32) / 3) as u8,
    ];
    let color_bits = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

    let mut out = [[0u8; 4]; 16];
    for i in 0..16 {
        let ci = ((color_bits >> (i * 2)) & 0x3) as usize;
        let rgb = palette[ci];
        out[i] = [rgb[0], rgb[1], rgb[2], alpha[alpha_idx[i] as usize]];
    }
    out
}

fn unpack_565(c: u16) -> (u8, u8, u8) {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    // Bit-replicate to 8 bits.
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    (r, g, b)
}
