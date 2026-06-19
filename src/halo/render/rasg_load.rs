//! rasg / bitmap loaders (lightmap atlases, material textures, default bitmaps, bitmap-view helpers).
//!
//! Split out of `render/mod.rs` as a behavior-preserving module carve-out.
//! These are additional inherent `impl Renderer` methods; child modules of
//! `render` can see `Renderer`'s private fields, so no visibility changes
//! are required.

use super::{shared, Renderer, SampleIntent};

impl Renderer {
    /// Load a tag-relative `.bitmap` to a sampled view. `None` if the
    /// path is empty or the bitmap fails to load.
    pub(crate) fn load_bitmap_view(
        &self,
        tags_root: &std::path::Path,
        rel: &str,
    ) -> Option<wgpu::TextureView> {
        self.load_bitmap_view_dims(tags_root, rel).map(|(v, _, _)| v)
    }

    /// Like [`Self::load_bitmap_view`] but also returns the base-mip
    /// `(width, height)` — needed to bake the particle sprite corner's
    /// bitmap-aspect factor.
    pub(crate) fn load_bitmap_view_dims(
        &self,
        tags_root: &std::path::Path,
        rel: &str,
    ) -> Option<(wgpu::TextureView, u32, u32)> {
        if rel.is_empty() {
            return None;
        }
        let abs = blam_tags::paths::resolve_tag_path(tags_root, rel, "bitmap");
        let tex = crate::halo::bitmaps::load(&abs)?;
        let (w, h) = (tex.width, tex.height);
        let view = crate::halo::structures::bsp_gpu::upload_inline_texture(
            &self.shared,
            &tex,
            crate::halo::render::SampleIntent::FollowCurve,
        );
        Some((view, w, h))
    }

    /// Load `LightmapBspData.lightprobe_texture` + `dominant_light_intensity_texture`
    /// as wgpu Texture2DArray views, upload `compression_vectors[2k].x` for
    /// k ∈ [0..4] into the lightmap_compress cbuffer, then rebuild the
    /// camera bind group so terrain shaders sample the new atlases.
    pub(crate) fn upload_lightmap_atlases(
        &mut self,
        tags_root: &std::path::Path,
        lbsp: &blam_tags::scenario_lightmap::LightmapBspData,
    ) {
        // Compression scalars — engine pack format from
        // `submit_extern_texture` case 11 in dllcache (verified IDA decompile,
        // see reference_dllcache_lightmap_path.md). 9 ranges from
        // `compression_vectors[2k].x` for k ∈ [0..9], packed 3-per-vec4 with
        // .w = 0. The HLSL `sample_lightprobe_texture` reads:
        //   constant_0.x/y/z → SH coefs 0/1/2 ranges
        //   constant_1.x     → SH coef 3 range
        //   constant_1.y     → dominant intensity range
        //   (constants_1.z + constant_2.* unused at order-2)
        let r = |i: usize| lbsp.compression_vectors.get(i).map(|v| v.i).unwrap_or(0.0);
        let compress: [[f32; 4]; 3] = [
            [r(0),  r(2),  r(4),  0.0],  // SH coefs 0/1/2
            [r(6),  r(8),  r(10), 0.0],  // SH coef 3, dominant intensity, (order-3 coef 5)
            [r(12), r(14), r(16), 0.0],  // (order-3 coefs 6/7/8 — zero at order-2)
        ];
        self.shared.queue.write_buffer(
            &self.shared.lightmap_compress_buffer,
            0,
            bytemuck::cast_slice(&compress),
        );

        // Load the two .bitmap tags referenced by the lightmap.
        // Read tag → upload via the helper, then assign the view.
        // Done as two explicit calls (not a closure) so the immutable
        // borrow of `&self.shared` ends before the mutable assignment
        // of `self.shared.lightprobe_atlas_view`.
        if !lbsp.lightprobe_texture.is_empty() {
            let p = blam_tags::paths::resolve_tag_path(tags_root, &lbsp.lightprobe_texture, "bitmap");
            if let Ok(tag) = blam_tags::file::TagFile::read(&p) {
                if let Ok(bitmap) = blam_tags::bitmap::Bitmap::new(&tag) {
                    if let Some(img) = bitmap.image(0) {
                        eprintln!(
                            "[lightmap]   lightprobe atlas: {}×{}, {} slices",
                            img.width(),
                            img.height(),
                            img.layer_count(),
                        );
                        if let Some(tex) = upload_bc3_array_bitmap(&self.shared, &img) {
                            self.shared.lightprobe_atlas_view = tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2Array),
                                ..Default::default()
                            });
                            self.shared.lightprobe_atlas = tex;
                        }
                    }
                }
            }
        }
        if !lbsp.dominant_light_intensity_texture.is_empty() {
            let p = blam_tags::paths::resolve_tag_path(tags_root, &lbsp.dominant_light_intensity_texture, "bitmap");
            if let Ok(tag) = blam_tags::file::TagFile::read(&p) {
                if let Ok(bitmap) = blam_tags::bitmap::Bitmap::new(&tag) {
                    if let Some(img) = bitmap.image(0) {
                        if let Some(tex) = upload_bc3_array_bitmap(&self.shared, &img) {
                            self.shared.lightprobe_intensity_atlas_view = tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2Array),
                                ..Default::default()
                            });
                            self.shared.lightprobe_intensity_atlas = tex;
                        }
                    }
                }
            }
        }
        self.shared.rebuild_camera_bind_group();
        eprintln!(
            "[lightmap] atlases bound: '{}' + '{}' (compression: {:.2}/{:.2}/{:.2}/{:.2}, dom={:.2})",
            lbsp.lightprobe_texture.split('\\').last().unwrap_or(""),
            lbsp.dominant_light_intensity_texture.split('\\').last().unwrap_or(""),
            r(0), r(2), r(4), r(6), r(8),
        );
    }

    /// Mirror of `c_rasterizer_globals::initialize` for the 3
    /// pre-integrated Cook-Torrance BRDF LUTs (`material_textures[]`).
    /// Sampled by `cook_torrance_fx.hlsl::sh_glossy_ct_3` at view /
    /// roughness UV to produce the specular distribution shape. Without
    /// these loaded, every cook_torrance material renders with a
    /// uniform white spec response (no shape / no roughness reaction).
    ///
    /// Engine path: rasg's `material_textures[0..2]` references three
    /// `rasterizer\<name>.bitmap` tags; `c_render_method::submit` routes
    /// these into the `g_sampler_cc0236/dd0236/c78d78` extern slots
    /// when the rmt2 declares the corresponding `RenderMethodExtern`
    /// values (24/25/26 — `TextureCookTorranceCc0236/Dd0236/C78d78`).
    pub(crate) fn load_material_textures_from_rasg(&mut self, tags_root: &std::path::Path) {
        let count = self.rasterizer_globals.material_textures.len();
        if count == 0 {
            return;
        }
        let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(count);
        let mut loaded = 0usize;
        for (idx, rel_path) in self.rasterizer_globals.material_textures.iter().enumerate() {
            if rel_path.is_empty() {
                views.push(self.shared.fallback_textures.white_view.clone());
                continue;
            }
            let abs = blam_tags::paths::resolve_tag_path(tags_root, rel_path, "bitmap");
            let Some(tex) = crate::halo::bitmaps::load(&abs) else {
                eprintln!(
                    "[scenario]   material_textures[{idx}] = '{rel_path}' — load failed, using white fallback",
                );
                views.push(self.shared.fallback_textures.white_view.clone());
                continue;
            };
            // Pre-integrated BRDF LUTs — physical reflectance / Schlick
            // fresnel terms, not display-encoded color → force linear.
            let view = crate::halo::structures::bsp_gpu::upload_inline_texture(
                &self.shared, &tex, SampleIntent::ForceLinear,
            );
            views.push(view);
            loaded += 1;
        }
        eprintln!(
            "[scenario]   material_textures loaded: {loaded}/{count}",
        );
        self.shared.fallback_textures.material_texture_views = views;
    }

    /// Mirror of `c_rasterizer_globals::initialize` — load the bitmap
    /// at every populated `default_bitmaps[]` slot from rasg into a
    /// wgpu cube/2D texture and stash on
    /// `FallbackTextures::default_bitmap_views`. Per-slot failures
    /// (missing tag, unsupported format) leave the slot's view as a
    /// black-cube/black-2D fallback so `default_bitmap_view` always
    /// returns something bindable. Rasg holds 12 entries (see
    /// `blam_tags::rasterizer_globals::DefaultBitmap`); slot 2
    /// (`DefaultDynamicCubeMap`) is what every cube-typed extern slot
    /// reads when the per-cluster lookup fails (always on MCC).
    pub(crate) fn load_default_bitmaps_from_rasg(&mut self, tags_root: &std::path::Path) {
        use blam_tags::rasterizer_globals::DefaultBitmap;
        let count = self.rasterizer_globals.default_bitmaps.len();
        if count == 0 {
            return;
        }
        let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(count);
        let mut loaded = 0usize;
        for (idx, rel_path) in self.rasterizer_globals.default_bitmaps.iter().enumerate() {
            let placeholder = if idx == DefaultBitmap::DefaultDynamicCubeMap as usize {
                self.shared.fallback_textures.black_cube_view.clone()
            } else if idx == DefaultBitmap::DefaultVector as usize {
                self.shared.fallback_textures.default_normal_view.clone()
            } else if idx == DefaultBitmap::ColorWhite as usize {
                self.shared.fallback_textures.white_view.clone()
            } else {
                self.shared.fallback_textures.black_view.clone()
            };
            if rel_path.is_empty() {
                views.push(placeholder);
                continue;
            }
            // rasg stores Halo-style paths with `\` separators
            // (`shaders\default_bitmaps\bitmaps\color_white`); use the
            // engine path resolver so the separator is normalized for
            // the host filesystem and the `.bitmap` extension is
            // stamped on.
            let abs = blam_tags::paths::resolve_tag_path(tags_root, rel_path, "bitmap");
            let Some(tex) = crate::halo::bitmaps::load(&abs) else {
                eprintln!(
                    "[scenario]   default_bitmaps[{idx}] = '{rel_path}' — load failed, using placeholder",
                );
                views.push(placeholder);
                continue;
            };
            // Trust the authored curve for color/env defaults — notably
            // `DefaultDynamicCubeMap` is RGBW gamma-encoded, so forcing it
            // linear was the same over-bright bug as the scenario cube atlas.
            // `DefaultVector` is a tangent-space normal → force linear.
            let intent = if idx == DefaultBitmap::DefaultVector as usize {
                SampleIntent::ForceLinear
            } else {
                SampleIntent::FollowCurve
            };
            let view = crate::halo::structures::bsp_gpu::upload_inline_texture(
                &self.shared, &tex, intent,
            );
            views.push(view);
            loaded += 1;
        }
        eprintln!(
            "[scenario]   default_bitmaps loaded: {loaded}/{count}",
        );
        self.shared.fallback_textures.default_bitmap_views = views;
    }
}

/// Upload a `BitmapImage` (expected DXT5 array) to a Texture2DArray.
/// One slice per layer, mip 0 only (atlases don't use mips). Returns
/// `None` if the bitmap isn't DXT5 or fails to slice.
fn upload_bc3_array_bitmap(
    shared: &shared::SharedResources,
    img: &blam_tags::bitmap::BitmapImage<'_>,
) -> Option<wgpu::Texture> {
    use blam_tags::bitmap::BitmapFormat;
    let format = img.format().ok()?;
    if !matches!(format, BitmapFormat::Dxt5) { return None; }
    let width = img.width();
    let height = img.height();
    let layers = img.layer_count();
    let mips = img.mipmap_levels().max(1);
    if layers == 0 { return None; }
    let pixels = img.pixel_bytes().ok()?;
    // Halo bitmap pixel layout: [layer0_full_mipchain, layer1_full_mipchain, ...].
    // The base mip is the first `block_w*block_h*16` bytes of each layer's chain;
    // skip the rest of each chain to get to the next layer's base.
    let bytes_per_layer = format.surface_bytes(width, height, mips) as usize;
    let block_w = ((width + 3) / 4).max(1);
    let block_h = ((height + 3) / 4).max(1);
    let base_mip_bytes = (block_w * block_h * 16) as usize;
    let total = bytes_per_layer.saturating_mul(layers as usize);
    if pixels.len() < total {
        eprintln!("[lightmap] atlas pixel slice {} < expected {}", pixels.len(), total);
        return None;
    }
    let texture = shared.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lightprobe_atlas"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc3RgbaUnorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for layer in 0..layers {
        let off = (layer as usize) * bytes_per_layer;
        let src = &pixels[off..off + base_mip_bytes];
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            src,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(block_w * 16),
                rows_per_image: Some(block_h),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
    Some(texture)
}
