//! A1 patchy fog — render pipeline + cbuffers.
//!
//! Engine: `c_patchy_fog::render_patchy_fog @ 0x1806B2160` builds a
//! 4-corner fullscreen quad in clip space and dispatches an explicit-
//! shader-56 (`patchy_fog`) draw with `_alpha_blend_multiply_add` blend
//! and `_z_buffer_mode_read`. The shader (ported in
//! `assets/shaders/patchy_fog.wgsl`) samples 8 fog sheets, integrates
//! optical depth, applies HG-phase Mie inscatter, and outputs
//! `(inscatter * exposure, extinction)`.
//!
//! Phase 3 (this file) builds the pipeline + BGL + cbuffer-backing
//! buffers + 4-corner geometry. Phase 4 will add the per-frame
//! sheet-UV-offset advancement and `render()` method. Phase 5 wires
//! the dispatch into `c_player_view::render` between water and
//! transparents via the TransparencyRenderer queue.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::halo::render::shared::SharedResources;

/// Per-frame patchy-fog parameters. Mirrors `PatchyFogParams` in
/// `patchy_fog.wgsl`. 272 bytes. Engine cbuffer slots `0x3F0000..0x3F0004`
/// + `0x400000..0x40000B`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct PatchyFogParams {
    pub inverse_z_transform: [f32; 4],
    pub texcoord_basis: [f32; 4],
    pub attenuation_data: [f32; 4],
    pub eye_position: [f32; 4],
    pub window_pixel_bounds: [f32; 4],
    pub sheet_fade_factors0: [f32; 4],
    pub sheet_fade_factors1: [f32; 4],
    pub sheet_depths0: [f32; 4],
    pub sheet_depths1: [f32; 4],
    pub tex_coord_transform0: [f32; 4],
    pub tex_coord_transform1: [f32; 4],
    pub tex_coord_transform2: [f32; 4],
    pub tex_coord_transform3: [f32; 4],
    pub tex_coord_transform4: [f32; 4],
    pub tex_coord_transform5: [f32; 4],
    pub tex_coord_transform6: [f32; 4],
    pub tex_coord_transform7: [f32; 4],
}

/// Per-frame atmosphere constants subset that `patchy_fog.wgsl` needs.
/// Mirrors `PatchyFogAtmosphere` in the WGSL. 112 bytes. Engine: derived
/// from `c_atmosphere_fog_interface::set_constant @ 0x1803AE530`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct PatchyFogAtmosphere {
    /// `(sun_dir.xyz, _)`.
    pub sun_dir: [f32; 4],
    /// `(SUN_INTENSITY / TOTAL_MIE, _)`.
    pub sun_intensity_over_tm: [f32; 4],
    /// `(_, _, _, HG_CONSTANT_PLUS_ONE)`.
    pub constant_2: [f32; 4],
    /// `(TOTAL_MIE × log2(e), _)`.
    pub total_mie_log2e: [f32; 4],
    pub _4: [f32; 4],
    /// `(MIE_THETA_PREFIX × (1 - g²), _)`.
    pub mie_theta_prefix_hgc: [f32; 4],
    /// `(HG_CONSTANT_TIMES_TWO, _, _, _)`.
    pub extra: [f32; 4],
}

/// `s_patchy_fog_vertex` (Ares 32B). Engine layout: `__m128 position`
/// (16B) + `real_point2d texcoord` (8B) + 8B padding to 32. We
/// preserve the layout for upload symmetry with the engine.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PatchyFogVertex {
    position: [f32; 4],
    texcoord: [f32; 2],
    _pad: [f32; 2],
}

impl PatchyFogVertex {
    fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBS: [wgpu::VertexAttribute; 2] = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 16,
                shader_location: 1,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBS,
        }
    }
}

/// Owns the patchy-fog pipeline + uniform buffers + 4-corner quad
/// geometry. Single instance per `Renderer`. Phase 3 stops at
/// "pipeline compiles, buffers exist"; Phase 4 adds the per-frame
/// state-update + render method.
pub struct PatchyFogPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,

    /// `PatchyFogParams` cbuffer — slots 0x3F0000..0x40000B folded.
    params_buffer: wgpu::Buffer,
    /// `PatchyFogAtmosphere` cbuffer — atmosphere constants subset.
    atmosphere_buffer: wgpu::Buffer,
    /// `inverse(View_Projection)` mat4. VS is a passthrough for clip
    /// corners (vertex buffer is already clip-space) and uses the inverse
    /// to reconstruct the world-space corner needed for view-direction +
    /// height-fade math.
    inverse_view_projection_buffer: wgpu::Buffer,
    /// `g_exposure` vec4 (only `.r` is used by the shader output).
    exposure_buffer: wgpu::Buffer,

    /// 4 corners of the screen-projected fog quad in clip space. Per
    /// engine, the VS multiplies these through View_Projection — we
    /// upload them in clip-space `(±1, ±1, 1, 1)` form and let the VS
    /// pass them through. Engine actually constructs them in world
    /// space (`inv_view_projection × clip-corner`) at render time but
    /// the math reduces to the same screen rect.
    vertex_buffer: wgpu::Buffer,
    /// 6-index strip = 2 triangles covering the quad.
    index_buffer: wgpu::Buffer,

    /// Loaded `sky.patchy_fog_texture` 2D bitmap. `None` until a scenario
    /// with patchy fog authored is loaded (or the bitmap fails to load
    /// — see `load_noise`). When `None`, callers should skip the fog
    /// dispatch (or fall back to a placeholder, as the wiring in
    /// `player_view` does for the env-var debug path).
    noise_view: Option<wgpu::TextureView>,
    /// Holds the GPU texture alive while `noise_view` references it.
    _noise_texture: Option<wgpu::Texture>,
}

impl PatchyFogPass {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("patchy_fog_bgl"),
            entries: &[
                // 0: PatchyFogParams (PS)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: PatchyFogAtmosphere (PS)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: tex_noise
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: smp_noise (wrap+bilinear per engine
                // `c_patchy_fog::set_rasterizer_state`)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 4: tex_scene_depth
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: smp_scene_depth (clamp+point per engine)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 6: View_Projection (VS)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: g_exposure (PS)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("patchy_fog_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("patchy_fog_wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/patchy_fog.wgsl").into(),
            ),
        });

        // Engine `_alpha_blend_multiply_add` per
        // `c_patchy_fog::set_rasterizer_state @ 0x1806B30E0`. Decoding
        // the packed D3D state value 0x3C9844C2 from
        // `set_alpha_blend_mode_no_cache @ 0x18066E930` yields
        // `ONE / INV_SRC_ALPHA / ADD` for both color and alpha — true
        // premultiplied-alpha blend. The shader already bakes extinction
        // into inscatter (`inscatter *= extinction` in patchy_fog.hlsl
        // line 217 and our port at patchy_fog.wgsl line 236), so the
        // factors here treat that .rgb as pre-multiplied and use src.a
        // to darken the existing scene by `(1 - extinction)`:
        //   dst_new.rgb = inscatter*g_exposure + dst.rgb * (1 - extinction)
        let multiply_add_color = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let multiply_add = wgpu::BlendState {
            color: multiply_add_color,
            alpha: multiply_add_color,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("patchy_fog_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[PatchyFogVertex::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // Engine `patchy_fog.hlsl:225` uses `convert_to_render_target`
                // → dual-MRT: RT0 = `_surface_screenshot_composite_depth`
                // (our `lighting_base`), RT1 = `_surface_screenshot_composite
                // _cubemap` (our `hdr_dark`, the bloom-extract feed). On MCC
                // PC `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0` so RT0 == RT1
                // content. Same blend on both — fog inscatter contributes to
                // auto-exposure feedback via the HDR target.
                // Write mask: RGB only — engine `_alpha_blend_multiply_add`
                // packed state `0x3C9844C2` bits 27-30 = 0b0111 = 7 = RGB
                // (no alpha write). The shader's `extinction` in src.a is
                // consumed by the blend factor (INV_SRC_ALPHA on dst) but
                // is NOT written to the render-target alpha. `ColorWrites::ALL`
                // would corrupt RT.a for downstream consumers that read it
                // (none today, but engine-faithful + future-proof).
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(multiply_add),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(multiply_add),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // Engine uses `_z_buffer_mode_read` — depth-test on, no
            // write — but the shader samples scene_depth as a texture
            // to compute near-surface fade, so we read depth via SRV
            // and skip the depth-stencil attachment entirely (matches
            // `shadow_apply_pass.rs` pattern). Visual result matches
            // engine because the smoothstep fade in the PS handles
            // the near-occluder transition.
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_params"),
            contents: bytemuck::bytes_of(&PatchyFogParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let atmosphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_atm"),
            contents: bytemuck::bytes_of(&PatchyFogAtmosphere::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let inverse_view_projection_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_inverse_view_projection"),
            contents: bytemuck::bytes_of(&[[0.0_f32; 4]; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let exposure_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_exposure"),
            contents: bytemuck::bytes_of(&[1.0_f32, 1.0, 1.0, 1.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 4-corner unit quad in clip-space (`±1, ±1, 1, 1`). With a
        // passthrough VS and wgpu's `+y up` NDC, screen UV at NDC `(x,y)`
        // is `(x*.5+.5, .5-y*.5)`. The PS reconstructs that via
        // `window_pixel_bounds = (.5, .5, .5, -.5)` and `texcoord ∈
        // [-1,1]²` — so `texcoord` must equal `position.xy`. Engine
        // `predefined_texcoords` order is for D3D screen-down convention;
        // we flip to match wgpu instead.
        let verts = [
            PatchyFogVertex { position: [ 1.0, -1.0, 1.0, 1.0], texcoord: [ 1.0, -1.0], _pad: [0.0, 0.0] },
            PatchyFogVertex { position: [ 1.0,  1.0, 1.0, 1.0], texcoord: [ 1.0,  1.0], _pad: [0.0, 0.0] },
            PatchyFogVertex { position: [-1.0,  1.0, 1.0, 1.0], texcoord: [-1.0,  1.0], _pad: [0.0, 0.0] },
            PatchyFogVertex { position: [-1.0, -1.0, 1.0, 1.0], texcoord: [-1.0, -1.0], _pad: [0.0, 0.0] },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_vertex_buffer"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patchy_fog_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            bgl,
            params_buffer,
            atmosphere_buffer,
            inverse_view_projection_buffer,
            exposure_buffer,
            vertex_buffer,
            index_buffer,
            noise_view: None,
            _noise_texture: None,
        }
    }

    /// Loaded noise view (`Some` once a scenario with `sky.patchy_fog_texture`
    /// has loaded successfully).
    pub fn noise_view(&self) -> Option<&wgpu::TextureView> {
        self.noise_view.as_ref()
    }

    /// Resolve `tag_path` against `tags_root`, load it as a Halo bitmap,
    /// upload to a 2D wgpu texture, and store the view. Empty / missing
    /// path becomes `None`. Engine: equivalent of
    /// `bitmap_group_get_bitmap_hardware_format(patchy_fog_texture.index, 0)`
    /// inside `c_patchy_fog::set_rasterizer_state`.
    pub fn load_noise(
        &mut self,
        shared: &SharedResources,
        tags_root: &std::path::Path,
        tag_path: &str,
    ) {
        if tag_path.is_empty() {
            self.noise_view = None;
            self._noise_texture = None;
            return;
        }
        let bitm_path = blam_tags::paths::resolve_tag_path(tags_root, tag_path, "bitmap");
        if !bitm_path.exists() {
            eprintln!(
                "[patchy_fog] noise bitmap path missing: {}",
                bitm_path.display(),
            );
            self.noise_view = None;
            self._noise_texture = None;
            return;
        }
        let Some(inline) = crate::halo::bitmaps::load(&bitm_path) else {
            eprintln!(
                "[patchy_fog] failed to load noise bitmap from {}",
                bitm_path.display(),
            );
            self.noise_view = None;
            self._noise_texture = None;
            return;
        };
        // Linear sampling — engine `set_sampler_filter_mode(0,
        // _sampler_filter_bilinear)`. Force `linear=true` so even sRGB-
        // tagged greyscale noise reads as linear gray.
        let format = crate::halo::render::resolve_wgpu_format(
            inline.format, inline.encoding, crate::halo::render::SampleIntent::ForceLinear,
        );
        let texture = shared.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("patchy_fog_noise"),
            size: wgpu::Extent3d {
                width: inline.width,
                height: inline.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: inline.mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // `inline.data` layout: layer-0 mip pyramid back-to-back. For a
        // single-layer 2D bitmap that's just the mip chain.
        let mut offset = 0usize;
        for mip in 0..inline.mip_count {
            let mw = (inline.width >> mip).max(1);
            let mh = (inline.height >> mip).max(1);
            let level_bytes = inline.format.level_bytes(mw, mh);
            let (extent_w, extent_h, rows, bytes_per_row);
            if inline.format.is_compressed() {
                // BC formats need 4×4-block-aligned extent.
                let bw = ((mw + 3) / 4).max(1);
                let bh = ((mh + 3) / 4).max(1);
                extent_w = (bw * 4).min(((inline.width + 3) / 4) * 4);
                extent_h = (bh * 4).min(((inline.height + 3) / 4) * 4);
                rows = bh;
                bytes_per_row = bw * inline.format.block_bytes();
            } else {
                extent_w = mw;
                extent_h = mh;
                rows = mh;
                bytes_per_row = mw * inline.format.bytes_per_pixel();
            }
            shared.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &inline.data[offset..offset + level_bytes],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d { width: extent_w, height: extent_h, depth_or_array_layers: 1 },
            );
            offset += level_bytes;
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        eprintln!(
            "[patchy_fog] loaded noise bitmap: {} ({}×{}, {} mips, {:?})",
            tag_path, inline.width, inline.height, inline.mip_count, inline.format,
        );
        self.noise_view = Some(view);
        self._noise_texture = Some(texture);
    }

    /// Phase 4 render entry. Uploads `params` + `atm` cbuffers, builds
    /// the per-frame bind group, opens a render pass on `target_view`
    /// (engine `_surface_screenshot_composite_depth` = our
    /// `lighting_base`), and draws the 4-corner fog quad with
    /// `_alpha_blend_multiply_add` blend.
    ///
    /// Engine: dispatched from `c_player_view::queue_patchy_fog @
    /// 0x18068BFB0` via TransparencyRenderer at the global tag's
    /// transparent_sort_layer / transparent_sort_distance. We invoke
    /// directly between water + transparents until that integration
    /// lands (Phase 5).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        hdr_dark_target_view: &wgpu::TextureView,
        scene_depth_view: &wgpu::TextureView,
        scene_depth_sampler: &wgpu::Sampler,
        noise_view: &wgpu::TextureView,
        noise_sampler: &wgpu::Sampler,
        params: &PatchyFogParams,
        atm: &PatchyFogAtmosphere,
        view_projection: glam::Mat4,
        exposure: f32,
        debug_mode: f32,
    ) {
        shared.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
        shared.queue.write_buffer(&self.atmosphere_buffer, 0, bytemuck::bytes_of(atm));
        let inverse_vp = view_projection.inverse();
        shared.queue.write_buffer(
            &self.inverse_view_projection_buffer,
            0,
            bytemuck::bytes_of(&inverse_vp.to_cols_array_2d()),
        );
        // .w repurposed as shader debug-mode selector (see patchy_fog.wgsl).
        shared.queue.write_buffer(
            &self.exposure_buffer,
            0,
            bytemuck::bytes_of(&[exposure, exposure, exposure, debug_mode]),
        );

        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("patchy_fog_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.atmosphere_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(noise_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(noise_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(scene_depth_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(scene_depth_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: self.inverse_view_projection_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.exposure_buffer.as_entire_binding() },
            ],
        });

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("patchy_fog"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // RT1 — `hdr_dark` (engine `_surface_screenshot_composite
                // _cubemap`, the bloom-extract feed). Engine `convert_to_render
                // _target` writes both per fragment.
                Some(wgpu::RenderPassColorAttachment {
                    view: hdr_dark_target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &bg, &[]);
        rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..1);
    }
}

/// Build `PatchyFogParams` from the current camera + accumulated
/// atmosphere parameters + sky-atmosphere globals + per-frame
/// `PatchyFog` state.
///
/// Phase 5.3 — per-setting fields (full_intensity_height,
/// half_intensity_height, reference_datum_plane, patchy_fog_density)
/// source from the [`WeightedAtmosphereParameters`] accumulator so
/// the cbuffer agrees with the engine-faithful pipeline rather than
/// the parallel single-setting path.
///
/// Phase 6 reads `state.lateral_offsets` / `state.vertical_offsets` for
/// per-sheet UV centers and `state.roll` for the noise basis rotation.
/// Caller updates these via `PatchyFog::update_per_frame` once per
/// frame before this call.
pub fn build_patchy_fog_params(
    params: &crate::halo::render::atmosphere_fog_interface::WeightedAtmosphereParameters,
    sky: &blam_tags::sky_atmosphere::SkyAtmosphere,
    state: &crate::halo::render::render_patchy_fog::PatchyFog,
    camera_position: glam::Vec3,
    camera_projection: glam::Mat4,
    screen_pixel_extent: (u32, u32),
    z_near: f32,
) -> PatchyFogParams {
    // inverse_z_transform — engine HLSL formula:
    //   vsd.x = iz.x * d + iz.z
    //   vsd.y = iz.y * d + iz.w
    //   view_distance = vsd.x / -vsd.y
    //
    // The engine extracts `inv(projection).r[2].zw` + `r[3].zw` (D3D row-
    // major) and packs them — but that math assumes D3D LH conventions
    // (depth 0 at near, 1 at far). protomorph uses glam's
    // `perspective_infinite_reverse_rh` (depth 1 at near, 0 at infinity,
    // RH looking -Z), so we derive iz directly from the matrix structure
    // for the formula to yield positive view-space distance.
    let p = camera_projection.to_cols_array_2d();
    let m22 = p[2][2];
    let m32 = p[3][2];
    let m23 = p[2][3];
    let inverse_z_transform = if m22.abs() < 1e-6 && m23 < -0.5 {
        // `perspective_infinite_reverse_rh`. clip.z = near (const),
        // clip.w = -view_z, depth = near/distance. iz = (0, -1, near, 0)
        // yields vsd.x/-vsd.y = near/d = distance.
        let near = m32.max(1e-4);
        [0.0_f32, -1.0, near, 0.0]
    } else {
        // Finite glam_perspective_rh: col2 = (0, 0, -far/(far-near), -1),
        // col3 = (0, 0, -near*far/(far-near), 0). Recover near + far,
        // build iz = (0, (far-near)/(near*far), 1, -1/near).
        let near = (m32 / m22).max(1e-4);
        let inv_far = (m22 + 1.0) / (m22 * near); // 0 for infinite far
        let inv_near = 1.0 / near;
        [0.0_f32, inv_near - inv_far, 1.0, -inv_near]
    };

    // Texcoord basis — engine cbuffer 0x3F0001 packs
    // `(cos(roll), sin(roll), -sin(roll), cos(roll))` so the HLSL's
    // `texcoord_basis.xy` (u-axis) and `.zw` (v-axis) form the 2D
    // rotation `[cos -sin; sin cos]` for accumulated m_roll. m_roll
    // tracks ACTUAL head-twist roll (Euler n[1]) — see
    // `PatchyFog::update_per_frame`'s `relative_euler_delta` call.
    let cos_r = state.roll.cos();
    let sin_r = state.roll.sin();
    let texcoord_basis = [cos_r, sin_r, -sin_r, cos_r];

    // Per HLSL line 197: `height_fade = min(1, exp(rate × (full_h - sheet_h)))`.
    // For Halo's "fog dense below ground, decays at altitude" semantic with
    // `half > full` (e.g. guardian: full=18.5, half=19.75), the rate must
    // be POSITIVE so that:
    //   h < full → arg > 0 → exp(positive) > 1 → clamps to 1 (full density)
    //   h > full → arg < 0 → exp(negative) < 1 (decay)
    //   h = half → exp(rate × -(half-full)) = 0.5 → rate = +ln(2)/(half-full)
    //
    // Engine slot 0x3F0002 packs (from `c_patchy_fog::render_patchy_fog @
    // 0x1806B2160` line "full_intensity_height + reference_datum_plane"):
    //   .x = full_intensity_height + reference_datum_plane (s_weighted_atm)
    //   .y = (computed; from xmm3 — the rate)
    //   .z = *(s_weighted_atm + 0x1C) = depth_fade_factor
    //   .w = *(s_weighted_atm + 0x18)
    //
    // `m_reference_datum_plane` (offset 32 in `c_atmosphere_setting`) is
    // the H3 MCC tag field labelled "Sea Level" — NOT stripped, just
    // renamed from Reach's "Reference plane height". Guardian authors it
    // as −10 and relies on the resulting effective_full_h = 18.5 + (−10)
    // = 8.5 to push the camera ~11 units above full intensity, so the
    // exponential height fade attenuates the (otherwise saturating)
    // density=500 setting to near zero. Skipping the add left effective
    // full_h ten units too high — origin sat near the camera, height_fade
    // ≈ 0.5 instead of ≈ 0.002, fog saturated everywhere.
    let height_diff = (params.half_intensity_height - params.full_intensity_height).max(0.001);
    let attenuation_rate = std::f32::consts::LN_2 / height_diff; // = +ln(2)/diff
    // A/B toggle for the engine `+ reference_datum_plane` addition.
    // `PROTOMORPH_PATCHY_FOG_NO_SEA_LEVEL=1` disables the add so
    // effective_full_h reverts to `full_intensity_height` (pre-7bcf8a5
    // behavior). Engine dllcache decompile shows the add IS present, so
    // this env var is purely for visual A/B against MCC reference.
    let add_sea_level = std::env::var("PROTOMORPH_PATCHY_FOG_NO_SEA_LEVEL")
        .map(|s| s != "1")
        .unwrap_or(true);
    let effective_full_h = if add_sea_level {
        params.full_intensity_height + params.reference_datum_plane
    } else {
        params.full_intensity_height
    };
    let attenuation_data = [
        effective_full_h,
        attenuation_rate,
        sky.depth_fade_factor,
        0.0,
    ];

    let eye_position = [camera_position.x, camera_position.y, camera_position.z, 1.0];

    // window_pixel_bounds — engine maps {-1,1}² texcoords to [0,1]² UVs.
    // Half pixel offset to match texture sampling convention.
    let (sw, sh) = (screen_pixel_extent.0 as f32, screen_pixel_extent.1 as f32);
    let window_pixel_bounds = [
        0.5 + 0.5 / sw,           // center.u
        0.5 + 0.5 / sh,           // center.v
        0.5,                       // half.u (flips to texcoord_x range)
        -0.5,                      // half.v (negative — wgpu y-down)
    ];

    // Sheet depths use engine-tracked `m_distance_to_first_sheet`
    // (kept in [z_near, z_near + sep] by ring-buffer ratcheting) so
    // the closest sheet sits where the engine actually places it.
    let sep = sky.distance_between_sheets.max(0.5);
    let dist0 = state.distance_to_first_sheet.max(z_near);
    let sheet_depths0 = [dist0, dist0 + sep, dist0 + 2.0 * sep, dist0 + 3.0 * sep];
    let sheet_depths1 = [dist0 + 4.0 * sep, dist0 + 5.0 * sep, dist0 + 6.0 * sep, dist0 + 7.0 * sep];

    // Pop-fade — engine `xmm8 = (dist - z_near) / sep` clamped to 1
    // (asm at 0x1806B2675), `xmm9 = 1 - xmm8`. Slot 0x400000 packs
    // (xmm8, 1, 1, 1) × density; slot 0x400001 packs (1, 1, 1, xmm9)
    // × density. That fades the closest visible sheet out as it
    // approaches z_near (about to ratchet) and fades the new
    // farthest sheet in after a backward ratchet.
    let phase = ((state.distance_to_first_sheet - z_near) / sep).clamp(0.0, 1.0);
    let pop_first = phase;
    let pop_last = 1.0 - phase;
    let density = params.patchy_fog_density;
    let sheet_fade_factors0 = [pop_first * density, density, density, density];
    let sheet_fade_factors1 = [density, density, density, pop_last * density];

    // Per-sheet UV transform — engine `c_patchy_fog::render_patchy_fog
    // @ 0x1806B2160` builds half-extent zw per sheet as:
    //   v235  = atanf(1 / projection_matrix[0][0])    // half h-FOV (rad)
    //   v237  = atanf(1 / projection_matrix[1][1])    // half v-FOV (rad)
    //   v156  = sky.texture_repeat × 0.15915494       // repeat / (2π)
    //   v143  = (i × sheet_separation + m_distance_to_first_sheet)
    //         / sheet_separation                       // sheet depth-norm
    //   tex_coord_transform.zw = (v156 × v235 × v143, v156 × v237 × v143)
    //
    // With repeat=6 and half-FOV ≈ 30°, deeper sheets sample noise at
    // an 8× wider scale than the closest visible sheet — that
    // decorrelates the 8 sheets so the same noise tile doesn't stack
    // coherently and produce a visible grid.
    //
    // We use `z_near` as the first-sheet distance (engine maintains a
    // separate `m_distance_to_first_sheet` ring-pivot that we don't
    // mirror yet — that's item #2 of the patchy-fog port).
    let p_arr = camera_projection.to_cols_array_2d();
    let half_h_fov = (1.0 / p_arr[0][0]).atan();
    let half_v_fov = (1.0 / p_arr[1][1]).atan();
    let inv_two_pi = 0.15915494_f32;
    let r = sky.texture_repeat_rate.max(0.01);
    let scale_const_x = r * half_h_fov * inv_two_pi;
    let scale_const_y = r * half_v_fov * inv_two_pi;
    use crate::halo::render::render_patchy_fog::{K_FOG_SHEET_OFFSET_STEPS, sheet_skip_count};
    let n_steps = K_FOG_SHEET_OFFSET_STEPS as i32;
    let start = state.starting_sheet_attribute_index.rem_euclid(n_steps) as usize;
    // Mirror `c_patchy_fog::render_patchy_fog @ 0x1806B2160`'s per-sheet
    // index = `n + v48`: `v91 = ((n8 + v48) × sep + dist) / sep` for
    // depth_norm, `v88 = (v48 + n8 + m_starting_sheet_attribute_index)
    // % 100` for ring slot. `sheet_depths0/1` above are NOT shifted by
    // skip — engine sets them as `(0..8) × sep + dist`, so the 8 quads
    // keep their geometric depths; only the UV scaling shifts deeper.
    let skip = sheet_skip_count(state.closest_fog_z, state.distance_to_first_sheet, sep);
    let make_xform = |n: usize| {
        let i = n + skip;
        let sheet_depth_norm = ((i as f32) * sep + dist0) / sep;
        let slot = (i + start) % K_FOG_SHEET_OFFSET_STEPS;
        [
            state.lateral_offsets[slot],
            state.vertical_offsets[slot],
            scale_const_x * sheet_depth_norm,
            scale_const_y * sheet_depth_norm,
        ]
    };

    PatchyFogParams {
        inverse_z_transform,
        texcoord_basis,
        attenuation_data,
        eye_position,
        window_pixel_bounds,
        sheet_fade_factors0,
        sheet_fade_factors1,
        sheet_depths0,
        sheet_depths1,
        tex_coord_transform0: make_xform(0),
        tex_coord_transform1: make_xform(1),
        tex_coord_transform2: make_xform(2),
        tex_coord_transform3: make_xform(3),
        tex_coord_transform4: make_xform(4),
        tex_coord_transform5: make_xform(5),
        tex_coord_transform6: make_xform(6),
        tex_coord_transform7: make_xform(7),
    }
}

/// Build a `PatchyFogAtmosphere` from the engine-faithful accumulated
/// [`WeightedAtmosphereParameters`]. Phase 5.3 of the atmosphere port —
/// retires the parallel single-setting path
/// (`from_atmosphere_setting`-style derivation) so patchy fog reads
/// from the SAME source as the main atmosphere cbuffer.
///
/// Mirrors `c_atmosphere_fog_interface::set_constant @ 0x1803AE530`
/// (the subset used by `patchy_fog.hlsl`):
/// ```text
/// constant_0.xyz = normalize(sun_direction)              [SUN_DIR]
/// constant_1.xyz = sun_intensity / (β_m + β_p)           [SUN_INTENSITY_OVER_TM]
/// constant_2.w   = heyey_greenstein + 1                  [HEYEY_GREENSTEIN_CONSTANT_PLUS_ONE]
/// constant_3.xyz = β_p × log2(e)                         [TOTAL_MIE_LOG2E]
/// constant_5.xyz = β_p_angular × (1 - g²)                [MIE_THETA_PREFIX_HGC]
/// constant_extra.x = 2g                                  [HEYEY_GREENSTEIN_CONSTANT_TIMES_TWO]
/// ```
///
/// The prior `from_atmosphere_setting`-style version had two bugs
/// relative to engine that this commit fixes:
/// 1. `constant_2.w` packed `1 + g²` (textbook HG denominator) instead
///    of `g + 1` (Halo's HG approximation per HLSL
///    `HEYEY_GREENSTEIN_CONSTANT_PLUS_ONE` line 66 of patchy_fog.hlsl).
///    For Guardian's g≈0.2 that was a ~15% delta in inscatter tint.
/// 2. `sun_intensity` was naive `setting.color × setting.intensity`
///    instead of the chromaticity-preserving xyY→XYZ→linear-RGB
///    conversion engine `get_sun_parameters @ 0x1803AF990` does. The
///    accumulator already runs settings through `get_sun_parameters`
///    (Phase 4), so reading from `params.sun_intensity` is byte-faithful.
pub fn build_patchy_fog_atmosphere(
    params: &crate::halo::render::atmosphere_fog_interface::WeightedAtmosphereParameters,
) -> PatchyFogAtmosphere {
    // Engine `normalize3d_with_default(sun_direction, global_down3d)`.
    let len_sq = params.sun_direction.i * params.sun_direction.i
        + params.sun_direction.j * params.sun_direction.j
        + params.sun_direction.k * params.sun_direction.k;
    let sd = if len_sq > 1.0e-12 {
        let inv = len_sq.sqrt().recip();
        [
            params.sun_direction.i * inv,
            params.sun_direction.j * inv,
            params.sun_direction.k * inv,
        ]
    } else {
        [0.0_f32, 0.0, -1.0]
    };
    let sun_dir = [sd[0], sd[1], sd[2], 0.0];

    // SUN_INTENSITY_OVER_TM = sun_intensity / (β_m + β_p) per channel.
    // Engine guards on `(β_m + β_p) > 0.0001` per channel — zero the
    // whole vector if any channel is sub-threshold.
    let sum_x = params.beta_m.i + params.beta_p.i;
    let sum_y = params.beta_m.j + params.beta_p.j;
    let sum_z = params.beta_m.k + params.beta_p.k;
    let sun = if sum_x <= 1.0e-4 || sum_y <= 1.0e-4 || sum_z <= 1.0e-4 {
        [0.0_f32, 0.0, 0.0, 0.0]
    } else {
        [
            params.sun_intensity.red / sum_x,
            params.sun_intensity.green / sum_y,
            params.sun_intensity.blue / sum_z,
            0.0,
        ]
    };

    // Henyey-Greenstein g (Halo's `m_hg`).
    let g = params.heyey_greenstein;
    let one_minus_g2 = 1.0 - g * g;
    // Engine packs `g + 1`, NOT `1 + g²`. Halo HG approximation.
    let constant_2 = [0.0_f32, 0.0, 0.0, 1.0 + g];
    let extra = [2.0 * g, 0.0, 0.0, 0.0];

    // TOTAL_MIE_LOG2E = β_p × log2(e). Per-channel.
    let log2e = std::f32::consts::LOG2_E;
    let total_mie = [
        params.beta_p.i * log2e,
        params.beta_p.j * log2e,
        params.beta_p.k * log2e,
        0.0,
    ];

    // MIE_THETA_PREFIX_HGC = β_p_angular × (1 - g²). Per-channel.
    let mie_theta = [
        params.beta_p_angular.i * one_minus_g2,
        params.beta_p_angular.j * one_minus_g2,
        params.beta_p_angular.k * one_minus_g2,
        0.0,
    ];

    PatchyFogAtmosphere {
        sun_dir,
        sun_intensity_over_tm: sun,
        constant_2,
        total_mie_log2e: total_mie,
        _4: [0.0; 4],
        mie_theta_prefix_hgc: mie_theta,
        extra,
    }
}
