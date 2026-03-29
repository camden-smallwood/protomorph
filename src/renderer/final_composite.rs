use crate::renderer::{
    create_fullscreen_pipeline, sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    tex_entry, uniform_entry,
};
use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as TextColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;

// ===========================================================================
// Final Composite Pass — FXAA + CubemapDebug + Text in a single render pass
// All three write to surface_view with no depth attachment.
// ===========================================================================

const FACE_LABELS: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DebugParams {
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    _pad: f32,
}

pub struct FinalCompositePass {
    // FXAA
    fxaa_pipeline: wgpu::RenderPipeline,
    fxaa_bgl: wgpu::BindGroupLayout,
    fxaa_bind_group: wgpu::BindGroup,

    // Cubemap debug
    debug_pipeline: wgpu::RenderPipeline,
    debug_bind_groups: [wgpu::BindGroup; 6],

    // Text
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    text_buffer: TextBuffer,
    shortcuts_buffer: TextBuffer,
    face_label_buffers: [TextBuffer; 6],
    fps_string: String,
    shortcuts_string: String,
    debug_cubemap: bool,
}

impl FinalCompositePass {
    pub fn new(
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        surface_format: wgpu::TextureFormat,
        face_views: [&wgpu::TextureView; 6],
        filtering_sampler: &wgpu::Sampler,
    ) -> Self {
        let device = &shared.device;
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        // --- FXAA ---
        let fxaa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa_bgl"),
            entries: &[
                tex_entry(0, filterable),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let fxaa_pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/fxaa.wgsl"),
            &[&fxaa_bgl],
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "fxaa_pipeline",
        );

        let fxaa_bind_group = Self::create_fxaa_bind_group(
            device,
            &fxaa_bgl,
            &intermediates.post_composite_view,
            &shared.bloom_sampler,
        );

        // --- Cubemap Debug ---
        let debug_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cubemap_debug_bgl"),
            entries: &[
                tex_entry(0, filterable),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                uniform_entry(
                    2,
                    size_of::<DebugParams>() as u64,
                    wgpu::ShaderStages::VERTEX,
                    false,
                ),
            ],
        });

        let debug_pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/cubemap_debug.wgsl"),
            &[&debug_bgl],
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "cubemap_debug_pipeline",
        );

        let scale = 0.12;
        let y = -0.75;
        let debug_bind_groups: [wgpu::BindGroup; 6] = std::array::from_fn(|face| {
            let x = -0.75 + face as f32 * 0.27;
            let params = DebugParams {
                offset_x: x,
                offset_y: y,
                scale,
                _pad: 0.0,
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("cubemap_debug_params_{face}")),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("cubemap_debug_bg_{face}")),
                layout: &debug_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(face_views[face]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(filtering_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        });

        // --- Text ---
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut text_atlas =
            TextAtlas::new(device, &shared.queue, &cache, shared.config.format);
        let text_renderer =
            TextRenderer::new(&mut text_atlas, device, wgpu::MultisampleState::default(), None);

        let mut text_buffer = TextBuffer::new(&mut font_system, Metrics::new(20.0, 24.0));
        text_buffer.set_size(&mut font_system, Some(300.0), Some(30.0));
        text_buffer.set_text(
            &mut font_system,
            "FPS: --",
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        text_buffer.shape_until_scroll(&mut font_system, false);

        let mut shortcuts_buffer = TextBuffer::new(&mut font_system, Metrics::new(20.0, 24.0));
        shortcuts_buffer.set_size(&mut font_system, Some(400.0), Some(300.0));
        shortcuts_buffer.set_text(
            &mut font_system,
            "",
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        shortcuts_buffer.shape_until_scroll(&mut font_system, false);

        let face_label_buffers = std::array::from_fn(|i| {
            let mut buf = TextBuffer::new(&mut font_system, Metrics::new(14.0, 16.0));
            buf.set_size(&mut font_system, Some(40.0), Some(20.0));
            buf.set_text(
                &mut font_system,
                FACE_LABELS[i],
                &Attrs::new().family(Family::Monospace),
                Shaping::Basic,
                None,
            );
            buf.shape_until_scroll(&mut font_system, false);
            buf
        });

        Self {
            fxaa_pipeline,
            fxaa_bgl,
            fxaa_bind_group,
            debug_pipeline,
            debug_bind_groups,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            text_buffer,
            shortcuts_buffer,
            face_label_buffers,
            fps_string: String::with_capacity(32),
            shortcuts_string: String::with_capacity(512),
            debug_cubemap: false,
        }
    }

    fn create_fxaa_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        input_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl RenderPass for FinalCompositePass {
    fn prepare(&mut self, ctx: &FrameContext) {
        use std::fmt::Write;

        let shared = ctx.shared;
        let game = ctx.game;
        let width = shared.config.width;
        let height = shared.config.height;

        self.debug_cubemap = game.debug_cubemap;

        // FPS text
        self.fps_string.clear();
        let _ = write!(
            self.fps_string,
            "FPS: {:.0}\n{}x{}",
            game.fps_counter.display_fps, width, height
        );
        self.text_buffer.set_text(
            &mut self.font_system,
            &self.fps_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.text_buffer.shape_until_scroll(&mut self.font_system, false);

        // Shortcuts text
        let on_off = |b: bool| if b { "ON" } else { "OFF" };
        self.shortcuts_string.clear();
        let _ = write!(
            self.shortcuts_string,
            "\
[G] Grunt Animation {}\n\
[T] Weapon Attached {}\n\
[H] Flashlight {}\n\
[K] Specular Occlusion {}\n\
[V] Vignette {}\n\
[P] Debug Cubemap {}\n\
[O] Cubemap Colors {}\n\
[Tab] Camera Mode: {}\n\
[1] Ready Anim\n\
[2] Reload Anim\n\
[3] Melee Anim\n\
[WASD] Move  [Space] Jump\n\
[Shift] Sprint\n\
[ESC] Release Cursor",
            if game.is_grunt_animation_paused() {
                "PAUSED"
            } else {
                "PLAYING"
            },
            on_off(!game.weapon_detached),
            on_off(game.is_flashlight_on()),
            on_off(game.enable_specular_occlusion),
            on_off(game.enable_vignette),
            on_off(game.debug_cubemap),
            on_off(game.debug_cubemap_colors),
            if game.is_flycam() { "FLYCAM" } else { "PLAYER" },
        );
        self.shortcuts_buffer.set_text(
            &mut self.font_system,
            &self.shortcuts_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.shortcuts_buffer
            .shape_until_scroll(&mut self.font_system, false);

        self.viewport
            .update(&shared.queue, Resolution { width, height });

        // Build text areas
        let mut areas: Vec<TextArea> = vec![TextArea {
            buffer: &self.text_buffer,
            left: 10.0,
            top: 10.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            },
            default_color: TextColor::rgb(255, 255, 255),
            custom_glyphs: &[],
        }];

        if self.debug_cubemap {
            let scale = 0.12_f32;
            for face in 0..6 {
                let ndc_x = -0.75 + face as f32 * 0.27;
                let ndc_y = -0.75;
                let px_left = ((ndc_x - scale + 1.0) / 2.0 * width as f32) + 2.0;
                let px_top = ((1.0 - (ndc_y + scale)) / 2.0 * height as f32) + 2.0;

                areas.push(TextArea {
                    buffer: &self.face_label_buffers[face],
                    left: px_left,
                    top: px_top,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: width as i32,
                        bottom: height as i32,
                    },
                    default_color: TextColor::rgb(255, 255, 0),
                    custom_glyphs: &[],
                });
            }
        }

        areas.push(TextArea {
            buffer: &self.shortcuts_buffer,
            left: 10.0,
            top: 58.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            },
            default_color: TextColor::rgb(255, 255, 255),
            custom_glyphs: &[],
        });

        self.text_renderer
            .prepare(
                &shared.device,
                &shared.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.viewport,
                areas,
                &mut self.swash_cache,
            )
            .unwrap();
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("final_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        // 1. FXAA
        rpass.set_pipeline(&self.fxaa_pipeline);
        rpass.set_bind_group(0, &self.fxaa_bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);

        // 2. Cubemap debug (conditional)
        if ctx.game.debug_cubemap {
            rpass.set_pipeline(&self.debug_pipeline);
            rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
            for face in 0..6 {
                rpass.set_bind_group(0, &self.debug_bind_groups[face], &[]);
                rpass.draw(0..6, 0..1);
            }
        }

        // 3. Text
        self.text_renderer
            .render(&self.text_atlas, &self.viewport, &mut rpass)
            .unwrap();
    }

    fn resize(
        &mut self,
        shared: &SharedResources,
        _gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.fxaa_bind_group = Self::create_fxaa_bind_group(
            &shared.device,
            &self.fxaa_bgl,
            &intermediates.post_composite_view,
            &shared.bloom_sampler,
        );
    }

    fn post_submit(&mut self) {
        self.text_atlas.trim();
    }
}
