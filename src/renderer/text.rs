use crate::renderer::shared::{FrameContext, RenderPass, SharedResources};
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as TextColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

const FACE_LABELS: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

pub struct TextPass {
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

impl TextPass {
    pub fn new(shared: &SharedResources) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&shared.device);
        let viewport = Viewport::new(&shared.device, &cache);
        let mut text_atlas = TextAtlas::new(&shared.device, &shared.queue, &cache, shared.config.format);
        let text_renderer = TextRenderer::new(&mut text_atlas, &shared.device, wgpu::MultisampleState::default(), None);
        let mut text_buffer = TextBuffer::new(&mut font_system, Metrics::new(20.0, 24.0));
        text_buffer.set_size(&mut font_system, Some(300.0), Some(30.0));
        text_buffer.set_text(&mut font_system, "FPS: --", &Attrs::new().family(Family::Monospace), Shaping::Basic, None);
        text_buffer.shape_until_scroll(&mut font_system, false);

        let mut shortcuts_buffer = TextBuffer::new(&mut font_system, Metrics::new(20.0, 24.0));
        shortcuts_buffer.set_size(&mut font_system, Some(400.0), Some(300.0));
        shortcuts_buffer.set_text(&mut font_system, "", &Attrs::new().family(Family::Monospace), Shaping::Basic, None);
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
}

impl RenderPass for TextPass {
    fn prepare(&mut self, ctx: &FrameContext) {
        use std::fmt::Write;

        let shared = ctx.shared;
        let game = ctx.game;
        let width = shared.config.width;
        let height = shared.config.height;

        self.debug_cubemap = game.debug_cubemap;

        self.fps_string.clear();
        let _ = write!(self.fps_string, "FPS: {:.0}\n{}x{}", game.fps_counter.display_fps, width, height);
        self.text_buffer.set_text(
            &mut self.font_system,
            &self.fps_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.text_buffer.shape_until_scroll(&mut self.font_system, false);

        let on_off = |b: bool| if b { "ON" } else { "OFF" };
        self.shortcuts_string.clear();
        let _ = write!(self.shortcuts_string, "\
[H] Flashlight {}\n\
[K] Specular Occlusion {}\n\
[P] Debug Cubemap {}\n\
[O] Cubemap Colors {}\n\
[1] Ready Anim\n\
[2] Reload Anim\n\
[3] Melee Anim\n\
[WASD] Move  [RF] Up/Down\n\
[Shift] Sprint\n\
[ESC] Release Cursor",
            on_off(game.is_flashlight_on()),
            on_off(game.enable_specular_occlusion),
            on_off(game.debug_cubemap),
            on_off(game.debug_cubemap_colors),
        );
        self.shortcuts_buffer.set_text(
            &mut self.font_system,
            &self.shortcuts_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.shortcuts_buffer.shape_until_scroll(&mut self.font_system, false);

        self.viewport.update(&shared.queue, Resolution { width, height });

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
            label: Some("text_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        self.text_renderer
            .render(&self.text_atlas, &self.viewport, &mut rpass)
            .unwrap();
    }

    fn post_submit(&mut self) {
        self.text_atlas.trim();
    }
}
