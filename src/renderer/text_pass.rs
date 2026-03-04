use crate::renderer::shared::SharedResources;
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
    face_label_buffers: [TextBuffer; 6],
    fps_string: String,
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
            face_label_buffers,
            fps_string: String::with_capacity(32),
            debug_cubemap: false,
        }
    }

    pub fn prepare(&mut self, shared: &SharedResources, fps: f32, width: u32, height: u32, debug_cubemap: bool) {
        use std::fmt::Write;
        self.debug_cubemap = debug_cubemap;

        self.fps_string.clear();
        let _ = write!(self.fps_string, "FPS: {:.0}\n{}x{}", fps, width, height);
        self.text_buffer.set_text(
            &mut self.font_system,
            &self.fps_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.text_buffer.shape_until_scroll(&mut self.font_system, false);

        self.viewport.update(&shared.queue, Resolution { width, height });

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

        if debug_cubemap {
            // Match the debug overlay quad positions from cubemap_debug_pass:
            // NDC: x = -0.75 + face * 0.27, y = -0.75, scale = 0.12
            // Convert NDC to pixel coords: px = (ndc + 1) / 2 * dimension
            let scale = 0.12_f32;
            for face in 0..6 {
                let ndc_x = -0.75 + face as f32 * 0.27;
                // Top-left of quad in NDC: (ndc_x - scale, -(ndc_y + scale))
                // The quad spans [ndc_x - scale, ndc_x + scale] in X
                //                [ndc_y - scale, ndc_y + scale] in Y
                // NDC y = -0.75, quad top in NDC Y = -0.75 + scale = -0.63
                // In pixel space (Y flipped): top = (1 - (-0.63)) / 2 * height
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

    pub fn record<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        self.text_renderer
            .render(&self.text_atlas, &self.viewport, rpass)
            .unwrap();
    }

    pub fn post_submit(&mut self) {
        self.text_atlas.trim();
    }
}
