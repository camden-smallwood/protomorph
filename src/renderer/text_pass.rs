use crate::renderer::shared::SharedResources;
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as TextColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

pub struct TextPass {
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    text_buffer: TextBuffer,
    fps_string: String,
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

        Self {
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            text_buffer,
            fps_string: String::with_capacity(32),
        }
    }

    pub fn prepare(&mut self, shared: &SharedResources, fps: f32, width: u32, height: u32) {
        use std::fmt::Write;
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
        self.text_renderer
            .prepare(
                &shared.device,
                &shared.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.viewport,
                [TextArea {
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
                }],
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
