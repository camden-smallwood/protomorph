use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod game;
mod halo;

use game::GameState;
use halo::render::Renderer;

const LOOK_SENSITIVITY: f32 = 5.0;

pub fn assets_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let candidate = exe_dir.join("assets");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Renderer>,
    scenario_path: PathBuf,
    game: Option<GameState>,
    keys_pressed: HashSet<KeyCode>,
    cursor_grabbed: bool,
    last_frame: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes().with_title("protomorph");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();

        let mut renderer = Renderer::new(Arc::clone(&window));

        let game = GameState::new(&mut renderer, size.width, size.height, self.scenario_path.clone());

        self.gpu = Some(renderer);
        self.game = Some(game);
        self.window = Some(window);

        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }

                if let Some(game) = self.game.as_mut() {
                    game.camera.handle_resize(size.width, size.height);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state.is_pressed() {
                        if key == KeyCode::Escape {
                            self.release_cursor();
                        }

                        self.keys_pressed.insert(key);
                    } else {
                        if let Some(game) = self.game.as_mut() {
                            match key {
                                KeyCode::KeyK => game.toggle_specular_occlusion(),
                                KeyCode::KeyV => game.toggle_vignette(),
                                _ => {}
                            }
                        }
                        if let Some(gpu) = self.gpu.as_mut() {
                            if key == KeyCode::KeyT {
                                let r = &mut gpu.transparency_renderer;
                                r.render_enabled = !r.render_enabled;
                                eprintln!(
                                    "[diag] transparency_renderer.render_enabled = {}",
                                    r.render_enabled,
                                );
                            }
                            if key == KeyCode::KeyP {
                                let d = &mut gpu.decal_gpu;
                                d.render_enabled = !d.render_enabled;
                                eprintln!(
                                    "[diag] decal_gpu.render_enabled = {}",
                                    d.render_enabled,
                                );
                            }
                        }

                        self.keys_pressed.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput { state, .. } => {
                if state.is_pressed() && !self.cursor_grabbed {
                    self.grab_cursor();
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                if let Some(game) = self.game.as_mut() {
                    game.update(&self.keys_pressed, dt);
                }

                if let (Some(gpu), Some(game)) = (self.gpu.as_mut(), self.game.as_ref()) {
                    gpu.render(game);
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if !self.cursor_grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(game) = self.game.as_mut() {
                game.camera.rotation.x += -delta.0 as f32 * 0.01 * LOOK_SENSITIVITY;
                game.camera.rotation.y += -delta.1 as f32 * 0.01 * LOOK_SENSITIVITY;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

impl App {
    fn grab_cursor(&mut self) {
        if let Some(window) = self.window.as_ref() {
            window.set_cursor_visible(false);

            let _ = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));

            self.cursor_grabbed = true;
        }
    }

    fn release_cursor(&mut self) {
        if let Some(window) = self.window.as_ref() {
            window.set_cursor_visible(true);

            let _ = window.set_cursor_grab(CursorGrabMode::None);

            self.cursor_grabbed = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut args = std::env::args();
    args.next();

    let scenario_path = match args.next() {
        Some(s) => PathBuf::from(s),
        None => {
            use rfd::FileDialog;

            let Some(scenario_path) = FileDialog::new()
                .add_filter("scenario", &["scenario"])
                .set_directory(".")
                .pick_file()
            else {
                return;
            };

            scenario_path
        }
    };

    let mut app = App {
        window: None,
        gpu: None,
        scenario_path,
        game: None,
        keys_pressed: HashSet::new(),
        cursor_grabbed: false,
        last_frame: Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}
