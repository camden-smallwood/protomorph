use std::{collections::HashSet, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod animation;
mod camera;
mod collision;
mod dds;
mod game;
mod lights;
mod materials;
mod models;
mod objects;
mod renderer;
mod sky;

use game::GameState;
use renderer::Renderer;

const LOOK_SENSITIVITY: f32 = 5.0;

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Renderer>,
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

        let game = GameState::new(&mut renderer, size.width, size.height);

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
                                KeyCode::KeyH => game.toggle_flashlight(),
                                KeyCode::KeyK => game.toggle_specular_occlusion(),
                                KeyCode::KeyP => game.toggle_debug_cubemap(),
                                KeyCode::KeyO => game.toggle_debug_cubemap_colors(),
                                KeyCode::Digit1 => game.trigger_weapon_animation("first_person ready"),
                                KeyCode::Digit2 => game.trigger_weapon_animation("first_person reload_empty"),
                                KeyCode::Digit3 => game.trigger_weapon_animation("first_person melee_strike_1"),
                                KeyCode::KeyG => game.toggle_grunt_animation_pause(),
                                KeyCode::KeyT => game.toggle_weapon_detach(),
                                KeyCode::KeyV => game.toggle_vignette(),
                                KeyCode::Tab => game.toggle_flycam(),
                                _ => {}
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

    let mut app = App {
        window: None,
        gpu: None,
        game: None,
        keys_pressed: HashSet::new(),
        cursor_grabbed: false,
        last_frame: Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}
