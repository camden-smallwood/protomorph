use crate::{
    animation::AnimationManager,
    camera::Camera,
    lights::{LightData, LightIndex, LightStore},
    models::ModelData,
    objects::{ObjectIndex, ObjectStore},
    renderer::{Renderer, env_probe::GpuAtmosphereData},
};
use glam::Vec3;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

const MOVEMENT_SPEED: f32 = 1.0;

// ---------------------------------------------------------------------------
// FPS counter
// ---------------------------------------------------------------------------

pub struct FpsCounter {
    frame_count: u32,
    elapsed: f32,
    pub display_fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self { frame_count: 0, elapsed: 0.0, display_fps: 0.0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.frame_count += 1;
        self.elapsed += dt;
        if self.elapsed >= 0.5 {
            self.display_fps = self.frame_count as f32 / self.elapsed;
            self.frame_count = 0;
            self.elapsed = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

pub struct GameState {
    pub objects: ObjectStore,
    pub lights: LightStore,
    pub camera: Camera,
    pub model_data: Vec<ModelData>,
    pub fps_counter: FpsCounter,
    pub atmosphere: GpuAtmosphereData,

    flashlight_index: LightIndex,
    weapon_index: ObjectIndex,
    grunt_index: ObjectIndex,
    weapon_model_index: usize,
    weapon_moving_anim_idx: Option<usize>,
    rotation_towards_grunt: f32,
    pub debug_cubemap: bool,
    pub debug_cubemap_colors: bool,
    pub enable_specular_occlusion: bool,
}

impl GameState {
    pub fn new(renderer: &mut Renderer, width: u32, height: u32) -> Self {
        let mut camera = Camera::new();
        camera.handle_resize(width, height);

        let mut lights = LightStore::new();

        // Flashlight (hidden by default)
        let mut flashlight = LightData::new_spot();
        flashlight.hidden = true;
        flashlight.diffuse_color = Vec3::splat(3.0);
        flashlight.ambient_color = Vec3::splat(0.05);
        flashlight.specular_color = Vec3::splat(3.0);
        flashlight.constant_atten = 1.0;
        flashlight.inner_cutoff = 10.0;
        flashlight.outer_cutoff = 25.0;
        flashlight.casts_shadow = true;
        let flashlight_index = lights.new_light(flashlight);

        // // Angled spot light over the grunt
        // let mut grunt_spot = LightData::new_spot();
        // grunt_spot.position = Vec3::new(-3.0, 2.0, 5.0);
        // grunt_spot.direction = (Vec3::new(-5.0, 0.0, 0.0) - grunt_spot.position).normalize();
        // grunt_spot.diffuse_color = Vec3::splat(2.0);
        // grunt_spot.ambient_color = Vec3::splat(0.1);
        // grunt_spot.specular_color = Vec3::splat(2.0);
        // grunt_spot.constant_atten = 1.0;
        // grunt_spot.linear_atten = 0.02;
        // grunt_spot.quadratic_atten = 0.005;
        // grunt_spot.inner_cutoff = 20.0;
        // grunt_spot.outer_cutoff = 35.0;
        // grunt_spot.casts_shadow = true;
        // lights.new_light(grunt_spot);

        // More angled sun angle
        let sun_dir = Vec3::new(-0.7, 0.4, -0.2).normalize();

        // More overhead sun angle:
        // let sun_dir = Vec3::new(-0.5, 0.3, -0.866).normalize();

        // Warm sunset — low sun from front-left
        let mut sun = LightData::new_directional();
        sun.direction = sun_dir;
        sun.diffuse_color = Vec3::new(1.4, 0.7, 0.3);
        sun.ambient_color = Vec3::new(0.12, 0.08, 0.06);
        sun.specular_color = Vec3::new(1.4, 0.7, 0.3);
        sun.casts_shadow = true;
        lights.new_light(sun);

        // // Noon summer sun
        // let mut sun = LightData::new_directional();
        // sun.direction = sun_dir;
        // sun.diffuse_color = Vec3::new(1.0, 0.95, 0.8);
        // sun.ambient_color = Vec3::new(0.15, 0.14, 0.12);
        // sun.specular_color = Vec3::new(1.0, 0.95, 0.8);
        // sun.casts_shadow = true;
        // lights.new_light(sun);

        let atmo_sun = -sun_dir; // direction TO the sun = negation of light travel
        let mut state = Self {
            objects: ObjectStore::new(),
            lights,
            camera,
            model_data: Vec::new(),
            fps_counter: FpsCounter::new(),
            atmosphere: GpuAtmosphereData {
                sun_direction: atmo_sun.to_array(),
                atmosphere_enable: 1.0,
                rayleigh_coefficients: [0.02, 0.05, 0.1],
                rayleigh_height_scale: 20.0,
                mie_coefficient: 0.01,
                mie_height_scale: 8.0,
                mie_g: 0.76,
                max_fog_thickness: 50.0,
                inscatter_scale: 1.0,
                reference_height: 0.0,
                _pad: [0.0; 2],
            },
            flashlight_index,
            weapon_index: ObjectIndex(0),
            grunt_index: ObjectIndex(0),
            weapon_model_index: 0,
            weapon_moving_anim_idx: None,
            rotation_towards_grunt: 0.0,
            debug_cubemap: false,
            debug_cubemap_colors: false,
            enable_specular_occlusion: true,
        };

        state.load_scene(renderer);
        state
    }

    fn load_scene(&mut self, renderer: &mut Renderer) {
        let manifest = env!("CARGO_MANIFEST_DIR");

        let (plane_model, plane_data) = renderer.load_model(&format!("{manifest}/assets/models/plane.fbx"));
        self.model_data.push(plane_data);

        let (grunt_model, grunt_data) = renderer.load_model(&format!("{manifest}/assets/models/grunt.fbx"));
        self.model_data.push(grunt_data);

        let (weapon_model, weapon_data) = renderer.load_model(&format!("{manifest}/assets/models/assault_rifle.fbx"));
        self.model_data.push(weapon_data);
        self.weapon_model_index = weapon_model;

        // Plane
        let plane = self.objects.new_object();
        self.objects.get_mut(plane).model_index = Some(plane_model);

        // Grunt
        let grunt = self.objects.new_object();
        self.objects.get_mut(grunt).position = Vec3::new(-5.0, 0.0, 0.0);
        self.objects.get_mut(grunt).scale = Vec3::splat(0.1);
        self.objects.get_mut(grunt).model_index = Some(grunt_model);
        self.init_animations(grunt, grunt_model);
        if let Some(anim) = self.objects.get_mut(grunt).animations.as_mut() {
            anim.set_active(0, true);
            anim.set_looping(0, true);
        }
        self.grunt_index = grunt;

        // Weapon (first-person viewmodel)
        let weapon = self.objects.new_object();
        self.objects.get_mut(weapon).scale = Vec3::splat(0.01);
        self.objects.get_mut(weapon).model_index = Some(weapon_model);
        self.init_animations(weapon, weapon_model);
        self.weapon_index = weapon;

        // Set "first_person moving" to looping (but not active) and cache the index
        if let Some(moving_idx) = self.model_data[weapon_model].find_animation_by_name("first_person moving") {
            let anim = self.objects.get_mut(weapon).animations.as_mut().unwrap();
            anim.set_looping(moving_idx, true);
            self.weapon_moving_anim_idx = Some(moving_idx);
        }

        // Play "first_person ready" once at startup
        if let Some(ready_idx) = self.model_data[weapon_model].find_animation_by_name("first_person ready") {
            let anim = self.objects.get_mut(weapon).animations.as_mut().unwrap();
            anim.set_active(ready_idx, true);
        }
    }

    fn init_animations(&mut self, obj_index: ObjectIndex, model_index: usize) {
        let model = &self.model_data[model_index];
        if model.nodes.is_empty() || model.animations.is_empty() {
            return;
        }

        let anim_mgr = AnimationManager::new(model);
        self.objects.get_mut(obj_index).animations = Some(anim_mgr);
    }

    pub fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        self.update_camera_rotation_towards_grunt(dt);
        self.update_movement(keys, dt);
        self.update_weapon_viewmodel();
        self.update_flashlight();
        self.update_weapon_animations();
        self.objects.update(&self.model_data, dt);
        self.fps_counter.update(dt);
    }

    pub fn toggle_debug_cubemap(&mut self) {
        self.debug_cubemap = !self.debug_cubemap;
    }

    pub fn toggle_debug_cubemap_colors(&mut self) {
        self.debug_cubemap_colors = !self.debug_cubemap_colors;
    }

    pub fn toggle_specular_occlusion(&mut self) {
        self.enable_specular_occlusion = !self.enable_specular_occlusion;
    }

    pub fn toggle_flashlight(&mut self) {
        let light = self.lights.get_mut(self.flashlight_index);
        light.hidden = !light.hidden;
    }

    pub fn is_flashlight_on(&self) -> bool {
        !self.lights.get(self.flashlight_index).hidden
    }

    pub fn trigger_weapon_animation(&mut self, name: &str) {
        if let Some(idx) = self.model_data[self.weapon_model_index].find_animation_by_name(name) {
            let anim = self.objects.get_mut(self.weapon_index).animations.as_mut().unwrap();
            anim.set_active(idx, true);
        }
    }

    // --- Private update methods ---

    fn update_camera_rotation_towards_grunt(&mut self, dt: f32) {
        if self.rotation_towards_grunt < 1.0 {
            let grunt_pos = self.objects.get(self.grunt_index).position;
            self.camera.rotate_towards_point(grunt_pos, self.rotation_towards_grunt);
            self.rotation_towards_grunt = (self.rotation_towards_grunt + dt * 2.0).min(1.0);
        }
    }

    fn update_movement(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        let mut move_dir = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) { move_dir += self.camera.forward; }
        if keys.contains(&KeyCode::KeyS) { move_dir -= self.camera.forward; }
        if keys.contains(&KeyCode::KeyA) { move_dir += self.camera.right; }
        if keys.contains(&KeyCode::KeyD) { move_dir -= self.camera.right; }
        if keys.contains(&KeyCode::KeyR) { move_dir += Vec3::Z; }
        if keys.contains(&KeyCode::KeyF) { move_dir -= Vec3::Z; }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        let mut speed = MOVEMENT_SPEED;
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            speed *= 2.0;
        }

        self.camera.velocity = move_dir * speed * dt;
        self.camera.update();
    }

    fn update_weapon_viewmodel(&mut self) {
        let weapon = self.objects.get_mut(self.weapon_index);
        weapon.position = self.camera.position + Vec3::new(0.0, 0.0, -0.015);
        weapon.rotation = Vec3::new(0.0, -self.camera.rotation.y, self.camera.rotation.x);
    }

    fn update_flashlight(&mut self) {
        let light = self.lights.get_mut(self.flashlight_index);
        light.position = self.camera.position + (self.camera.forward * 0.25);
        light.direction = self.camera.forward;
    }

    fn update_weapon_animations(&mut self) {
        let is_moving = self.camera.velocity.length_squared() > 0.0;

        if let Some(moving_idx) = self.weapon_moving_anim_idx {
            let anim = self.objects.get_mut(self.weapon_index).animations.as_mut().unwrap();
            let was_active = anim.is_active(moving_idx);

            if !was_active && is_moving {
                anim.set_active(moving_idx, true);
            } else if was_active && !is_moving {
                anim.set_active(moving_idx, false);
            }

            anim.set_speed(moving_idx, if is_moving { 1.0 } else { 0.0 });
        }
    }
}
