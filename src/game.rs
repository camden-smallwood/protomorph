use crate::{
    animation::AnimationManager,
    camera::Camera,
    collision::{
        self, CollisionMesh, PlayerPhysics, build_collision_mesh,
        EYE_HEIGHT, GRAVITY, GROUND_SNAP, JUMP_VELOCITY, PLAYER_RADIUS,
    },
    lights::{LightData, LightIndex, LightStore},
    models::ModelData,
    objects::{ObjectIndex, ObjectStore},
    renderer::{Renderer, env_probe::GpuAtmosphereData},
    sky::SkyConfig,
};
use glam::{Vec2, Vec3};
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
    pub sky: SkyConfig,

    flashlight_index: LightIndex,
    weapon_index: ObjectIndex,
    grunt_index: ObjectIndex,
    weapon_model_index: usize,
    weapon_idle_anim_idx: Option<usize>,
    weapon_moving_anim_idx: Option<usize>,
    rotation_towards_grunt: f32,
    pub weapon_detached: bool,
    pub debug_cubemap: bool,
    pub debug_cubemap_colors: bool,
    pub enable_specular_occlusion: bool,
    pub enable_vignette: bool,
    pub total_time: f32,
    collision_mesh: CollisionMesh,
    physics: PlayerPhysics,
    pub flycam: bool,
    is_moving: bool,
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

        let sky = SkyConfig::default();

        // Sun light from sky config
        let mut sun = LightData::new_directional();
        sun.direction = sky.sun_direction;
        sun.diffuse_color = sky.sun_diffuse_color;
        sun.ambient_color = sky.sun_ambient_color;
        sun.specular_color = sky.sun_specular_color;
        sun.casts_shadow = true;
        lights.new_light(sun);

        let mut state = Self {
            objects: ObjectStore::new(),
            lights,
            camera,
            model_data: Vec::new(),
            fps_counter: FpsCounter::new(),
            sky,
            flashlight_index,
            weapon_index: ObjectIndex(0),
            grunt_index: ObjectIndex(0),
            weapon_model_index: 0,
            weapon_idle_anim_idx: None,
            weapon_moving_anim_idx: None,
            rotation_towards_grunt: 0.0,
            weapon_detached: false,
            debug_cubemap: false,
            debug_cubemap_colors: false,
            enable_specular_occlusion: true,
            enable_vignette: true,
            total_time: 0.0,
            collision_mesh: CollisionMesh {
                floor_triangles: Vec::new(),
                wall_segments: Vec::new(),
            },
            physics: PlayerPhysics::new(),
            flycam: false,
            is_moving: false,
        };

        state.load_scene(renderer);
        state
    }

    fn load_scene(&mut self, renderer: &mut Renderer) {
        let models_dir = crate::assets_dir().join("models");
        let model_path = |name: &str| models_dir.join(name).to_string_lossy().into_owned();

        // let (plane_model, plane_data) = renderer.load_model_with_uv_scale(&model_path("plane.fbx"), 10.0);
        let (plane_model, plane_data) = renderer.load_model(&model_path("plane2.fbx"));
        self.model_data.push(plane_data);

        let (grunt_model, grunt_data) = renderer.load_model(&model_path("grunt.fbx"));
        self.model_data.push(grunt_data);

        let (weapon_model, weapon_data) = renderer.load_model(&model_path("assault_rifle.fbx"));
        self.model_data.push(weapon_data);
        self.weapon_model_index = weapon_model;

        // Plane
        let plane = self.objects.new_object();
        // self.objects.get_mut(plane).scale = Vec3::splat(10.0);
        self.objects.get_mut(plane).model_index = Some(plane_model);

        // Build collision mesh from plane geometry
        let model_matrix = self.objects.get(plane).model_matrix();
        self.collision_mesh = build_collision_mesh(&self.model_data[plane_model], model_matrix);

        // Grunt
        let grunt = self.objects.new_object();
        self.objects.get_mut(grunt).position = Vec3::new(-5.0, 0.0, 0.0);
        self.objects.get_mut(grunt).scale = Vec3::splat(0.01);
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

        // Set "first_person idle" to looping (but not active) and cache the index
        if let Some(idle_idx) = self.model_data[weapon_model].find_animation_by_name("first_person idle") {
            let anim = self.objects.get_mut(weapon).animations.as_mut().unwrap();
            anim.set_looping(idle_idx, true);
            self.weapon_idle_anim_idx = Some(idle_idx);
        }

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
        self.total_time += dt;
    }

    pub fn toggle_debug_cubemap(&mut self) {
        self.debug_cubemap = !self.debug_cubemap;
    }

    pub fn toggle_debug_cubemap_colors(&mut self) {
        self.debug_cubemap_colors = !self.debug_cubemap_colors;
    }

    pub fn atmosphere_data(&self) -> GpuAtmosphereData {
        GpuAtmosphereData::from_sky_config(&self.sky)
    }

    pub fn toggle_specular_occlusion(&mut self) {
        self.enable_specular_occlusion = !self.enable_specular_occlusion;
    }

    pub fn toggle_vignette(&mut self) {
        self.enable_vignette = !self.enable_vignette;
    }


    pub fn is_grunt_animation_paused(&self) -> bool {
        self.objects.get(self.grunt_index).animations.as_ref()
            .map_or(false, |a| a.is_paused(0))
    }

    pub fn toggle_grunt_animation_pause(&mut self) {
        if let Some(anim) = self.objects.get_mut(self.grunt_index).animations.as_mut() {
            let paused = anim.is_paused(0);
            anim.set_paused(0, !paused);
        }
    }

    pub fn toggle_weapon_detach(&mut self) {
        self.weapon_detached = !self.weapon_detached;
    }

    pub fn toggle_flycam(&mut self) {
        self.flycam = !self.flycam;
        if !self.flycam {
            // Re-entering player mode: reset vertical velocity
            self.physics.vertical_velocity = 0.0;
            self.physics.is_grounded = false;
        }
    }

    pub fn is_flycam(&self) -> bool {
        self.flycam
    }

    pub fn toggle_flashlight(&mut self) {
        let light = self.lights.get_mut(self.flashlight_index);
        light.hidden = !light.hidden;
    }

    pub fn is_flashlight_on(&self) -> bool {
        !self.lights.get(self.flashlight_index).hidden
    }

    pub fn trigger_weapon_animation(&mut self, name: &str) {
        let animations = self.objects.get_mut(self.weapon_index).animations.as_mut().unwrap();
        
        if let Some(idle_index) = self.weapon_idle_anim_idx
            && animations.is_active(idle_index)
        {
            animations.set_active(idle_index, false);
        }
        
        if let Some(moving_index) = self.weapon_moving_anim_idx
            && animations.is_active(moving_index)
        {
            animations.set_active(moving_index, false);
        }
        
        if let Some(idx) = self.model_data[self.weapon_model_index].find_animation_by_name(name) {
            animations.set_active(idx, true);
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
        if self.flycam {
            self.update_movement_flycam(keys, dt);
        } else {
            self.update_movement_player(keys, dt);
        }
    }

    fn update_movement_flycam(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
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

        self.is_moving = move_dir.length_squared() > 0.0;
        self.camera.velocity = move_dir * speed * dt;
        self.camera.update();
    }

    fn update_movement_player(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        // Horizontal input — project forward/right onto XY plane
        let forward_xy = Vec2::new(self.camera.forward.x, self.camera.forward.y).normalize_or_zero();
        let right_xy = Vec2::new(self.camera.right.x, self.camera.right.y).normalize_or_zero();

        let mut move_dir = Vec2::ZERO;
        if keys.contains(&KeyCode::KeyW) { move_dir += forward_xy; }
        if keys.contains(&KeyCode::KeyS) { move_dir -= forward_xy; }
        if keys.contains(&KeyCode::KeyA) { move_dir += right_xy; }
        if keys.contains(&KeyCode::KeyD) { move_dir -= right_xy; }
        self.is_moving = move_dir.length_squared() > 0.0;
        if self.is_moving {
            move_dir = move_dir.normalize();
        }

        let mut speed = MOVEMENT_SPEED;
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            speed *= 2.0;
        }

        // Apply horizontal movement + wall collision
        let new_xy = self.camera.position.truncate() + move_dir * speed * dt;
        let feet_z = self.camera.position.z - EYE_HEIGHT;
        let resolved_xy = collision::collide_and_slide(
            new_xy,
            feet_z,
            &self.collision_mesh.wall_segments,
            PLAYER_RADIUS,
        );
        self.camera.position.x = resolved_xy.x;
        self.camera.position.y = resolved_xy.y;

        // Jump
        if keys.contains(&KeyCode::Space) && self.physics.is_grounded {
            self.physics.vertical_velocity = JUMP_VELOCITY;
            self.physics.is_grounded = false;
        }

        // Gravity
        if !self.physics.is_grounded {
            self.physics.vertical_velocity -= GRAVITY * dt;
        }

        // Apply vertical movement
        self.camera.position.z += self.physics.vertical_velocity * dt;

        // Ground check
        if let Some(ground_z) = collision::ground_raycast(
            self.camera.position,
            &self.collision_mesh.floor_triangles,
        ) {
            let feet_z = self.camera.position.z - EYE_HEIGHT;
            if feet_z <= ground_z + GROUND_SNAP && self.physics.vertical_velocity <= 0.0 {
                self.camera.position.z = ground_z + EYE_HEIGHT;
                self.physics.vertical_velocity = 0.0;
                self.physics.is_grounded = true;
            } else if feet_z > ground_z + GROUND_SNAP {
                self.physics.is_grounded = false;
            }
        } else {
            self.physics.is_grounded = false;
        }

        self.camera.velocity = Vec3::ZERO;
        self.camera.update();
    }

    fn update_weapon_viewmodel(&mut self) {
        if self.weapon_detached {
            return;
        }
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
        let object = self.objects.get_mut(self.weapon_index);
        let animations = object.animations.as_mut().unwrap();

        for (index, animation) in self.model_data[self.weapon_model_index].animations.iter().enumerate() {
            match animation.name.as_str() {
                "first_person ready" | "first_person idle" | "first_person moving" => continue,

                _ => {
                    if animations.is_active(index) {
                        return;
                    }
                }
            }
        }

        match (self.weapon_idle_anim_idx, self.weapon_moving_anim_idx) {
            (Some(idle_idx), Some(moving_idx)) => {
                let was_moving_active = animations.is_active(moving_idx);
                let was_idle_active = animations.is_active(idle_idx);

                if self.is_moving {
                    if was_idle_active {
                        animations.set_active(idle_idx, false);
                    }
                    if !was_moving_active {
                        animations.set_active(moving_idx, true);
                    }
                } else {
                    if was_moving_active {
                        animations.set_active(moving_idx, false);
                    }
                    if !was_idle_active {
                        animations.set_active(idle_idx, true);
                    }
                }

                animations.set_speed(moving_idx, if self.is_moving { 1.0 } else { 0.0 });
            }

            _ => todo!(),
        }
    }
}
