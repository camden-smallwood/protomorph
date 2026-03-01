use bytemuck::{Pod, Zeroable};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Light types
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
}

// ---------------------------------------------------------------------------
// Light data (port of C's light_data)
// ---------------------------------------------------------------------------

pub struct LightData {
    pub light_type: LightType,
    pub hidden: bool,
    pub casts_shadow: bool,
    pub position: Vec3,
    pub direction: Vec3,
    pub diffuse_color: Vec3,
    pub ambient_color: Vec3,
    pub specular_color: Vec3,
    pub constant_atten: f32,
    pub linear_atten: f32,
    pub quadratic_atten: f32,
    pub inner_cutoff: f32, // degrees
    pub outer_cutoff: f32, // degrees
}

impl LightData {
    pub fn new_point() -> Self {
        Self {
            light_type: LightType::Point,
            hidden: false,
            casts_shadow: false,
            position: Vec3::ZERO,
            direction: Vec3::ZERO,
            diffuse_color: Vec3::ONE,
            ambient_color: Vec3::splat(0.05),
            specular_color: Vec3::ONE,
            constant_atten: 1.0,
            linear_atten: 0.009,
            quadratic_atten: 0.0032,
            inner_cutoff: 0.0,
            outer_cutoff: 0.0,
        }
    }

    pub fn new_spot() -> Self {
        Self {
            light_type: LightType::Spot,
            hidden: false,
            casts_shadow: false,
            position: Vec3::ZERO,
            direction: Vec3::ZERO,
            diffuse_color: Vec3::ONE,
            ambient_color: Vec3::splat(0.05),
            specular_color: Vec3::ONE,
            constant_atten: 1.0,
            linear_atten: 0.1,
            quadratic_atten: 0.32,
            inner_cutoff: 12.5,
            outer_cutoff: 17.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Light store (mirrors ObjectStore pattern)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct LightIndex(pub usize);

pub struct LightStore {
    lights: Vec<Option<LightData>>,
}

impl LightStore {
    pub fn new() -> Self {
        Self {
            lights: Vec::new(),
        }
    }

    pub fn new_light(&mut self, light: LightData) -> LightIndex {
        for (i, slot) in self.lights.iter().enumerate() {
            if slot.is_none() {
                self.lights[i] = Some(light);
                return LightIndex(i);
            }
        }

        let index = self.lights.len();
        self.lights.push(Some(light));
        LightIndex(index)
    }

    pub fn get(&self, index: LightIndex) -> &LightData {
        self.lights[index.0]
            .as_ref()
            .expect("light slot is empty")
    }

    pub fn get_mut(&mut self, index: LightIndex) -> &mut LightData {
        self.lights[index.0]
            .as_mut()
            .expect("light slot is empty")
    }

    pub fn iter(&self) -> impl Iterator<Item = (LightIndex, &LightData)> {
        self.lights
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|data| (LightIndex(i), data)))
    }
}

// ---------------------------------------------------------------------------
// GPU uniform structs (must match WGSL alignment)
// ---------------------------------------------------------------------------

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuLightData {
    pub position: [f32; 3],
    pub light_type: u32,
    pub direction: [f32; 3],
    pub constant_atten: f32,
    pub diffuse_color: [f32; 3],
    pub linear_atten: f32,
    pub ambient_color: [f32; 3],
    pub quadratic_atten: f32,
    pub specular_color: [f32; 3],
    pub inner_cutoff: f32,
    pub outer_cutoff: f32,
    pub shadow_index: i32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuLightingUniforms {
    pub camera_position: [f32; 3],
    pub light_count: u32,
    pub camera_direction: [f32; 3],
    pub _pad: f32,
    pub lights: [GpuLightData; MAX_LIGHTS],
}

impl GpuLightingUniforms {
    pub fn from_scene(
        camera_position: Vec3,
        camera_direction: Vec3,
        store: &LightStore,
        shadow_assignments: &[(usize, i32)],
    ) -> Self {
        let mut uniforms = Self::zeroed();
        uniforms.camera_position = camera_position.into();
        uniforms.camera_direction = camera_direction.into();

        let mut count = 0u32;

        for (_idx, light) in store.iter() {
            if light.hidden {
                continue;
            }

            if count as usize >= MAX_LIGHTS {
                break;
            }

            let gpu_slot = count as usize;

            let shadow_index = shadow_assignments
                .iter()
                .find(|(slot, _)| *slot == gpu_slot)
                .map(|(_, si)| *si)
                .unwrap_or(-1);

            uniforms.lights[gpu_slot] = GpuLightData {
                position: light.position.into(),
                light_type: light.light_type as u32,
                direction: light.direction.into(),
                constant_atten: light.constant_atten,
                diffuse_color: light.diffuse_color.into(),
                linear_atten: light.linear_atten,
                ambient_color: light.ambient_color.into(),
                quadratic_atten: light.quadratic_atten,
                specular_color: light.specular_color.into(),
                inner_cutoff: light.inner_cutoff.to_radians().cos(),
                outer_cutoff: light.outer_cutoff.to_radians().cos(),
                shadow_index,
                _pad: [0.0; 2],
            };

            count += 1;
        }
        
        uniforms.light_count = count;
        uniforms
    }
}
