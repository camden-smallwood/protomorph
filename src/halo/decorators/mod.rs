//! Mirror of `Ares/source/decorators/decorator_tag_definitions.h`
//! (142 lines).
//!
//! Halo's decorator system places GPU-instanced foliage clusters
//! (grass, plants, scattered rocks) across BSP cluster space. Each
//! `s_decorator_set` (the `dctr` tag) references one render_model
//! plus a texture + shader-flavor selection; the runtime instances
//! it across `s_decorator_runtime_block`s in cluster-aligned grids.
//!
//! Decorators render in the opaque pass with a dedicated shader
//! variant (one of 6 — `_render_shader_*`). The placements are
//! compressed (16B per placement: position_xyz + quaternion +
//! RGBE color) so 48,000 placements per cluster stays under
//! cache-line pressure.

use glam::{Vec3, Vec4};

pub mod bake_material_lookup;
pub mod decorator_tag_definitions;
pub mod light_placement;

pub const MAXIMUM_DECORATOR_SETS_PER_SCENARIO: usize = 48;
pub const MAXIMUM_CLUSTER_WIDTH_IN_BLOCKS: usize = 10;
pub const MAXIMUM_CLUSTER_DEPTH_IN_BLOCKS: usize = 10;
pub const MAXIMUM_CLUSTER_HEIGHT_IN_BLOCKS: usize = 10;
pub const MAXIMUM_CLUSTER_BLOCKS_PER_SET: usize = 1000;
pub const MAXIMUM_BLOCKS_PER_CLUSTER: usize = 48000;
pub const MAXIMUM_PLACEMENTS_PER_BLOCK: usize = 4096;

/// `s_decorator_set::e_render_flags` (h:32-37).
pub mod decorator_render_flags {
    pub const TWO_SIDED: u8 = 1 << 0;
    pub const DONT_SAMPLE_LIGHTING_THROUGH_GEOMETRY: u8 = 1 << 1;
}

/// `s_decorator_set::e_render_shader` (h:38-47). One of 6 dedicated
/// decorator shader variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DecoratorRenderShader {
    WindDynamicLights = 0,
    DynamicLights = 1,
    Static = 2,
    DominantLightOnly = 3,
    WavyDynamicLights = 4,
    ShadedDynamicLights = 5,
}

/// `s_decorator_set::e_lighting_sample_pattern` (h:48-53).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LightingSamplePattern {
    Ground = 0,
    Hanging = 1,
}

/// `s_decorator_set` (h:25-76, 128B). Tag-side decorator definition.
#[derive(Debug, Clone, Default)]
pub struct DecoratorSet {
    pub render_model_path: String,
    pub render_model_instance_names: Vec<String>,
    pub render_model_instance_name_valid_count: i32,
    pub texture_path: String,
    pub render_flags: u8,
    pub render_shader: u8,
    pub lighting_sample_pattern: u8,
    pub _unused: u8,
    pub translucency: Vec4,
    pub wave_flow: Vec4,
    pub contrast: Vec4,
    pub start_fade_distance: f32,
    pub end_fade_distance: f32,
    pub early_cull: f32,
    pub cull_block_size: f32,
    pub decorator_types: Vec<u8>,
}

/// `s_decorator_runtime_placement` (h:78-117, 16B). Compressed
/// per-placement record. Halo's quaternion is 4× signed bytes;
/// color is RGBE in 4 bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecoratorRuntimePlacement {
    pub position_x: u16,
    pub position_y: u16,
    pub position_z: u16,
    /// Halo unions `(motion_scale, subpart_index)` with `position_w`.
    /// Stored together as u16; high byte = subpart_index, low = motion_scale.
    pub position_w_or_pair: u16,
    /// Compressed quaternion: i8/i8/i8/i8 packed into u32.
    pub orientation: u32,
    /// RGBE color: r/g/b/exponent (or `ground_tint`) packed into u32.
    pub rgbe_color: u32,
}

impl DecoratorRuntimePlacement {
    pub fn motion_scale(&self) -> u8 {
        (self.position_w_or_pair & 0xFF) as u8
    }
    pub fn subpart_index(&self) -> u8 {
        ((self.position_w_or_pair >> 8) & 0xFF) as u8
    }

    /// Halo: `compress_quaternion_component` (h:114). Signed-byte
    /// component encoding `[-1, +1] → [-127, +127]`.
    pub fn compress_quaternion_component(value: f32) -> i8 {
        (value.clamp(-1.0, 1.0) * 127.0) as i8
    }

    /// Halo: `decompress_quaternion_component` (h:115).
    pub fn decompress_quaternion_component(byte: i8) -> f32 {
        byte as f32 / 127.0
    }
}

/// `s_decorator_runtime_block` (h:119-131, 60B). Per-block runtime
/// metadata — bounds + count + instance vertex buffer offset.
#[derive(Debug, Clone, Default)]
pub struct DecoratorRuntimeBlock {
    pub block_decorator_placement_count: u16,
    pub bsp_decorator_set_index: u8,
    pub bsp_instance_vertex_buffer_index: u8,
    pub instance_vertex_buffer_byte_offset: i32,
    pub position_bounds_0: Vec3,
    pub bounding_sphere_radius: f32,
    pub position_bounds_1: Vec3,
    pub bounding_sphere_center: Vec3,
    pub model_start_index: Vec<u32>,
}
