//! Mirror of `Ares/source/models/render_model_definitions.h`
//! (140 lines).
//!
//! Tag-side `render_model_definition` (`mode` tag, version 5,
//! 460B) — Halo's per-character/scenery skeletal mesh package.
//! Holds:
//!   - regions/permutations selection (variant-driven)
//!   - skeleton nodes (parent + first_child + next_sibling tree)
//!   - marker groups for attachment points + IK
//!   - materials list
//!   - embedded `s_render_geometry`
//!   - per-model default lightprobe SH coefficients (16 each for R/G/B)
//!   - lightgen_lights (sky-gen light directions for rare characters)
//!   - volume_samples (PRT volume probes for high-quality static
//!     lighting on hero characters)

use crate::halo::geometry::render_geometry::RenderGeometry;
use glam::{Mat4, Quat, Vec3};

/// `mode` tag identifier (h:17). Note: 4 chars `'mode'`, NOT
/// `'rmod'` — Bungie's tag IDs read backward.
pub const RENDER_MODEL_DEFINITION_TAG: u32 = u32::from_be_bytes(*b"mode");
pub const RENDER_MODEL_DEFINITION_VERSION: i16 = 5;

/// `e_render_model_flags` (h:21-27).
pub mod render_model_flags {
    pub const HAS_NODE_MAPS: u16 = 1 << 2;
}

/// `s_sky_gen_light` (h:31-36, 28B). Sky-generated directional
/// light driving SH probe generation for the model.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkyGenLight {
    pub direction: Vec3,
    pub intensity: Vec3,
    pub solid_angle: f32,
}

/// `s_volume_sample` (h:39-43, 336B). PRT volume-sample point. The
/// 81-element `radiance_transfer_matrix` is a 9×9 SH transfer for
/// quadratic PRT.
#[derive(Debug, Clone, Copy)]
pub struct VolumeSample {
    pub sample_position: Vec3,
    pub radiance_transfer_matrix: [f32; 81],
}

impl Default for VolumeSample {
    fn default() -> Self {
        Self {
            sample_position: Vec3::ZERO,
            radiance_transfer_matrix: [0.0; 81],
        }
    }
}

/// `render_model_region` (h:73-77, 16B).
#[derive(Debug, Clone, Default)]
pub struct RenderModelRegion {
    pub name: i32,
    pub permutations: Vec<RenderModelPermutation>,
}

/// `render_model_permutation` (h:80-86, 16B).
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderModelPermutation {
    pub name: i32,
    pub mesh_start_index: i16,
    pub mesh_count: u16,
    /// 64-bit instance placement bitmask.
    pub instance_bitmask: [u32; 2],
}

/// `render_model_node` (h:89-100, 96B). One bone in the skeleton.
#[derive(Debug, Clone, Copy)]
pub struct RenderModelNode {
    pub name: i32,
    pub parent_node_index: i16,
    pub first_child_node_index: i16,
    pub next_sibling_node_index: i16,
    pub _pad: u16,
    pub default_translation: Vec3,
    pub default_rotation: Quat,
    pub default_inverse_matrix: Mat4, // real_matrix4x3 in Halo
    pub distance_from_parent: f32,
}

impl Default for RenderModelNode {
    fn default() -> Self {
        Self {
            name: 0,
            parent_node_index: -1,
            first_child_node_index: -1,
            next_sibling_node_index: -1,
            _pad: 0,
            default_translation: Vec3::ZERO,
            default_rotation: Quat::IDENTITY,
            default_inverse_matrix: Mat4::IDENTITY,
            distance_from_parent: 0.0,
        }
    }
}

/// `render_model_marker` (h:103-112, 36B). Attachment point on the
/// model. Used for FX positioning + IK + first-person grip.
#[derive(Debug, Clone, Copy)]
pub struct RenderModelMarker {
    pub region_index: i8,
    pub permutation_index: i8,
    pub node_index: u8,
    pub _pad: u8,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for RenderModelMarker {
    fn default() -> Self {
        Self {
            region_index: -1,
            permutation_index: -1,
            node_index: 0,
            _pad: 0,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

/// `render_model_marker_group` (h:115-119, 16B).
#[derive(Debug, Clone, Default)]
pub struct RenderModelMarkerGroup {
    pub name: i32,
    pub markers: Vec<RenderModelMarker>,
}

/// `render_model_definition` (h:46-70, 460B). The `mode` tag.
#[derive(Debug, Clone, Default)]
pub struct RenderModelDefinition {
    pub name: i32,
    pub flags: u16,
    pub _pad1: u16,
    pub runtime_import_info_checksum: i32,
    pub regions: Vec<RenderModelRegion>,
    pub section_group_indices: [i8; 2],
    pub _pad2: u16,
    pub instance_mesh_index: i32,
    pub instance_placements: Vec<u8>,
    pub node_list_checksum: u32,
    pub nodes: Vec<RenderModelNode>,
    pub marker_groups: Vec<RenderModelMarkerGroup>,
    pub materials: Vec<u8>,
    pub errors: Vec<u8>,
    pub dont_draw_over_cosine_angle: f32,
    pub render_geometry: RenderGeometry,
    pub lightgen_lights: Vec<SkyGenLight>,
    /// Per-model default SH probe (16 coefficients × R/G/B).
    pub default_lightprobe_r: [f32; 16],
    pub default_lightprobe_g: [f32; 16],
    pub default_lightprobe_b: [f32; 16],
    pub volume_samples: Vec<VolumeSample>,
    pub default_node_orientations: Vec<u8>,
}
