//! Mirror of `Ares/source/models/model_definitions.h` (~325 lines).
//!
//! `hlmt` tag — Halo's per-object model package. References the
//! render_model, collision_model, animation_graph, physics_model;
//! holds variants/regions/permutations + LOD distances + damage
//! info + occlusion spheres + per-region shadow overrides.

/// `hlmt` tag identifier (h:15).
pub const MODEL_DEFINITION_TAG: u32 = u32::from_be_bytes(*b"hlmt");
pub const MODEL_DEFINITION_VERSION: i16 = 1;

/// `e_bounding_sphere_description` (h:19-25).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum BoundingSphereDescription {
    #[default]
    None = 0,
    User = 1,
    Autogen = 2,
}

pub const MAXIMUM_CATEGORIES_PER_VARIANT: usize = 16;
pub const MAXIMUM_VARIANTS_PER_MODEL: usize = 64;
pub const MAXIMUM_OBJECTS_PER_MODEL_VARIANT: usize = 16;

/// `e_model_flags` (h:34-43).
pub mod model_flags {
    pub const ACTIVE_CAMO_ALWAYS_ON: u32 = 1 << 0;
    pub const ACTIVE_CAMO_NEVER: u32 = 1 << 1;
    pub const HAS_SHIELD_IMPACT_EFFECT: u32 = 1 << 2;
    pub const USE_SKY_LIGHTING: u32 = 1 << 3;
    pub const IS_INCONSEQUENTIAL_TARGET: u32 = 1 << 4;
    pub const USE_AIRPROBE_LIGHTING: u32 = 1 << 5;
}

/// `e_prt_shadow_receive_mode` (h:51-57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PrtShadowReceiveMode {
    All = 0,
    SelfShadowOnly = 1,
    None = 2,
}

/// `e_model_target_lock_on_flags` (h:79-89).
pub mod model_target_flags {
    pub const LOCK_ON_HUMAN: u32 = 1 << 0;
    pub const LOCK_ON_PLASMA: u32 = 1 << 1;
    pub const HEADSHOT: u32 = 1 << 2;
    pub const VULNERABLE: u32 = 1 << 3;
    pub const LOCK_ON_PLASMA_ALWAYS: u32 = 1 << 4;
    pub const IGNORED_ON_LOCAL_PHYSICS: u32 = 1 << 5;
    pub const NETWORK_LEAD_VECTOR_ONLY: u32 = 1 << 6;
}

use glam::Vec3;

/// `s_model_level_of_detail` (h:99-107, 36B).
#[derive(Debug, Clone, Default)]
pub struct ModelLevelOfDetail {
    pub max_draw_distance: f32,
    pub begin_fade_distance: f32,
    pub animation_lod_distance: f32,
    pub cutoff_distance: f32,
    pub instance_max_draw_distance: f32,
    pub lod_render_model_path: String,
}

/// `s_model_bounding_sphere` (h:110-116, 20B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelBoundingSphere {
    pub description: u16,
    pub _pad: u16,
    pub offset: Vec3,
    pub radius: f32,
}

/// `s_model_object_data` (h:119-122, 20B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelObjectData {
    pub bounding_sphere: ModelBoundingSphere,
}

/// `s_prt_region_shadow_cast_override` (h:125-129, 8B).
#[derive(Debug, Clone, Copy, Default)]
pub struct PrtRegionShadowCastOverride {
    pub region_name: i32,
    pub shadow_cast_permutation_name: i32,
}

/// `s_prt_region_shadow_receive_override` (h:132-136, 8B).
#[derive(Debug, Clone, Copy, Default)]
pub struct PrtRegionShadowReceiveOverride {
    pub region_name: i32,
    pub shadow_receive_mode: i32,
}

/// `s_model_occlusion_sphere` (h:139-146, 20B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelOcclusionSphere {
    pub marker_one_name: i32,
    pub marker_one_index: i32,
    pub marker_two_name: i32,
    pub marker_two_index: i32,
    pub radius: f32,
}

/// `s_model_definition` (h:149-181, 392B). The hlmt tag.
#[derive(Debug, Clone, Default)]
pub struct ModelDefinition {
    pub render_model_path: String,
    pub collision_model_path: String,
    pub animation_graph_path: String,
    pub physics_model_path: String,
    pub level_of_detail: ModelLevelOfDetail,
    pub variants: Vec<ModelVariant>,
    pub instance_groups: Vec<u8>,
    pub materials: Vec<u8>,
    pub damage_info: Vec<u8>,
    pub targets: Vec<u8>,
    pub runtime_regions: Vec<u8>,
    pub runtime_nodes: Vec<u8>,
    pub runtime_node_list_checksum: u32,
    pub model_object_data: Vec<ModelObjectData>,
    pub default_dialogue_path: String,
    pub default_dialogue_female_path: String,
    pub flags: u32,
    pub default_dialogue_sound_effect_id: i32,
    /// `c_static_flags<256>` — 256-bit flag block for which nodes
    /// are render-only.
    pub render_only_node_flags: [u8; 32],
    pub sections_requiring_render_only_nodes: [u8; 32],
    pub private_flags: u32,
    pub scenario_load_parameters: Vec<u8>,
    pub prt_shadow_detail: u8,
    pub prt_shadow_bounces: u8,
    pub _pad: u16,
    pub prt_region_shadow_cast_overrides: Vec<PrtRegionShadowCastOverride>,
    pub prt_region_shadow_receive_overrides: Vec<PrtRegionShadowReceiveOverride>,
    pub model_occlusion_spheres: Vec<ModelOcclusionSphere>,
    pub shield_impact_parameter_override_path: String,
    pub shield_impact_parameter_override_first_person_path: String,
}

/// `s_model_variant` (h:184-192, 56B).
#[derive(Debug, Clone, Default)]
pub struct ModelVariant {
    pub name: i32,
    pub runtime_variant_region_indices: [i8; 16],
    pub regions: Vec<ModelVariantRegion>,
    pub objects: Vec<u8>,
    pub instance_group_index: i32,
    pub _pad: [i32; 2],
}

/// `s_model_variant_region` (h:195-204, 24B).
#[derive(Debug, Clone, Default)]
pub struct ModelVariantRegion {
    pub region_name: i32,
    pub runtime_region_index: i8,
    pub flags: u8,
    pub parent_variant_index: i16,
    pub permutations: Vec<ModelVariantPermutation>,
    pub sort_offset_enum: i16,
    pub _pad: i16,
}

/// `s_model_variant_permutation` (h:207-216, 36B).
#[derive(Debug, Clone, Default)]
pub struct ModelVariantPermutation {
    pub permutation_name: i32,
    pub runtime_permutation_index: i8,
    pub flags: u8,
    pub _pad: [u8; 2],
    pub probability: f32,
    pub states: Vec<ModelVariantState>,
    pub _pad2: [u8; 12],
}

/// `s_model_variant_state` (h:228-237, 32B). The version-2 variant
/// state with looping effect support; v1 (8B) is older Halo content.
#[derive(Debug, Clone, Default)]
pub struct ModelVariantState {
    pub permutation_name: i32,
    pub runtime_permutation_index: i8,
    pub state_property_flags: u8,
    pub state: u16,
    pub looping_effect_path: String,
    pub looping_effect_marker_name: i32,
    pub initial_probability: f32,
}
