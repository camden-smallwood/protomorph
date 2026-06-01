//! Mirror of `Ares/source/models/model_constants.h` (84 lines).
//!
//! Per-model fixed-array caps used by the model loader + animation
//! subsystem.

/// `e_damage_part` (h:9-23). Halo's body damage segmentation —
/// each region in a humanoid biped is mapped to one of these for
/// ragdoll + dismemberment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DamagePart {
    Gut = 0,
    Chest = 1,
    Head = 2,
    LeftShoulder = 3,
    LeftArm = 4,
    LeftLeg = 5,
    LeftFoot = 6,
    RightShoulder = 7,
    RightArm = 8,
    RightLeg = 9,
    RightFoot = 10,
}

pub const NUMBER_OF_DAMAGE_PARTS: usize = 11;

pub const MODEL_ROOT_NODE_INDEX: usize = 0;
pub const MAXIMUM_NODES_PER_MODEL: usize = 255;
pub const MAXIMUM_NODES_PER_FIRST_PERSON_MODEL: usize = 64;
pub const MAXIMUM_NODES_UPLOADED_SIMULTANEOUSLY: usize = 70;
pub const MAXIMUM_NODES_PER_RENDER_MODEL_MESH: usize = 70;
pub const MAXIMUM_NODES_PER_MESH_FOR_PRT: usize = 2;
pub const MAXIMUM_REGIONS_PER_MODEL: usize = 16;
pub const MAXIMUM_PERMUTATIONS_PER_MODEL_REGION: usize = 32;
pub const MAXIMUM_STATES_PER_MODEL_PERMUTATION: usize = 10;
pub const MAXIMUM_MATERIALS_PER_MODEL: usize = 32;
pub const MAXIMUM_DAMAGE_SECTIONS_PER_MODEL: usize = 16;
pub const MAXIMUM_RESPONSES_PER_DAMAGE_SECTION: usize = 16;
pub const MAXIMUM_DAMAGE_SEAT_INFOS_PER_MODEL: usize = 16;
pub const MAXIMUM_DAMAGE_CONSTRAINT_INFOS_PER_MODEL: usize = 16;
pub const MAXIMUM_MODEL_TARGETS_PER_MODEL: usize = 32;

/// `e_model_state` (h:44-53).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelState {
    Default = 0,
    MinorDamage = 1,
    MediumDamage = 2,
    MajorDamage = 3,
    Destroyed = 4,
}

pub const NUMBER_OF_MODEL_STATES: usize = 5;
pub const MAXIMUM_NUMBER_OF_MODEL_STATES: usize = 12;

/// `e_model_state_property` (h:55-62).
pub mod model_state_property_flags {
    pub const BLURRED: u32 = 1 << 0;
    pub const VERY_BLURRED: u32 = 1 << 1;
    pub const SHIELD_INACTIVE: u32 = 1 << 2;
    pub const BATTERY_DEPLETED: u32 = 1 << 3;
}

pub const MAXIMUM_SECTIONS_PER_RENDER_MODEL: usize = 255;
pub const MAXIMUM_DSQ_VERTICES_PER_RENDER_MODEL_SECTION: usize = 65_536;
pub const MAXIMUM_DSQ_STRIP_INDICES_PER_RENDER_MODEL_SECTION: usize = 262_144;
pub const MAXIMUM_DSQ_SILHOUETTE_QUADS_PER_RENDER_MODEL_SECTION: usize = 65_536;
pub const MAXIMUM_MARKER_GROUPS_PER_RENDER_MODEL: usize = 4096;
pub const MAXIMUM_MARKERS_PER_RENDER_MODEL_MARKER_GROUP: usize = 256;
