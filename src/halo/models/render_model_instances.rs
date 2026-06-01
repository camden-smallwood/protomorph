//! Mirror of `Ares/source/models/render_model_instances.h` (56 lines).
//!
//! Instance groups within a render_model — Halo's mechanism for
//! placing repeated sub-meshes (e.g. computers in a UNSC base, racks
//! of weapons) at fixed positions within a model. Resolved at
//! variant-selection time.

pub const MAXIMUM_INSTANCE_PLACEMENTS_PER_RENDER_MODEL: usize = 64;
pub const RENDER_MODEL_INSTANCE_BITFIELD_BIT_COUNT: usize = 64;
pub const RENDER_MODEL_INSTANCE_BITFIELD_DWORD_COUNT: usize = 2;
pub const MAXIMUM_INSTANCE_GROUPS_PER_MODEL: usize = 256;
pub const MAXIMUM_MEMBERS_PER_INSTANCE_GROUP: usize = 32;

/// `c_model_instance_group_member` (h:22-29, 20B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelInstanceGroupMember {
    pub subgroup_block_index: i32,
    pub instance_name_reference: i32,
    pub probability: f32,
    /// `unsigned long instance_placement_mask[2]` — 64-bit
    /// placement mask (one bit per placement slot).
    pub instance_placement_mask: [u32; 2],
}

/// `c_model_instance_group::e_choice` (h:35-39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelInstanceGroupChoice {
    ChooseOneMember = 0,
    ChooseAllMembers = 1,
}

/// `c_model_instance_group` (h:32-44, 24B).
#[derive(Debug, Clone, Default)]
pub struct ModelInstanceGroup {
    pub group_name: i32,
    pub choice: i32,
    pub member_list: Vec<ModelInstanceGroupMember>,
    pub total_probability: f32,
}
