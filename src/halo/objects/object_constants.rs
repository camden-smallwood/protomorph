//! Mirror of `Ares/source/objects/object_constants.h` (render-
//! relevant subset).
//!
//! Halo's object system has 14 concrete types. The renderer needs
//! the type enum + the mask-bit table (used by visibility filters
//! and the per-pass entry-point dispatch — e.g. bipeds + giants
//! get the unit static-lighting path; scenery + crates get the
//! generic one).

/// `e_object_type` (object_constants.h:36-55).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum ObjectType {
    Biped = 0,
    Vehicle = 1,
    Weapon = 2,
    Equipment = 3,
    Terminal = 4,
    Projectile = 5,
    Scenery = 6,
    Machine = 7,
    Control = 8,
    SoundScenery = 9,
    Crate = 10,
    Creature = 11,
    Giant = 12,
    EffectScenery = 13,
}

impl ObjectType {
    pub const COUNT: usize = 14;
    pub const NONE: i8 = -1;

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            0 => Some(Self::Biped),
            1 => Some(Self::Vehicle),
            2 => Some(Self::Weapon),
            3 => Some(Self::Equipment),
            4 => Some(Self::Terminal),
            5 => Some(Self::Projectile),
            6 => Some(Self::Scenery),
            7 => Some(Self::Machine),
            8 => Some(Self::Control),
            9 => Some(Self::SoundScenery),
            10 => Some(Self::Crate),
            11 => Some(Self::Creature),
            12 => Some(Self::Giant),
            13 => Some(Self::EffectScenery),
            _ => None,
        }
    }
}

/// Object mask constants (object_constants.h:62-108). Used by
/// visibility/AI filters to enable subsets of types in one bitwise
/// test. Render-side cares about a few:
///   - `MASK_HAS_PERMUTATION` — types that have variant-driven mesh
///     permutation (bipeds/vehicles/weapons/equipment/scenery/crates).
///   - `MASK_PLACED_PER_BSP` — scenery (only scenery is BSP-anchored).
pub mod object_mask {
    pub const ALL: u32 = 0xFFFF_FFFF;
    pub const UNIT: u32 = 0x1003;
    pub const BIPED: u32 = 0x0001;
    pub const VEHICLE: u32 = 0x0002;
    pub const GIANT: u32 = 0x1000;
    pub const ITEM: u32 = 0x000C;
    pub const WEAPON: u32 = 0x0004;
    pub const EQUIPMENT: u32 = 0x0008;
    pub const PROJECTILE: u32 = 0x0020;
    pub const SCENERY: u32 = 0x0040;
    pub const CRATE: u32 = 0x0400;
    pub const CREATURE: u32 = 0x0800;
    pub const SOUND_SCENERY: u32 = 0x0200;
    pub const EFFECT_SCENERY: u32 = 0x2000;
    pub const DEVICE: u32 = 0x0190;
    pub const MACHINE: u32 = 0x0080;
    pub const TERMINAL: u32 = 0x0010;
    pub const CONTROL: u32 = 0x0100;
    pub const HAS_PERMUTATION: u32 = 0x1447;
    pub const PLACED_PER_BSP: u32 = 0x0040;
    pub const COMPUTES_OWN_NODE_MATRICES: u32 = 0x1080;
    pub const SIGHTBLOCKING: u32 = 0x14C2;
}

/// `e_object_source` (object_constants.h:110-119). Where this object
/// came from — affects render-side decisions like
/// `_object_source_sky` getting routed to sky rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum ObjectSource {
    StructureObject = 0,
    Editor = 1,
    Dynamic = 2,
    Legacy = 3,
    Sky = 4,
    Parent = 5,
}

impl ObjectSource {
    pub const NONE: i8 = -1;
}

/// Change-color reference enum (object_constants.h pre-line 33).
/// Used by the change_colors block to remap the four artist-facing
/// color slots. Render reads this in
/// `c_object_renderer::render_object_context_mesh_part` to drive
/// the rmsh `change_color_*` cbuffer slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum ChangeColorReference {
    Primary = 0,
    Secondary = 1,
    Tertiary = 2,
    QuaternaryObsolete = 3,
}

/// `c_object_identifier` (object_constants.h:138-166, 8B).
/// Unique handle assigned per object. Render code uses
/// `get_type` + `get_source` to dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ObjectIdentifier {
    pub unique_id: i32,         // 0x0
    pub origin_bsp_index: i16,  // 0x4
    pub object_type: i8,        // 0x6
    pub source: i8,             // 0x7
}

impl ObjectIdentifier {
    pub fn get_type(&self) -> Option<ObjectType> {
        ObjectType::from_i8(self.object_type)
    }
    pub fn is_scenario_object(&self) -> bool {
        matches!(
            self.source,
            x if x == ObjectSource::StructureObject as i8 || x == ObjectSource::Editor as i8 || x == ObjectSource::Legacy as i8
        )
    }
}

/// `object_header_block_reference` (object_constants.h:168-173, 4B).
/// Halo's per-object dynamic-data slots (node matrices, region info,
/// attachments, etc.) live in a heap allocator + this struct points
/// into it via `(size, offset)`. The renderer accesses them through
/// `c_objects::get_node_matrices(object_index)` etc., not by reading
/// these directly — the references just exist for the lookup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectHeaderBlockReference {
    pub size: i16,
    pub offset: i16,
}
