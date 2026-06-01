//! `object_datum` runtime state — engine `object_datum` (352 bytes,
//! `objects.h:420-425`) plus the per-type extension layers
//! (`unit_datum`, `weapon_datum`, `biped_datum`, etc.).
//!
//! ## Engine layout (Ares headers)
//!
//! Ares mirrors the C++ class hierarchy via flat composition — each
//! derived type lists every base body in order:
//! ```text
//!   object_datum   : definition_index, _object_datum object
//!   item_datum     : definition_index, _object_datum object, _item_datum item
//!   weapon_datum   : definition_index, _object_datum object, _item_datum item, _weapon_datum weapon
//!   scenery_datum  : definition_index, _object_datum object, _scenery_datum scenery
//!   biped_datum    : definition_index, _object_datum object, _mover_datum mover, _unit_datum unit, _biped_datum biped
//!   ... etc
//! ```
//!
//! ## Protomorph layout (Rust-idiomatic)
//!
//! Composition by layer body, leaf-type variants encoded via a sum
//! type. Avoids the 18-distinct-struct explosion that Ares needs in
//! C, while keeping every engine field name reachable through a
//! single accessor (`datum.object.body_vitality`,
//! `datum.as_weapon().map(|w| w.heat)`).
//!
//! ```text
//!   ObjectDatum {
//!     definition_index: i32,                  // Ares: definition_index
//!     object: ObjectBody,                     // Ares: _object_datum
//!     layers: TypeLayers,                     // sum of inheritance chain bodies
//!   }
//!
//!   TypeLayers::Weapon(ItemBody, WeaponBody)  // matches weapon_datum
//!   TypeLayers::Biped(MoverBody, UnitBody, BipedBody)
//!   TypeLayers::Scenery(SceneryBody)
//!   TypeLayers::SoundScenery                  // pure object (no leaf body)
//!   ... 14 leaf variants total
//! ```
//!
//! Per-type-layer body structs (`ItemBody`, `UnitBody`, `WeaponBody`,
//! …) start as empty placeholders. Fields land as their respective
//! `<type>_compute_function_value` cases get ported. Initial focus is
//! `ObjectBody` since the chain dispatcher routes through
//! `object_compute_function_value` first on every type.

use crate::halo::objects::object_type::ObjectType;
use blam_tags::math::{RealPoint3d, RealVector3d};
use blam_tags::scenario::ObjectPlacement;

// ---------------------------------------------------------------------------
// Helper substructs (small, used inline)
// ---------------------------------------------------------------------------

/// Engine `c_object_identifier` (8 bytes). Identifies the spawn — the
/// `m_type` byte gates several compute-function cases (`shield_depleted`,
/// `phantom_on`, `jetpack_exhaust`, the teleporter cases).
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectIdentifier {
    pub unique_id: i32,
    pub origin_bsp_index: i16,
    /// `e_object_type` discriminant — matches [`ObjectType`].
    pub m_type: i8,
    pub m_source: i8,
}

/// Engine `s_location` (4 bytes). The placement's BSP cluster + leaf.
#[derive(Debug, Clone, Copy, Default)]
pub struct Location {
    pub cluster_reference: u16,
    pub leaf_index: u16,
}

/// Engine `s_damage_owner` (12 bytes). Tracks who caused damage to
/// this object — relevant once damage simulation lands.
#[derive(Debug, Clone, Copy, Default)]
pub struct DamageOwner {
    pub damage_owner_player_index: i32,
    pub damage_owner_object_index: i32,
    pub damage_owner_team_index: i8,
    pub pad: [u8; 3],
}

/// Engine `object_header_block_reference` (4 bytes). Index+size into
/// the per-object block heap (animation, attachments, node matrices,
/// etc.). Static placements default to (0, 0); the heap entries get
/// allocated when the engine adds a runtime feature.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectHeaderBlockReference {
    pub size: i16,
    pub offset: i16,
}

// ---------------------------------------------------------------------------
// ObjectBody — engine `_object_datum` (348 bytes)
// ---------------------------------------------------------------------------

/// Engine `_object_datum` (`objects.h:325-411`, 348 bytes). Every
/// object's base state — read by `object_compute_function_value` and
/// (via inheritance) by every derived `*_compute_function_value`.
///
/// Field order + names mirror Ares verbatim. Bitflag fields stay as
/// raw `u32` / `u8` until typed wrappers are needed.
#[derive(Debug, Clone)]
pub struct ObjectBody {
    /// `c_object_data_flags flags` (32 bits).
    pub flags: u32,
    pub next_object_index: i32,
    pub first_child_object_index: i32,
    pub parent_object_index: i32,
    pub parent_node_index: i8,
    /// `c_object_inhibited_flags inhibited_flags` (4-bit field, u8 storage).
    pub inhibited_flags: u8,
    pub scenario_datum_index: i16,
    pub map_variant_index: i16,
    pub pad0: i16,
    pub location: Location,
    pub bounding_sphere_center: RealPoint3d,
    pub bounding_sphere_radius: f32,
    pub attached_bounds_center: RealPoint3d,
    pub attached_bounds_radius: f32,
    pub dynamic_light_bounding_sphere_center: RealPoint3d,
    pub dynamic_light_bounding_sphere_radius: f32,
    pub first_cluster_reference_index: i32,
    pub relative_position: RealPoint3d,
    pub forward: RealVector3d,
    pub up: RealVector3d,
    pub translational_velocity: RealVector3d,
    pub angular_velocity: RealVector3d,
    pub scale: f32,
    pub object_identifier: ObjectIdentifier,
    pub name_index: i16,
    /// `c_object_bsp_placement_policy bsp_placement_policy` (i8 enum).
    pub bsp_placement_policy: i8,
    pub keyframed_object_collision_damage_ticks: u8,
    pub havok_component_index: i32,
    pub local_physics_space_object_index: i32,
    pub last_motion_time: i32,
    /// `c_object_physics_flags physics_flags` (24-bit field, u32 storage).
    /// Bit `0x20000` = phantom-on (engine `object_compute_function_value`
    /// case `phantom_on`).
    pub physics_flags: u32,
    pub high_frequency_physics_deactivation_count: u8,
    pub low_frequency_physics_deactivation_count: u8,
    /// `object_compute_function_value` case `phantom_power`.
    pub phantom_power_scale: u8,
    /// `object_compute_function_value` case `phantom_speed`.
    pub phantom_speed_scale: u8,
    pub in_water_ticks: i16,
    pub inertia_tensor_scale: u8,
    pub collision_damage_armor_scale: u8,
    pub environment_interpenetration_ticks: i16,
    /// `object_compute_function_value` case `variant`.
    pub variant_index: i8,
    /// `c_object_ai_flags ai_flags` (3-bit field, u8 storage).
    pub ai_flags: u8,
    pub melee_damage_unique_identifier: u32,
    pub damage_owner: DamageOwner,
    pub structure_bsp_fake_lightprobe_block_index: i16,
    pub gamestate_index: i32,
    pub owner_team_index: i16,
    /// `c_object_simulation_flags simulation_flags` (4-bit field, u8 storage).
    pub simulation_flags: u8,
    pub child_variant_index: i8,
    pub simulation_last_position_time: i32,
    pub simulation_last_position: RealPoint3d,
    pub first_widget_index: i32,
    pub destroyed_constraints: u16,
    pub loosened_constraints: u16,
    pub maximum_body_vitality: f32,
    /// `object_compute_function_value` case `shield_depleted`
    /// (gate: `maximum_shield_vitality > 0`).
    pub maximum_shield_vitality: f32,
    /// `object_compute_function_value` case `body_vitality`.
    pub body_vitality: f32,
    /// `object_compute_function_value` cases `shield_vitality` and
    /// `object_overshield_amount` (the latter = `shield_vitality - 1`).
    pub shield_vitality: f32,
    /// `object_compute_function_value` case `current_shield_damage`.
    pub current_shield_damage: f32,
    /// `object_compute_function_value` case `current_body_damage`.
    pub current_body_damage: f32,
    pub recent_shield_damage: f32,
    pub recent_body_damage: f32,
    pub shield_stun_ticks: i16,
    pub body_stun_ticks: i16,
    /// `c_object_damage_flags damage_flags` (17-bit field, u32 storage).
    /// Bit `0x04` = killed (`object_compute_function_value` case `alive`
    /// returns 1.0 when this bit is CLEAR).
    pub damage_flags: u32,
    pub shield_damage_decay_timer: i8,
    pub body_damage_decay_timer: i8,
    /// `c_object_recycling_flags recycling_flags` (6-bit field).
    pub recycling_flags: u8,
    pub next_recycling_object_index: i32,
    pub recycling_time: i32,
    pub parent_recycling_group: i32,
    pub next_recycling_group_member: i32,
    /// `c_static_flags<16> types_created` — 16-bit static flags (stored
    /// as u32 with 3-byte padding per Ares layout).
    pub types_created: u32,
    pub unfortunate_padding: [u8; 1],
    pub original_node_orientations: ObjectHeaderBlockReference,
    pub node_orientations: ObjectHeaderBlockReference,
    pub node_matrices: ObjectHeaderBlockReference,
    pub region_information: ObjectHeaderBlockReference,
    pub attachments: ObjectHeaderBlockReference,
    pub damage_sections: ObjectHeaderBlockReference,
    pub change_colors: ObjectHeaderBlockReference,
    pub animation: ObjectHeaderBlockReference,
    pub multiplayer: ObjectHeaderBlockReference,
    pub air_probe_index: i32,
    pub stock_probe_index: i32,
}

impl Default for ObjectBody {
    /// Engine-faithful "freshly-spawned, untouched, full-health" defaults.
    /// Matches the state produced by `c_object::initialize` for a newly
    /// instantiated static placement with no damage simulation yet.
    fn default() -> Self {
        Self {
            flags: 0,
            next_object_index: -1,
            first_child_object_index: -1,
            parent_object_index: -1,
            parent_node_index: -1,
            inhibited_flags: 0,
            scenario_datum_index: -1,
            map_variant_index: -1,
            pad0: 0,
            location: Location::default(),
            bounding_sphere_center: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            bounding_sphere_radius: 0.0,
            attached_bounds_center: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            attached_bounds_radius: 0.0,
            dynamic_light_bounding_sphere_center: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            dynamic_light_bounding_sphere_radius: 0.0,
            first_cluster_reference_index: -1,
            relative_position: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            forward: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
            up: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
            translational_velocity: RealVector3d { i: 0.0, j: 0.0, k: 0.0 },
            angular_velocity: RealVector3d { i: 0.0, j: 0.0, k: 0.0 },
            scale: 1.0,
            object_identifier: ObjectIdentifier::default(),
            name_index: -1,
            bsp_placement_policy: 0,
            keyframed_object_collision_damage_ticks: 0,
            havok_component_index: -1,
            local_physics_space_object_index: -1,
            last_motion_time: 0,
            physics_flags: 0,
            high_frequency_physics_deactivation_count: 0,
            low_frequency_physics_deactivation_count: 0,
            phantom_power_scale: 0,
            phantom_speed_scale: 0,
            in_water_ticks: 0,
            inertia_tensor_scale: 0,
            collision_damage_armor_scale: 0,
            environment_interpenetration_ticks: 0,
            variant_index: 0,
            ai_flags: 0,
            melee_damage_unique_identifier: 0,
            damage_owner: DamageOwner::default(),
            structure_bsp_fake_lightprobe_block_index: -1,
            gamestate_index: -1,
            owner_team_index: -1,
            simulation_flags: 0,
            child_variant_index: 0,
            simulation_last_position_time: 0,
            simulation_last_position: RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
            first_widget_index: -1,
            destroyed_constraints: 0,
            loosened_constraints: 0,
            // Engine `c_object::initialize` copies `maximum_body_vitality`
            // from `object_definition`. Until that plumbing lands,
            // default to 1.0 so `body_vitality / max = 1.0` makes sense.
            maximum_body_vitality: 1.0,
            // 0.0 = no shields (most static placements). Triggers the
            // `shield_depleted` short-circuit correctly.
            maximum_shield_vitality: 0.0,
            // Full-health = 1.0 (engine compute_function_value case 496).
            body_vitality: 1.0,
            shield_vitality: 1.0,
            current_shield_damage: 0.0,
            current_body_damage: 0.0,
            recent_shield_damage: 0.0,
            recent_body_damage: 0.0,
            shield_stun_ticks: 0,
            body_stun_ticks: 0,
            // Killed bit (0x04) CLEAR → `alive` case returns 1.0.
            damage_flags: 0,
            shield_damage_decay_timer: 0,
            body_damage_decay_timer: 0,
            recycling_flags: 0,
            next_recycling_object_index: -1,
            recycling_time: 0,
            parent_recycling_group: -1,
            next_recycling_group_member: -1,
            types_created: 0,
            unfortunate_padding: [0; 1],
            original_node_orientations: ObjectHeaderBlockReference::default(),
            node_orientations: ObjectHeaderBlockReference::default(),
            node_matrices: ObjectHeaderBlockReference::default(),
            region_information: ObjectHeaderBlockReference::default(),
            attachments: ObjectHeaderBlockReference::default(),
            damage_sections: ObjectHeaderBlockReference::default(),
            change_colors: ObjectHeaderBlockReference::default(),
            animation: ObjectHeaderBlockReference::default(),
            multiplayer: ObjectHeaderBlockReference::default(),
            air_probe_index: -1,
            stock_probe_index: -1,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-type body re-exports
// ---------------------------------------------------------------------------
//
// Each layer's body lives in the protomorph file that mirrors its
// Ares header location. Re-exported here so `TypeLayers` variants can
// reference them by short name. Empty placeholders remain in this
// file for layers whose bodies haven't been surfaced yet — fill them
// in (or move to their proper Ares-mapped file) as their
// `*_compute_function_value` gets ported.
//
// Engine struct → protomorph file mapping:
//   _item_datum       → `src/halo/items/items.rs`             (Ares items.h:34)
//   _weapon_datum     → `src/halo/items/weapons.rs`           (Ares weapons.h:216)
//   _equipment_datum  → `src/halo/items/equipment.rs`         (Ares equipment.h:18)
//   _projectile_datum → `src/halo/items/projectiles.rs`       (Ares projectiles.h:64)
//   _unit_datum       → `src/halo/units/units.rs`             (Ares units.h:179)
//   _mover_datum      → `src/halo/units/units.rs`             (Ares embedded)
//   _biped_datum      → `src/halo/units/bipeds.rs`            (Ares bipeds.h:150)
//   _vehicle_datum    → `src/halo/units/vehicles.rs`          (Ares vehicles.h:63)
//   _giant_datum      → `src/halo/units/giants.rs`            (Ares giants.h:110)
//   _device_datum     → `src/halo/devices/devices.rs`         (Ares devices.h:108)
//   _machine_datum    → `src/halo/devices/device_machines.rs` (Ares devices.h:134)
//   _terminal_datum   → `src/halo/devices/device_terminals.rs`(Ares devices.h:154)
//   _control_datum    → `src/halo/devices/`                   (Ares devices.h:197)
//   _scenery_datum    → `src/halo/objects/`                   (Ares scenery.h:71)
//   _crate_datum      → `src/halo/objects/crates.rs`          (Ares crates.h:24)
//   _creature_datum   → `src/halo/creatures/creatures.rs`     (Ares creatures.h:69)

pub use crate::halo::creatures::creatures::CreatureBody;
pub use crate::halo::devices::device_controls::ControlBody;
pub use crate::halo::devices::device_machines::MachineBody;
pub use crate::halo::devices::device_terminals::TerminalBody;
pub use crate::halo::devices::devices::DeviceBody;
pub use crate::halo::items::equipment::EquipmentBody;
pub use crate::halo::items::items::ItemBody;
pub use crate::halo::items::projectiles::ProjectileBody;
pub use crate::halo::items::weapons::WeaponBody;
pub use crate::halo::units::bipeds::BipedBody;
pub use crate::halo::units::units::UnitBody;
pub use crate::halo::units::vehicles::VehicleBody;

#[derive(Debug, Clone, Default)] pub struct MoverBody {}
#[derive(Debug, Clone, Default)] pub struct GiantBody {}
#[derive(Debug, Clone, Default)] pub struct SceneryBody {}
#[derive(Debug, Clone, Default)] pub struct CrateBody {}

// ---------------------------------------------------------------------------
// TypeLayers — inheritance-chain sum type
// ---------------------------------------------------------------------------

/// The per-type layers that follow `ObjectBody` in the engine's
/// composite layouts (Ares `<type>_datum` structs). Each variant
/// captures the additional bodies a leaf type adds beyond
/// [`ObjectBody`] — order matches Ares field order so a code reader
/// can walk variant fields top-to-bottom against the engine layout.
///
/// Mirrors the engine `part_definitions[]` chains decoded into
/// [`reference_object_type_compute_function_chains`] — the variants
/// here correspond 1:1 to the rows in that table.
#[derive(Debug, Clone)]
pub enum TypeLayers {
    /// Abstract `object` (`obje`) — no leaf body. Engine rarely
    /// instantiates this directly; here for completeness.
    Object,
    /// `scenery` — `_object_datum` + `_scenery_datum`.
    Scenery(SceneryBody),
    /// `crate` (`bloc`) — `_object_datum` + `_crate_datum`.
    Crate(CrateBody),
    /// `sound_scenery` — pure `_object_datum`.
    SoundScenery,
    /// `effect_scenery` — pure `_object_datum`.
    EffectScenery,
    /// `projectile` — `_object_datum` + `_projectile_datum`.
    Projectile(ProjectileBody),
    /// `creature` — `_object_datum` + `_creature_datum`. Engine
    /// `part_definitions[]` also includes `motor` between but its
    /// compute pointer is NULL.
    Creature(CreatureBody),
    /// `equipment` — `_object_datum` + `_item_datum` + `_equipment_datum`.
    Equipment(ItemBody, EquipmentBody),
    /// `weapon` — `_object_datum` + `_item_datum` + `_weapon_datum`.
    Weapon(ItemBody, WeaponBody),
    /// `device_machine` — `_object_datum` + `_device_datum` + `_machine_datum`.
    Machine(DeviceBody, MachineBody),
    /// `device_control` — `_object_datum` + `_device_datum` + `_control_datum`.
    Control(DeviceBody, ControlBody),
    /// `device_terminal` — `_object_datum` + `_device_datum` + `_terminal_datum`.
    Terminal(DeviceBody, TerminalBody),
    /// `biped` — `_object_datum` + `_mover_datum` + `_unit_datum` + `_biped_datum`.
    Biped(MoverBody, UnitBody, BipedBody),
    /// `vehicle` — `_object_datum` + `_mover_datum` + `_unit_datum` + `_vehicle_datum`.
    Vehicle(MoverBody, UnitBody, VehicleBody),
    /// `giant` — `_object_datum` + `_mover_datum` + `_unit_datum` + `_giant_datum`.
    Giant(MoverBody, UnitBody, GiantBody),
}

impl TypeLayers {
    /// Instantiate the default-state layers for `object_type`. Used by
    /// scenario load to populate `ObjectDatum::layers` for each placement.
    pub fn default_for(object_type: ObjectType) -> Self {
        match object_type {
            ObjectType::Biped         => Self::Biped(MoverBody::default(), UnitBody::default(), BipedBody::default()),
            ObjectType::Vehicle       => Self::Vehicle(MoverBody::default(), UnitBody::default(), VehicleBody::default()),
            ObjectType::Weapon        => Self::Weapon(ItemBody::default(), WeaponBody::default()),
            ObjectType::Equipment     => Self::Equipment(ItemBody::default(), EquipmentBody::default()),
            ObjectType::Terminal      => Self::Terminal(DeviceBody::default(), TerminalBody::default()),
            ObjectType::Projectile    => Self::Projectile(ProjectileBody::default()),
            ObjectType::Scenery       => Self::Scenery(SceneryBody::default()),
            ObjectType::Machine       => Self::Machine(DeviceBody::default(), MachineBody::default()),
            ObjectType::Control       => Self::Control(DeviceBody::default(), ControlBody::default()),
            ObjectType::SoundScenery  => Self::SoundScenery,
            ObjectType::Crate         => Self::Crate(CrateBody::default()),
            ObjectType::Creature      => Self::Creature(CreatureBody::default()),
            ObjectType::Giant         => Self::Giant(MoverBody::default(), UnitBody::default(), GiantBody::default()),
            ObjectType::EffectScenery => Self::EffectScenery,
        }
    }
}

// ---------------------------------------------------------------------------
// ObjectDatum — top-level runtime state
// ---------------------------------------------------------------------------

/// Engine `object_datum` (352 bytes wrapper) + the inheritance chain
/// of derived bodies. One per scenario placement; stored on the
/// per-placement [`crate::halo::objects::object_header_data::ObjectHeaderDatum`].
#[derive(Debug, Clone)]
pub struct ObjectDatum {
    /// `definition_index` — engine `object_datum.definition_index`
    /// (offset 0). Tag-index of the `.scenery`/`.weapon`/etc. tag the
    /// placement instantiated. `-1` when unset.
    pub definition_index: i32,
    /// Base `_object_datum` body (engine layer 0). Always populated.
    pub object: ObjectBody,
    /// Per-type extension layers (engine layers 1..N).
    pub layers: TypeLayers,
}

impl ObjectDatum {
    /// Construct a default-state datum for `object_type`. Engine
    /// equivalent: `c_object::initialize` on a freshly-allocated slot
    /// in `object_data` pool — protomorph mirrors the
    /// "untouched, full-health, idle" output state for static
    /// placements.
    ///
    /// `definition_index` is the tag handle of the placement's source
    /// tag (i32 — engine signed). Currently unused by
    /// `object_compute_function_value` cases that aren't yet wired
    /// (variant lookup needs it); pass the placement's palette
    /// definition handle once available, or `-1` to signal "unbound".
    pub fn new(object_type: ObjectType, definition_index: i32) -> Self {
        let mut object = ObjectBody::default();
        // Stash the type discriminant into object_identifier.m_type so
        // engine cases that gate on it (`phantom_on`, `jetpack_exhaust`,
        // teleporters) see the right value.
        object.object_identifier.m_type = object_type as i8;
        Self {
            definition_index,
            object,
            layers: TypeLayers::default_for(object_type),
        }
    }

    /// Construct from a scenario placement, populating every field
    /// derivable from existing data. Engine equivalent: the path
    /// `object_placement_data_new_from_scenario_object` (`0x1808189D0`)
    /// → `object_new` (`0x1807D51D0`) — copies the placement transform
    /// + identifiers into the runtime datum.
    ///
    /// Fields populated from the placement:
    /// - `object_identifier.m_type` ← `object_type as i8`
    /// - `object_identifier.m_unique_id` ← `header_index as i32`
    ///   (engine assigns a runtime UID; protomorph reuses the handle)
    /// - `object_identifier.m_source` ← 0 (scenario-placed)
    /// - `scenario_datum_index` ← `placement_index_within_type as i16`
    /// - `name_index` ← `placement.name_index`
    /// - `forward` / `up` ← engine euler→matrix math
    ///   (`vectors3d_from_euler_angles3d`)
    /// - `scale` ← `placement.object_data.scale` (0.0 → engine
    ///   default of 1.0)
    /// - `relative_position` / `bounding_sphere_center` ←
    ///   `placement.object_data.position`
    ///
    /// Fields left at default (handled in later phases):
    /// - `definition_index` — palette index → tag handle resolution
    /// - `maximum_body_vitality` / `maximum_shield_vitality` — sourced
    ///   from `damage_info` tag (not in blam-tags today)
    /// - `variant_index` — needs `model.variants[]` lookup
    ///
    /// `definition_bounding_radius` is the object_definition tag's
    /// `bounding_radius` field — engine `c_object::initialize` copies
    /// this into the runtime `object_datum.bounding_sphere_radius` at
    /// object creation. Used by `lights_distant_lighting_at_point_new
    /// @ 0x1808A3220` to derive the cast-ray scales (`v28 = max(r,
    /// 0.4)`, `v29 = max(r, 0.05)`). Pass `0.0` for the auto-bake
    /// fallback (caller can backfill from `.model`'s `model_object_data[0]`
    /// auto-bake — see `loader.rs::read_model_object_data_sphere`).
    pub fn from_placement(
        object_type: ObjectType,
        placement: &ObjectPlacement,
        variant_names: &[String],
        placement_index_within_type: u32,
        header_index: u32,
        definition_bounding_radius: f32,
    ) -> Self {
        let mut datum = Self::new(object_type, -1);

        let rot = placement.object_data.rotation;
        // Engine `matrix4x3_rotation_from_angles` rows 0 + 2 →
        // `forward` and `up`. Verbatim port of dllcache 0x18019xxxx.
        // Inputs are radians (blam-tags `RealEulerAngles3d` already
        // surfaces them that way per the comment at math.rs:127).
        let (cy, sy) = (rot.yaw.cos(), rot.yaw.sin());
        let (cp, sp) = (rot.pitch.cos(), rot.pitch.sin());
        let (cr, sr) = (rot.roll.cos(), rot.roll.sin());
        let forward = RealVector3d {
            i: cy * cp,
            j: sy * cr - sp * sr * cy,
            k: sp * cr * cy + sy * sr,
        };
        let up = RealVector3d {
            i: -sp,
            j: -cp * sr,
            k: cp * cr,
        };

        // Engine convention (per `PlacementObjectData.scale` doc + the
        // `if (data->scale <= 0.0)` gate at object_new:L174): 0.0
        // means "use default scale". Resolve to 1.0 here so downstream
        // code never sees the 0 sentinel.
        let scale = if placement.object_data.scale > 0.0 {
            placement.object_data.scale
        } else {
            1.0
        };

        datum.object.object_identifier.unique_id = header_index as i32;
        // Engine `c_object_identifier::e_source` — 0 = scenario placement.
        // Other values: dynamic, scripting, prefab, etc. Static-only for now.
        datum.object.object_identifier.m_source = 0;
        datum.object.scenario_datum_index = placement_index_within_type as i16;
        datum.object.name_index = placement.name_index;
        datum.object.forward = forward;
        datum.object.up = up;
        datum.object.scale = scale;
        datum.object.relative_position = placement.object_data.position;
        // Engine `calculate_object_starting_bounding_sphere` derives
        // this from `position + rotated_offset * scale` using the
        // object_definition's bounding_offset. We use just the
        // position as an initial approximation; refine when the
        // bounding-offset transform lands.
        datum.object.bounding_sphere_center = placement.object_data.position;
        // Engine `c_object::initialize` copies the tag's bounding
        // radius into the runtime datum at +0x2C (inner +40). The
        // engine accepts 0 here — `lights_distant_lighting_at_point_new`'s
        // v28 formula has a 0.4 default for `m_radius < 0.05`. Caller
        // is expected to have backfilled from `.model`'s auto-bake
        // when the .object tag itself authored 0.
        datum.object.bounding_sphere_radius = definition_bounding_radius;

        // Resolve `placement.permutation_data.variant_name` (string_id)
        // against the model's `variants[]` block. Engine equivalent:
        // `c_object::initialize` walks the loaded hlmt for a matching
        // string_id and stores the index in `object_datum.variant_index`.
        // Drives `object_compute_function_value` case `variant` (sid 501).
        // Defaults to 0 when name is empty or not found in the list.
        let variant_name = &placement.permutation_data.variant_name;
        datum.object.variant_index = if !variant_name.is_empty() {
            variant_names
                .iter()
                .position(|n| n == variant_name)
                .map(|i| i as i8)
                .unwrap_or(0)
        } else {
            0
        };

        datum
    }

    // -- typed accessors for the per-type bodies --
    // Each `as_*` returns `Some` only when the datum's leaf type
    // includes that layer. Compute functions use these to read their
    // type-specific state without manual pattern matching.

    pub fn as_item(&self) -> Option<&ItemBody> {
        match &self.layers {
            TypeLayers::Equipment(i, _) | TypeLayers::Weapon(i, _) => Some(i),
            _ => None,
        }
    }
    pub fn as_weapon(&self) -> Option<&WeaponBody> {
        match &self.layers {
            TypeLayers::Weapon(_, w) => Some(w),
            _ => None,
        }
    }
    pub fn as_equipment(&self) -> Option<&EquipmentBody> {
        match &self.layers {
            TypeLayers::Equipment(_, e) => Some(e),
            _ => None,
        }
    }
    pub fn as_projectile(&self) -> Option<&ProjectileBody> {
        match &self.layers {
            TypeLayers::Projectile(p) => Some(p),
            _ => None,
        }
    }
    pub fn as_unit(&self) -> Option<&UnitBody> {
        match &self.layers {
            TypeLayers::Biped(_, u, _)
            | TypeLayers::Vehicle(_, u, _)
            | TypeLayers::Giant(_, u, _) => Some(u),
            _ => None,
        }
    }
    pub fn as_mover(&self) -> Option<&MoverBody> {
        match &self.layers {
            TypeLayers::Biped(m, _, _)
            | TypeLayers::Vehicle(m, _, _)
            | TypeLayers::Giant(m, _, _) => Some(m),
            _ => None,
        }
    }
    pub fn as_biped(&self) -> Option<&BipedBody> {
        match &self.layers {
            TypeLayers::Biped(_, _, b) => Some(b),
            _ => None,
        }
    }
    pub fn as_vehicle(&self) -> Option<&VehicleBody> {
        match &self.layers {
            TypeLayers::Vehicle(_, _, v) => Some(v),
            _ => None,
        }
    }
    pub fn as_giant(&self) -> Option<&GiantBody> {
        match &self.layers {
            TypeLayers::Giant(_, _, g) => Some(g),
            _ => None,
        }
    }
    pub fn as_device(&self) -> Option<&DeviceBody> {
        match &self.layers {
            TypeLayers::Machine(d, _) | TypeLayers::Control(d, _) | TypeLayers::Terminal(d, _) => Some(d),
            _ => None,
        }
    }
    pub fn as_machine(&self) -> Option<&MachineBody> {
        match &self.layers {
            TypeLayers::Machine(_, m) => Some(m),
            _ => None,
        }
    }
    pub fn as_control(&self) -> Option<&ControlBody> {
        match &self.layers {
            TypeLayers::Control(_, c) => Some(c),
            _ => None,
        }
    }
    pub fn as_terminal(&self) -> Option<&TerminalBody> {
        match &self.layers {
            TypeLayers::Terminal(_, t) => Some(t),
            _ => None,
        }
    }
    pub fn as_scenery(&self) -> Option<&SceneryBody> {
        match &self.layers {
            TypeLayers::Scenery(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_crate(&self) -> Option<&CrateBody> {
        match &self.layers {
            TypeLayers::Crate(c) => Some(c),
            _ => None,
        }
    }
    pub fn as_creature(&self) -> Option<&CreatureBody> {
        match &self.layers {
            TypeLayers::Creature(c) => Some(c),
            _ => None,
        }
    }
}
