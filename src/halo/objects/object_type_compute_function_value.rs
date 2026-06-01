//! `object_type_compute_function_value @ dllcache 0x18081CCB0` — the
//! TYPE DISPATCHER. Walks the `object_type_definition::part_definitions[]`
//! vtable-like array in order, calling each step's
//! `datum_compute_function_value` until one returns `true`.
//!
//! Ares source: `source/objects/object_types.cpp:763`.
//!
//! Engine body (verified 2026-05-20 via dllcache IDA bridge):
//! ```c
//! v11 = object_type_definition_get(<type>);
//! part_definitions = v11->part_definitions;
//! *value = 0.0;  *active = false;
//! while (*part_definitions) {
//!     fn = (*part_definitions)->datum_compute_function_value;
//!     if (fn) v12 = fn(object_index, function, owner, value, active, det);
//!     ++part_definitions;
//!     if (v12) return v12;   // first non-false return wins
//! }
//! *value = 0.0; *active = false; return 0;
//! ```
//!
//! The `part_definitions[]` array is initialized at boot from a static
//! global. Per-type chains read from `object_type_definitions@0x1811abe10`
//! + each type's `part_definitions` slot @ offset 328:
//!
//! ```text
//!   biped/vehicle/giant            : object → unit → leaf
//!   creature                       : object → creature
//!   weapon/equipment               : object → leaf
//!   machine/control/terminal       : object → device → leaf
//!   scenery/crate/projectile/      : object → leaf
//!     sound_scenery/effect_scenery
//! ```
//!
//! `motor` and `item` are intermediate steps with NULL compute pointers
//! (skipped by the `if (fn)` gate). They're elided from the table above.

use crate::halo::creatures::creatures::creature_compute_function_value;
use crate::halo::devices::device_machines::machine_compute_function_value;
use crate::halo::devices::device_terminals::device_terminal_compute_function_value;
use crate::halo::devices::devices::device_compute_function_value;
use crate::halo::items::equipment::equipment_compute_function_value;
use crate::halo::items::projectiles::projectile_compute_function_value;
use crate::halo::items::weapons::weapon_compute_function_value;
use crate::halo::objects::crates::crate_compute_function_value;
use crate::halo::objects::object_compute_function_value::object_compute_function_value;
use crate::halo::objects::object_header_data::get_object_type;
use crate::halo::objects::object_type::ObjectType;
use crate::halo::units::bipeds::biped_compute_function_value;
use crate::halo::units::giants::giant_compute_function_value;
use crate::halo::units::units::unit_compute_function_value;
use crate::halo::units::vehicles::vehicle_compute_function_value;

/// Shared signature for every `*_compute_function_value` in a chain.
/// Engine `bool (__fastcall *)(int, int, int, float *, bool *, bool *)`.
type ComputeFn = fn(u32, &str, u32, &mut f32, &mut bool, &mut bool) -> bool;

/// Walk an inheritance chain (engine `part_definitions[]`) and return
/// on first `true`. Engine: while-loop in `object_type_compute_function_value
/// @ 0x18081CCB0`. NULL slots (`motor`, `item`) are pre-filtered out by
/// the chain definitions in [`object_type_compute_function_value`].
fn try_chain(
    chain: &[ComputeFn],
    object_index: u32,
    function: &str,
    function_owner_tag_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    for compute in chain {
        let handled = compute(
            object_index,
            function,
            function_owner_tag_index,
            value,
            active,
            deterministic,
        );
        if handled {
            return true;
        }
    }
    false
}

/// Engine `object_type_compute_function_value` body. Reads the object's
/// type from `object_header_data` then walks the type's
/// `part_definitions[]` chain (base-first, derived-last) — first
/// non-false return wins.
pub fn object_type_compute_function_value(
    object_index: u32,
    function: &str,
    function_owner_tag_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    // Engine: `*value = 0.0; *active = 0;` before the part_definitions
    // walk (`0x18081CCB0:18-19`).
    *value = 0.0;
    *active = false;

    let object_type = get_object_type(object_index);

    // The 14 chains, decoded from `object_type_definitions` data @
    // dllcache 0x1811abe10 (2026-05-20). NULL `motor` and `item` slots
    // pre-filtered.
    let chain: &[ComputeFn] = match object_type {
        ObjectType::Biped         => &[object_compute_function_value, unit_compute_function_value, biped_compute_function_value],
        ObjectType::Vehicle       => &[object_compute_function_value, unit_compute_function_value, vehicle_compute_function_value],
        ObjectType::Giant         => &[object_compute_function_value, unit_compute_function_value, giant_compute_function_value],
        ObjectType::Creature      => &[object_compute_function_value, creature_compute_function_value],
        ObjectType::Weapon        => &[object_compute_function_value, weapon_compute_function_value],
        ObjectType::Equipment     => &[object_compute_function_value, equipment_compute_function_value],
        ObjectType::Projectile    => &[object_compute_function_value, projectile_compute_function_value],
        ObjectType::Machine       => &[object_compute_function_value, device_compute_function_value, machine_compute_function_value],
        ObjectType::Control       => &[object_compute_function_value, device_compute_function_value],
        ObjectType::Terminal      => &[object_compute_function_value, device_compute_function_value, device_terminal_compute_function_value],
        ObjectType::Crate         => &[object_compute_function_value, crate_compute_function_value],
        ObjectType::Scenery
        | ObjectType::SoundScenery
        | ObjectType::EffectScenery => &[object_compute_function_value],
    };

    try_chain(
        chain,
        object_index,
        function,
        function_owner_tag_index,
        value,
        active,
        deterministic,
    )
}
