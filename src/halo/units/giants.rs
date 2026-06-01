//! `giant_compute_function_value @ dllcache 0x1808814B0`.
//! Ares source: `source/units/giants.cpp:405`.
//!
//! Verbatim port — engine has 2 cases:
//!   - sid 845 `buckle` → reads `giant_buckling_magnitude_get` engine helper.
//!   - Default → forwards to `weapon_compute_function_value` on the
//!     weapon held in unit slot `_unit_datum[+602]` (engine reads a
//!     `char` at +602 = unit_datum.weapon_object_indices[some_slot]).
//!
//! Chain position: `object → unit → giant`. The default-arm weapon
//! forward duplicates what `unit_compute_function_value` does for
//! the 576/578/579 weapon-forwarding cases, but giant forwards EVERY
//! function — making giants effectively delegate any unrecognized
//! sid to their wielded weapon.

use crate::halo::objects::object_header_data::get_object_datum;

/// Engine `giant_buckling_magnitude_get(giant_index) -> f32`.
/// Stubbed at 0 until giant buckling animation state is simulated.
fn giant_buckling_magnitude_get(_giant_index: u32) -> f32 {
    0.0
}

/// Engine `unit_inventory_get_weapon(unit_index, slot) -> i32` —
/// returns weapon object_index for the given slot, or -1.
/// Stubbed at -1 until inventory simulation lands.
fn unit_inventory_get_weapon(_unit_index: u32, _slot: i8) -> i32 {
    -1
}

pub fn giant_compute_function_value(
    giant_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    if function == "buckle" {
        // Engine sid 845. Read buckling magnitude, clamp to [0, 1].
        let mut v = giant_buckling_magnitude_get(giant_index);
        if v <= 0.0 {
            v = 0.0;
        } else if v >= 1.0 {
            v = 1.0;
        }
        *value = v;
        *active = v > 0.0;
        return true;
    }

    // Default arm — forward to weapon_compute_function_value on the
    // giant's currently-wielded weapon. Engine reads slot index at
    // `_unit_datum + offset 602` (a char field); without inventory
    // simulation `unit_inventory_get_weapon → -1` and we return false.
    let datum_lock = get_object_datum(giant_index);
    let datum = datum_lock.read().unwrap();
    // Engine reads byte at unit_datum+602. Our UnitBody surfaces a
    // limited subset — use weapon_indices[0] as the closest analog
    // since the engine slot fetch lands on inventory state we don't
    // simulate yet.
    let slot = datum
        .as_unit()
        .map(|u| u.current_weapon_set.weapon_indices[0])
        .unwrap_or(-1);
    drop(datum);

    let weapon = unit_inventory_get_weapon(giant_index, slot);
    if weapon == -1 {
        return false;
    }
    crate::halo::items::weapons::weapon_compute_function_value(
        weapon as u32,
        function,
        function_owner_definition_index,
        value,
        active,
        deterministic,
    )
}
