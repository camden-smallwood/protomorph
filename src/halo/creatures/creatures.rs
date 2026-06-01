//! `_creature_datum` body + `creature_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/creatures/creatures.h:69-85` (_creature_datum, 212B)
//!   - body:   `source/creatures/creatures.cpp:248`
//! Engine:
//!   - struct verified via `get_type_info?name=_creature_datum` (12 fields).
//!   - compute body verbatim port of dllcache `0x18087BF80` (29 lines, 1 case).
//!
//! Chain position: `object → creature`. Only one creature-specific
//! input case: `gear_power` (sid 1189). Everything else falls through
//! to `object_compute_function_value`.

use crate::halo::objects::object_header_data::get_object_datum;

// ---------------------------------------------------------------------------
// _creature_datum body
// ---------------------------------------------------------------------------

/// Engine `_creature_datum` (212 bytes, 12 fields). Field set today
/// is minimal — only `control_data_flags` is read by compute. Add
/// fields as consumers come online.
///
/// Ares: `source/creatures/creatures.h:69-85`.
#[derive(Debug, Clone, Default)]
pub struct CreatureBody {
    /// `flags` (u16 @ offset 0).
    pub flags: u16,
    /// `team_index` (i16 @ offset 2).
    pub team_index: i16,
    /// Byte 20 of the `s_creature_control_data` substruct (which
    /// itself starts at `_creature_datum` offset 12 — so this byte
    /// sits at `_creature_datum` offset 32). Bit 0x20 is the
    /// `gear_power` predicate read by compute case sid 1189. Full
    /// `s_creature_control_data` layout deferred until a consumer
    /// needs the surrounding fields.
    pub control_data_flags_byte: u8,
}

// ---------------------------------------------------------------------------
// creature_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `creature_compute_function_value @
/// 0x18087BF80`. Single case:
///   sid 1189 `gear_power` — reads `(control_data_flags_byte & 0x20)
///   != 0 → 1.0 else 0.0`.
pub fn creature_compute_function_value(
    creature_index: u32,
    function: &str,
    _function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    if function != "gear_power" {
        return false;
    }
    let datum_lock = get_object_datum(creature_index);
    let datum = datum_lock.read().unwrap();
    let Some(c) = datum.as_creature() else {
        return false;
    };
    let v = if (c.control_data_flags_byte & 0x20) != 0 {
        1.0
    } else {
        0.0
    };
    *value = v;
    *active = v > 0.0;
    true
}
