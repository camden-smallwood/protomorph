//! `_item_datum` body + `item_compute_function_value` stub. Engine
//! layout mirrors `source/items/items.h:34-44`. Sits between `_object_datum`
//! and `_weapon_datum` / `_equipment_datum` in the composite layouts.
//!
//! No `item_compute_function_value` exists in the engine — `item` is
//! a pure layout intermediate (NULL slot in the `part_definitions[]`
//! chain — see [[reference_object_type_compute_function_chains]]).
//! Code that needs item-state reads via `datum.as_item()` directly.

/// Engine `_item_datum` (20 bytes, 7 fields). Verified 2026-05-20
/// via `get_type_info?name=_item_datum`.
///
/// Ares: `source/items/items.h:34-44`.
#[derive(Debug, Clone, Default)]
pub struct ItemBody {
    /// `c_flags<e_item_datum_flags, u8, 5>` — runtime item flags.
    /// Read by various item-state predicates (`item_in_hand`,
    /// `item_stowed`, etc.). Bit names per the engine enum.
    pub flags: u8,
    /// `unsigned char inventory_state` — engine `e_item_inventory_state`
    /// enum (in_world, in_inventory, in_hand, stowed, etc.). Read by
    /// item_in_hand / item_stowed helpers.
    pub inventory_state: u8,
    pub detonation_ticks: i16,
    pub ignore_object_index: i32,
    pub last_relevant_time: i32,
    pub inventory_unit_index: i32,
    pub last_inventory_unit_index: i32,
}

// Engine `e_item_inventory_state` values — used by `item_in_hand` /
// `item_stowed` predicates. Stub helpers read `ItemBody.inventory_state`
// and compare against these.
pub const ITEM_INVENTORY_STATE_IN_WORLD:     u8 = 0;
pub const ITEM_INVENTORY_STATE_IN_INVENTORY: u8 = 1;
pub const ITEM_INVENTORY_STATE_IN_HAND:      u8 = 2;
pub const ITEM_INVENTORY_STATE_STOWED:       u8 = 3;
