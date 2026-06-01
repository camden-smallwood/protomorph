//! `_equipment_datum` body + `equipment_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/items/equipment.h:18-28` (_equipment_datum, 28B)
//!   - body:   `source/items/equipment.cpp:394` (equipment_compute_function_value)
//! Engine:
//!   - struct verified via `get_type_info?name=_equipment_datum` (6 fields).
//!   - compute body verbatim port of dllcache `0x180840BE0` (50-line
//!     pseudocode, 2 cases — `age` and `death`).
//!
//! Chain position: `object → equipment` (item layer has NULL compute,
//! so the engine `part_definitions[]` chain skips it for equipment;
//! see [[reference_object_type_compute_function_chains]]).

use crate::halo::objects::object_datum::DamageOwner;
use crate::halo::objects::object_header_data::{get_object_datum, get_tag_path};
use crate::halo::tags::tag_get;
use blam_tags::equipment::{EquipmentDefinition, EQUIPMENT_TYPE_PROXIMITY_MINE};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// _equipment_datum body
// ---------------------------------------------------------------------------

/// Engine `_equipment_datum` (28 bytes, 6 fields). Verified
/// 2026-05-20 via `get_type_info?name=_equipment_datum`. Field order
/// + names mirror Ares `source/items/equipment.h:18-28`.
#[derive(Debug, Clone, Default)]
pub struct EquipmentBody {
    /// Game-tick stamp when this equipment was spawned. Drives the
    /// `age` compute case — engine math:
    /// `(game_time_get() - game_time_at_creation) / 30 / 5.0` clamped
    /// to `[0, 1]`.
    pub game_time_at_creation: i32,
    /// Game-tick stamp of the most recent activation. Drives the
    /// `death` compute case (proximity-mine fuse readout).
    pub last_use_time: i32,
    pub charges_used: u8,
    pub proximity_triggered_counter: u8,
    pub creator_damage_owner: DamageOwner,
    pub looping_effect_index: i32,
}

// ---------------------------------------------------------------------------
// Engine helpers
// ---------------------------------------------------------------------------

/// Engine `game_time_get` — current tick. Stubbed at 0 until tick
/// simulation wires in (matches "no time has elapsed since load").
fn game_time_get() -> i32 {
    0
}

/// Engine `game_ticks_to_seconds(ticks) -> f32` — 1 tick = 1/30 s.
fn game_ticks_to_seconds(ticks: i32) -> f32 {
    (ticks as f32) / 30.0
}

/// Resolve the `EquipmentDefinition` for `equipment_index` via the
/// tag cache. Engine equivalent: dereferences
/// `global_tag_instances[object_datum.definition_index]` to an
/// `equipment_definition*`. `None` when the placement's palette
/// tag_path is unset.
fn equipment_definition_for(equipment_index: u32) -> Option<Arc<EquipmentDefinition>> {
    let tag_path = get_tag_path(equipment_index);
    if tag_path.is_empty() {
        return None;
    }
    tag_get(*b"eqip", &tag_path).and_then(|t| t.as_equipment_definition())
}

// ---------------------------------------------------------------------------
// equipment_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `equipment_compute_function_value @
/// 0x180840BE0`. Only two cases on the engine side:
///   - `age` (sid 529) — elapsed-since-creation / 5 seconds.
///   - `death` (sid 530) — proximity-mine fuse readout; routes
///     through the equipment_definition's proximity_mine sub-block.
pub fn equipment_compute_function_value(
    object_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let _ = function_owner_definition_index;

    let datum_lock = get_object_datum(object_index);
    let datum = datum_lock.read().unwrap();
    let Some(eq) = datum.as_equipment() else {
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        // Case 529 `age` —
        //   v12 = game_ticks_to_seconds(game_time - game_time_at_creation) * 0.2
        // (i.e. seconds-since-creation / 5.0, clamped to [0, 1] by the
        // LABEL_30 final clamp).
        "age" => {
            let elapsed_ticks = game_time_get() - eq.game_time_at_creation;
            magnitude = game_ticks_to_seconds(elapsed_ticks) * 0.2;
        }

        // Case 530 `death` — proximity-mine fuse readout. Engine:
        //   if !equipment_definition_has_type(def, 3 /* proximity mine */) → 0
        //   pm_def = equipment_get_proximity_mine_definition(def);
        //   v11 = max(0, pm_def.self_destruct_time -
        //          game_ticks_to_seconds(game_time - last_use_time))
        //   v12 = 1.0 - v11 * 0.2     // 1 - remaining_fuse/5
        // For untouched equipment (last_use_time = 0) the elapsed
        // term is large → v11 saturates to 0 → v12 = 1.0 (fully
        // dead). Only fires for proximity mines.
        "death" => {
            magnitude = 0.0;
            if let Some(ed) = equipment_definition_for(object_index) {
                if ed.has_type(EQUIPMENT_TYPE_PROXIMITY_MINE) {
                    if let Some(pm) = ed.proximity_mine_blocks.first() {
                        let elapsed_seconds =
                            game_ticks_to_seconds(game_time_get() - eq.last_use_time);
                        let mut v11 = pm.self_destruct_time - elapsed_seconds;
                        if v11 < 0.0 {
                            v11 = 0.0;
                        }
                        magnitude = 1.0 - v11 * 0.2;
                    }
                }
            }
        }

        _ => recognized = false,
    }

    // Engine LABEL_30 clamp: positive → [0, 1], else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
