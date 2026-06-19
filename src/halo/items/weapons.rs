//! `_weapon_datum` body + `weapon_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/items/weapons.h:216-254` (_weapon_datum, 280B)
//!   - body:   `source/items/weapons.cpp:2016` (weapon_compute_function_value)
//! Engine:
//!   - struct verified via `get_type_info?name=_weapon_datum` (35 fields).
//!   - compute body verbatim port of dllcache `0x1808205E0` (413-line
//!     pseudocode), case-by-case across ~30 string_ids.
//!
//! Chain position: `object → weapon` (item layer has NULL compute, so
//! `object_type_compute_function_value` dispatch skips it for weapon).
//! `weapon` is the leaf in `[object, weapon]` per
//! [[reference_object_type_compute_function_chains]].

use crate::halo::items::items::{
    ITEM_INVENTORY_STATE_IN_HAND, ITEM_INVENTORY_STATE_STOWED,
};
use crate::halo::objects::object_header_data::{get_object_datum, get_object_definition};
use crate::halo::tags::tag_get;
use blam_tags::weapon::WeaponDefinition;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Sub-struct bodies (used as elements of WeaponBody's arrays)
// ---------------------------------------------------------------------------

/// Engine `weapon_barrel` (52 bytes, 20 fields). One per barrel; max 2.
/// Ares: `source/items/weapons.h` (`weapon_barrel` struct).
#[derive(Debug, Clone, Default)]
pub struct WeaponBarrel {
    pub firing_idle_ticks: i8,
    /// Read by `primary_barrel_firing*` cases (533-537) via `BYTE1`.
    pub state: i8,
    pub state_timer: i16,
    pub flags: u16,
    pub fire_count: u16,
    pub firing_effects_used_flags: u16,
    pub firing_effect_index: i16,
    pub firing_effect_shots_remaining: i16,
    pub sequential_non_tracer_rounds: i16,
    /// `primary/secondary_rate_of_fire` (cases 523/524) — and the
    /// numerator for case 579 (`barrel_spin_rate`, reads barrels[0]).
    pub rate_of_fire: f32,
    /// `primary/secondary_ejection_port` (cases 521/522).
    pub ejection_port_position: f32,
    /// Read by case 516 (`illumination`) as part of the max-of-barrels
    /// calculation.
    pub illumination: f32,
    pub current_error: f32,
    pub angle_change_scale: f32,
    pub bonus_shot_fraction: f32,
    pub recovery_overflow: f32,
    pub bonus_shot_count: u16,
    pub staggered_marker_offset: u8,
    pub predicted_recovery_timer: u8,
    pub effect_index: i32,
}

/// Engine `weapon_trigger` (12 bytes, 7 fields).
/// Ares: `source/items/weapons.h` (`weapon_trigger` struct).
#[derive(Debug, Clone, Default)]
pub struct WeaponTrigger {
    pub state: i8,
    pub utility_timer: u8,
    pub recent_spew_timer: u8,
    pub pad: u8,
    pub state_timer: i16,
    pub flags: u16,
    pub charging_effect_index: i32,
}

/// Engine `weapon_magazine` (24 bytes, 12 fields).
/// Ares: `source/items/weapons.h` (`weapon_magazine` struct).
#[derive(Debug, Clone, Default)]
pub struct WeaponMagazine {
    pub state: i16,
    pub state_timer: i16,
    pub original_time: i16,
    pub rounds_inventory: i16,
    pub rounds_inventory_delayed: i16,
    /// `primary/secondary_ammunition`-family cases (517-520) — engine
    /// derives the normalized ammo readout from this.
    pub rounds_loaded: i16,
    pub rounds_loaded_delayed: i16,
    pub rounds_fractional_recharged: i16,
    pub reload_timer: i16,
    pub reload_duration: i16,
    pub reload_weapon_disable_duration: i16,
    pub shotgun_chambered_timer: i16,
}

// ---------------------------------------------------------------------------
// _weapon_datum body
// ---------------------------------------------------------------------------

/// Engine `_weapon_datum` (280 bytes, 35 fields). Verified 2026-05-20
/// via `get_type_info?name=_weapon_datum`. Field order + names mirror
/// Ares `source/items/weapons.h:216-254`.
#[derive(Debug, Clone, Default)]
pub struct WeaponBody {
    /// `c_flags<e_weapon_flags, u16, 13>`. Bit 2 is the
    /// `overheated` predicate gate (case 106).
    pub flags: u16,
    /// `c_flags<e_weapon_control_flags, u16, 9>`. Bits 0x02 and 0x04
    /// drive `primary/secondary_trigger_down` (cases 531/532).
    pub control_flags: u16,
    pub primary_trigger: u8,
    pub last_primary_trigger: u8,
    pub last_hill_or_valley: u8,
    pub primary_trigger_direction: i8,
    pub primary_trigger_down_ticks: i8,
    /// `barrel_spin` case (578) reads this as `u8 / 255`.
    pub barrel_spin: u8,
    pub tracked_model_target_index: i8,
    pub predicted_delay_timer: u8,
    pub state: i16,
    pub state_timer: i16,
    pub weapon_disabled_by_reload_timer: i16,
    pub multiplayer_weapon_identifier: i16,
    pub simulation_weapon_identifier: i16,
    pub turn_on_timer: u8,
    pub ready_for_use_timer: u8,
    /// `heat` case (513) — direct read.
    pub heat: f32,
    /// `age` case (529), `turned_on` (514), `turning_on` (515),
    /// `weapon_battery_empty` (577) — all funnel through `weapon_get_age`
    /// which reads this field.
    pub age: f32,
    pub delayed_age: f32,
    /// Read by `overheated` case (106) — engine uses it as the
    /// over-recovery accumulator.
    pub overcharged: f32,
    /// `power` case (133) — direct read.
    pub current_power: f32,
    pub desired_power: f32,
    pub tracked_object_index: i32,
    pub recoil_angular_velocity: f32,
    pub recoil_recovery_time: i16,
    pub shots_until_demotion: i16,
    pub alternate_shots_loaded: i16,
    /// Two-element arrays per Ares — `[primary, secondary]` semantics
    /// match the dispatch pattern in cases 521-528 / 533-537.
    pub barrels: [WeaponBarrel; 2],
    pub triggers: [WeaponTrigger; 2],
    pub magazines: [WeaponMagazine; 2],
    pub overheated_effect_index: i32,
    /// Case 525/526 (`primary/secondary_firing`) tests the elapsed
    /// game-time since this stamp against a 2-frame window.
    pub game_time_last_fired: i32,
    pub tracked_object_last_target_acquisition_time: i32,
    // `first_person_emulation: weapon_first_person_emulation` (28B,
    // 4 fields) — used by first-person camera code, not compute_function_value.
    // Surface when a consumer needs it.
}

// ---------------------------------------------------------------------------
// Engine helpers — currently stubbed against the datum
// ---------------------------------------------------------------------------
//
// Engine equivalents live in `source/items/weapons.cpp` /
// `source/objects/objects.cpp`. Each helper reads runtime state we
// haven't simulated (inventory ownership, barrel firing predicates,
// trigger charging). Until simulation lands, helpers either read
// `ItemBody.inventory_state` / `WeaponBody.*` or return engine-default
// "untouched, on the ground" values.

/// Engine `item_in_hand` — true when the inventory_state == in_hand.
fn item_in_hand(weapon_index: u32) -> bool {
    let datum = get_object_datum(weapon_index);
    let d = datum.read().unwrap();
    d.as_item()
        .map(|i| i.inventory_state == ITEM_INVENTORY_STATE_IN_HAND)
        .unwrap_or(false)
}

/// Engine `item_stowed` — true when the inventory_state == stowed.
fn item_stowed(weapon_index: u32) -> bool {
    let datum = get_object_datum(weapon_index);
    let d = datum.read().unwrap();
    d.as_item()
        .map(|i| i.inventory_state == ITEM_INVENTORY_STATE_STOWED)
        .unwrap_or(false)
}

/// Engine `weapon_get_age` — reads `_weapon_datum.age`.
fn weapon_get_age(weapon_index: u32) -> f32 {
    let datum = get_object_datum(weapon_index);
    let d = datum.read().unwrap();
    d.as_weapon().map(|w| w.age).unwrap_or(0.0)
}

/// Engine `weapon_trigger_get_charged_fraction` — engine computes
/// the charge ramp based on trigger state + timer. Until trigger
/// simulation lands, return 0.0 (trigger not charged).
fn weapon_trigger_get_charged_fraction(_weapon_index: u32, _trigger_index: i32) -> f32 {
    0.0
}

/// Engine `weapon_barrel_can_fire` — runtime gate combining reload
/// state, ammo, etc. Ground placements default to false (not firing).
fn weapon_barrel_can_fire(_weapon_index: u32, _barrel_index: i32, _flag: i32) -> bool {
    false
}

/// Engine `game_time_get` — current tick. Read by case 525/526's
/// "fired recently?" test. Returns 0 until tick simulation wires in
/// (matches "no firing event has happened yet").
fn game_time_get() -> i32 {
    0
}

/// Engine `game_seconds_to_ticks_real(seconds) -> int` — 30 ticks/sec.
fn game_seconds_to_ticks_real(seconds: f32) -> i32 {
    (seconds * 30.0).round() as i32
}

/// Resolve the `WeaponDefinition` for `weapon_index` via the tag
/// cache. Engine equivalent: `global_tag_instances[object_datum.
/// definition_index]` resolved to a `weapon_definition*`. Returns
/// `None` when no weapon tag is loaded for the placement (e.g.
/// definition_index unresolved or palette tag_path empty).
fn weapon_definition_for(weapon_index: u32) -> Option<Arc<WeaponDefinition>> {
    let obj_def = get_object_definition(weapon_index);
    // ObjectDefinition holds the tag path indirectly via the cache —
    // protomorph stores it on `ObjectHeaderDatum.tag_path`. Look it up.
    let tag_path = crate::halo::objects::object_header_data::get_tag_path(weapon_index);
    if tag_path.is_empty() {
        let _ = obj_def;
        return None;
    }
    tag_get(*b"weap", &tag_path).and_then(|t| t.as_weapon_definition())
}

// ---------------------------------------------------------------------------
// weapon_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `weapon_compute_function_value @ 0x1808205E0`
/// (Ares `source/items/weapons.cpp:2016`). Each case reads from the
/// per-placement `ObjectDatum` via typed accessors (`datum.as_weapon`,
/// `datum.as_item`). Returns `true` when the case was recognized (engine
/// `v14 = 1`); the chain dispatcher uses this to short-circuit.
///
/// Engine return semantics — same shape as
/// `object_compute_function_value`:
/// - recognized AND value > 0 → return `true`, `*active = true`
/// - recognized AND value <= 0 → return `true`, `*active = false`
/// - not recognized → return `false`, fall through to next chain step
///   (object_compute, or LABEL_82 in `object_get_function_value`).
pub fn weapon_compute_function_value(
    weapon_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    let _ = function_owner_definition_index;

    let datum_lock = get_object_datum(weapon_index);
    let datum = datum_lock.read().unwrap();
    let Some(weapon) = datum.as_weapon() else {
        // Not a weapon datum — engine `object_type_compute_function_value`
        // wouldn't have routed here, but guard defensively.
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        // Case 38 `ready` — magnitude = 1.0 iff in-hand.
        "ready" => {
            if item_in_hand(weapon_index) {
                magnitude = 1.0;
            }
        }

        // Case 106 `overheated`. Engine:
        //   if ((flags & 2) == 0) → 0.0   (overheated flag bit)
        //   threshold = weapon_definition.heat_recovery_threshold
        //   if (threshold == 1.0) → 0.0
        //   magnitude = (heat - threshold) / (1.0 - threshold)
        // Maps the current heat to a [0, 1] over-recovery progress
        // bar — at threshold = 0.0, at heat=1.0 = 1.0.
        "overheated" => {
            if (weapon.flags & 2) != 0 {
                if let Some(wd) = weapon_definition_for(weapon_index) {
                    let threshold = wd.heat_recovery_threshold;
                    if threshold != 1.0 {
                        magnitude = (weapon.heat - threshold) / (1.0 - threshold);
                    }
                }
            }
        }

        // Case 513 `heat` — direct read.
        "heat" => magnitude = weapon.heat,

        // Case 514 `turned_on`. Engine:
        //   if stowed || (in_hand && (flags & 0x20) == 0) → 0.0
        //   else → weapon_get_age
        "turned_on" => {
            let block = item_stowed(weapon_index)
                || (item_in_hand(weapon_index) && (weapon.flags & 0x20) == 0);
            if !block {
                magnitude = weapon_get_age(weapon_index);
            }
        }

        // Case 515 `turning_on`. Engine:
        //   if !in_hand || (flags & 0x20) != 0 → 0.0
        //   else: age = weapon_get_age; LABEL_39: if age >= 1.0 → 0.0 else 1.0 if age > 0 else 0.0
        // LABEL_39 collapses to "1.0 iff 0 < age < 1.0".
        "turning_on" => {
            if item_in_hand(weapon_index) && (weapon.flags & 0x20) == 0 {
                let age = weapon_get_age(weapon_index);
                if age > 0.0 && age < 1.0 {
                    magnitude = 1.0;
                }
            }
        }

        // Case 516 `illumination`. Engine takes the max of (verbatim
        // from dllcache 0x1808205E0:121-193):
        //   1. Per runtime barrel `i` in [0, weapon_def.barrels.count):
        //      max := max(max, weapon.barrels[i].illumination)
        //   2. Per runtime trigger `i` in [0, weapon_def.triggers.count):
        //      let td = weapon_def.triggers[i];
        //      if td.charging.charging_time > 0.0:
        //          charged_frac := weapon_trigger_get_charged_fraction(weapon, i);
        //          max := max(max, charged_frac * td.charging.charged_illumination)
        //      if weapon.triggers[i].state == 2 (overcharged):
        //          max := max(max,
        //                     (1 - td.charging.charged_illumination)
        //                       * weapon.overcharged
        //                       + td.charging.charged_illumination)
        //   3. max := max(max, weapon_def.heat_illumination * weapon.heat)
        "illumination" => {
            let mut mx = 0.0f32;
            if let Some(wd) = weapon_definition_for(weapon_index) {
                // Barrel illumination (per-runtime-barrel field).
                for i in 0..wd.barrels.len().min(weapon.barrels.len()) {
                    if weapon.barrels[i].illumination > mx {
                        mx = weapon.barrels[i].illumination;
                    }
                }
                // Trigger charging + overcharge contributions.
                for i in 0..wd.triggers.len().min(weapon.triggers.len()) {
                    let td = &wd.triggers[i];
                    if td.charging.charging_time > 0.0 {
                        let cf = weapon_trigger_get_charged_fraction(weapon_index, i as i32);
                        let contrib = cf * td.charging.charged_illumination;
                        if contrib > mx {
                            mx = contrib;
                        }
                    }
                    // Runtime trigger state 2 = overcharged (engine
                    // weapon_trigger.state byte at offset 0).
                    if weapon.triggers[i].state == 2 {
                        let ci = td.charging.charged_illumination;
                        let contrib = (1.0 - ci) * weapon.overcharged + ci;
                        if contrib > mx {
                            mx = contrib;
                        }
                    }
                }
                // Heat illumination.
                let hi = wd.heat_illumination * weapon.heat;
                if hi > mx {
                    mx = hi;
                }
            }
            magnitude = mx;
        }

        // Cases 517/520 `primary/secondary_ammunition`. Engine:
        //   if (idx >= weapon_def.magazines.len()) → 0.0
        //   max_loaded = weapon_def.magazines[idx].rounds_loaded_maximum
        //   if (!max_loaded) → 0.0
        //   magnitude = runtime_rounds_loaded / max_loaded
        "primary_ammunition" | "secondary_ammunition" => {
            let idx = if function == "primary_ammunition" { 0 } else { 1 };
            magnitude = 0.0;
            if let Some(wd) = weapon_definition_for(weapon_index) {
                if let Some(mag_def) = wd.magazines.get(idx) {
                    let max_loaded = mag_def.rounds_loaded_maximum as f32;
                    if max_loaded > 0.0 {
                        let loaded = weapon.magazines[idx].rounds_loaded as f32;
                        magnitude = loaded / max_loaded;
                    }
                }
            }
        }

        // Case 518 `primary_ammunition_ones`. Engine:
        //   if (weapon_def.magazines.count <= 0) → 0.0
        //   else (rounds_loaded % 10) / 10
        "primary_ammunition_ones" => {
            if weapon_definition_for(weapon_index)
                .map(|wd| !wd.magazines.is_empty())
                .unwrap_or(false)
            {
                let loaded = weapon.magazines[0].rounds_loaded as i32;
                magnitude = (loaded.rem_euclid(10) as f32) * 0.1;
            }
        }

        // Case 519 `primary_ammunition_tens`. Engine:
        //   if (weapon_def.magazines.count <= 0) → 0.0
        //   else (rounds_loaded / 10 % 10) / 10
        "primary_ammunition_tens" => {
            if weapon_definition_for(weapon_index)
                .map(|wd| !wd.magazines.is_empty())
                .unwrap_or(false)
            {
                let loaded = weapon.magazines[0].rounds_loaded as i32;
                magnitude = ((loaded / 10).rem_euclid(10) as f32) * 0.1;
            }
        }

        // Cases 521/522 `primary/secondary_ejection_port`. Engine
        // gates on `idx < weapon_def.barrels.count`.
        "primary_ejection_port" | "secondary_ejection_port" => {
            let idx = if function == "primary_ejection_port" { 0 } else { 1 };
            if weapon_definition_for(weapon_index)
                .map(|wd| idx < wd.barrels.len())
                .unwrap_or(false)
            {
                magnitude = weapon.barrels[idx].ejection_port_position;
            }
        }

        // Cases 523/524 `primary/secondary_rate_of_fire`. Same gate
        // as the ejection-port cases.
        "primary_rate_of_fire" | "secondary_rate_of_fire" => {
            let idx = if function == "primary_rate_of_fire" { 0 } else { 1 };
            if weapon_definition_for(weapon_index)
                .map(|wd| idx < wd.barrels.len())
                .unwrap_or(false)
            {
                magnitude = weapon.barrels[idx].rate_of_fire;
            }
        }

        // Cases 525/526 `primary/secondary_firing`. Engine:
        //   if (idx >= weapon_def.barrels.count) → 0.0
        //   if (game_time - game_time_last_fired <= 1 frame in ticks)
        //       → barrels[idx].rate_of_fire
        //   else → 0.0
        "primary_firing" | "secondary_firing" => {
            let idx = if function == "primary_firing" { 0 } else { 1 };
            magnitude = 0.0;
            if weapon_definition_for(weapon_index)
                .map(|wd| idx < wd.barrels.len())
                .unwrap_or(false)
            {
                let threshold = game_seconds_to_ticks_real(1.0 / 30.0);
                let elapsed = game_time_get() - weapon.game_time_last_fired;
                if elapsed <= threshold {
                    magnitude = weapon.barrels[idx].rate_of_fire;
                }
            }
        }

        // Cases 527/528 `primary/secondary_charged`. Engine gates on
        // `idx < weapon_def.triggers.count` then asks the engine
        // helper `weapon_trigger_get_charged_fraction`.
        "primary_charged" | "secondary_charged" => {
            let idx = if function == "primary_charged" { 0 } else { 1 };
            if weapon_definition_for(weapon_index)
                .map(|wd| idx < wd.triggers.len())
                .unwrap_or(false)
            {
                magnitude = weapon_trigger_get_charged_fraction(weapon_index, idx as i32);
            }
        }

        // Case 529 `age` — engine helper read.
        "age" => magnitude = weapon_get_age(weapon_index),

        // Cases 531/532 `primary/secondary_trigger_down`. Engine reads
        // control_flags bits 0x02 and 0x04. Case 531 also gates on
        // weapon_barrel_can_fire.
        "primary_trigger_down" => {
            if (weapon.control_flags & 0x02) != 0
                && weapon_barrel_can_fire(weapon_index, 0, 0)
            {
                magnitude = 1.0;
            }
        }
        "secondary_trigger_down" => {
            if (weapon.control_flags & 0x04) != 0 {
                magnitude = 1.0;
            }
        }

        // Cases 533/534 `primary/secondary_barrel_firing`. Engine:
        //   if (idx >= weapon_def.barrels.count) → 0.0
        //   BYTE1(barrels[idx].state) == 1 → 1.0 else 0.0
        // The state field's "firing" sub-byte is at offset 1 of the
        // weapon_barrel runtime struct (Ares: `char state; // 0x1`).
        "primary_barrel_firing" | "secondary_barrel_firing" => {
            let idx = if function == "primary_barrel_firing" { 0 } else { 1 };
            if weapon_definition_for(weapon_index)
                .map(|wd| idx < wd.barrels.len())
                .unwrap_or(false)
                && weapon.barrels[idx].state == 1
            {
                magnitude = 1.0;
            }
        }

        // Cases 535/536/537 `primary_barrel_firing_single/_dual_left/_right`.
        // Engine resolves the holding unit's left+right weapons and
        // matches against this weapon_index to decide which firing
        // mode applies. Until unit-inventory simulation lands, mirror
        // the engine "no unit" path: case takes LABEL_98 → magnitude=0.
        "primary_barrel_firing_single"
        | "primary_barrel_firing_dual_left"
        | "primary_barrel_firing_dual_right" => magnitude = 0.0,

        // Case 577 `weapon_battery_empty` — age >= 1.0 ? 1.0 : 0.0.
        "weapon_battery_empty" => {
            if weapon_get_age(weapon_index) >= 1.0 {
                magnitude = 1.0;
            }
        }

        // Case 578 `barrel_spin` — u8 / 255.
        "barrel_spin" => magnitude = (weapon.barrel_spin as f32) * (1.0 / 255.0),

        // Case 579 `barrel_spin_rate` — barrels[0].rate_of_fire iff
        // weapon_def.barrels.count > 0.
        "barrel_spin_rate" => {
            if weapon_definition_for(weapon_index)
                .map(|wd| !wd.barrels.is_empty())
                .unwrap_or(false)
            {
                magnitude = weapon.barrels[0].rate_of_fire;
            }
        }

        // Case 738 `bomb_arming_amount` and 739/740/741 (bomb state).
        // Engine: requires multiplayer + the bomb-is-this-weapon
        // gametype predicate. Untouched-state default = 0.0 / inactive.
        "bomb_arming_amount" | "bomb_is_unarmed" | "bomb_is_arming" | "bomb_is_armed" => {
            magnitude = 0.0;
        }

        // Case 832 `warthog_chaingun_illumination` — engine recurses
        // into `weapon_compute_function_value(weapon_index,
        // "illumination", ...)` when the global "warthog chaingun
        // light enabled" toggle is on. Default toggle is on; recurse.
        "warthog_chaingun_illumination" => {
            drop(datum);
            return weapon_compute_function_value(
                weapon_index,
                "illumination",
                function_owner_definition_index,
                value,
                active,
                deterministic,
            );
        }

        // Case 973 `stowed` — `1.0 iff item_stowed`.
        "stowed" => {
            if item_stowed(weapon_index) {
                magnitude = 1.0;
            }
        }

        // Default arm — engine returns 0 (recognized=false) so the chain
        // dispatcher falls through to the next step (object_compute_function_value).
        _ => recognized = false,
    }

    // Engine LABEL_142/30 clamp: [0, 1] for positive, else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
