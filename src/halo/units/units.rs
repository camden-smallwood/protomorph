//! `_unit_datum` body + `unit_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/units/units.h:179-292` (_unit_datum, 732B)
//!   - body:   `source/units/units.cpp:802` (unit_compute_function_value)
//! Engine:
//!   - struct verified via `get_type_info?name=_unit_datum` (109 fields).
//!   - compute body verbatim port of dllcache `0x1807EE290`
//!     (242-line pseudocode). ~20 cases covering seat/aiming/throttle/
//!     mouth/illumination/camo/EMP/open-amount/emblem/weapon-forwarding.
//!
//! Chain position: `object → unit → biped/vehicle/giant`. unit_compute
//! handles most state shared across the 3 derived types; leaf computes
//! handle type-specific cases. The `weapon_illumination` / `barrel_spin`
//! family (sids 576/578/579) is forwarded to the currently-held weapon's
//! compute.

use blam_tags::math::RealVector3d;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::halo::objects::object_datum::ObjectDatum;
use crate::halo::objects::object_header_data::get_object_datum;

// ---------------------------------------------------------------------------
// Helper substructs
// ---------------------------------------------------------------------------

/// Engine `s_unit_weapon_set` (4 bytes) — references into the unit's
/// runtime `weapon_object_indices[]` table. Used by compute to find
/// the currently-held weapon for `weapon_illumination` / `barrel_spin*`
/// forwarding.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitWeaponSet {
    pub set_identifier: i16,
    /// Two `char` indices (one per weapon-slot in this set).
    pub weapon_indices: [i8; 2],
}

// ---------------------------------------------------------------------------
// _unit_datum body
// ---------------------------------------------------------------------------

/// Walked subset of engine `_unit_datum` (732 bytes, 109 fields).
/// Currently surfaces the 14 fields read by
/// `unit_compute_function_value` — the rest (AI/animation/seat-state)
/// stays deferred until a consumer needs it. Field names + offsets
/// mirror Ares `source/units/units.h:179-292`.
#[derive(Debug, Clone, Default)]
pub struct UnitBody {
    /// `flags` (engine `c_flags<e_unit_flags, u32, 31>` @ offset 8).
    /// Bit 0x2 used by `driver_seat_occupied` case (sid 568).
    /// Bit 0x20000 used by `can_blink` case (sid 567).
    pub flags: u32,
    /// `throttle` (engine `real_vector3d` @ offset 132). The z
    /// component (`throttle.k`) is read by `vertical_thrust` (sid 722).
    pub throttle: RealVector3d,
    /// `horizontal_aiming_change` (u8 @ offset 215). Read by sid 564
    /// (`horizontal_aiming_change`) — divided by 255.
    pub horizontal_aiming_change: u8,
    /// `mouth_aperture` (f32 @ offset 228). Direct read by sid 565.
    pub mouth_aperture: f32,
    /// `current_weapon_set` (s_unit_weapon_set @ offset 236). Engine
    /// reads `weapon_indices[0]` for the weapon-forwarding cases.
    pub current_weapon_set: UnitWeaponSet,
    /// `aiming_change` (u8 @ offset 295). Read by sid 563
    /// (`aiming_change`) — divided by 255.
    pub aiming_change: u8,
    /// `motion_control_unit_index` (i32 @ offset 300). -1 = no driver.
    /// Used by sid 568 (`driver_seat_occupied`).
    pub motion_control_unit_index: i32,
    /// `weapon_control_unit_index` (i32 @ offset 304). -1 = no gunner.
    /// Used by sid 569 (`gunner_seat_occupied`).
    pub weapon_control_unit_index: i32,
    /// `seat_power_valid_flags` (u8 @ offset 319). Bit 1 = driver
    /// seat, bit 2 = gunner seat. Used by sids 561/562.
    pub seat_power_valid_flags: u8,
    /// `seat_power[2]` (f32[2] @ offset 320). Used by sids 561/562.
    pub seat_power: [f32; 2],
    /// `integrated_light_power` (f32 @ offset 328). Read by sid 566.
    pub integrated_light_power: f32,
    /// `active_camouflage` (f32 @ offset 492). Read by sid 142
    /// (`active_camouflage`) with optional debug frequency modulator.
    pub active_camouflage: f32,
    /// `emp_timer` (i16 @ offset 510). 0 = no EMP active. Read by
    /// sid 193 (`emp_disabled`).
    pub emp_timer: i16,
    /// `crouch` (f32 @ offset 516). Direct read by sid 141.
    pub crouch: f32,
}

// ---------------------------------------------------------------------------
// Engine helpers (stubbed against runtime state we don't simulate)
// ---------------------------------------------------------------------------

/// Engine `unit_get_open_amount(unit_index) -> f32` — reads from
/// animation state. Stubbed at 0 (units are "closed" by default).
fn unit_get_open_amount(_unit_index: u32) -> f32 {
    0.0
}

/// Engine `unit_compute_boost_fraction(unit_index, ...)`. Stubbed
/// at 0 (no boost active for untouched units).
fn unit_compute_boost_fraction(_unit_index: u32) -> f32 {
    0.0
}

/// Engine `unit_get_weapon(unit_datum, weapon_set_index)` — looks up
/// `weapon_object_indices[weapon_set_index]`. Returns -1 for
/// invalid sets (no weapon held). Stubbed at -1 until inventory
/// simulation lands.
fn unit_get_weapon(_datum: &ObjectDatum, _weapon_set_index: i8) -> i32 {
    -1
}

/// Engine `game_engine_object_get_emblem_player(unit_index) -> i32`.
/// Returns -1 for non-player units (which is everyone in MP scenarios
/// at scenario-load time).
fn game_engine_object_get_emblem_player(_unit_index: u32) -> i32 {
    -1
}

/// Engine `player_function_get_function_value(player_index, function,
/// out_value, out_active) -> bool`. Returns false for invalid players;
/// stubbed at false.
fn player_function_get_function_value(
    _player_index: i32,
    _function: &str,
    _value: &mut f32,
    _active: &mut bool,
) -> bool {
    false
}

/// Engine debug knob (typically 0). When non-zero, modulates
/// active_camouflage by a cosine over game-time. Stubbed at 0.
const DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR: f32 = 0.0;
/// Companion bias knob; engine reads when modulator > 0.
const DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR_BIAS: f32 = 0.0;

/// Engine `game_time_get` — current tick. 0 until simulation wires.
fn game_time_get() -> i32 {
    static T: AtomicI32 = AtomicI32::new(0);
    T.load(Ordering::Relaxed)
}

/// Engine `game_ticks_to_seconds(ticks)` — 1/30 s per tick.
fn game_ticks_to_seconds(ticks: i32) -> f32 {
    (ticks as f32) / 30.0
}

// ---------------------------------------------------------------------------
// unit_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `unit_compute_function_value @
/// 0x1807EE290`. Cases covered (sid → name):
///   141 crouch, 142 active_camouflage, 193 emp_disabled,
///   541 boost, 561 driver_seat_power, 562 gunner_seat_power,
///   563 aiming_change, 564 horizontal_aiming_change,
///   565 mouth_aperture, 566 integrated_light_power,
///   567 can_blink, 568 driver_seat_occupied,
///   569 gunner_seat_occupied, 708 unit_open, 709 unit_closed,
///   722 vertical_thrust, 576/578/579 weapon-forward (illumination/
///   barrel_spin/barrel_spin_rate).
/// Emblem cases (628-636) require player-emblem state we don't
/// simulate — stubbed to return false (chain falls through).
pub fn unit_compute_function_value(
    unit_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    let datum_lock = get_object_datum(unit_index);
    let datum = datum_lock.read().unwrap();
    let Some(unit) = datum.as_unit() else {
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        // Case 141 `crouch` — direct read.
        "crouch" => magnitude = unit.crouch,

        // Case 142 `active_camouflage` — direct read with optional
        // debug modulation. Engine: if camo <= 0 → 0; else apply
        // cosine modulation when debug knob > 0.
        "active_camouflage" => {
            magnitude = unit.active_camouflage;
            if magnitude > 0.0 && DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR != 0.0 {
                let t_sec = game_ticks_to_seconds(game_time_get());
                let modulator = (1.0
                    - (t_sec
                        * std::f32::consts::TAU
                        * DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR)
                        .cos())
                    * 0.5;
                magnitude *= modulator;
                if DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR_BIAS > 0.0 {
                    magnitude = (magnitude * 2.0 - 1.0)
                        * DEBUG_UNIT_ACTIVE_CAMO_FREQUENCY_MODULATOR_BIAS
                        + magnitude;
                }
            }
        }

        // Case 193 `emp_disabled` — engine: `emp_timer == 0 → 0 else 1`.
        "emp_disabled" => {
            magnitude = if unit.emp_timer != 0 { 1.0 } else { 0.0 };
        }

        // Case 541 `boost` — engine helper.
        "boost" => magnitude = unit_compute_boost_fraction(unit_index),

        // Cases 561/562 — driver/gunner seat power.
        "driver_seat_power" => {
            if (unit.seat_power_valid_flags & 1) != 0 {
                magnitude = unit.seat_power[0];
            }
        }
        "gunner_seat_power" => {
            if (unit.seat_power_valid_flags & 2) != 0 {
                magnitude = unit.seat_power[1];
            }
        }

        // Cases 563/564 — aiming change normalized to [0, 1].
        "aiming_change" => {
            magnitude = (unit.aiming_change as f32) * (1.0 / 255.0);
        }
        "horizontal_aiming_change" => {
            magnitude = (unit.horizontal_aiming_change as f32) * (1.0 / 255.0);
        }

        // Case 565 `mouth_aperture` — direct read.
        "mouth_aperture" => magnitude = unit.mouth_aperture,

        // Case 566 `integrated_light_power` — direct read.
        "integrated_light_power" => magnitude = unit.integrated_light_power,

        // Case 567 `can_blink` — engine: 1.0 if killed flag OR unit
        // flag 0x20000 set, else 0.
        "can_blink" => {
            let killed = (datum.object.damage_flags & 0x4) != 0;
            let unit_bit = (unit.flags & 0x20000) != 0;
            magnitude = if killed || unit_bit { 1.0 } else { 0.0 };
        }

        // Case 568 `driver_seat_occupied`. Engine:
        //   if motion_control_unit_index != -1 → 1.0
        //   else if !(flags & 2) → 0
        //   else → 1.0
        "driver_seat_occupied" => {
            if unit.motion_control_unit_index != -1 {
                magnitude = 1.0;
            } else if (unit.flags & 0x2) != 0 {
                magnitude = 1.0;
            }
        }

        // Case 569 `gunner_seat_occupied`. Engine:
        //   weapon_control_unit_index != -1 → 1.0 else 0.0
        "gunner_seat_occupied" => {
            magnitude = if unit.weapon_control_unit_index != -1 { 1.0 } else { 0.0 };
        }

        // Cases 708/709 — engine animation helpers.
        "unit_open" => magnitude = unit_get_open_amount(unit_index),
        "unit_closed" => magnitude = 1.0 - unit_get_open_amount(unit_index),

        // Case 722 `vertical_thrust` — throttle z component.
        "vertical_thrust" => magnitude = unit.throttle.k,

        // Emblem cases (628-636) — engine reads from
        // `game_engine_object_get_emblem_player(unit) → s_emblem_info`.
        // Untouched units have no emblem player → engine LABEL_69
        // path which forwards to weapon compute. We mirror by
        // returning false (chain proceeds).
        "emblem_active"
        | "emblem_foreground_index"
        | "emblem_background_index"
        | "emblem_alternate_foreground_enabled"
        | "emblem_flip_foreground_image"
        | "emblem_flip_background_image"
        | "emblem_primary_color_index"
        | "emblem_secondary_color_index"
        | "emblem_background_color_index" => {
            let _ = game_engine_object_get_emblem_player(unit_index);
            // Stub: defer to player_function_get_function_value (also
            // stubbed → false). Drop into the recognized=false arm so
            // the chain dispatcher tries the next step.
            recognized = false;
        }

        // Weapon-forwarding cases — engine remaps the sid to the
        // weapon-side compute and calls weapon_compute_function_value
        // on the current weapon.
        "weapon_illumination" | "barrel_spin" | "barrel_spin_rate" => {
            let weapon_sid = match function {
                "weapon_illumination" => "illumination",
                "barrel_spin" => "barrel_spin",
                _ => "barrel_spin_rate",
            };
            let weapon = unit_get_weapon(&datum, unit.current_weapon_set.weapon_indices[0]);
            if weapon == -1 {
                // No weapon held — engine returns function_value
                // (which is false at this point in the engine).
                return false;
            }
            drop(datum);
            return crate::halo::items::weapons::weapon_compute_function_value(
                weapon as u32,
                weapon_sid,
                function_owner_definition_index,
                value,
                active,
                deterministic,
            );
        }

        _ => recognized = false,
    }

    // Engine LABEL_47 final clamp: positive → [0, 1], else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
