//! `_biped_datum` body + `biped_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/units/bipeds.h:150-196` (_biped_datum, 620B)
//!   - body:   `source/units/bipeds.cpp:647`
//! Engine:
//!   - struct verified via `get_type_info?name=_biped_datum` (43 fields).
//!   - compute body verbatim port of dllcache `0x180834450` (88-line
//!     pseudocode, 4 cases — `stunned`, `flight`, `combat_status`,
//!     `flying_speed`).
//!
//! Chain position: `object → unit → biped`. Reads `_biped_datum`
//! fields + engine helpers (`motor_system_stunned`,
//! `biped_get_movement_type`) + biped's `character_physics_flying.max_velocity`
//! authored field.

use crate::halo::objects::object_header_data::{get_object_datum, get_tag_path};
use crate::halo::tags::tag_get;

/// Walked subset of engine `_biped_datum` (620 bytes, 43 fields).
/// Field set today is minimal — only the byte read by compute
/// (`ai_combat_status`). Add fields as consumers come online.
///
/// Ares: `source/units/bipeds.h:150-196`.
#[derive(Debug, Clone, Default)]
pub struct BipedBody {
    /// `ai_combat_status` (char @ offset 619). Read by compute case
    /// `combat_status` (sid 482) — engine scales by 0.1.
    pub ai_combat_status: i8,
}

// ---------------------------------------------------------------------------
// Engine helpers
// ---------------------------------------------------------------------------

/// Engine `motor_system_stunned(biped_index) -> bool`. Returns true
/// when the biped is in a stunned animation state. Stubbed false
/// (untouched units aren't stunned).
fn motor_system_stunned(_biped_index: u32) -> bool {
    false
}

/// Engine `biped_get_movement_type(biped_index) -> u32` —
/// `e_biped_movement_type` enum (walking=0, crouching=1, flying=2,
/// other states). Stubbed at 0 (walking).
fn biped_get_movement_type(_biped_index: u32) -> u32 {
    0
}

// ---------------------------------------------------------------------------
// biped_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `biped_compute_function_value @ 0x180834450`.
/// 4 cases (engine sids):
///   186 `stunned`, 476 `flight`, 482 `combat_status`, 570 `flying_speed`.
pub fn biped_compute_function_value(
    biped_index: u32,
    function: &str,
    _function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let datum_lock = get_object_datum(biped_index);
    let datum = datum_lock.read().unwrap();
    let Some(biped) = datum.as_biped() else {
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        // Case 186 `stunned` — 1.0 iff motor_system_stunned, else
        // engine returns 0 (function_value = false), i.e. unrecognized.
        // We mirror by returning false when not stunned (chain
        // continues); 1.0 + recognized when stunned.
        "stunned" => {
            if motor_system_stunned(biped_index) {
                magnitude = 1.0;
            } else {
                return false;
            }
        }

        // Case 476 `flight` — 1.0 iff movement_type == flying (2).
        "flight" => {
            if biped_get_movement_type(biped_index) == 2 {
                magnitude = 1.0;
            }
        }

        // Case 482 `combat_status` — ai_combat_status × 0.1.
        "combat_status" => {
            magnitude = (biped.ai_combat_status as f32) * 0.1;
        }

        // Case 570 `flying_speed` — sqrt(|translational_velocity|²)
        // / biped_def.physics.flying.max_velocity.
        // Engine gates on max_velocity > 0 (untouched bipeds at zero
        // velocity → result 0 regardless).
        "flying_speed" => {
            let v = datum.object.translational_velocity;
            let speed_sq = v.i * v.i + v.j * v.j + v.k * v.k;
            let tag_path = get_tag_path(biped_index);
            let max_velocity = tag_get(*b"bipd", &tag_path)
                .and_then(|t| t.as_biped_definition())
                .map(|bd| bd.physics.flying.max_velocity)
                .unwrap_or(0.0);
            if max_velocity > 0.0 {
                magnitude = speed_sq.sqrt() / max_velocity;
            }
        }

        _ => recognized = false,
    }

    // Engine LABEL_18/25 clamp.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
