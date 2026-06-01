//! `object_compute_function_value @ dllcache 0x1807DB310` — the BASE
//! `*_compute_function_value`. Handles function names that any object
//! type can resolve (`body_vitality`, `compass`, `random_constant`,
//! `electrical_power`, `time_*`, `scripted_object_function_*`,
//! `teleporter_*`, etc.). First chain step in the engine
//! `object_type_compute_function_value` dispatcher for every type.
//!
//! Ares source: `source/objects/objects.cpp:5063`.
//!
//! Engine body: 293-line switch on `function` (a `string_id`). Each
//! recognized case sets a magnitude (`v15` in IDA), then falls into
//! the `LABEL_93` clamp:
//!
//! ```c
//! LABEL_93:
//!   if (v15 > 0.0) { if (v15 >= 1.0) v15 = 1.0; }
//!   else            v15 = 0.0;
//!   *value = v15;
//!   *active = v15 > 0.0;
//!   return v14;          // 1 = recognized, 0 = unknown (LABEL_82)
//! ```
//!
//! Three semantic outputs:
//! - **recognized AND value > 0** → return `true`, `*active = true`
//! - **recognized AND value <= 0** → return `true`, `*active = false`
//!   (e.g. case `zero` / case `current_body_damage` on untouched object)
//! - **unrecognized** → return `false`, `*active = false`, fall
//!   through to the next chain step (or LABEL_82 in the walker)
//!
//! ## Protomorph static-placement defaults
//!
//! Engine cases read live object_datum state (vitality, damage, phantom
//! input, etc.). Protomorph doesn't simulate any of that yet — every
//! placement is a freshly-spawned, untouched, full-health, idle object.
//! Each case below returns the engine-faithful default value for that
//! initial state, with the IDA field offset documented so the read
//! becomes a one-line edit once `ObjectDatum` lands.

use crate::halo::objects::object_header_data::{get_object_datum, get_object_definition};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cases marked `goto $LN84_6` in the engine ⇒ always-1.0 outputs
/// (`one`, `electrical_power`, `morse_code`). The clamp at LABEL_93
/// reproduces the same 1.0 result for any caller via `v15 = FLOAT_1_0`.
const ALWAYS_ON_NAMES: &[&str] = &[
    "one",            // engine sid 574 — `object_get_function_value` shortcircuits before reaching us
    "electrical_power", // engine sid 714 — always-on power signal
    "morse_code",     // engine sid 719 — always-on signal (engine returns 1.0)
];

pub fn object_compute_function_value(
    object_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    deterministic: &mut bool,
) -> bool {
    let _ = function_owner_definition_index;

    // Engine `v15` — the magnitude before the LABEL_93 clamp.
    let mut magnitude: f32;
    // Engine `v14` — the `recognized` return. Starts true; cleared by the
    // default arm so the chain dispatcher knows to try the next step.
    let mut recognized = true;

    // Engine reads from `(char*)object_header_data->data[idx].datum →
    // object_datum.object` for the per-object state. Protomorph mirror:
    // get_object_datum looks up the same per-placement struct.
    let datum_lock = get_object_datum(object_index);
    let datum = datum_lock.read().unwrap();
    let obj = &datum.object;

    match function {
        // -- damage / vitality state (engine struct `object_datum::object`) --
        "current_body_damage"      => magnitude = obj.current_body_damage,    // case 494
        "current_shield_damage"    => magnitude = obj.current_shield_damage,  // case 495
        "body_vitality"            => magnitude = obj.body_vitality,          // case 496
        "shield_vitality"          => magnitude = obj.shield_vitality,        // case 497
        "object_overshield_amount" => magnitude = obj.shield_vitality - 1.0,  // case 498

        // -- always-on signal cases ($LN84_6 in IDA) --
        n if ALWAYS_ON_NAMES.contains(&n) => magnitude = 1.0,

        // -- always-off shortcut (engine sid 575 → $LN197_0) --
        "zero" => magnitude = 0.0,

        // -- alive flag (engine sid 499) --
        // Engine: `(damage_flags & 4) == 0` means alive (the `& 4` bit is
        // the "killed" flag). Returns 1.0 when CLEAR.
        "alive" => magnitude = if (obj.damage_flags & 0x4) == 0 { 1.0 } else { 0.0 },

        // -- random per-object hash, pure compute (engine sid 580) --
        // Engine: v20 = (float)((1664525u32 * object_index + 1013904223u32) >> 16);
        //        v15 = v20 * 0.000015259022;   (= 1.0/65535.0)
        "random_constant" => {
            let hashed = 1664525u32
                .wrapping_mul(object_index)
                .wrapping_add(1013904223)
                >> 16;
            magnitude = (hashed as f32) * (1.0 / 65535.0);
        }

        // -- compass — engine sid 500 --
        // Engine: `object_get_node_matrix(object_index, 0)` returns
        // the root-node matrix; row 0 is the forward vector. If
        // `|forward.k| >= 0.995` the heading is degenerate (facing
        // straight up/down) and the engine takes the $LN197_0 path
        // (= 0.0). Otherwise:
        //   heading = atan2f(forward.i, forward.j);
        //   diff = signed_angular_difference(global_scenario->local_north, heading);
        //   if (diff < 0) diff += 2*PI;
        //   result = diff / (2*PI);   // normalized to [0, 1]
        //
        // Protomorph reads `forward` straight off the datum
        // (`ObjectDatum::from_placement` populates it via the engine
        // euler→matrix math). `local_north` comes from the
        // `global_scenario` Arc.
        "compass" => {
            if obj.forward.k.abs() >= 0.995 {
                magnitude = 0.0;
            } else {
                let heading = obj.forward.i.atan2(obj.forward.j);
                let local_north =
                    crate::halo::scenario::globals::get_local_north();
                let mut diff = signed_angular_difference(local_north, heading);
                if diff < 0.0 {
                    diff += std::f32::consts::TAU;
                }
                magnitude = diff / std::f32::consts::TAU;
            }
        }

        // -- variant — engine sid 501 --
        // Engine: variant_count = model.variants[].len();
        //   result = clamp(variant_index + 1, 0, count) / count
        // The model is reached via the object's tag → object_definition
        // .model (path) → tag_get(b"hlmt", model_path) → LoadedModel.
        // Returns 0.0 when no model is bound or it has no variants.
        "variant" => {
            let count: usize = {
                let obj_def = get_object_definition(object_index);
                if obj_def.model.is_empty() {
                    0
                } else {
                    crate::halo::tags::tag_get(*b"hlmt", &obj_def.model)
                        .and_then(|t| t.as_model())
                        .map(|m| m.variant_names.len())
                        .unwrap_or(0)
                }
            };
            if count == 0 {
                magnitude = 0.0;
            } else {
                let idx_plus_one =
                    (obj.variant_index as i32 + 1).max(0).min(count as i32) as f32;
                magnitude = idx_plus_one / count as f32;
            }
        }

        // -- shield_depleted — engine sid 582 --
        // Engine: returns 1.0 only when all three conditions hold:
        //   maximum_shield_vitality > 0 &&
        //   shield_recharge_rate_percentage > 0 (MP-only; default 1.0) &&
        //   shield_vitality == 0.0
        // Recharge-rate plumbing is MP-only and not yet wired — assume
        // 1.0 (engine default for non-MP). The other two come from datum.
        "shield_depleted" => {
            let max_shield = obj.maximum_shield_vitality;
            let recharge = 1.0f32;
            magnitude = if max_shield > 0.0 && recharge > 0.0 && obj.shield_vitality == 0.0 {
                1.0
            } else {
                0.0
            };
        }

        // -- phantom input (vehicle phantom — Halo MP gametype) --
        // Engine fields:
        //   case 682 → object_datum->object.phantom_speed_scale (u8) / 255
        //   case 683 → object_datum->object.phantom_power_scale (u8) / 255
        //   case 684 → object_datum->object.physics_flags.m_flags & 0x20000
        "phantom_speed" => magnitude = obj.phantom_speed_scale as f32 / 255.0,
        "phantom_power" => magnitude = obj.phantom_power_scale as f32 / 255.0,
        "phantom_on"    => magnitude = if (obj.physics_flags & 0x20000) != 0 { 1.0 } else { 0.0 },

        // -- jetpack_exhaust — engine sid 685 --
        // Engine: 1.0 only when `(1 << object_identifier.m_type) & 1`
        // (m_type == biped) AND `biped_get_gravity_scale(...) < 1.0`.
        // Gravity-scale needs biped runtime state we don't simulate.
        // Default: no jetpack burning → 0.0.
        "jetpack_exhaust" => magnitude = 0.0,

        // -- wall-clock time helpers (deterministic=false per engine) --
        "time_hours" => {
            magnitude = wall_clock_hours();
            *deterministic = false;
        }
        "time_minutes" => {
            magnitude = wall_clock_minutes();
            *deterministic = false;
        }
        "time_seconds" => {
            magnitude = wall_clock_seconds();
            *deterministic = false;
        }
        "holiday" => {
            // Engine: compute_holiday() — checks scenario calendar against
            // wall-clock date. Default: not a holiday. Marks non-deterministic.
            magnitude = 0.0;
            *deterministic = false;
        }

        // -- environmental scene state --
        "all_quiet"           => magnitude = 0.0,    // engine compute_all_quiet()
        "flashlights_allowed" => magnitude = 0.0,    // engine game_flashlight_magnitude()

        // -- scripted globals (HSC-driven, default 0 in unscripted scenes) --
        "scripted_object_function_a"
        | "scripted_object_function_b"
        | "scripted_object_function_c"
        | "scripted_object_function_d" => magnitude = 0.0,

        // -- MP gametype state --
        "oddball_effect" | "mp_goal_active" => magnitude = 0.0,

        // -- teleporter state (MP) --
        "teleporter_active"
        | "teleporter_blocked"
        | "teleporter_cannot_send"
        | "teleporter_recently_teleported" => magnitude = 0.0,

        // -- default arm (engine LABEL_82) --
        // Engine logs a c_event ("objects:function: object '%s' failed
        // to find function '%s'") under `(v13[1] & 4) == 0`, sets v14=0,
        // falls into $LN197_0 (magnitude=0). The chain dispatcher checks
        // the false return to try the next chain step.
        _ => {
            magnitude = 0.0;
            recognized = false;
            let _ = object_index;
        }
    }

    // Engine LABEL_93 clamp: [0, 1] for positive, else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}

/// Wall-clock helper for engine `compute_time_hours`. Returns
/// `hours_of_day / 24` (so the input cycles through [0, 1) over a day).
fn wall_clock_hours() -> f32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Local-day seconds approximation — engine uses local time but a
    // UTC approximation is fine for the curve sample (the difference
    // is a phase shift the artist can't observe).
    let day_secs = secs % 86_400;
    (day_secs / 3600) as f32 / 24.0
}

/// Wall-clock helper for engine `compute_time_minutes`. Returns
/// `minutes_of_hour / 60`.
fn wall_clock_minutes() -> f32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let m = (secs % 3600) / 60;
    m as f32 / 60.0
}

/// Wall-clock helper for engine `compute_time_seconds`. Returns
/// `seconds_of_minute / 60`.
fn wall_clock_seconds() -> f32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (secs % 60) as f32 / 60.0
}

/// Engine `signed_angular_difference(a, b)` — returns the shortest
/// signed angular delta from `a` to `b`, wrapped into `(-π, π]`. Used
/// by `object_compute_function_value` case `compass` to compute the
/// heading offset against `global_scenario->local_north`.
fn signed_angular_difference(a: f32, b: f32) -> f32 {
    let mut d = b - a;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d <= -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}
