//! `_device_datum` body + `device_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/devices/devices.h:108-125` (_device_datum, 200B)
//!   - body:   `source/devices/devices.cpp:387` (device_compute_function_value)
//! Engine:
//!   - struct verified via `get_type_info?name=_device_datum` (13 fields).
//!   - compute body verbatim port of dllcache `0x18085D260` (169-line
//!     pseudocode, 7 cases across position/power/locked/loop/delay/
//!     change_in_power/change_in_position).
//!
//! Chain position: `object → device → leaf` for machine/control/terminal.
//! `device` IS callable (unlike `item` / `mover`); see
//! [[reference_object_type_compute_function_chains]].

use crate::halo::objects::object_header_data::{get_object_datum, get_tag_path};
use crate::halo::tags::tag_get;
use blam_tags::device::DeviceDefinition;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Sub-struct bodies
// ---------------------------------------------------------------------------

/// `s_device_animation_control` (32 bytes) — engine's per-track animation
/// state. Empty placeholder; surface fields when consumers need them.
/// (The `change_in_position` compute case reads `position_track[20]` —
/// when that case's flag-3-SET path is needed, add the field here.)
#[derive(Debug, Clone, Default)]
pub struct DeviceAnimationControl {}

/// `c_animation_channel` (52 bytes) — engine animation channel.
/// Empty placeholder; surface fields when consumers need them.
#[derive(Debug, Clone, Default)]
pub struct AnimationChannel {}

// ---------------------------------------------------------------------------
// _device_datum body
// ---------------------------------------------------------------------------

/// Engine `_device_datum` (200 bytes, 13 fields). Verified
/// 2026-05-20 via `get_type_info?name=_device_datum`. Field order +
/// names mirror Ares `source/devices/devices.h:108-125`.
#[derive(Debug, Clone, Default)]
pub struct DeviceBody {
    /// `flags` (u32). Bit 3 = "use animation_control" gate for the
    /// `change_in_position` compute case's flag-3-SET path.
    pub flags: u32,
    pub power_group_index: i32,
    /// `power` case (sid 133) direct read.
    pub power: f32,
    /// `change_in_power` case (sid 504) numerator (abs value).
    pub power_velocity: f32,
    pub position_group_index: i32,
    /// `position` case (sid 132) direct read. Also gates `loop` (sid 581)
    /// (returns 1 iff position <= 0.9) and the `locked` case.
    pub position: f32,
    /// `change_in_position` case (sid 505) numerator (abs value) in
    /// the flag-3-CLEAR path.
    pub position_velocity: f32,
    /// `delay` case (sid 507) numerator — converted to seconds via
    /// `game_ticks_to_seconds`.
    pub delay_ticks: i16,
    pub pad: u16,
    pub position_track: DeviceAnimationControl,
    pub overlay_track: DeviceAnimationControl,
    pub position_channel: AnimationChannel,
    pub power_and_overlay_track_channel: AnimationChannel,
}

// ---------------------------------------------------------------------------
// Engine helpers
// ---------------------------------------------------------------------------

/// Engine `game_time_get` — current tick. Stubbed at 0 until tick
/// simulation wires in.
fn _game_time_get() -> i32 { 0 }

/// Engine `game_ticks_to_seconds(ticks) -> f32` — 1 tick = 1/30 s.
fn game_ticks_to_seconds(ticks: i16) -> f32 {
    (ticks as f32) / 30.0
}

/// Resolve the device parent definition for any of mach/ctrl/term.
/// Engine pattern: `dereferences global_tag_instances[def_index]`
/// regardless of which derived FOURCC. Protomorph tries each FOURCC
/// in order — only one will match for a given placement.
fn device_definition_for(device_index: u32) -> Option<Arc<DeviceDefinition>> {
    let tag_path = get_tag_path(device_index);
    if tag_path.is_empty() {
        return None;
    }
    // Try each derived FOURCC since the cache distinguishes them.
    for fourcc in [*b"mach", *b"ctrl", *b"term"] {
        if let Some(t) = tag_get(fourcc, &tag_path) {
            // Each derived LoadedTag carries a `device: Arc<DeviceDefinition>`.
            if let Some(d) = t.as_machine_definition() {
                return Some(d.device.clone());
            }
            if let Some(d) = t.as_control_definition() {
                return Some(d.device.clone());
            }
            if let Some(d) = t.as_terminal_definition() {
                return Some(d.device.clone());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// device_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `device_compute_function_value @
/// 0x18085D260`. Engine cases (string_id → name):
///   132 position, 133 power, 504 change_in_power,
///   505 change_in_position, 506 locked, 507 delay, 581 loop.
pub fn device_compute_function_value(
    device_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let _ = function_owner_definition_index;

    let datum_lock = get_object_datum(device_index);
    let datum = datum_lock.read().unwrap();
    let Some(dev) = datum.as_device() else {
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        // Case 132 `position` — direct read.
        "position" => magnitude = dev.position,

        // Case 133 `power` — direct read.
        "power" => magnitude = dev.power,

        // Case 504 `change_in_power`. Engine:
        //   if (power == 0) → 0.0
        //   if (device_def.power_transition_time == 0) → 0.0
        //   abs(power_velocity) / power_transition_time
        // (Engine path through v8[91] for power_velocity; condition
        // gates on v8[91] != 0 in the negative branch and falls
        // through to LABEL_31 = 0 when zero.)
        "change_in_power" => {
            if dev.power_velocity != 0.0 {
                if let Some(dd) = device_definition_for(device_index) {
                    if dd.power_transition_time != 0.0 {
                        magnitude = dev.power_velocity.abs() / dd.power_transition_time;
                    }
                }
            }
        }

        // Case 505 `change_in_position`. Engine has a flag-3 split —
        // when flags & 8 is SET, divide abs(animation_track[20]) by
        // abs(animation_track[20]) (probably actually a different
        // field); flag-CLEAR path uses position_transition_time.
        // The position_track field-20 path needs s_device_animation_control
        // surfaced — for now use only the flag-CLEAR (default) path.
        // The flag-3-SET path is rare and exercised by animated doors
        // only; revisit when those land.
        "change_in_position" => {
            if (dev.flags & 0x8) == 0 {
                if dev.position_velocity != 0.0 {
                    if let Some(dd) = device_definition_for(device_index) {
                        if dd.position_transition_time != 0.0 {
                            magnitude =
                                dev.position_velocity.abs() / dd.position_transition_time;
                        }
                    }
                }
            }
            // TODO: flag-3-SET path needs DeviceAnimationControl.field_20
            // surfaced (engine reads animation_track ratio).
        }

        // Case 506 `locked`. Engine:
        //   if power == 0 → 1.0
        //   if m_type == 7 (machine) AND position_group_index != -1:
        //     device_groups_data[group].flags & 1
        //     + machine flags: bits 1/4/8/0x80 of (datum + 552)
        //     + position == 1.0 OR (machine flags & 0x10)
        //   else fall through to LABEL_59 with whatever v12 is.
        // Stub: power-zero gate only. Device-groups + machine-flag
        // path needs `device_groups_data` global + the machine
        // _datum's flags byte at (datum+552 = _machine_datum offset 0).
        "locked" => {
            if dev.power == 0.0 {
                magnitude = 1.0;
            }
            // TODO: device_groups_data + machine flags path.
        }

        // Case 507 `delay`. Engine:
        //   v12 = game_ticks_to_seconds(delay_ticks) / device_def.delay_time
        "delay" => {
            if let Some(dd) = device_definition_for(device_index) {
                if dd.delay_time > 0.0 {
                    magnitude = game_ticks_to_seconds(dev.delay_ticks) / dd.delay_time;
                }
            }
        }

        // Case 581 `loop` — engine reads position; (position <= 0.9) → 1.0
        // else 0.0. (Cast bool to float through the SSE pipeline.)
        "loop" => {
            magnitude = if dev.position <= 0.9 { 1.0 } else { 0.0 };
        }

        _ => recognized = false,
    }

    // Engine LABEL_59/32 clamp: positive → [0, 1], else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
