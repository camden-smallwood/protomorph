//! `_terminal_datum` body + `device_terminal_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/devices/devices.h:154-187` (_terminal_datum, 120B)
//!   - body:   `source/devices/device_terminals.cpp:535`
//! Engine:
//!   - struct verified via `get_type_info?name=_terminal_datum` (29 fields).
//!   - compute body verbatim port of dllcache `0x180886950` (79-line
//!     pseudocode, 11 cases — all sid 1002-1012 terminal-specific).
//!
//! Chain position: `object → device → terminal`. Many compute cases
//! fall through to the device layer; terminal-specific cases live here.

use crate::halo::objects::object_header_data::get_object_datum;

// ---------------------------------------------------------------------------
// _terminal_datum body
// ---------------------------------------------------------------------------

/// Engine `_terminal_datum` (120 bytes, 29 fields). Verified
/// 2026-05-20 via `get_type_info?name=_terminal_datum`. Field order +
/// names mirror Ares `source/devices/devices.h:154-187`.
#[derive(Debug, Clone, Default)]
pub struct TerminalBody {
    /// `c_flags<e_terminal_flags, i16, 4>` — runtime terminal flags.
    pub flags: i16,
    /// `c_enum<e_terminal_state, i16, 0, 3>` — state machine
    /// (closed/opening/open/closing).
    pub state: i16,
    pub state_timer: i16,
    /// `c_enum<e_terminal_screen_state, i16, 0, 6>` — screen content.
    pub screen_state: i16,
    pub screen_state_timer: i16,
    pub reader_unit_index: i32,
    /// `terminal_error` case (sid 1002) — direct read.
    pub error_value: f32,
    /// `activation` case (sid 1003) — direct read. Also gates
    /// `hex_intensity` (sid 1008) which takes min(error_value, this).
    pub activation_value: f32,
    /// `frame_color` case (sid 1011) — direct read.
    pub frame_color_value: f32,
    /// `scroll_velocity` vector (8 bytes — x and y). The
    /// `scroll_speed` case (sid 1012) reads `sqrt(x*x + y*y) / 6`.
    pub scroll_velocity_x: f32,
    pub scroll_velocity_y: f32,
    /// `scroll_position` point — drives `layer0_x` / `layer0_y`
    /// (sids 1004 / 1005).
    pub scroll_position_x: f32,
    pub scroll_position_y: f32,
    /// `reading_mode` case (sid 1010) — direct read.
    pub reading_mode: f32,
    pub desired_reading_mode: bool,
    pub reading_mode_transition_ticks: i16,
    pub reading_mode_max_transition_ticks: i16,
    /// `overlay_intensity` case (sid 1009) — direct read.
    pub overlay_intensity: f32,
    /// `hex_intensity` case (sid 1008) raw value; engine takes
    /// `min(activation_value, this)`.
    pub hex_intensity: f32,
    pub currently_seeking: bool,
    pub finished_seek: bool,
    pub seek_transition_ticks: i16,
    pub seek_transition_max_ticks: i16,
    pub seek_ticks: i16,
    pub seek_desired_velocity_x: f32,
    pub seek_desired_velocity_y: f32,
    pub seek_original_velocity_x: f32,
    pub seek_original_velocity_y: f32,
    pub seek_transitioning_to_zero: bool,
    pub initial_position_x: f32,
    pub initial_position_y: f32,
    /// Bezier point B (vector2d).
    pub bezier_b_x: f32,
    pub bezier_b_y: f32,
    /// Bezier point C (vector2d).
    pub bezier_c_x: f32,
    pub bezier_c_y: f32,
    pub can_exit: bool,
}

// ---------------------------------------------------------------------------
// device_terminal_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Globals `layer1_x_value` and `layer1_y_value` are HUD-driven
/// floats updated by the terminal screen renderer. Engine reads them
/// in the `layer1_x` / `layer1_y` compute cases. Stubbed at 0.0
/// until terminal UI rendering wires them in.
fn layer1_x_value() -> f32 { 0.0 }
fn layer1_y_value() -> f32 { 0.0 }

/// Verbatim port of dllcache `device_terminal_compute_function_value
/// @ 0x180886950`. All cases are sid 1002..1012.
pub fn device_terminal_compute_function_value(
    terminal_index: u32,
    function: &str,
    function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let _ = function_owner_definition_index;

    let datum_lock = get_object_datum(terminal_index);
    let datum = datum_lock.read().unwrap();
    let Some(term) = datum.as_terminal() else {
        return false;
    };

    let mut magnitude: f32 = 0.0;
    let mut recognized = true;

    match function {
        "terminal_error"    => magnitude = term.error_value,         // sid 1002
        "activation"        => magnitude = term.activation_value,    // sid 1003
        "layer0_x"          => magnitude = term.scroll_position_x,   // sid 1004
        "layer0_y"          => magnitude = term.scroll_position_y,   // sid 1005
        "layer1_x"          => magnitude = layer1_x_value(),         // sid 1006
        "layer1_y"          => magnitude = layer1_y_value(),         // sid 1007
        // sid 1008 — engine: `v9 = hex_intensity; if (activation_value <= v9) v9 = activation_value;`
        // i.e. `min(activation_value, hex_intensity)`.
        "hex_intensity"     => magnitude = term.hex_intensity.min(term.activation_value),
        "overlay_intensity" => magnitude = term.overlay_intensity,   // sid 1009
        "reading_mode"      => magnitude = term.reading_mode,        // sid 1010
        "frame_color"       => magnitude = term.frame_color_value,   // sid 1011
        // sid 1012 — engine: sqrt(vx*vx + vy*vy) * (1/6).
        "scroll_speed" => {
            let vx = term.scroll_velocity_x;
            let vy = term.scroll_velocity_y;
            magnitude = (vx * vx + vy * vy).sqrt() / 6.0;
        }
        _ => recognized = false,
    }

    // Engine LABEL_22 clamp: positive → [0, 1], else 0.
    let clamped = if magnitude > 0.0 {
        magnitude.min(1.0)
    } else {
        0.0
    };
    *value = clamped;
    *active = clamped > 0.0;
    recognized
}
