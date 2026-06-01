//! `_control_datum` body — no compute function exists.
//!
//! Ares source: `source/devices/devices.h:197-204` (_control_datum, 8B).
//! Engine `part_definitions[]` for control = `[object, device]` (no
//! leaf compute slot, per
//! [[reference_object_type_compute_function_chains]]). All control
//! state-reads happen at the device layer.

/// Engine `_control_datum` (8 bytes, 3 fields).
#[derive(Debug, Clone, Default)]
pub struct ControlBody {
    pub flags: u32,
    pub hud_override_index: i16,
    pub pad: u16,
}
