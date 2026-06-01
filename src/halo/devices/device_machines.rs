//! `_machine_datum` body + `machine_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/devices/devices.h:134-144` (_machine_datum, 16B)
//!   - body:   `source/devices/device_machines.cpp:390`
//! Engine:
//!   - struct verified via `get_type_info?name=_machine_datum` (6 fields).
//!   - compute body: dllcache `0x18087D690` is a single-line
//!     `return 0` (engine has no machine-specific cases — chain
//!     dispatcher falls through to `device_compute_function_value`).

/// Engine `_machine_datum` (16 bytes, 6 fields). Ares
/// `source/devices/devices.h:134-144`.
#[derive(Debug, Clone, Default)]
pub struct MachineBody {
    /// `flags` — machine-specific (door-open bits etc.). Engine reads
    /// some bits from this in the `locked` compute case at the device
    /// layer.
    pub flags: u32,
    pub door_open_ticks: i32,
    pub pathfinding_policy: i16,
    pub door_machine_first_portal_index: i16,
    pub door_machine_portal_index_count: i16,
    pub door_machine_portal_structure_bsp_index: i16,
}

/// Engine `machine_compute_function_value @ 0x18087D690` is a no-op
/// in dllcache (`return 0;`). The chain dispatcher therefore falls
/// through to `device_compute_function_value` for every input —
/// machines have no machine-specific cases at the leaf level.
pub fn machine_compute_function_value(
    _machine_index: u32,
    _function: &str,
    _function_owner_definition_index: u32,
    _value: &mut f32,
    _active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    false
}
