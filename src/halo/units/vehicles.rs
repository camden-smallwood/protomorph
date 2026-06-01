//! `_vehicle_datum` body + `vehicle_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/units/vehicles.h:63-105` (_vehicle_datum, 828B)
//!   - body:   `source/units/vehicles.cpp:927`
//! Engine:
//!   - struct verified via `get_type_info?name=_vehicle_datum` (38 fields).
//!   - compute body verbatim port of dllcache `0x18082FF30` (132-line
//!     pseudocode). Dispatches first to
//!     `c_vehicle_type_component::compute_function_value`, then falls
//!     through to direct cases (`jump`, `boost`, `wingtip_contrail`,
//!     `mean_antigrav`, and 3 action cases).
//!
//! Chain position: `object → unit → vehicle`.

use crate::halo::objects::object_header_data::get_object_datum;
use blam_tags::math::RealVector3d;

// ---------------------------------------------------------------------------
// _vehicle_datum body
// ---------------------------------------------------------------------------

/// `c_vehicle_type_component` (100 bytes) — engine vehicle-type
/// dispatch component (chopper/banshee/warthog/etc.). Stub
/// placeholder until per-type behavior lands.
#[derive(Debug, Clone, Default)]
pub struct VehicleTypeComponent {}

/// Walked subset of engine `_vehicle_datum` (828 bytes, 38 fields).
/// Field set today is minimal — only the fields read by compute
/// (`mean_antigrav_fraction` and the type_component placeholder).
///
/// Ares: `source/units/vehicles.h:63-105`.
#[derive(Debug, Clone, Default)]
pub struct VehicleBody {
    /// `flags` (engine `c_flags<e_vehicle_flags, u16, 10>` @ offset 0).
    pub flags: u16,
    /// `mean_antigrav_fraction` (f32 @ offset 12). Read by compute
    /// case `mean_antigrav` (sid 560).
    pub mean_antigrav_fraction: f32,
    /// `type_component` (c_vehicle_type_component @ offset 16). Engine
    /// dispatches compute cases through this before falling through
    /// to direct case handling.
    pub type_component: VehicleTypeComponent,
}

// ---------------------------------------------------------------------------
// Engine helpers
// ---------------------------------------------------------------------------

/// Engine `c_vehicle_type_component::compute_function_value` — vehicle-
/// type-specific dispatch. Returns true when the type handled the
/// case (writes magnitude + force_active). Stubbed false to defer to
/// direct cases until per-type-component bodies land.
fn vehicle_type_component_compute(
    _component: &VehicleTypeComponent,
    _vehicle_index: u32,
    _function: &str,
    _magnitude: &mut f32,
    _force_active: &mut bool,
) -> bool {
    false
}

/// Engine `action_executing(vehicle_index, action) -> bool` — checks
/// if a specific action animation/state is active. Stubbed false
/// (no actions active on a freshly-spawned vehicle).
#[derive(Debug, Clone, Copy)]
enum VehicleAction {
    EmitAi,
    Surge,
    Phase,
}
fn action_executing(_vehicle_index: u32, _action: VehicleAction) -> bool {
    false
}

/// Engine `unit_compute_boost_fraction(vehicle_index, ...)`. Stubbed
/// at 0.
fn unit_compute_boost_fraction(_vehicle_index: u32) -> f32 {
    0.0
}

/// Engine `vehicle_function_safe_divide(num, denom)` — returns
/// `num/denom` when `denom != 0`, else 0.
fn vehicle_function_safe_divide(num: f32, denom: f32) -> f32 {
    if denom != 0.0 { num / denom } else { 0.0 }
}

/// Engine `component_vectors_from_normal3d(velocity, normal, &parallel,
/// &perpendicular)`. Decomposes `velocity` along `normal`:
///   parallel = (velocity · normal) * normal
///   perpendicular = velocity - parallel
fn component_vectors_from_normal3d(
    velocity: &RealVector3d,
    normal: &RealVector3d,
) -> (RealVector3d, RealVector3d) {
    let dot = velocity.i * normal.i + velocity.j * normal.j + velocity.k * normal.k;
    let parallel = RealVector3d {
        i: dot * normal.i,
        j: dot * normal.j,
        k: dot * normal.k,
    };
    let perpendicular = RealVector3d {
        i: velocity.i - parallel.i,
        j: velocity.j - parallel.j,
        k: velocity.k - parallel.k,
    };
    (parallel, perpendicular)
}

// ---------------------------------------------------------------------------
// vehicle_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `vehicle_compute_function_value @
/// 0x18082FF30`. Engine flow:
///   1. Call `c_vehicle_type_component::compute_function_value`;
///      if it recognized the case, use its magnitude.
///   2. Otherwise dispatch direct cases:
///      540 `jump`, 541 `boost`, 554 `wingtip_contrail`,
///      560 `mean_antigrav`, 660 `enter_a`, 873 `surge`, 874 `phase`.
pub fn vehicle_compute_function_value(
    vehicle_index: u32,
    function: &str,
    _function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let datum_lock = get_object_datum(vehicle_index);
    let datum = datum_lock.read().unwrap();
    let Some(vehicle) = datum.as_vehicle() else {
        return false;
    };

    // Stage 1: vehicle_type_component dispatch.
    let mut magnitude: f32 = 0.0;
    let mut force_active = false;
    if vehicle_type_component_compute(
        &vehicle.type_component,
        vehicle_index,
        function,
        &mut magnitude,
        &mut force_active,
    ) {
        // Type component handled it — use its magnitude.
        let v = magnitude;
        let final_v = if v <= 0.0 { 0.0 } else if v >= 1.0 { 1.0 } else { v };
        *value = final_v;
        *active = final_v > 0.0 || force_active;
        return true;
    }

    // Stage 2: direct case dispatch.
    // Engine reads `unit.control_flags` (offset 28 in _unit_datum, which
    // protomorph hasn't surfaced as a field on UnitBody yet — read
    // via the chain). For simplicity stub at 0 for now; static vehicles
    // have no control inputs.
    let control_flags: u32 = 0;

    let recognized;
    let mut v17: f32 = 0.0;

    match function {
        // Case 540 `jump` — control_flags & 2 → 1.0 else 0.
        "jump" => {
            recognized = true;
            v17 = if (control_flags & 0x2) == 0 { 0.0 } else { 1.0 };
        }

        // Case 541 `boost` —
        //   if control_flags & 0x1000 → 1.0
        //   else → unit_compute_boost_fraction
        "boost" => {
            recognized = true;
            if (control_flags & 0x1000) != 0 {
                v17 = 1.0;
            } else {
                v17 = unit_compute_boost_fraction(vehicle_index);
            }
        }

        // Case 554 `wingtip_contrail` —
        //   decompose translational_velocity along forward;
        //   v17 = (|perpendicular| / 0.3)²
        "wingtip_contrail" => {
            recognized = true;
            let (_, perp) = component_vectors_from_normal3d(
                &datum.object.translational_velocity,
                &datum.object.forward,
            );
            let mag_sq = perp.i * perp.i + perp.j * perp.j + perp.k * perp.k;
            let mag = mag_sq.sqrt();
            let div = vehicle_function_safe_divide(mag, 0.3);
            v17 = div * div;
        }

        // Case 560 `mean_antigrav` — direct read of mean_antigrav_fraction.
        "mean_antigrav" => {
            recognized = true;
            v17 = vehicle.mean_antigrav_fraction;
        }

        // Cases 660/873/874 — action_executing predicates.
        "enter_a" => {
            recognized = true;
            v17 = if action_executing(vehicle_index, VehicleAction::EmitAi) { 1.0 } else { 0.0 };
        }
        "surge" => {
            recognized = true;
            v17 = if action_executing(vehicle_index, VehicleAction::Surge) { 1.0 } else { 0.0 };
        }
        "phase" => {
            recognized = true;
            v17 = if action_executing(vehicle_index, VehicleAction::Phase) { 1.0 } else { 0.0 };
        }

        _ => recognized = false,
    }

    if !recognized {
        return false;
    }

    // Engine LABEL_31/23/24 final clamp.
    let final_v = if v17 <= 0.0 { 0.0 } else if v17 >= 1.0 { 1.0 } else { v17 };
    *value = final_v;
    *active = final_v > 0.0 || force_active;
    true
}
