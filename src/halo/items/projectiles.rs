//! `_projectile_datum` body + `projectile_compute_function_value`.
//!
//! Ares source:
//!   - struct: `source/items/projectiles.h:64-100` (_projectile_datum, 152B)
//!   - body:   `source/items/projectiles.cpp:1191`
//! Engine:
//!   - struct verified via `get_type_info?name=_projectile_datum` (32 fields).
//!   - compute body verbatim port of dllcache `0x180845470` (100-line
//!     pseudocode, 5 cases — all sid 508-512).
//!
//! Chain position: `object → projectile`. No intermediate. Reads
//! both `_projectile_datum` (runtime) AND `ProjectileDefinition`
//! authored fields (range/velocity/acceleration math).

use crate::halo::objects::object_header_data::{get_object_datum, get_tag_path};
use crate::halo::tags::tag_get;
use blam_tags::projectile::ProjectileDefinition;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// _projectile_datum body
// ---------------------------------------------------------------------------

/// Walked subset of engine `_projectile_datum` (152 bytes, 32 fields).
/// Currently surfaces the 3 fields read by
/// `projectile_compute_function_value` (flags, detonation_timer,
/// odometer). The rest defers — collision/targeting/rotation state
/// not consumed by compute.
///
/// Ares: `source/items/projectiles.h:64-100`.
#[derive(Debug, Clone, Default)]
pub struct ProjectileBody {
    /// `flags` — bit 1 = tracer, bit 3 = projectile_attach,
    /// bit 8 = max-acceleration-range (gates compute cases 510/511/512).
    pub flags: u32,
    /// `detonation_timer` — direct read by `time_remaining` (sid 508).
    pub detonation_timer: f32,
    /// `odometer` — used by `range_remaining` (509) and
    /// `acceleration_range` (511).
    pub odometer: f32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the `ProjectileDefinition` for `projectile_index` via the
/// tag cache. Returns `None` when the placement's palette tag_path
/// is unset (the case typically — projectiles are dynamic-spawned,
/// not placed in scenarios).
fn projectile_definition_for(projectile_index: u32) -> Option<Arc<ProjectileDefinition>> {
    let tag_path = get_tag_path(projectile_index);
    if tag_path.is_empty() {
        return None;
    }
    tag_get(*b"proj", &tag_path).and_then(|t| t.as_projectile_definition())
}

// ---------------------------------------------------------------------------
// projectile_compute_function_value — verbatim engine port
// ---------------------------------------------------------------------------

/// Verbatim port of dllcache `projectile_compute_function_value @
/// 0x180845470`. 5 cases (all sid 508-512). Reads both runtime
/// `_projectile_datum` state AND authored `ProjectileDefinition`
/// fields for the range/acceleration math.
pub fn projectile_compute_function_value(
    projectile_index: u32,
    function: &str,
    _function_owner_definition_index: u32,
    value: &mut f32,
    active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    let datum_lock = get_object_datum(projectile_index);
    let datum = datum_lock.read().unwrap();
    let Some(proj) = datum.as_projectile() else {
        return false;
    };

    let mut magnitude: f32 = 1.0; // Engine initializes magnitude = 1.0
    let mut recognized = true;

    match function {
        // Case 508 `time_remaining` — direct read of detonation_timer.
        "time_remaining" => magnitude = proj.detonation_timer,

        // Case 509 `range_remaining` — odometer / detonation_maximum_range.
        // Returns 0 when range is 0 (untimed projectile).
        "range_remaining" => {
            magnitude = 0.0;
            if let Some(pd) = projectile_definition_for(projectile_index) {
                if pd.detonation_maximum_range != 0.0 {
                    magnitude = proj.odometer / pd.detonation_maximum_range;
                }
            }
        }

        // Case 510 `tracer` — flags bit 1 → 1.0 else 0.0.
        "tracer" => {
            magnitude = if (proj.flags & (1 << 1)) != 0 { 1.0 } else { 0.0 };
        }

        // Case 511 `acceleration_range`. Engine logic:
        //   if (flags bit 8 set) → 1.0
        //   else if (initial_velocity == 0 || final_velocity == 0) → 1.0
        //   else:
        //     t = (odometer - initial_velocity_end_distance) * runtime_oo_acceleration_distance
        //     if t <= 0 → 0.0
        //     else if t >= 1 → 1.0
        //     else → t
        "acceleration_range" => {
            if (proj.flags & (1 << 8)) != 0 {
                magnitude = 1.0;
            } else if let Some(pd) = projectile_definition_for(projectile_index) {
                if pd.initial_velocity == 0.0 || pd.final_velocity == 0.0 {
                    magnitude = 1.0;
                } else {
                    let t = (proj.odometer - pd.initial_velocity_end_distance)
                        * pd.runtime_oo_acceleration_distance;
                    magnitude = if t <= 0.0 {
                        0.0
                    } else if t >= 1.0 {
                        1.0
                    } else {
                        t
                    };
                }
            } else {
                magnitude = 1.0;
            }
        }

        // Case 512 `projectile_attach` — flags bit 3 → 1.0 else 0.0.
        "projectile_attach" => {
            magnitude = if (proj.flags & (1 << 3)) != 0 { 1.0 } else { 0.0 };
        }

        _ => recognized = false,
    }

    if recognized {
        *value = magnitude;
        *active = magnitude > 0.0;
    }
    recognized
}
