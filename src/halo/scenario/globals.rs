//! Engine `global_scenario` — pointer to the currently loaded
//! `.scenario` tag definition. Set by `LoadedScenario::load` at
//! scenario load and cleared between scenarios. Read by code that
//! needs scenario-scoped data without threading the scenario through
//! every callsite (compute_function_value, lighting setup, etc.).
//!
//! The engine carries this as a single global pointer. Protomorph
//! mirrors with an `Arc<Scenario>` so the consumer can hold a shared
//! reference without taking a read lock for the duration of a
//! computation.

use blam_tags::scenario::Scenario;
use std::sync::{Arc, RwLock};

static GLOBAL_SCENARIO: RwLock<Option<Arc<Scenario>>> = RwLock::new(None);

/// Replace the global pointer with `scenario`. Call once per scenario
/// load.
pub fn set_global_scenario(scenario: Arc<Scenario>) {
    *GLOBAL_SCENARIO.write().unwrap() = Some(scenario);
}

/// Convenience accessor for `global_scenario->local_north` (radians).
/// Drives `object_compute_function_value` case `compass` (engine sid
/// 500). Returns 0.0 when no scenario is loaded.
pub fn get_local_north() -> f32 {
    GLOBAL_SCENARIO
        .read()
        .unwrap()
        .as_ref()
        .map(|s| s.local_north)
        .unwrap_or(0.0)
}
