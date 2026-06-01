//! Scenario subsystem — load and render `.scenario` (scnr) tags.
//!
//! Mirror of `Ares/source/scenario/`. Today: only the sky subsystem
//! is populated (carryover from the old single-object viewer's
//! atmosphere config). Phase D adds the scenario loader, zone sets,
//! lightmap data, decorators, cubemaps.

pub mod globals;
pub mod loader;
pub mod location;
pub mod runtime_lights;
pub mod scenario_pvs;

pub use loader::{LoadedScenario, ScenarioLoadError};
pub use location::{
    scenario_location_from_camera_point, scenario_location_from_point, CameraClusterCache,
};
pub use runtime_lights::{RuntimeLight, RuntimeLightLoadResult};
pub use scenario_pvs::{
    scenario_zone_set_pvs_row_fill, scenario_zone_set_pvs_row_set, scenario_zone_set_pvs_row_test,
    scenario_zone_set_pvs_write_closed_row, scenario_zone_set_pvs_write_open_row,
    SGameClusterBitVectors, CLUSTER_BIT_WORDS_PER_BSP, MAX_BSPS, MAX_CLUSTERS_PER_BSP,
};
