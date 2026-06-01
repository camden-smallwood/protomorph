//! Mirror of `scenario_location_from_point @ 0x18017BFE0` and the
//! per-frame camera→cluster cache that consumes it.
//!
//! Engine flow:
//! 1. Iterate `g_active_structure_bsp_mask` — each set bit = one
//!    active BSP in the current zone set.
//! 2. For each active BSP, call `bsp3d_test_point(bsp.collision_bsp,
//!    0, point)` — recursive plane-test descent.
//! 3. If it returns a leaf index (≥ 0): look up
//!    `bsp.leaves[leaf_index].cluster` (i8) → cluster_reference.
//! 4. Stop on first hit; if no BSP matches, return
//!    [`Location::NONE`].
//!
//! The result populates [`Location::cluster_reference`] which is then
//! handed to `c_visibility_collection::compute(...)` (Phase F) as the
//! seed cluster for the projections walker / PVS path.

use blam_tags::math::RealPoint3d;

use crate::halo::structures::clusters::ClusterReference;
use crate::halo::scenario::loader::LoadedScenario;
use crate::halo::structures::{bsp3d_test_point, BSP3D_ROOT_NODE_INDEX};

/// `s_location` (Ares `cseries/location.h:25-30`, 4 bytes).
///
/// World-position cluster + leaf identifier. Used by the observer
/// + view code so per-frame the renderer knows which BSP/cluster the
/// camera is in (for choosing sky / fog / lighting probes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Location {
    pub cluster_reference: ClusterReference,
    pub leaf_index: u16,
}

impl Location {
    pub const NONE: Self = Self {
        cluster_reference: ClusterReference::NONE,
        leaf_index: u16::MAX,
    };
}

/// `scenario_location_from_point @ 0x18017BFE0`. Convert a world
/// position into the cluster + leaf the position falls into.
///
/// Returns [`Location::NONE`] when the position is not inside any
/// active BSP (engine returns a sentinel `s_location` with
/// `bsp_index = -1`).
pub fn scenario_location_from_point(scenario: &LoadedScenario, point: RealPoint3d) -> Location {
    for bsp in &scenario.active_bsps {
        let Some(bsp3d) = &bsp.sbsp.collision_bsp else {
            continue;
        };
        let Some(leaf_index) = bsp3d_test_point(bsp3d, BSP3D_ROOT_NODE_INDEX, point) else {
            continue;
        };
        // leaf_index < 0 isn't possible (bsp3d_test_point would have
        // returned None), but guard anyway in case the leaf table is
        // shorter than the BSP3D would suggest.
        let Some(leaf) = bsp.sbsp.leaves.get(leaf_index as usize) else {
            continue;
        };
        return Location {
            cluster_reference: ClusterReference {
                bsp_index: bsp.scenario_bsp_index as i16,
                cluster_index: leaf.cluster as i16,
            },
            leaf_index: leaf_index as u16,
        };
    }
    Location::NONE
}

/// Camera-friendly variant of [`scenario_location_from_point`]. Engine
/// never does a cold BSP3D test on the camera (it uses
/// `scenario_location_from_line` to walk from the previous frame's
/// cached location). Protomorph's free-fly camera *does* go cold, and
/// commonly lands in solid space (above ceilings, below floors,
/// outside the play volume). When [`scenario_location_from_point`]
/// returns [`Location::NONE`], fall back to:
///
/// 1. Cluster AABB containment — find a cluster whose `bounds_x/y/z`
///    encloses the point. Engine does this implicitly when the player
///    pawn's tracked location is updated.
/// 2. Nearest cluster center — squared distance to each cluster's
///    AABB center. Always returns a valid cluster as long as the
///    scenario has at least one cluster.
pub fn scenario_location_from_camera_point(
    scenario: &LoadedScenario,
    point: RealPoint3d,
) -> Location {
    let primary = scenario_location_from_point(scenario, point);
    if primary.cluster_reference.bsp_index >= 0 {
        return primary;
    }

    // Fallback 1 — cluster AABB containment.
    for bsp in &scenario.active_bsps {
        for (cluster_idx, cluster) in bsp.sbsp.clusters.iter().enumerate() {
            if point.x >= cluster.bounds_x.lower
                && point.x <= cluster.bounds_x.upper
                && point.y >= cluster.bounds_y.lower
                && point.y <= cluster.bounds_y.upper
                && point.z >= cluster.bounds_z.lower
                && point.z <= cluster.bounds_z.upper
            {
                return Location {
                    cluster_reference: ClusterReference {
                        bsp_index: bsp.scenario_bsp_index as i16,
                        cluster_index: cluster_idx as i16,
                    },
                    leaf_index: 0,
                };
            }
        }
    }

    // Fallback 2 — nearest cluster center.
    let mut best: Option<(f32, i16, i16)> = None;
    for bsp in &scenario.active_bsps {
        for (cluster_idx, cluster) in bsp.sbsp.clusters.iter().enumerate() {
            let cx = (cluster.bounds_x.lower + cluster.bounds_x.upper) * 0.5;
            let cy = (cluster.bounds_y.lower + cluster.bounds_y.upper) * 0.5;
            let cz = (cluster.bounds_z.lower + cluster.bounds_z.upper) * 0.5;
            let dx = point.x - cx;
            let dy = point.y - cy;
            let dz = point.z - cz;
            let d2 = dx * dx + dy * dy + dz * dz;
            match best {
                None => best = Some((d2, bsp.scenario_bsp_index as i16, cluster_idx as i16)),
                Some((bd2, _, _)) if d2 < bd2 => {
                    best = Some((d2, bsp.scenario_bsp_index as i16, cluster_idx as i16));
                }
                _ => {}
            }
        }
    }
    if let Some((_, bsp_index, cluster_index)) = best {
        return Location {
            cluster_reference: ClusterReference {
                bsp_index,
                cluster_index,
            },
            leaf_index: 0,
        };
    }
    Location::NONE
}

/// Per-player-view camera-cluster cache. Stash the most-recent
/// position + result; recompute only when the position changes (or
/// the position-change tolerance is exceeded). Engine equivalent
/// lives in `s_player_view::camera_cluster_reference`; we keep it
/// per-view since each view has its own camera.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraClusterCache {
    /// Last position the cluster was computed for (`f32::NAN` =
    /// uninitialised).
    last_point: RealPoint3d,
    last_result: Location,
    /// `true` once `update` has been called at least once.
    initialized: bool,
}

impl CameraClusterCache {
    pub const POSITION_EPSILON: f32 = 1.0e-3;

    pub fn new() -> Self {
        Self::default()
    }

    /// Recompute if the camera moved more than [`POSITION_EPSILON`].
    /// Engine recomputes every frame regardless; we deduplicate to
    /// save per-frame BSP3D walks. Pass-through to
    /// [`scenario_location_from_point`].
    pub fn update(&mut self, scenario: &LoadedScenario, point: RealPoint3d) -> Location {
        if self.initialized {
            let dx = point.x - self.last_point.x;
            let dy = point.y - self.last_point.y;
            let dz = point.z - self.last_point.z;
            if dx.abs() < Self::POSITION_EPSILON
                && dy.abs() < Self::POSITION_EPSILON
                && dz.abs() < Self::POSITION_EPSILON
            {
                return self.last_result;
            }
        }
        self.last_point = point;
        self.last_result = scenario_location_from_camera_point(scenario, point);
        self.initialized = true;
        self.last_result
    }

    pub fn last(&self) -> Location {
        self.last_result
    }
}
