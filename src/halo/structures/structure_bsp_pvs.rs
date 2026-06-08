//! Mirror of `structure_bsp_compute_cluster_active_pvs @ 0x180334F80`
//! (Ares `source/structures/structure_bsp_definitions.cpp:273`).
//!
//! Combines the precomputed OPEN PVS row with door-portal-driven
//! expansion of the CLOSED row, producing the per-frame "active PVS"
//! bitvector for one cluster.
//!
//! Protomorph note: today no door machines exist in the gameplay
//! layer, so the door-portal expansion is a no-op (TODO when machines
//! land). The OPEN-row fast path is engine-faithful.

use crate::halo::structures::clusters::ClusterReference;
use crate::halo::scenario::{
    scenario_zone_set_pvs_write_open_row, LoadedScenario, SGameClusterBitVectors,
};

/// `structure_bsp_cluster_pvs_flags`.
///
/// Bit indices match Ares `structure_bsp_definitions.h:39-42`.
pub mod cluster_pvs_flags {
    pub const PVS_AFFECTED_BY_ONE_WAY_PORTAL: u16 = 1 << 0;
    pub const PVS_AFFECTED_BY_DOOR_PORTAL: u16 = 1 << 1;
}

/// `structure_bsp_compute_cluster_active_pvs @ 0x180334F80`. Compute
/// the active cluster-PVS bitvector for `cluster_reference` in the
/// current zone set.
///
/// Engine flow:
/// 1. Zero `destination_pvs`.
/// 2. Bail if `cluster_reference.bsp_index == -1`.
/// 3. Load the BSP + cluster; check
///    `cluster.flags & PVS_AFFECTED_BY_DOOR_PORTAL`.
/// 4. Read the OPEN PVS row → temporary bitvector.
/// 5. If door-portal flag set: read CLOSED row, walk
///    `g_active_portal_bitvectors` to expand the open mask through
///    active door portals into the closed mask. (Phase D5b — needs
///    door support; stubbed for now.)
/// 6. Else: copy OPEN row directly into `destination_pvs`.
pub fn structure_bsp_compute_cluster_active_pvs(
    scenario: &LoadedScenario,
    cluster_reference: ClusterReference,
    destination_pvs: &mut SGameClusterBitVectors,
) {
    *destination_pvs = SGameClusterBitVectors::default();
    if cluster_reference.bsp_index < 0 {
        return;
    }
    // Resolve the BSP + cluster to read its flags.
    let bsp = match scenario
        .active_bsps
        .iter()
        .find(|b| b.scenario_bsp_index as i16 == cluster_reference.bsp_index)
    {
        Some(b) => b,
        None => return,
    };
    let cluster = match bsp.sbsp.clusters.get(cluster_reference.cluster_index as usize) {
        Some(c) => c,
        None => return,
    };
    let door_affected = cluster
        .flags
        .contains(blam_tags::structure_bsp::StructureClusterFlags::DoorPortal);

    // OPEN row — always required.
    scenario_zone_set_pvs_write_open_row(
        &scenario.scenario,
        scenario.active_zone_set,
        cluster_reference,
        destination_pvs,
    );

    if door_affected {
        // TODO(phase-d5b): port door-portal expansion. Needs:
        //   1. CLOSED row read (scenario_zone_set_pvs_write_closed_row)
        //   2. Walk this BSP's cluster_portals[]; for each portal that's
        //      in the door-portal mapping AND `portal_activation`-active,
        //      the destination cluster (front/back, whichever isn't us)
        //      gets unioned into our active PVS via its CLOSED row.
        //   3. Recurse one level (engine uses a worklist + visited
        //      bitvector).
        //
        // Until door support lands at the gameplay layer, all portals
        // are treated as open and the OPEN row is the active PVS —
        // already what the function above wrote.
    }
}
