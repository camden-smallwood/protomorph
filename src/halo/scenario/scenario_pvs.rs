//! Mirror of `Ares/source/scenario/scenario_pvs.cpp` — accessors for
//! the precomputed cluster-PVS bit vectors stored in
//! `scenario.zone_set_pvs[]`. The data was parsed by blam-tags in
//! Phase A4; this module exposes the runtime read/write API that
//! drives `structure_bsp_compute_cluster_active_pvs` (Phase D5) and
//! `visibility_build_region_from_pvs` (Phase D6).

use blam_tags::scenario::{Scenario, ZoneSetClusterPvs};

use crate::halo::structures::clusters::ClusterReference;

// =============================================================================
// `s_game_cluster_bit_vectors` — runtime per-BSP × per-cluster bit
// vector. Engine layout (deduced from stack offset 0x200 = 512B and
// access patterns `bit_vectors.m_elements[bsp].m_flags[word]`):
//
//   c_static_array<c_static_flags<256>, 16> bit_vectors;
//
// = 16 BSPs × 256-bit cluster vector = 16 × 8 u32 words = 512 bytes.
// =============================================================================

pub const MAX_BSPS: usize = 16;
pub const MAX_CLUSTERS_PER_BSP: usize = 256;
pub const CLUSTER_BIT_WORDS_PER_BSP: usize = MAX_CLUSTERS_PER_BSP / 32; // = 8

/// `s_game_cluster_bit_vectors` (512B, runtime). One bit per
/// (bsp, cluster) pair indicating "this cluster is in the active set."
/// Indexing: `bit_vectors[bsp_index][cluster_index >> 5] & (1 <<
/// (cluster_index & 31))`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SGameClusterBitVectors {
    pub bit_vectors: [[u32; CLUSTER_BIT_WORDS_PER_BSP]; MAX_BSPS],
}

const _: () = assert!(std::mem::size_of::<SGameClusterBitVectors>() == 512);

impl Default for SGameClusterBitVectors {
    fn default() -> Self {
        Self { bit_vectors: [[0; CLUSTER_BIT_WORDS_PER_BSP]; MAX_BSPS] }
    }
}

// =============================================================================
// `scenario_zone_set_pvs_get_row @ 0x180333630`
// `scenario_zone_set_pvs_write_row @ 0x180333930`
//
// In the engine, `s_scenario_pvs_row` (192B) is an intermediate that
// holds 16 pointers + 16 counts referencing the parsed scenario tag
// data. `write_row` then copies those bytes into a
// `SGameClusterBitVectors`. Since we already have the parsed data via
// `Scenario::zone_set_pvs`, we collapse the two-step pattern into a
// single function: read the row + write to destination.
// =============================================================================

/// Resolve `(zone_set_index, cluster_reference)` into a
/// [`ZoneSetClusterPvs`] row from the parsed scenario data. Returns
/// `None` when the zone set / pvs index / bsp_mask doesn't admit the
/// reference (matches engine behavior of leaving destination zeroed).
fn resolve_cluster_pvs_row<'a>(
    scenario: &'a Scenario,
    zone_set_index: usize,
    cluster_reference: ClusterReference,
    doors_closed: bool,
) -> Option<(&'a ZoneSetClusterPvs, u32)> {
    let zone_set = scenario.zone_sets.get(zone_set_index)?;
    let pvs_idx = zone_set.pvs_index;
    if pvs_idx < 0 {
        return None;
    }
    let zone_set_pvs = scenario.zone_set_pvs.get(pvs_idx as usize)?;
    let bsp_mask = zone_set_pvs.structure_bsp_mask;
    let bsp_index = cluster_reference.bsp_index;
    if bsp_index < 0 || (bsp_index as usize) >= MAX_BSPS {
        return None;
    }
    if (bsp_mask & (1u32 << bsp_index)) == 0 {
        return None;
    }
    // Active position of the camera's BSP within structure_bsp_pvs[].
    let active_pos = (bsp_mask & ((1u32 << bsp_index) - 1)).count_ones() as usize;
    let bsp_pvs = zone_set_pvs.structure_bsp_pvs.get(active_pos)?;
    let cluster_pvs_array = if doors_closed {
        &bsp_pvs.cluster_pvs_doors_closed
    } else {
        &bsp_pvs.cluster_pvs
    };
    let cluster_pvs = cluster_pvs_array.get(cluster_reference.cluster_index as usize)?;
    Some((cluster_pvs, bsp_mask))
}

/// `scenario_zone_set_pvs_write_row @ 0x180333930` — write the OPEN PVS
/// row for `cluster_reference` into `pvs`. `pvs` is fully zeroed on
/// entry; only bits that the row covers get set.
pub fn scenario_zone_set_pvs_write_open_row(
    scenario: &Scenario,
    zone_set_index: usize,
    cluster_reference: ClusterReference,
    pvs: &mut SGameClusterBitVectors,
) {
    *pvs = SGameClusterBitVectors::default();
    let Some((cluster_pvs, bsp_mask)) =
        resolve_cluster_pvs_row(scenario, zone_set_index, cluster_reference, false)
    else {
        return;
    };
    write_cluster_pvs_into(cluster_pvs, bsp_mask, pvs);
}

fn write_cluster_pvs_into(
    cluster_pvs: &ZoneSetClusterPvs,
    bsp_mask: u32,
    pvs: &mut SGameClusterBitVectors,
) {
    // For each active BSP b' (in global-index order), pull its
    // bit-vector from the cluster_pvs row and place it into
    // pvs.bit_vectors[b'].
    let mut other_active_pos = 0usize;
    for b in 0..MAX_BSPS {
        if (bsp_mask & (1u32 << b)) == 0 {
            continue;
        }
        if let Some(bit_vec) = cluster_pvs.bsp_bit_vectors.get(other_active_pos) {
            for (i, &word) in bit_vec.iter().take(CLUSTER_BIT_WORDS_PER_BSP).enumerate() {
                pvs.bit_vectors[b][i] = word;
            }
        }
        other_active_pos += 1;
    }
}

