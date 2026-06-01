//! Mirror of `Ares/source/visibility/visibility_portal_activation.h`
//! and `.cpp`.
//!
//! Per-frame portal-activation state — for each portal in each BSP,
//! a single bit ("active" = open, "inactive" = closed). Driven by
//! gameplay (machine doors set portal bits; everything else is
//! always-open).
//!
//! Engine layout: thread-local
//! `c_static_array<c_static_flags<512>, 16> g_active_portal_bitvectors`
//! = 16 BSPs × 512 portal-bits each = 1024 bytes total. Engine
//! initializes each BSP-row to all-active when the BSP loads
//! (`portal_activation_initialize_for_new_structure_bsp`); machines
//! flip individual bits when their doors open/close.
//!
//! Protomorph today has no door machines, so the bitvector is
//! initialized all-active and stays that way. The plumbing is here
//! so that when door support lands, the visibility port doesn't
//! need re-architecting.

use crate::halo::scenario::MAX_BSPS;

pub const MAX_PORTALS_PER_BSP: usize = 512;
pub const PORTAL_FLAG_WORDS_PER_BSP: usize = MAX_PORTALS_PER_BSP / 32;
pub const MAX_TRANSFORMED_PORTALS: usize = 512;

/// `s_portal_reference` (Ares `cseries/location.h:32-37`, 2B
/// bit-packed). `short bsp_index : 6` + `short portal_index : 10`.
/// Use [`PortalReference::new`] to construct safely; accessor methods
/// extract sign-extended values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PortalReference {
    /// Raw 16-bit packed form. Low 6 bits = signed bsp_index;
    /// upper 10 bits = signed portal_index.
    pub raw: u16,
}

const _: () = assert!(std::mem::size_of::<PortalReference>() == 2);

impl PortalReference {
    pub fn new(bsp_index: i16, portal_index: i16) -> Self {
        let bsp_bits = (bsp_index as u16) & 0x3F;
        let portal_bits = (portal_index as u16) & 0x3FF;
        Self { raw: (portal_bits << 6) | bsp_bits }
    }

    /// Sign-extended 6-bit BSP index. Engine: `(short)(raw << 10) >> 10`.
    pub fn bsp_index(self) -> i16 {
        ((self.raw as i16) << 10) >> 10
    }

    /// Sign-extended 10-bit portal index. Engine: `(short)raw >> 6`.
    pub fn portal_index(self) -> i16 {
        (self.raw as i16) >> 6
    }
}

/// `c_static_array<c_static_flags<512>, 16>` — 16 BSPs × 512 portal
/// bits.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ActivePortalBitvectors {
    pub bits: [[u32; PORTAL_FLAG_WORDS_PER_BSP]; MAX_BSPS],
}

const _: () = assert!(std::mem::size_of::<ActivePortalBitvectors>() == 1024);

impl Default for ActivePortalBitvectors {
    fn default() -> Self {
        // Engine initializes each loaded BSP to all-active; we apply
        // that to all 16 slots since `init_for_new_structure_bsp`
        // would do it on load anyway.
        Self { bits: [[u32::MAX; PORTAL_FLAG_WORDS_PER_BSP]; MAX_BSPS] }
    }
}

impl ActivePortalBitvectors {
    pub fn new_all_active() -> Self {
        Self::default()
    }

    pub fn new_all_inactive() -> Self {
        Self { bits: [[0; PORTAL_FLAG_WORDS_PER_BSP]; MAX_BSPS] }
    }

    /// `portal_activation_set_portal_active @ ?`. Set/clear one
    /// portal bit.
    pub fn set_active(&mut self, portal: PortalReference, active: bool) {
        let bsp = portal.bsp_index();
        let p = portal.portal_index();
        if bsp < 0 || (bsp as usize) >= MAX_BSPS
            || p < 0 || (p as usize) >= MAX_PORTALS_PER_BSP
        {
            return;
        }
        let word = &mut self.bits[bsp as usize][p as usize >> 5];
        let mask = 1u32 << (p & 31);
        if active {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    /// `portal_activation_portal_is_active @ ?`. Read one portal bit.
    pub fn is_active(&self, portal: PortalReference) -> bool {
        let bsp = portal.bsp_index();
        let p = portal.portal_index();
        if bsp < 0 || (bsp as usize) >= MAX_BSPS
            || p < 0 || (p as usize) >= MAX_PORTALS_PER_BSP
        {
            return false;
        }
        let word = self.bits[bsp as usize][p as usize >> 5];
        (word >> (p & 31)) & 1 != 0
    }

    /// `portal_activation_initialize_for_new_structure_bsp @ ?`. Mark
    /// all portals in the given BSPs as active. `mask` has bit `i`
    /// set for each BSP index to initialize.
    pub fn init_for_new_bsps(&mut self, mask: u32) {
        for b in 0..MAX_BSPS {
            if (mask & (1u32 << b)) != 0 {
                self.bits[b] = [u32::MAX; PORTAL_FLAG_WORDS_PER_BSP];
            }
        }
    }

    /// `portal_activation_dispose_from_old_structure_bsp @ ?`. Mark
    /// all portals in the given BSPs as inactive (zero).
    pub fn dispose_from_bsps(&mut self, mask: u32) {
        for b in 0..MAX_BSPS {
            if (mask & (1u32 << b)) != 0 {
                self.bits[b] = [0; PORTAL_FLAG_WORDS_PER_BSP];
            }
        }
    }
}
