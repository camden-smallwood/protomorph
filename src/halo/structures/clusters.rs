
/// `s_cluster_reference` (Ares `cseries/location.h:18-22`, 2 bytes).
///
/// Halo packs this as `(char bsp_index, unsigned char cluster_index)`.
/// We widen to `i16/i16` for ergonomic comparisons (no asserts on
/// exact-byte parity since it's a runtime in-memory type, never
/// serialized).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ClusterReference {
    pub bsp_index: i16,
    pub cluster_index: i16,
}

impl ClusterReference {
    /// `_invalid_cluster_reference` sentinel — both fields negative.
    pub const NONE: Self = Self {
        bsp_index: -1,
        cluster_index: -1,
    };

    pub fn is_valid(&self) -> bool {
        self.bsp_index >= 0 && self.cluster_index >= 0
    }
}

impl std::fmt::Display for ClusterReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(bsp={}, cluster={})", self.bsp_index, self.cluster_index)
    }
}

/// `s_cluster_reference` packed to engine-exact 2-byte layout
/// (`char bsp_index; unsigned char cluster_index;`). Used inside
/// engine-faithful structs (visibility region/cluster/projection) where
/// `static_assert(sizeof) == N` parity matters; convert to/from the
/// widened ergonomic [`ClusterReference`] at boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PackedClusterReference {
    pub bsp_index: i8,
    pub cluster_index: u8,
}

const _: () = assert!(std::mem::size_of::<PackedClusterReference>() == 2);

impl PackedClusterReference {
    pub const NONE: Self = Self { bsp_index: -1, cluster_index: 0xFF };
}

impl From<ClusterReference> for PackedClusterReference {
    fn from(c: ClusterReference) -> Self {
        Self { bsp_index: c.bsp_index as i8, cluster_index: c.cluster_index as u8 }
    }
}

impl From<PackedClusterReference> for ClusterReference {
    fn from(p: PackedClusterReference) -> Self {
        Self {
            bsp_index: p.bsp_index as i16,
            cluster_index: if p.bsp_index < 0 { -1 } else { p.cluster_index as i16 },
        }
    }
}
