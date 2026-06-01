//! `effects/decals.cpp` — runtime decal state.
//!
//! Mirrors Ares `effects/decals.{h,cpp}`:
//!   - [`CDecalSystem`] (sizeof 80) — per-placement runtime state. Owns
//!     the decal-system's projection matrix + cluster references; one
//!     instance per scenario decal placement (or runtime spawn).
//!   - [`CDecal`] (sizeof 244) — per-collision runtime state. Each
//!     `CDecalSystem` can have 1..=16 children (one per collision hit
//!     returned by `collide`). Owns the per-decal mesh allocation
//!     handles + fade state + animated sprite bounds.
//!
//! Engine bodies for the methods on these types are decompiled from
//! dllcache (port 13372); each `impl` block lists the EA + comments
//! the algorithm step-by-step against the disassembly. The existing
//! mesh-build free functions in `mesh_builder.rs`, `orchestrator.rs`,
//! `smoother.rs`, `writer.rs`, `projection_builder.rs`, and
//! `edge_iterator.rs` are wired as helpers under these methods —
//! they're correct ports of the engine algorithms; what was missing
//! was the class structure to hang them on.

use blam_tags::math::{RealPoint2d, RealPoint3d, RealVector2d, RealVector3d};

use blam_tags::math::RealMatrix4x3;

use super::writer::RasterizerVertexWorld;

// ============================================================================
// `s_cluster_reference` — `cseries/cluster_reference.h` (sizeof 2; 16-bit
// packed structure_bsp_index + cluster_index_in_bsp). Engine uses `-1`
// (`0xFFFF`) as the sentinel.
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SClusterReference(pub u16);

impl SClusterReference {
    pub const SENTINEL: Self = Self(0xFFFF);
}

impl Default for SClusterReference {
    fn default() -> Self {
        Self::SENTINEL
    }
}

// ============================================================================
// `c_tag_index` — `tag_files/tag_index.h` (sizeof 4). 16-bit datum-style
// index with an upper-16-bits salt; engine treats `0xFFFFFFFF` as invalid.
// We mirror as `u32` since the salt only matters for runtime spawn paths.
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CTagIndex(pub u32);

impl CTagIndex {
    pub const NONE: Self = Self(0xFFFFFFFF);
    pub fn index(&self) -> u16 {
        self.0 as u16
    }
    pub fn is_none(&self) -> bool {
        self.0 == 0xFFFFFFFF
    }
}

impl Default for CTagIndex {
    fn default() -> Self {
        Self::NONE
    }
}

// ============================================================================
// `c_decal_system` — `effects/decals.h:37` (sizeof 80).
//
// Engine field layout (from Ares):
//   0x00  short                        identifier
//   0x04  c_tag_index                  m_definition_index
//   0x08  long                         m_first_decal_index   (datum index into c_decal::x_data_array; chain head)
//   0x0C  long                         m_flags               (bit 0 = preplaced, 1 = u_mirror, 2 = v_mirror)
//   0x10  s_cluster_reference[4]       m_cluster_refs        (filled at create-time by structure_clusters_in_sphere)
//   0x18  real_matrix4x3               m_projection          (basis[0]=forward, basis[1]=left, basis[2]=up; scale + position)
//   0x4C  float                        m_rotation            (rad — random Z-rotation for sprite variation)
// ============================================================================

bitflags::bitflags! {
    /// `c_decal_system::m_flags`. Engine bits:
    ///   bit 0 — preplaced
    ///   bit 1 — u_mirror   (set by `create_at_index` if random_u_mirror)
    ///   bit 2 — v_mirror
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct CDecalSystemFlags: u32 {
        const PREPLACED = 1 << 0;
        const U_MIRROR  = 1 << 1;
        const V_MIRROR  = 1 << 2;
    }
}

#[derive(Debug, Clone)]
pub struct CDecalSystem {
    pub identifier: i16,
    pub m_definition_index: CTagIndex,
    pub m_first_decal_index: i32,
    pub m_flags: CDecalSystemFlags,
    pub m_cluster_refs: [SClusterReference; 4],
    pub m_projection: RealMatrix4x3,
    pub m_rotation: f32,
}

impl CDecalSystem {
    /// Engine `get_preplaced` (`effects/decals.cpp` accessor).
    pub fn get_preplaced(&self) -> bool {
        self.m_flags.contains(CDecalSystemFlags::PREPLACED)
    }
    /// Engine `get_u_mirror`.
    pub fn get_u_mirror(&self) -> bool {
        self.m_flags.contains(CDecalSystemFlags::U_MIRROR)
    }
    /// Engine `get_v_mirror`.
    pub fn get_v_mirror(&self) -> bool {
        self.m_flags.contains(CDecalSystemFlags::V_MIRROR)
    }
    /// Engine `get_left_handed` — `(u_mirror XOR v_mirror)`. Used by
    /// `s_decal_projection_builder::initialize` to set the
    /// left-handedness flag on the projection.
    pub fn get_left_handed(&self) -> bool {
        self.get_u_mirror() ^ self.get_v_mirror()
    }
    /// Engine `get_projection_forward` — `&m_projection.basis[0]`.
    pub fn get_projection_forward(&self) -> RealVector3d {
        self.m_projection.forward
    }
    /// Engine `get_projection_left` — `&m_projection.basis[1]`.
    pub fn get_projection_left(&self) -> RealVector3d {
        self.m_projection.left
    }
    /// Engine `get_projection_up` — `&m_projection.basis[2]`.
    pub fn get_projection_up(&self) -> RealVector3d {
        self.m_projection.up
    }
    /// Engine `get_center` — `&m_projection.position`.
    pub fn get_center(&self) -> RealPoint3d {
        self.m_projection.position
    }
    /// Engine `get_rotation`.
    pub fn get_rotation(&self) -> f32 {
        self.m_rotation
    }
}

// ============================================================================
// `c_decal` — `effects/decals.cpp:168` (sizeof 244).
//
// Engine field layout (from Ares):
//   0x00  short                          identifier
//   0x04  long                           m_definition_block_index   (which c_decal_definition in the system's `definitions[]`)
//   0x08  long                           m_parent_system_index      (datum into c_decal_system::x_data_array)
//   0x0C  long                           m_next_sibling_index       (chain to next decal in the parent system)
//   0x10  long                           m_vertex_allocation_index  (lruv_block_index into preplaced/runtime c_vertex_allocator)
//   0x14  long                           m_index_allocation_index   (lruv_block_index into preplaced/runtime c_index_allocator)
//   0x18  unsigned short                 m_vertex_count
//   0x1A  unsigned short                 m_index_count
//   0x1C  long                           m_flags                    (bit 0 = alive, 1 = floating)
//   0x20  float                          m_age
//   0x24  float                          m_lifespan
//   0x28  float                          m_decay_period
//   0x2C  real_vector2d                  m_texture_scale            (set by choose_sprite from bitmap_aspect)
//   0x34  real_point2d                   m_sprite_corner            (animated sprite UV origin)
//   0x3C  real_vector2d                  m_sprite_size              (animated sprite UV size)
//   0x44  rasterizer_vertex_world[4]     m_floating_vertices        (immediate-mode quad for `m_floating` decals)
// ============================================================================

bitflags::bitflags! {
    /// `c_decal::m_flags`. Engine bits:
    ///   bit 0 — alive    (datum-allocated, in chain)
    ///   bit 1 — floating (use `m_floating_vertices` instead of allocator pool)
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct CDecalFlags: u32 {
        const ALIVE    = 1 << 0;
        const FLOATING = 1 << 1;
    }
}

#[derive(Debug, Clone)]
pub struct CDecal {
    pub identifier: i16,
    pub m_definition_block_index: i32,
    pub m_parent_system_index: i32,
    pub m_next_sibling_index: i32,
    pub m_vertex_allocation_index: i32,
    pub m_index_allocation_index: i32,
    pub m_vertex_count: u16,
    pub m_index_count: u16,
    pub m_flags: CDecalFlags,
    pub m_age: f32,
    pub m_lifespan: f32,
    pub m_decay_period: f32,
    pub m_texture_scale: RealVector2d,
    pub m_sprite_corner: RealPoint2d,
    pub m_sprite_size: RealVector2d,
    pub m_floating_vertices: [RasterizerVertexWorld; 4],
}

impl Default for CDecal {
    fn default() -> Self {
        Self {
            identifier: 0,
            m_definition_block_index: 0,
            m_parent_system_index: -1,
            m_next_sibling_index: -1,
            m_vertex_allocation_index: -1,
            m_index_allocation_index: -1,
            m_vertex_count: 0,
            m_index_count: 0,
            m_flags: CDecalFlags::ALIVE,
            m_age: 0.0,
            m_lifespan: 0.0,
            m_decay_period: 1.0,
            m_texture_scale: RealVector2d { i: 1.0, j: 1.0 },
            m_sprite_corner: RealPoint2d { x: 0.0, y: 0.0 },
            m_sprite_size: RealVector2d { i: 1.0, j: 1.0 },
            m_floating_vertices: [RasterizerVertexWorld::default(); 4],
        }
    }
}

impl CDecal {
    /// Engine `get_alive` (`effects/decals.cpp`).
    pub fn get_alive(&self) -> bool {
        self.m_flags.contains(CDecalFlags::ALIVE)
    }
    /// Engine `get_floating`.
    pub fn get_floating(&self) -> bool {
        self.m_flags.contains(CDecalFlags::FLOATING)
    }
    /// Engine `get_next_sibling_index`.
    pub fn get_next_sibling_index(&self) -> i32 {
        self.m_next_sibling_index
    }
    /// Engine `get_vertex_count`.
    pub fn get_vertex_count(&self) -> i32 {
        self.m_vertex_count as i32
    }
    /// Engine `get_index_count`.
    pub fn get_index_count(&self) -> i32 {
        self.m_index_count as i32
    }
}

// ============================================================================
// `c_decal_definition::e_pass` — `effects/decal_definitions.h:26`.
// Per-decal classification of which engine sub-pass renders this decal.
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EPass {
    PostAlbedo = 0,
    PostStaticLighting = 1,
}

impl EPass {
    pub fn from_index(i: i32) -> Self {
        match i {
            1 => Self::PostStaticLighting,
            _ => Self::PostAlbedo,
        }
    }
}
