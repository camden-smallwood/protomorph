//! Engine extern table — `g_extern_descriptions @ 0x180EACD70` (dllcache).
//!
//! Mirror of the 49-entry `s_extern_description[49]` static table the
//! engine walks in `render_method_submit_externs @ 0x180685580` to bind
//! per-frame / per-pass shader inputs (G-buffer textures, light dirs,
//! change colors, etc.) to per-shader cbuffer slots.
//!
//! ## Engine struct layout (24B each)
//!
//! ```c
//! struct s_extern_description {
//!     e_extern_volatility_level m_volatility_level;  // i32, +0
//!     uint32_t                  m_register;          // u32, +4 (ALWAYS 0xFFFFFFFF in the table)
//!     e_render_method_parameter_type m_type;         // i32, +8
//!     uint32_t                  reserved;            // u32, +12
//!     const char*               m_name;              // i64, +16
//! };
//! ```
//!
//! `m_register == 0xFFFFFFFF` for every entry (audit-verified by byte-dumping
//! all 49 rows on 2026-05-13). The per-shader register routing comes
//! from `c_render_method_template::m_current_platform.m_routing_info[]`,
//! NOT from this table. So protomorph drops the m_register field.
//!
//! `reserved` is unused; dropped too.
//!
//! ## Cross-reference
//!
//! - `reference_engine_49_externs_full.md` — full audit doc with the
//!   per-loop walker decompile + pair-stride hypothesis.
//! - `reference_engine_shader_constant_routing.md` — cbuffer slot-ID
//!   bit-packing for `set_shader_constant` targets.

use crate::halo::rasterizer::Surface;

/// Number of extern entries — array length for `ExternState` storage.
pub const K_NUMBER_OF_EXTERNS: usize = 49;

// =============================================================================
// ExternState — runtime storage
// =============================================================================

/// Runtime values for each of the 49 externs. Mirrors what the engine
/// keeps in static storage (texture pointers + vec4 constants) and
/// reads back via `submit_extern_texture` / `get_extern_constant`.
///
/// Array-indexed storage (49 slots) matches the engine's
/// `g_extern_descriptions[i]`-keyed access pattern. Unused-slot waste
/// is ~1.5KB total — acceptable for the simpler indexing.
///
/// Owned by [`crate::halo::rasterizer::Rasterizer`]. P5.2 wires the
/// walker; P5.4 writes per-frame values; P5.5 binds the textures into
/// bind groups.
#[derive(Debug, Clone)]
pub struct ExternState {
    /// `Texture`-type externs point at a [`Surface`] in the rasterizer's
    /// `SurfaceTable`. `Vec4`-type externs leave this `None` (storage
    /// is in [`Self::constants`]).
    pub textures: [Option<Surface>; K_NUMBER_OF_EXTERNS],
    /// `Vec4`-type externs hold their value here. `Texture`-type externs
    /// leave this zeroed.
    pub constants: [[f32; 4]; K_NUMBER_OF_EXTERNS],
}

impl Default for ExternState {
    fn default() -> Self {
        Self {
            textures: [None; K_NUMBER_OF_EXTERNS],
            constants: [[0.0; 4]; K_NUMBER_OF_EXTERNS],
        }
    }
}

impl ExternState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mirror of engine `render_method_clear_externs @ 0x180684440`.
    ///
    /// Engine body on PC is just an assertion walk (verifies every entry's
    /// `m_volatility_level != -1`) plus `g_render_method_last_bsp_used = -1;`.
    /// No data is reset. Protomorph mirror is a no-op for now — the
    /// `last_bsp_used` cache invariant lives elsewhere when ported.
    pub fn clear_externs(&mut self) {
        // Engine PC behavior: NO clearing. The per-pass walker overwrites
        // PerPass-volatility entries every pass; Persistent entries are
        // updated when the underlying source state changes (object load,
        // light setup, etc.).
    }
}

// =============================================================================
// Pending submissions — populated by the walker, consumed by P5.5 bind-group builder
// =============================================================================

/// Per-pass texture-binding intent: the walker calls
/// `submit_extern_texture(stage, extern_idx, dest_register)`; that
/// resolves to a `(register, Surface)` pairing that the bind-group
/// builder consumes when materializing the wgpu descriptor.
///
/// `stage` mirrors engine arg `submit_extern_texture(stage=0, ...)`
/// — always 0 in the walker (PS stage by convention).
#[derive(Debug, Clone, Copy)]
pub struct PendingTextureBinding {
    pub stage: u32,
    pub extern_idx: u32,
    pub dest_register: u32,
    pub surface: Surface,
}
