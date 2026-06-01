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
use blam_tags::render_method::RenderMethodTemplate;

// =============================================================================
// Enum mirrors
// =============================================================================

/// `e_extern_volatility_level` — when the engine refreshes the extern's
/// stored value.
///
/// Only two levels appear across all 49 entries; the engine source enum
/// likely has more for x360-era debug builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ExternVolatility {
    /// Cleared rarely — object-attached params, light setup, change-color.
    /// Loop guard: `volatility_level == g_extern_descriptions[i].m_volatility_level`.
    Persistent = 0,
    /// Cleared / refreshed every pass — G-buffer textures, water memexport,
    /// screen constants, lightmap atlas references.
    PerPass = 1,
}

/// `e_render_method_parameter_type` — the engine dispatch in
/// `render_method_submit_single_extern @ 0x1806868A0` accepts types
/// `{0, 1, 2, 5, 6}` but only types 0 and 6 appear in the 49-entry table.
///
/// Type 0 routes to `submit_extern_texture(stage, source_index, dest_register)`.
/// Type 6 routes to `get_extern_constant(idx, &vec4) + set_shader_constant(...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ExternParameterType {
    /// Texture binding — dest_register is a sampler register index.
    Texture = 0,
    /// Vec4 constant — dest_register decodes via `((cbuffer_idx << 8) | entry_idx)`
    /// then re-packs to a 32-bit `set_shader_constant` slot ID.
    Vec4 = 6,
}

/// `s_extern_description` (engine 24B; protomorph drops the always-`0xFFFFFFFF`
/// `m_register` and the 4B `reserved` slot, leaving the meaningful fields).
#[derive(Debug, Clone, Copy)]
pub struct ExternDescription {
    pub volatility: ExternVolatility,
    pub parameter_type: ExternParameterType,
    pub name: &'static str,
}

// =============================================================================
// Extern indices — `e_render_method_extern` enum
// =============================================================================

/// Indices into [`EXTERN_DESCRIPTIONS`] and [`ExternState`] arrays.
///
/// Names match the engine's `e_render_method_extern` enum (verified
/// 2026-05-13 by byte-dumping `g_extern_descriptions` via IDA). The
/// `texture_*` prefix on entries 4-10, 24-26, 38, 44 is misleading —
/// those entries are `Vec4` type, carrying packed sampler-state vec4s,
/// not actual texture bindings.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Extern {
    None = 0,
    TextureGlobalTargetTexaccum = 1,
    TextureGlobalTargetNormal = 2,
    TextureGlobalTargetZ = 3,
    TextureGlobalTargetShadowBuffer1 = 4,
    TextureGlobalTargetShadowBuffer2 = 5,
    TextureGlobalTargetShadowBuffer3 = 6,
    TextureGlobalTargetShadowBuffer4 = 7,
    TextureGlobalTargetTextureCamera = 8,
    TextureGlobalTargetReflection = 9,
    TextureGlobalTargetRefraction = 10,
    TextureLightprobeTexture = 11,
    TextureDominantLightIntensityMap = 12,
    TextureUnused1 = 13,
    TextureUnused2 = 14,
    ObjectChangeColorPrimary = 15,
    ObjectChangeColorSecondary = 16,
    ObjectChangeColorTertiary = 17,
    ObjectChangeColorQuaternary = 18,
    ObjectEmblemColorBackground = 19,
    ObjectEmblemColorPrimary = 20,
    ObjectEmblemColorSecondary = 21,
    TextureDynamicEnvironmentMap0 = 22,
    TextureDynamicEnvironmentMap1 = 23,
    TextureCookTorranceCc0236 = 24,
    TextureCookTorranceDd0236 = 25,
    TextureCookTorranceC78d78 = 26,
    LightDir0 = 27,
    LightColor0 = 28,
    LightDir1 = 29,
    LightColor1 = 30,
    LightDir2 = 31,
    LightColor2 = 32,
    LightDir3 = 33,
    LightColor3 = 34,
    TextureUnused3 = 35,
    TextureUnused4 = 36,
    TextureUnused5 = 37,
    TextureDynamicLightGel0 = 38,
    FlatEnvmapMatrixX = 39,
    FlatEnvmapMatrixY = 40,
    FlatEnvmapMatrixZ = 41,
    DebugTint = 42,
    ScreenConstants = 43,
    ActiveCamoDistortionTexture = 44,
    SceneLdrTexture = 45,
    SceneHdrTexture = 46,
    WaterMemoryExportAddress = 47,
    TreeAnimationTimer = 48,
}

/// Number of extern entries — array length for both the descriptions
/// table and `ExternState` storage.
pub const K_NUMBER_OF_EXTERNS: usize = 49;

// =============================================================================
// `g_extern_descriptions` — the 49-entry table
// =============================================================================

/// Mirror of `g_extern_descriptions[49] @ 0x180EACD70`.
///
/// **Audit-verified 2026-05-13** by dumping the table bytes via IDA
/// (`run_script` over the 49 × 24B entries at `0x180EACD70`). Names,
/// volatility levels, and types all match 1:1 with the engine.
pub const EXTERN_DESCRIPTIONS: [ExternDescription; K_NUMBER_OF_EXTERNS] = {
    use ExternParameterType::{Texture, Vec4};
    use ExternVolatility::{PerPass, Persistent};
    [
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_none" },
        ExternDescription { volatility: PerPass,    parameter_type: Texture, name: "_render_method_extern_texture_global_target_texaccum" },
        ExternDescription { volatility: PerPass,    parameter_type: Texture, name: "_render_method_extern_texture_global_target_normal" },
        ExternDescription { volatility: PerPass,    parameter_type: Texture, name: "_render_method_extern_texture_global_target_z" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_shadow_buffer1" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_shadow_buffer2" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_shadow_buffer3" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_shadow_buffer4" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_texture_camera" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_reflection" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_global_target_refraction" },
        ExternDescription { volatility: Persistent, parameter_type: Texture, name: "_render_method_extern_texture_lightprobe_texture" },
        ExternDescription { volatility: Persistent, parameter_type: Texture, name: "_render_method_extern_texture_dominant_light_intensity_map" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_unused1" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_unused2" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_change_color_primary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_change_color_secondary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_change_color_tertiary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_change_color_quaternary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_emblem_color_background" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_emblem_color_primary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_object_emblem_color_secondary" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_dynamic_environment_map_0" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_dynamic_environment_map_1" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_cook_torrance_cc0236" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_cook_torrance_dd0236" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_texture_cook_torrance_c78d78" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_dir_0" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_color_0" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_dir_1" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_color_1" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_dir_2" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_color_2" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_dir_3" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_light_color_3" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_unused_3" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_unused_4" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_unused_5" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_texture_dynamic_light_gel_0" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_flat_envmap_matrix_x" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_flat_envmap_matrix_y" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_flat_envmap_matrix_z" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_debug_tint" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_screen_constants" },
        ExternDescription { volatility: Persistent, parameter_type: Vec4,    name: "_render_method_extern_active_camo_distortion_texture" },
        ExternDescription { volatility: PerPass,    parameter_type: Texture, name: "_render_method_extern_scene_ldr_texture" },
        ExternDescription { volatility: PerPass,    parameter_type: Texture, name: "_render_method_extern_scene_hdr_texture" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_water_memory_export_address" },
        ExternDescription { volatility: PerPass,    parameter_type: Vec4,    name: "_render_method_extern_tree_animation_timer" },
    ]
};

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

    /// Mirror of engine `get_extern_constant(idx, &out)`. Returns the
    /// vec4 stored in `constants[idx]` (or zero if the slot has never
    /// been populated). Engine asserts on out-of-range; we panic on
    /// invalid idx since it's a programmer error.
    pub fn get_constant(&self, extern_idx: u32) -> [f32; 4] {
        self.constants[extern_idx as usize]
    }

    /// Set a `Vec4`-type extern's vec4 value. Volatility-aware writes
    /// happen in the per-frame / per-pass scaffolding (P5.4) — engine
    /// uses one of `update_constants_for_*` helpers per source kind.
    pub fn set_constant(&mut self, extern_idx: Extern, value: [f32; 4]) {
        debug_assert_eq!(
            EXTERN_DESCRIPTIONS[extern_idx as usize].parameter_type,
            ExternParameterType::Vec4,
            "set_constant called on Texture-typed extern {:?}", extern_idx,
        );
        self.constants[extern_idx as usize] = value;
    }

    /// Bind a `Texture`-type extern to a [`Surface`]. The per-pass
    /// walker reads `textures[extern_idx]` and routes it to the
    /// shader's expected sampler register.
    pub fn set_texture(&mut self, extern_idx: Extern, surface: Surface) {
        debug_assert_eq!(
            EXTERN_DESCRIPTIONS[extern_idx as usize].parameter_type,
            ExternParameterType::Texture,
            "set_texture called on Vec4-typed extern {:?}", extern_idx,
        );
        self.textures[extern_idx as usize] = Some(surface);
    }

    pub fn clear_texture(&mut self, extern_idx: Extern) {
        self.textures[extern_idx as usize] = None;
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

// =============================================================================
// `render_method_submit_externs` — the walker
// =============================================================================

/// Mirror of `render_method_submit_externs @ 0x180685580` (dllcache).
///
/// Walks three per-pass dep ranges from the rmt2's `routing_info[]`
/// array — one for textures, two for vec4 constants — and dispatches
/// each entry to either `submit_extern_texture` (texture path) or
/// `get_extern_constant + set_shader_constant` (constant path).
///
/// ## Tag-form mapping (audit 2026-05-13)
///
/// The engine reads three packed `(count<<10 | base&0x3FF)` ranges
/// straddling two 16B `s_render_method_pass` runtime structs. blam-tags
/// parses the H3 MCC schema's 12-`TagBlockIndex` per-pass form, so the
/// protomorph walker skips the runtime-side compaction entirely and
/// iterates the three extern-specific slots directly:
///
/// | Engine walker loop | tag-form slot |
/// |---|---|
/// | 1 (textures) | `RenderMethodTemplatePass::extern_bitmaps` |
/// | 2 (constants A) | `RenderMethodTemplatePass::extern_vertex_real_constants` |
/// | 3 (constants B) | `RenderMethodTemplatePass::extern_pixel_real_constants` |
///
/// The VS-vs-PS ordering of loops 2 and 3 is inferred from the
/// cache-build packer's 3-iteration HLSL compile loop (`sub_140C574F0`
/// in tool.exe) — the rotating `__ROL4__(n2_2, 1)` flag drives 3 D3D
/// reflection queries in extern_bitmaps / VS_real / PS_real order. If
/// this turns out to be wrong, the runtime debug-assert below
/// (`assert_constant_routing`) catches it on the first per-pass write.
///
/// ## Volatility filter
///
/// Walker only submits entries whose `EXTERN_DESCRIPTIONS[idx].volatility`
/// matches the caller's `level`. Engine separates volatile (PerPass)
/// from persistent (Persistent) submission so per-frame state and
/// per-pass state can be wired independently. Caller passes the level
/// it's currently servicing.
pub fn submit_externs_for_shader(
    rasterizer: &mut crate::halo::rasterizer::Rasterizer,
    rmt2: &RenderMethodTemplate,
    pass_idx: usize,
    level: ExternVolatility,
) {
    let pass = match rmt2.passes.get(pass_idx) {
        Some(p) => p,
        None => return,
    };

    // Loop 1 — textures (extern_bitmaps).
    for entry in &rmt2.routing_info[pass.extern_bitmaps.range()] {
        let src = entry.source_index as usize;
        if src >= K_NUMBER_OF_EXTERNS {
            continue;
        }
        if EXTERN_DESCRIPTIONS[src].volatility != level {
            continue;
        }
        // dest_register for textures is the raw u16 sampler slot
        // (no bit-packing decode — engine: `submit_extern_texture(0, src, dest_index)`).
        rasterizer.submit_extern_texture(0, src as u32, entry.destination_index as u32);
    }

    // Loop 2 — vertex-stage extern real constants.
    submit_constant_range(
        rasterizer,
        rmt2,
        &rmt2.routing_info[pass.extern_vertex_real_constants.range()],
        level,
        ExpectedStage::Vertex,
    );

    // Loop 3 — pixel-stage extern real constants.
    submit_constant_range(
        rasterizer,
        rmt2,
        &rmt2.routing_info[pass.extern_pixel_real_constants.range()],
        level,
        ExpectedStage::Pixel,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedStage {
    Vertex,
    Pixel,
}

fn submit_constant_range(
    rasterizer: &mut crate::halo::rasterizer::Rasterizer,
    _rmt2: &RenderMethodTemplate,
    entries: &[blam_tags::render_method::RenderMethodRoutingInfo],
    level: ExternVolatility,
    stage: ExpectedStage,
) {
    for entry in entries {
        let src = entry.source_index as usize;
        if src >= K_NUMBER_OF_EXTERNS {
            continue;
        }
        if EXTERN_DESCRIPTIONS[src].volatility != level {
            continue;
        }
        // Engine decode: slot_id = (high_byte << 16) | low_byte. See
        // `dest_index_to_slot_id` for the verbatim engine derivation.
        let slot_id = dest_index_to_slot_id(entry.destination_index);
        assert_constant_routing(slot_id, src, stage);
        let value = rasterizer.extern_state.get_constant(src as u32);
        rasterizer.set_shader_constant(slot_id, 1, &[value]);
    }
}

/// Audit-style debug guard for the VS/PS ordering hypothesis. Catches
/// the case where Loop 2 entries route to a PS-only cbuffer pool index
/// (or vice versa). Engine cbuffer naming pattern: `*VS` / `*PS`
/// suffix per pool slot, with VS-stage cbuffers clustered around
/// pool indices 0x27..=0x2B (`_ViewVS`/`_ExposureVS`/`_AtmosphereVS`/
/// `_LightingVS`/`_ShadowProjVS`) plus 0x15 (`_DecoratorsLightsVS`),
/// 0x6A (`_DynamicLightClipVS`); PS-stage cbuffers cluster around
/// 0x2C..=0x33 plus a handful of postprocess slots.
///
/// We don't have the full 112-entry inventory as data in protomorph
/// yet, so this guard checks the few clear-cut bands only and ignores
/// the rest. A misrouting at the band boundary stays loud (the wrong
/// stage's shader won't see the value); the guard's job is to catch
/// the EASY case (e.g. Loop 2 trying to write to MiscPS).
/// Pure-math: encode an rmt2 routing-info `destination_index` into the
/// 32-bit slot_id format `set_shader_constant` expects. Engine math
/// (from `render_method_submit_externs` decompile) is:
///
/// ```c
/// register = ((16 * (uint8_t)dest) >> 4)
///          | ((((uint8_t)dest << 24) | (dest & 0xFF00)) << 8);
/// ```
///
/// Reduces to `(high_byte << 16) | low_byte`. Exposed for tests; the
/// walker inlines the same expression.
#[inline]
pub fn dest_index_to_slot_id(dest_index: u16) -> u32 {
    let dest = dest_index as u32;
    ((dest & 0xFF00) << 8) | (dest & 0x00FF)
}

fn assert_constant_routing(slot_id: u32, src: usize, stage: ExpectedStage) {
    let cbuffer_idx = (slot_id >> 16) & 0x7FF;
    // Bands derived from `reference_engine_cbuffer_inventory.md`.
    let pure_vs_band = matches!(cbuffer_idx, 0x15 | 0x27 | 0x29 | 0x2A | 0x2B | 0x6A);
    let pure_ps_band = matches!(cbuffer_idx, 0x2C..=0x33 | 0x37 | 0x42 | 0x47 | 0x48);
    match stage {
        ExpectedStage::Vertex => {
            assert!(
                !pure_ps_band,
                "VS-extern routing from extern {} ({:?}) lands in PS cbuffer 0x{:02X}: \
                 the Loop-2 = extern_vertex_real_constants hypothesis may be wrong",
                src,
                EXTERN_DESCRIPTIONS[src].name,
                cbuffer_idx,
            );
        }
        ExpectedStage::Pixel => {
            assert!(
                !pure_vs_band,
                "PS-extern routing from extern {} ({:?}) lands in VS cbuffer 0x{:02X}: \
                 the Loop-3 = extern_pixel_real_constants hypothesis may be wrong",
                src,
                EXTERN_DESCRIPTIONS[src].name,
                cbuffer_idx,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim test against the engine bit-mangle. Engine code:
    ///   register = ((16 * (uint8_t)dest) >> 4)
    ///            | ((((uint8_t)dest << 24) | (dest & 0xFF00)) << 8);
    /// — should equal `(high_byte << 16) | low_byte`.
    fn engine_reference(dest_index: u16) -> u32 {
        let dest = dest_index as u32;
        let low = (16 * (dest & 0xFF)) >> 4; // = dest & 0xFF
        let high = ((dest & 0xFF) << 24).wrapping_mul(0); // <<32 overflows on i32 — engine truncates
        let mid = (dest & 0xFF00) << 8;
        low | high | mid
    }

    #[test]
    fn dest_index_decode_matches_engine_for_sample_routes() {
        // Cases from `reference_engine_shader_constant_routing.md` real-world slot IDs.
        // `actually_calc_albedo`-flavored ones (with sub_byte) only happen through
        // set_shader_constant_bool; here we test the routing-info plain decode.
        let cases = [
            // dest_index → expected slot_id
            (0x2F05_u16, 0x002F_0005_u32), // MiscPS entry 5
            (0x2800_u16, 0x0028_0000_u32), // ExposureVS entry 0
            (0x2D00_u16, 0x002D_0000_u32), // ExposurePS entry 0
            (0x3100_u16, 0x0031_0000_u32), // SimpleLightsPS entry 0
            (0x3101_u16, 0x0031_0001_u32), // SimpleLightsPS entry 1
            (0x4201_u16, 0x0042_0001_u32), // PostProcessPS entry 1
            (0x1500_u16, 0x0015_0000_u32), // DecoratorsLightsVS entry 0
        ];
        for (dest, expected) in cases {
            let got = dest_index_to_slot_id(dest);
            assert_eq!(
                got, expected,
                "dest 0x{dest:04X} → got 0x{got:08X}, expected 0x{expected:08X}",
            );
            // Cross-check against the verbatim engine math
            assert_eq!(got, engine_reference(dest));
        }
    }

    #[test]
    fn extern_table_has_49_entries() {
        assert_eq!(EXTERN_DESCRIPTIONS.len(), 49);
        assert_eq!(EXTERN_DESCRIPTIONS.len(), K_NUMBER_OF_EXTERNS);
    }

    #[test]
    fn texture_typed_externs_match_audit() {
        // Audit-verified texture indices (m_type == 0): 1, 2, 3, 11, 12, 45, 46.
        let textures: Vec<usize> = (0..K_NUMBER_OF_EXTERNS)
            .filter(|&i| EXTERN_DESCRIPTIONS[i].parameter_type == ExternParameterType::Texture)
            .collect();
        assert_eq!(textures, vec![1, 2, 3, 11, 12, 45, 46]);
    }
}
