//! Halo `c_rasterizer` mirror.
//!
//! Ports `Ares/source/rasterizer/rasterizer.h` (the runtime
//! GPU-state wrapper). Halo's `c_rasterizer` is hundreds of static
//! methods over D3D11; we mirror the public surface so call sites
//! in ported render code read like Ares, with wgpu underneath.
//!
//! Submodules mirror Ares' `rasterizer/` directory layout. The
//! `dx11/` submodule (mirrored from `Ares/source/rasterizer/dx11/`)
//! has the per-state caches (blend / depth-stencil / rasterizer /
//! input-layout) — Halo's PC port has the same layout.

pub mod dx11;
pub mod rasterizer_constants;
pub mod rasterizer_dynamic_render_targets;
pub mod rasterizer_render_targets;

pub use rasterizer_constants::*;
pub use rasterizer_dynamic_render_targets::{DynamicRenderTargets, SurfaceTable};

/// Mirrors `c_rasterizer` (Ares
/// `source/rasterizer/rasterizer.h:145`). Halo's class is all
/// statics over a global D3D11 device; we own the device + queue
/// here and pass `&mut self` through the pass orchestration.
///
/// State methods (`set_alpha_blend_mode`, `set_z_buffer_mode`, ...)
/// stash the requested state on this struct; they're applied at
/// the next `draw_indexed_primitive`. Mirrors Halo's deferred-state
/// approach in `dx11/`.
///
/// `wgpu::Device` and `wgpu::Queue` are internally `Arc`'d already
/// (cheap `Clone`), so we hold them by value — no extra `Arc<>`
/// wrapping needed.
pub struct Rasterizer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // State caches — mirror `Ares/source/rasterizer/dx11/`
    pub blend_state_cache: dx11::BlendStateCache,
    pub depth_stencil_cache: dx11::DepthStencilStateCache,
    pub rasterizer_state_cache: dx11::RasterizerStateCache,
    pub input_layout_cache: dx11::InputLayoutCache,

    /// 112-slot engine cbuffer pool (`e_constantbuffers`). Holds the
    /// per-frame / per-pass values that `set_shader_constant` writes
    /// (Camera_Position_PS, g_exposure, actually_calc_albedo, etc.).
    /// `flush_cbuffers` uploads dirty slots to GPU before draws.
    pub cbuffer_pool: dx11::CbufferPool,

    /// Engine `d3d11_sampler_state_cache @ 0x1806F1C70` mirror — caches
    /// `wgpu::Sampler` instances per `(filter, address_u, address_v,
    /// address_w)` key. Per-material sampler resolution lives in
    /// downstream BG construction code (`bsp_gpu::resolve_material_resources`
    /// today; broader paths as they migrate off `shared.filtering_sampler`).
    pub sampler_state_cache: dx11::SamplerStateCache,

    // Surface storage. `SurfaceTable` owns the wgpu textures
    // backing the static `Surface` enum;
    // `DynamicRenderTargets` owns runtime texture-camera /
    // mirror / refraction allocations.
    pub surface_table: SurfaceTable,
    pub dynamic_render_targets: DynamicRenderTargets,

    // Pending state — applied at draw call time. Mirrors c_rasterizer's
    // cached-state members tracked across calls.
    pending_alpha_blend: AlphaBlendMode,
    pending_separate_alpha_blend: SeparateAlphaBlendMode,
    pending_z_buffer: ZBufferMode,
    pending_stencil: StencilMode,
    pending_stencil_value: u8,
    pending_stencil_write_mask: u8,
    pending_stencil_ref: u8,
    pending_cull: CullMode,
    pending_fill: FillMode,
    pending_color_write_enable: [u32; K_NUMBER_OF_COLOR_SURFACES],
    pending_alpha_to_coverage: bool,
    pending_blend_factor: [f32; 4],

    // Pending render-target state — set by `set_render_target` and
    // `set_depth_stencil_surface`. Read by `begin_render_pass` /
    // pass-issue helpers when opening a wgpu rpass.
    pending_render_targets: [Surface; K_NUMBER_OF_COLOR_SURFACES],
    pending_depth_stencil: Surface,
    pending_using_albedo_sampler: bool,

    /// 49-entry runtime extern table (engine `g_extern_descriptions` slot
    /// values, separated into texture pointers + vec4 constants). Walker
    /// in `render_method_submit_externs` (P5.2) reads from here and
    /// writes through to `set_shader_constant`. See
    /// `crate::halo::render_methods::externs`.
    pub extern_state: crate::halo::render_methods::externs::ExternState,

    /// Texture-binding intents recorded by `submit_extern_texture`.
    /// Cleared between draws by `clear_pending_texture_bindings`.
    /// P5.5 bind-group builder consumes this list.
    pub pending_texture_bindings:
        Vec<crate::halo::render_methods::externs::PendingTextureBinding>,
}

impl Rasterizer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        display_width: u32,
        display_height: u32,
    ) -> Self {
        let mut s = Self {
            device,
            queue,
            blend_state_cache: dx11::BlendStateCache::new(),
            depth_stencil_cache: dx11::DepthStencilStateCache::new(),
            rasterizer_state_cache: dx11::RasterizerStateCache::new(),
            input_layout_cache: dx11::InputLayoutCache::new(),
            cbuffer_pool: dx11::CbufferPool::new(),
            sampler_state_cache: dx11::SamplerStateCache::new(),
            surface_table: SurfaceTable::new(display_width, display_height),
            dynamic_render_targets: DynamicRenderTargets::new(),
            pending_alpha_blend: AlphaBlendMode::Opaque,
            pending_separate_alpha_blend: SeparateAlphaBlendMode::Off,
            pending_z_buffer: ZBufferMode::Write,
            pending_stencil: StencilMode::Off,
            pending_stencil_value: 0,
            pending_stencil_write_mask: 0xFF,
            pending_stencil_ref: 0,
            pending_cull: CullMode::Cw,
            pending_fill: FillMode::Solid,
            pending_color_write_enable: [color_write_enable::ALL; K_NUMBER_OF_COLOR_SURFACES],
            pending_alpha_to_coverage: false,
            pending_blend_factor: [1.0; 4],
            pending_render_targets: [Surface::None; K_NUMBER_OF_COLOR_SURFACES],
            pending_depth_stencil: Surface::None,
            pending_using_albedo_sampler: false,
            extern_state: crate::halo::render_methods::externs::ExternState::new(),
            pending_texture_bindings: Vec::new(),
        };
        // Pre-allocate the cbuffer pool slots that need an immediate GPU
        // buffer for bind-group construction (no caller has had a chance
        // to write yet, but the BG needs a real buffer to reference).
        //
        // 0x33 _Kernel5PS — `actually_calc_albedo` at byte 88 + other PS
        //                   bool flags. Bound into camera_bgl @ 14.
        let device = s.device.clone();
        s.cbuffer_pool.ensure_slot_allocated(&device, 0x33, 256);
        s
    }

    // -------- Shader constants (Ares rasterizer.h:330+)
    //
    // Halo's shader-constant API is the unified upload to GPU
    // cbuffers. Constants are addressed by an integer index that
    // encodes the cbuffer slot + offset (the runtime extern table
    // documents this layout — see reference_h3_extern_table.md).

    /// `c_rasterizer::set_shader_constant @ 0x1806BEA70` — thin pass-through
    /// to `d3d11_shader_constants::set_shader_constant(slot_id, constants, 16 * count)`.
    /// Routes `count` vec4s into the cbuffer pool slot decoded from the
    /// 32-bit slot_id (bit layout in [`dx11::CbufferPool::set_shader_constant`]).
    pub fn set_shader_constant(&mut self, slot_id: u32, count: i32, constants: &[[f32; 4]]) {
        self.cbuffer_pool
            .set_shader_constant(slot_id, count as u32, constants);
    }
    /// `c_rasterizer::set_shader_constant_bool @ 0x1806BEB20` — ORs
    /// `0x8000000` (bit 27) into `slot_id` and writes `count` × 4-byte
    /// bool ints.
    pub fn set_shader_constant_bool(&mut self, slot_id: u32, count: i32, constants: &[i32]) {
        let as_u32: &[u32] = bytemuck::cast_slice(constants);
        self.cbuffer_pool
            .set_shader_constant_bool(slot_id, count as u32, as_u32);
    }

    /// Upload dirty CPU staging from the cbuffer pool to GPU. Call once
    /// per frame (or per pass setup) prior to recording draws that read
    /// from these uniform buffers.
    pub fn flush_cbuffers(&mut self) {
        self.cbuffer_pool.flush_dirty(&self.device, &self.queue);
    }

    /// `c_rasterizer::set_using_albedo_sampler` — writes `actually_calc_albedo`
    /// bool cbuffer slot. Engine signature is `void set_using_albedo_sampler(bool)`;
    /// callers pass int 0/1 directly per the dllcache convention.
    ///
    /// When enabled, static-lighting shaders sample the G-buffer
    /// (`_surface_accum_HDR`) instead of recomputing albedo. Cbuffer
    /// routing: `set_shader_constant_bool(0x80330005, 1, &value)` —
    /// Kernel5PS (pool 0x33) entry 5 byte 8. Slot routing detailed in
    /// `reference_engine_shader_constant_routing.md`.
    pub fn set_using_albedo_sampler(&mut self, enable: i32) {
        let enabled = enable != 0;
        self.pending_using_albedo_sampler = enabled;
        // Engine body @ 0x18069EF80:
        //   c_rasterizer::g_using_albedo_sampler = value;
        //   actually_calc_albedo = !value;
        //   set_shader_constant_bool(-2144403451, 1, &actually_calc_albedo);
        // -2144403451 = 0x80330005 → Kernel5PS pool 0x33 entry 5 byte 8.
        let actually_calc_albedo: i32 = if enabled { 0 } else { 1 };
        self.set_shader_constant_bool(0x8033_0005, 1, &[actually_calc_albedo]);
    }

    /// `c_rasterizer::setup_render_target_globals_with_exposure(
    /// view_exposure, illum_scale, HDR_stops, alpha_blend) @
    /// 0x180670AD0`. Writes 5 cbuffer entries:
    /// - g_exposure  = (view_exposure, 2^HDR_stops, 1, 1) → cb 0x28 entry 0
    /// - v_exposure  = same → cb 0x2D entry 0
    /// - g_alt_exposure = (illum_scale, illum_scale * view_exposure, 0, 0) → cb 0x28 entry 1
    /// - v_alt_exposure = same → cb 0x2D entry 1
    /// - LDR_gamma2 = HDR_gamma2 = FALSE → cb 0x2F entry 5 (bool overlay)
    ///
    /// PC build ALWAYS writes FALSE for the gamma2 booleans — the
    /// `safe_sqrt(LDR_gamma2 ? gamma_encode : pass_through)` paths in
    /// `render_target_fx.hlsl` resolve to `pass_through`. Decoded in
    /// `reference_engine_frame_loop_full.md`.
    ///
    /// P5.4 — engine-faithful body. Writes 5 cbuffer entries verbatim
    /// from `reference_engine_frame_loop_full.md` lines 203-218:
    /// - `0x280000` → `_ExposureVS.entries[0]` = (view_exposure, 2^stops, 1, 1)
    /// - `0x2D0000` → `_ExposurePS.entries[0]` = same (cross-stage redundant)
    /// - `0x280001` → `_ExposureVS.entries[1]` = (illum_scale, illum_scale*ve, 0, 0)
    /// - `0x2D0001` → `_ExposurePS.entries[1]` = same
    /// - `0x2F0005` → `_MiscPS.entries[5]` two bools = [0, 0] (gamma2 always FALSE on PC)
    ///
    /// `alpha_blend` is ignored — engine's PC build doesn't gate any of
    /// these writes on it (the XBox `restore_last_*` path used it).
    pub fn setup_render_target_globals_with_exposure(
        &mut self,
        view_exposure: f32,
        illum_scale: f32,
        hdr_target_stops: f32,
        _alpha_blend: bool,
    ) {
        let exposure = [view_exposure, 2.0f32.powf(hdr_target_stops), 1.0, 1.0];
        self.set_shader_constant(0x280000, 1, &[exposure]);
        self.set_shader_constant(0x2D0000, 1, &[exposure]);

        let alt_exposure = [illum_scale, illum_scale * view_exposure, 0.0, 0.0];
        self.set_shader_constant(0x280001, 1, &[alt_exposure]);
        self.set_shader_constant(0x2D0001, 1, &[alt_exposure]);

        // LDR_gamma2 + HDR_gamma2 = FALSE on PC (sqrt-encode is XBox-only).
        self.set_shader_constant_bool(0x2F0005, 2, &[0, 0]);
    }
}
