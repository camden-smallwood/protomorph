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
pub mod rasterizer_clear;
pub mod rasterizer_constants;
pub mod rasterizer_dynamic_render_targets;
pub mod rasterizer_hue_saturation;
pub mod rasterizer_lut_control;
pub mod rasterizer_occlusion_queries;
pub mod rasterizer_render_targets;
pub mod rasterizer_resource_definitions;
pub mod rasterizer_stipple;
pub mod rasterizer_vertex_attributes;

pub use rasterizer_constants::*;
pub use rasterizer_dynamic_render_targets::{
    DynamicRenderTarget, DynamicRenderTargetType, DynamicRenderTargets, SurfaceEntry,
    SurfaceTable, K_MAXIMUM_DYNAMIC_RENDER_TARGETS,
};
pub use rasterizer_render_targets::{
    resolve_surface_size, DxgiFormat, SurfaceDescription, SurfaceSpecializationFlags,
    SURFACE_DESCRIPTIONS,
};

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

    // -------- State setters (most-called subset of Ares c_rasterizer)
    //
    // Bodies stash pending state for the next draw. wgpu pipeline
    // construction uses these via the state caches.

    /// `c_rasterizer::set_alpha_blend_mode`
    pub fn set_alpha_blend_mode(&mut self, mode: AlphaBlendMode) {
        self.pending_alpha_blend = mode;
    }
    /// `c_rasterizer::set_alpha_blend_mode_no_cache` — same as
    /// set_alpha_blend_mode for our wgpu impl (no separate cache
    /// path).
    pub fn set_alpha_blend_mode_no_cache(&mut self, mode: AlphaBlendMode) {
        self.pending_alpha_blend = mode;
    }
    /// `c_rasterizer::set_alpha_to_coverage_enabled`
    pub fn set_alpha_to_coverage_enabled(&mut self, enabled: bool, _object_key: i32) {
        self.pending_alpha_to_coverage = enabled;
    }
    /// `c_rasterizer::set_blend_factor` (float[4] overload).
    pub fn set_blend_factor(&mut self, factor: &[f32; 4]) {
        self.pending_blend_factor = *factor;
    }
    /// `c_rasterizer::set_blend_factor` (packed u32 overload).
    pub fn set_blend_factor_argb(&mut self, color: u32) {
        let a = ((color >> 24) & 0xFF) as f32 / 255.0;
        let r = ((color >> 16) & 0xFF) as f32 / 255.0;
        let g = ((color >> 8) & 0xFF) as f32 / 255.0;
        let b = (color & 0xFF) as f32 / 255.0;
        self.pending_blend_factor = [r, g, b, a];
    }
    /// `c_rasterizer::set_separate_alpha_blend_mode`
    pub fn set_separate_alpha_blend_mode(&mut self, mode: SeparateAlphaBlendMode) {
        self.pending_separate_alpha_blend = mode;
    }
    /// `c_rasterizer::set_z_buffer_mode`
    pub fn set_z_buffer_mode(&mut self, mode: ZBufferMode) {
        self.pending_z_buffer = mode;
    }
    /// `c_rasterizer::get_z_buffer_mode`
    pub fn get_z_buffer_mode(&self) -> ZBufferMode {
        self.pending_z_buffer
    }
    /// `c_rasterizer::set_stencil_mode`
    pub fn set_stencil_mode(&mut self, mode: StencilMode) {
        self.pending_stencil = mode;
    }
    /// `c_rasterizer::set_stencil_mode_with_value`
    pub fn set_stencil_mode_with_value(&mut self, mode: StencilMode, value: u8) {
        self.pending_stencil = mode;
        self.pending_stencil_value = value;
    }
    /// `c_rasterizer::set_stencil_write_mask`
    pub fn set_stencil_write_mask(&mut self, mask: u8) {
        self.pending_stencil_write_mask = mask;
    }
    /// `c_rasterizer::set_stencil_ref`
    pub fn set_stencil_ref(&mut self, value: u8) {
        self.pending_stencil_ref = value;
    }
    /// `c_rasterizer::set_color_write_enable`
    pub fn set_color_write_enable(&mut self, target_index: i32, enable: u32) {
        if let Some(slot) = self.pending_color_write_enable.get_mut(target_index as usize) {
            *slot = enable;
        }
    }
    /// `c_rasterizer::set_cull_mode`
    pub fn set_cull_mode(&mut self, mode: CullMode) {
        self.pending_cull = mode;
    }
    pub fn get_cull_mode(&self) -> CullMode { self.pending_cull }
    /// `c_rasterizer::set_fill_mode`
    pub fn set_fill_mode(&mut self, mode: FillMode) {
        self.pending_fill = mode;
    }
    pub fn get_fill_mode(&self) -> FillMode { self.pending_fill }

    /// `c_rasterizer::set_scissor_rect`. Phase fills body via
    /// `wgpu::RenderPass::set_scissor_rect`.
    pub fn set_scissor_rect(&mut self, _region: Option<crate::halo::render::render_cameras::Rectangle2d>) {
        todo!("set_scissor_rect — apply at draw time via RenderPass")
    }

    /// `c_rasterizer::invalidate_render_state_cache`
    pub fn invalidate_render_state_cache(&mut self) {
        // wgpu has no cache to invalidate per-call; pipelines are
        // created up front and looked up from caches at draw time.
    }
    /// `c_rasterizer::invalidate_shader_state`
    pub fn invalidate_shader_state(&mut self) {}

    // -------- Sampler state (Ares rasterizer.h:265+)
    //
    // Halo binds samplers + textures by index slot; we mirror the
    // slot-based API. Internal storage is the per-bind-group state
    // that the next draw call materializes into a wgpu BindGroup.

    /// `c_rasterizer::set_sampler_address_mode(slot, mode)` —
    /// uvw = mode triple. The 4-arg overload that takes mode_u/v/w
    /// is the same call with three modes per axis.
    pub fn set_sampler_address_mode(&mut self, _sampler_index: i32, _mode: SamplerAddressMode) {
        todo!("set_sampler_address_mode — phase 2 late: stash in pending sampler table")
    }
    /// `c_rasterizer::set_sampler_address_mode` (per-axis overload).
    pub fn set_sampler_address_mode_uvw(
        &mut self,
        _sampler_index: i32,
        _mode_u: SamplerAddressMode,
        _mode_v: SamplerAddressMode,
        _mode_w: SamplerAddressMode,
    ) {
        todo!("set_sampler_address_mode (uvw) — phase 2 late")
    }
    /// `c_rasterizer::set_sampler_filter_mode`
    pub fn set_sampler_filter_mode(&mut self, _sampler_index: i32, _mode: SamplerFilterMode) {
        todo!("set_sampler_filter_mode — phase 2 late")
    }
    /// `c_rasterizer::set_sampler_texture(slot, c_rasterizer_texture_ref)`
    /// — binds the texture handle. Phase 3 wires the SurfaceTable
    /// behind c_rasterizer_texture_ref.
    pub fn set_sampler_texture(&mut self, _sampler_index: i32, _texture: TextureRef) {
        todo!("set_sampler_texture — phase 3")
    }
    /// `c_rasterizer::clear_sampler_texture_cache`
    pub fn clear_sampler_texture_cache(&mut self, _sampler_index: i32) {
        todo!("clear_sampler_texture_cache")
    }
    /// `c_rasterizer::clear_sampler_textures(samplers_flag)` —
    /// 0xFFFF clears all.
    pub fn clear_sampler_textures(&mut self, _samplers_flag: u32) {
        todo!("clear_sampler_textures")
    }

    /// Vertex sampler variants (Ares rasterizer.h:282-287).
    pub fn set_vertex_sampler_address_mode(&mut self, _i: i32, _m: SamplerAddressMode) { todo!() }
    pub fn set_vertex_sampler_address_mode_uvw(
        &mut self, _i: i32, _u: SamplerAddressMode, _v: SamplerAddressMode, _w: SamplerAddressMode,
    ) { todo!() }
    pub fn set_vertex_sampler_filter_mode(&mut self, _i: i32, _m: SamplerFilterMode) { todo!() }
    pub fn set_vertex_sampler_texture(&mut self, _i: i32, _t: TextureRef) { todo!() }

    /// Compute sampler variants.
    pub fn set_compute_sampler_address_mode(&mut self, _i: i32, _m: SamplerAddressMode) { todo!() }
    pub fn set_compute_sampler_address_mode_uvw(
        &mut self, _i: i32, _u: SamplerAddressMode, _v: SamplerAddressMode, _w: SamplerAddressMode,
    ) { todo!() }
    pub fn set_compute_sampler_filter_mode(&mut self, _i: i32, _m: SamplerFilterMode) { todo!() }
    pub fn set_compute_sampler_texture(&mut self, _i: i32, _t: TextureRef) { todo!() }

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
    /// `c_rasterizer::set_single_shader_constant` — engine convenience
    /// for `set_shader_constant(constant_index, 1, &constant)`.
    pub fn set_single_shader_constant(&mut self, slot_id: u32, constant: &[f32; 4]) {
        self.cbuffer_pool
            .set_shader_constant(slot_id, 1, std::slice::from_ref(constant));
    }
    /// `c_rasterizer::set_shader_constant_single(slot, float)` — single
    /// scalar in the `.x` lane (writes 4 bytes at the entry's byte 0).
    pub fn set_shader_constant_single(&mut self, slot_id: u32, value: f32) {
        let v = [value, 0.0, 0.0, 0.0];
        self.cbuffer_pool
            .set_shader_constant(slot_id, 1, std::slice::from_ref(&v));
    }
    /// `c_rasterizer::set_shader_constant_int` — wire format not yet
    /// audited from the dllcache. Engine has this function but no PC
    /// callsites are yet exercised by the protomorph frame loop, so the
    /// concrete bit-pattern (does it OR a flag like `_bool` does?) is
    /// unverified. Stub kept as `todo!()` so callers panic loudly rather
    /// than silently miswrite.
    pub fn set_shader_constant_int(&mut self, _slot_id: u32, _count: i32, _constants: &[i32]) {
        todo!("set_shader_constant_int — audit engine wrapper at impl time")
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

    /// Mirror of engine `submit_extern_texture(stage, extern_idx, dest_register)`
    /// invoked from `render_method_submit_externs`'s texture loop. Records
    /// the binding intent into `pending_texture_bindings` — P5.5 bind-group
    /// builder consumes the list at draw time.
    ///
    /// If `extern_state.textures[extern_idx]` is `None` (the extern was
    /// never populated for this frame), the call is silently dropped —
    /// engine path does the same (skips the bind).
    pub fn submit_extern_texture(&mut self, stage: u32, extern_idx: u32, dest_register: u32) {
        let Some(surface) = self.extern_state.textures[extern_idx as usize] else {
            return;
        };
        self.pending_texture_bindings.push(
            crate::halo::render_methods::externs::PendingTextureBinding {
                stage,
                extern_idx,
                dest_register,
                surface,
            },
        );
    }

    /// Drain the per-draw pending texture bindings. Call after building
    /// the wgpu bind group for one draw, before the next draw's
    /// `submit_externs_for_shader` repopulates.
    pub fn clear_pending_texture_bindings(&mut self) {
        self.pending_texture_bindings.clear();
    }

    /// VS-specific shader constants.
    pub fn set_vertex_shader_constant(&mut self, _i: i32, _count: i32, _constants: &[[f32; 4]]) { todo!() }
    pub fn set_vertex_shader_constant_owned(&mut self, _i: i32, _count: i32, _constants: &[[f32; 4]]) { todo!() }
    pub fn set_vertex_shader_constant_int(&mut self, _i: i32, _count: i32, _constants: &[i32]) { todo!() }
    pub fn set_vertex_shader_constant_bool(&mut self, _i: i32, _count: i32, _constants: &[i32]) { todo!() }

    /// PS-specific shader constants.
    pub fn set_pixel_shader_constant(&mut self, _i: i32, _count: i32, _constants: &[[f32; 4]]) { todo!() }
    pub fn set_pixel_shader_constant_single(&mut self, _i: i32, _value: f32) { todo!() }
    pub fn set_pixel_shader_constant_int(&mut self, _i: i32, _count: i32, _constants: &[i32]) { todo!() }

    // -------- Render targets (Ares rasterizer.h:300+)

    /// `c_rasterizer::set_render_target(target_index, surface, mip_level)`.
    /// Stashes the surface in `pending_render_targets[target_index]`.
    /// `mip_level` is ignored on wgpu — pass-time TextureView creation
    /// picks the base mip; engine PC almost always uses base mip too.
    pub fn set_render_target(&mut self, target_index: u32, surface: Surface, _mip_level: i32) {
        if let Some(slot) = self.pending_render_targets.get_mut(target_index as usize) {
            *slot = surface;
        }
    }
    /// `c_rasterizer::set_depth_stencil_surface`. Stashes in
    /// `pending_depth_stencil`.
    pub fn set_depth_stencil_surface(&mut self, surface: Surface, _mip_level: f32) {
        self.pending_depth_stencil = surface;
    }
    /// Read-back accessor for the frame loop to inspect pending state
    /// when opening a wgpu rpass.
    pub fn current_render_target(&self, target_index: u32) -> Surface {
        self.pending_render_targets
            .get(target_index as usize)
            .copied()
            .unwrap_or(Surface::None)
    }
    /// Read-back accessor for the frame loop's depth-stencil binding.
    pub fn current_depth_stencil(&self) -> Surface {
        self.pending_depth_stencil
    }
    /// `c_rasterizer::resolve_surface(surface, sample, color_only, depth_only)`
    /// — Xbox MSAA resolve. wgpu resolves via render-pass color attachments;
    /// we no-op for non-MSAA paths.
    pub fn resolve_surface(&mut self, _surface: Surface, _sample: i32, _color_only: i32, _depth_only: i32) {
        // no-op for our forward pipeline
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
    /// Read-back for the `actually_calc_albedo` cbuffer flag.
    pub fn using_albedo_sampler(&self) -> bool {
        self.pending_using_albedo_sampler
    }

    /// `c_rasterizer::esram_push(surface, ...)` / `esram_pop` —
    /// Xbox 360 ESRAM tile management. wgpu has no equivalent;
    /// no-op.
    pub fn esram_push(&mut self, _surface: Surface, _other: Surface) {}
    pub fn esram_pop(&mut self, _surface: Surface) {}
    /// `c_rasterizer::copy_to_dram(surface)` — Xbox 360 ESRAM →
    /// RAM resolve. No-op on PC.
    pub fn copy_to_dram(&mut self, _surface: Surface) {}

    /// `c_rasterizer::surface_valid(surface)`.
    pub fn surface_valid(&self, surface: Surface) -> bool {
        self.surface_table.surface_valid(surface)
    }

    /// `c_rasterizer::implicit_geometry_submit` — flushes any
    /// implicit geometry submissions queued via the implicit
    /// geometry path. Phase 12 wires.
    pub fn implicit_geometry_submit(&mut self) {}

    // -------- Render-pass open helpers
    //
    // The wgpu render-pass API borrows from the `CommandEncoder` and
    // from each attachment's `TextureView`. Returning a fully-built
    // `wgpu::RenderPass` from a `&self` method is impractical for
    // lifetime reasons. Instead these helpers return attachment
    // arrays that the caller plugs into a `RenderPassDescriptor`
    // inline.

    /// Resolve `pending_render_targets[]` to a `RenderPassColorAttachment`
    /// array sized to `K_NUMBER_OF_COLOR_SURFACES`. Each slot is
    /// `None` if the bound `Surface` is `Surface::None` or
    /// unallocated; otherwise present with `LoadOp::Load` (or `Clear`
    /// per `clears[]`).
    pub fn color_attachments<'a>(
        &'a self,
        clears: &[Option<wgpu::Color>; K_NUMBER_OF_COLOR_SURFACES],
    ) -> [Option<wgpu::RenderPassColorAttachment<'a>>; K_NUMBER_OF_COLOR_SURFACES] {
        std::array::from_fn(|i| {
            let surface = self.pending_render_targets[i];
            if surface == Surface::None {
                return None;
            }
            let entry = self.surface_table.get(surface)?;
            Some(wgpu::RenderPassColorAttachment {
                view: &entry.view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: match clears[i] {
                        Some(c) => wgpu::LoadOp::Clear(c),
                        None => wgpu::LoadOp::Load,
                    },
                    store: wgpu::StoreOp::Store,
                },
            })
        })
    }

    /// Resolve `pending_depth_stencil` to a depth-stencil attachment.
    /// `depth_clear`/`stencil_clear` = `Some(value)` to clear,
    /// `None` to load. Returns `None` if no depth-stencil surface is
    /// bound or its texture isn't allocated.
    pub fn depth_stencil_attachment<'a>(
        &'a self,
        depth_clear: Option<f32>,
        stencil_clear: Option<u32>,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'a>> {
        let surface = self.pending_depth_stencil;
        if surface == Surface::None {
            return None;
        }
        let entry = self.surface_table.get(surface)?;
        let depth_ops = wgpu::Operations {
            load: match depth_clear {
                Some(d) => wgpu::LoadOp::Clear(d),
                None => wgpu::LoadOp::Load,
            },
            store: wgpu::StoreOp::Store,
        };
        let stencil_ops = if entry.format.has_stencil_aspect() {
            Some(wgpu::Operations {
                load: match stencil_clear {
                    Some(s) => wgpu::LoadOp::Clear(s),
                    None => wgpu::LoadOp::Load,
                },
                store: wgpu::StoreOp::Store,
            })
        } else {
            None
        };
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: &entry.view,
            depth_ops: Some(depth_ops),
            stencil_ops,
        })
    }

    /// Engine `c_rasterizer::clear(target_bits, color_argb, depth,
    /// stencil) @ 0x18066D8B0`. `target_bits` decoder:
    /// - bit 0 (0x01): RT0
    /// - bit 1 (0x02): RT1
    /// - bit 2 (0x04): RT2
    /// - bit 3 (0x08): RT3
    /// - bit 4 (0x10): depth
    /// - bit 5 (0x20): stencil
    ///
    /// Builds a `PassClears` from the engine's packed ARGB color +
    /// per-target bit mask. Caller passes to
    /// `color_attachments` / `depth_stencil_attachment` to open the
    /// rpass. Engine call sites: `setup_targets_albedo` uses
    /// `clear(target_bits|0x10, 0x8D1B3C8D, 1.0, 0)` for combined
    /// RT0+RT1+depth clear, then `clear(2, 0x008080FFu, 1.0, 0)` to
    /// re-clear RT1.
    pub fn clear_pass_args(
        &self,
        target_bits: u32,
        color_argb: u32,
        depth: f32,
        stencil: u32,
    ) -> PassClears {
        let mut colors = [None; K_NUMBER_OF_COLOR_SURFACES];
        let argb = unpack_argb(color_argb);
        for i in 0..K_NUMBER_OF_COLOR_SURFACES {
            if (target_bits & (1 << i)) != 0 {
                colors[i] = Some(argb);
            }
        }
        let depth_clear = if (target_bits & 0x10) != 0 { Some(depth) } else { None };
        let stencil_clear = if (target_bits & 0x20) != 0 { Some(stencil) } else { None };
        PassClears { colors, depth: depth_clear, stencil: stencil_clear }
    }

    // -------- Setup-targets helpers (Ares c_rasterizer)
    //
    // These mirror the engine's high-level `setup_targets_*`
    // sequences. Each is a fixed sequence of state-setter calls that
    // configure the active render-target bind set for a specific
    // pass. They don't open wgpu render passes — the frame loop
    // (P4) reads `pending_render_targets` + `pending_depth_stencil`
    // when it opens its rpasses.

    /// `c_rasterizer::setup_targets_albedo(clear_stencil, is_clear) @
    /// 0x180670CF0` — binds the G-buffer MRT:
    /// - RT0 = `_surface_accum_HDR` (albedo accumulator)
    /// - RT1 = `_surface_post_HDR`  (normal/specular target)
    /// - RT2, RT3 = `_surface_none`
    /// - DS = `_surface_depth_stencil`
    ///
    /// When `is_clear`, engine clears RT0+RT1+depth (color 0x8D1B3C8D,
    /// depth 1.0) then re-clears RT1 to 0x008080FF (encoded zero
    /// normal). Clear-state is stashed via `set_pending_clears`;
    /// the frame loop applies it when opening the rpass.
    pub fn setup_targets_albedo(&mut self, _clear_stencil: bool, _is_clear: bool) {
        self.set_depth_stencil_surface(Surface::DepthStencil, 0.0);
        self.set_render_target(3, Surface::None, -1);
        self.set_render_target(0, Surface::AccumHdr, -1);
        self.set_render_target(1, Surface::PostHdr, -1);
        self.set_render_target(2, Surface::None, -1);
        // viewport/scissor restore + clear-state stash happen in the
        // frame loop when it opens the rpass.
    }

    /// `c_rasterizer::setup_targets_static_lighting(view_exposure,
    /// illum_scale, render_to_HDR_target, HDR_stops, clear,
    /// copy_albedo_pc) @ 0x180670DB0` — binds the SL accumulator
    /// MRT:
    /// - RT0 = `_surface_screenshot_composite_depth`
    /// - RT1 = `_surface_screenshot_composite_cubemap` (if HDR)
    /// - RT2, RT3 = `_surface_none`
    /// - DS = `_surface_depth_stencil`
    ///
    /// `copy_albedo_pc` triggers a PC-only blit accum_HDR →
    /// composite_depth (handled by frame loop / postprocess code).
    /// Exposure cbuffer writes happen via
    /// `setup_render_target_globals_with_exposure` — separate call.
    pub fn setup_targets_static_lighting(
        &mut self,
        _view_exposure: f32,
        _illum_scale: f32,
        render_to_hdr_target: bool,
        _hdr_target_stops: f32,
        _clear: bool,
        _copy_albedo_pc: bool,
    ) {
        self.set_depth_stencil_surface(Surface::DepthStencil, 0.0);
        self.set_render_target(1, Surface::None, -1);
        self.set_render_target(2, Surface::None, -1);
        self.set_render_target(3, Surface::None, -1);
        self.set_render_target(0, Surface::ScreenshotCompositeDepth, -1);
        if render_to_hdr_target {
            self.set_render_target(1, Surface::ScreenshotCompositeCubemap, -1);
        }
    }

    /// `c_rasterizer::setup_targets_static_lighting_alpha_blend(render_to_HDR, _) @ 0x180670FC0`.
    /// Same target set as `setup_targets_static_lighting` but caller
    /// pairs with `set_alpha_blend_mode(_alpha_blend)` for the
    /// transparent / lens-flare passes.
    pub fn setup_targets_static_lighting_alpha_blend(&mut self, render_to_hdr_target: bool, _unused: bool) {
        self.setup_targets_static_lighting(1.0, 1.0, render_to_hdr_target, 0.0, false, false);
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

/// Mirror of `c_rasterizer_texture_ref` (Ares
/// `rasterizer_textures.h`). Halo's runtime opaque texture handle
/// — wraps a tag index + cached D3D11 SRV pointer. We map this to
/// a Surface or a sampler-bound `wgpu::TextureView` indirection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TextureRef {
    pub datum_ref: i32,
}

impl TextureRef {
    pub const INVALID: TextureRef = TextureRef { datum_ref: -1 };
    pub fn is_valid(self) -> bool { self.datum_ref >= 0 }
}

/// Aggregated clear state for the next render-pass. Built by
/// `Rasterizer::clear_pass_args` from an engine-style
/// `c_rasterizer::clear(target_bits, color_argb, depth, stencil)`
/// call. Passed to `color_attachments` / `depth_stencil_attachment`
/// to populate the `LoadOp` of each attachment.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassClears {
    pub colors: [Option<wgpu::Color>; K_NUMBER_OF_COLOR_SURFACES],
    pub depth: Option<f32>,
    pub stencil: Option<u32>,
}

/// Decode the engine's `c_rasterizer::clear` packed ARGB u32 into a
/// `wgpu::Color`. Engine convention from
/// `reference_engine_clear_conventions.md`:
/// - `0x8D1B3C8D` = (A=0.553, R=0.106, G=0.235, B=0.553) muddy debug
/// - `0x008080FF` = (A=0.000, R=0.500, G=0.500, B=1.000) zero normal
fn unpack_argb(packed: u32) -> wgpu::Color {
    let a = ((packed >> 24) & 0xFF) as f64 / 255.0;
    let r = ((packed >> 16) & 0xFF) as f64 / 255.0;
    let g = ((packed >> 8) & 0xFF) as f64 / 255.0;
    let b = (packed & 0xFF) as f64 / 255.0;
    wgpu::Color { r, g, b, a }
}

/// Helper trait for stencil-aspect detection — used by
/// `Rasterizer::depth_stencil_attachment` to decide whether to emit
/// `stencil_ops` in the attachment.
trait DepthFormatExt {
    fn has_stencil_aspect(self) -> bool;
}

impl DepthFormatExt for wgpu::TextureFormat {
    fn has_stencil_aspect(self) -> bool {
        matches!(
            self,
            wgpu::TextureFormat::Depth24PlusStencil8
                | wgpu::TextureFormat::Depth32FloatStencil8
                | wgpu::TextureFormat::Stencil8
        )
    }
}
