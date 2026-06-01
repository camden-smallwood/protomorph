//! `c_decal::render_all` + `c_decal::render` — per-BSP packed
//! decal mesh storage and (eventually) the draw dispatch.
//!
//! This module owns the post-mesh-build state: each scenario decal
//! placement becomes a contiguous slice of `RasterizerVertexWorld`
//! vertices + `u16` indices in its parent BSP's packed buffer. At
//! render time the player_view albedo pass binds the pipeline and
//! issues one indexed draw per decal.
//!
//! Anchored against dllcache:
//!   `c_decal_system::render_all @ 0x180399310`  (frame entry)
//!   `c_decal::render_all        @ 0x18039b040`  (per-pass loop)
//!   `c_decal::render            @ 0x18039b100`  (per-decal draw)

use std::path::Path;
use std::rc::Rc;

use blam_tags::decal_system::DecalPass;
use wgpu::util::DeviceExt;

use crate::halo::decal::writer::RasterizerVertexWorld;
use crate::halo::render::shared::SharedResources;
use crate::halo::render_methods::material_bindings::{
    fallback_view_for_param, is_linear_param_name,
};
use crate::halo::render_methods::materials::{InlineTexture, MaterialData};
use crate::halo::render_methods::pipeline_cache::{
    stub_choices, HaloPipelineCache, VariantArtifacts,
};
use crate::halo::render_methods::serializer::serialize as serialize_material;
use crate::halo::render_methods::HaloEntryPoint;

/// Per-decal record within a `LoadedBspDecals` buffer. Indexes a
/// contiguous slice of the parent BSP's packed vertex + index
/// buffers (44-byte `RasterizerVertexWorld` vertices and `u16`
/// indices). The palette index points back into
/// `scenario.decal_palette[]` so D4c can resolve the per-decal
/// base_map at render time.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadedDecal {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
    /// Index into `scenario.decal_palette[]` — resolves the
    /// `.decal_system` tag + its first definition's `actual_shader`.
    pub palette_index: u16,
}

/// Per-BSP packed decal buffer. Engine mirror: the per-BSP `c_decal::
/// c_vertex_allocator` + `c_index_allocator` LRU pool that
/// `c_decal::build_mesh` writes through. For preplaced decals (cyberdyne)
/// the pool is fixed-size; eviction/repack only matters for runtime
/// impact decals which are not in scope.
#[derive(Debug, Default)]
pub struct LoadedBspDecals {
    pub vertices: Vec<RasterizerVertexWorld>,
    pub indices: Vec<u16>,
    pub decals: Vec<LoadedDecal>,
}

impl LoadedBspDecals {
    /// Append a freshly packed decal into this BSP's buffer. Returns
    /// the `LoadedDecal` record (already pushed onto `self.decals`),
    /// in case the caller wants to inspect the offsets.
    pub fn append(
        &mut self,
        packed_vertices: &[RasterizerVertexWorld],
        packed_indices: &[u16],
        palette_index: u16,
    ) -> LoadedDecal {
        let vertex_start = self.vertices.len() as u32;
        let index_start = self.indices.len() as u32;
        let base = vertex_start;
        self.vertices.extend_from_slice(packed_vertices);
        // Engine writes per-fragment indices that reference the
        // per-fragment vertex range (0..fragment.output_vertex_count).
        // When we concatenate all decals into one BSP-scoped pool, we
        // rebase each decal's indices by the running vertex cursor.
        // (For a single-fragment decal — the cyberdyne preplaced case
        // — fragment.starting_vertex == 0 so we add `base` directly.)
        self.indices.extend(packed_indices.iter().map(|i| i.wrapping_add(base as u16)));
        let rec = LoadedDecal {
            vertex_start,
            vertex_count: packed_vertices.len() as u32,
            index_start,
            index_count: packed_indices.len() as u32,
            palette_index,
        };
        self.decals.push(rec);
        rec
    }
}

/// Per-scenario decal renderer state — one `LoadedBspDecals` per
/// active BSP slot (aligned 1:1 with `LoadedScenario::active_bsps`).
/// D4b adds GPU buffer fields + a wgpu pipeline; D4c adds per-decal
/// material bind groups.
#[derive(Debug, Default)]
pub struct DecalRenderer {
    pub per_bsp: Vec<LoadedBspDecals>,
}

impl DecalRenderer {
    pub fn new(bsp_count: usize) -> Self {
        let mut per_bsp = Vec::with_capacity(bsp_count);
        for _ in 0..bsp_count {
            per_bsp.push(LoadedBspDecals::default());
        }
        Self { per_bsp }
    }

    pub fn total_decals(&self) -> usize {
        self.per_bsp.iter().map(|b| b.decals.len()).sum()
    }
    pub fn total_vertices(&self) -> usize {
        self.per_bsp.iter().map(|b| b.vertices.len()).sum()
    }
    pub fn total_indices(&self) -> usize {
        self.per_bsp.iter().map(|b| b.indices.len()).sum()
    }
}

// ============================================================================
// GPU-side decal state — built post-load via `bake_decals_gpu`.
// ============================================================================

/// Per-draw decal constants — engine `set_shader_constant(0x140000,
/// fade_vec4)` + `set_shader_constant(0x130000, m_sprite_corner_vec4)`
/// in `c_decal::render @ 0x18039B100`. Written into the per-frame
/// ring buffer; the per-`draw_indexed` dynamic offset picks each
/// decal's slice.
///
/// Layout matches the WGSL `DecalConstants` struct in
/// `material_bindings::emit_wgsl`. 32 bytes natural — padded out to
/// `min_uniform_buffer_offset_alignment` (256 on most adapters) at
/// the ring-buffer level.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DecalConstantsGpu {
    /// Engine `set_shader_constant(0x140000, fade_vec4)`. Per-decal
    /// alpha multiplier from `c_decal_system::get_distance_fade *
    /// lifespan_distance_fade`. Preplaced decals default 1.0.
    pub fade: f32,
    /// Engine `pixel_kill_enabled` bool (slot 1245185). True for the
    /// default sprite branch; false skips the un-sprited UV write in
    /// the VS.
    pub pixel_kill_enabled: u32,
    /// Engine `set_shader_constant(... u_tiles ...)` / `v_tiles`.
    /// Tile-count multipliers for the sprited UV. Default 1.0.
    pub u_tiles: f32,
    pub v_tiles: f32,
    /// Engine `set_shader_constant(0x130000, m_sprite_corner_vec4)`.
    /// Sprite atlas sub-rect `(u0, v0, du, dv)` chosen by
    /// `c_decal_system::create_at_index` from the definition's
    /// sprite block. Default `(0, 0, 1, 1)` (full bitmap).
    pub sprite_corner: [f32; 4],
}

impl Default for DecalConstantsGpu {
    fn default() -> Self {
        Self {
            fade: 1.0,
            pixel_kill_enabled: 1,
            u_tiles: 1.0,
            v_tiles: 1.0,
            sprite_corner: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

const _: () = assert!(std::mem::size_of::<DecalConstantsGpu>() == 32);

/// Resolved per-palette GPU resources. Mirrors the engine's per-decal
/// `render_method_submit(parent_def_idx, &m_shader, ...)` —
/// `c_decal::render @ 0x18039b100`. Each cyberdyne palette entry's
/// `definitions[0]` carries one rmd shader; we ensure-pipeline that
/// shader once and reuse it for every placement bound to that palette.
pub struct GpuDecalPalette {
    /// Pipeline + cbuffer layout + material bgl from
    /// `HaloPipelineCache::ensure(..., HaloEntryPoint::Decal)` —
    /// post-SL composition variant. Targets `[lighting_base,
    /// hdr_dark]`. Used for `c_decal_definition::m_pass ==
    /// _pass_post_static_lighting` (additive overlays).
    pub artifacts: Rc<VariantArtifacts>,
    /// Material bind group for the post-SL pipeline.
    pub material_bind_group: wgpu::BindGroup,
    /// Pipeline variant for `_pass_post_albedo` decals — targets
    /// `[albedo_view, normal_view]` so the decal modifies the
    /// albedo G-buffer that SL re-samples + lights. Engine
    /// `c_decal_system::render_all(_pass_post_albedo)` inside
    /// `c_player_view::render_albedo`.
    pub albedo_artifacts: Rc<VariantArtifacts>,
    /// Material bind group for the post_albedo pipeline. Bound to
    /// the same texture + cbuffer resources as `material_bind_group`
    /// but addressed against `albedo_artifacts.material_bgl` (wgpu
    /// requires per-pipeline-layout bind groups).
    pub albedo_material_bind_group: wgpu::BindGroup,
    /// Identity model bind group — @group(1). Decal vertices live in
    /// world space (engine `write_mesh_fragment` outputs world coords),
    /// so the model matrix is identity. Same shape as
    /// `bsp_gpu::build_instance_bind_groups` for the rigid case.
    pub identity_model_bg: wgpu::BindGroup,
    /// Identity 1-bone node-matrices bind group — @group(3). Decals
    /// don't skin; we still need to bind the slot because the pipeline
    /// layout includes it.
    pub identity_nm_bg: wgpu::BindGroup,
    /// Engine `c_decal_definition::m_pass` — picks which sub-pass
    /// draws this decal. cyberdyne's 46 palette entries are ALL
    /// `PreLighting` (post_albedo).
    pub pass: DecalPass,
    /// Engine `c_decal_definition::flags` — bit 0 = specular_modulate,
    /// 1 = bump_modulate, 2 = has_sprite. Drives the second-pass
    /// alpha-write for bump_modulate (deferred until Step 2's RT1).
    pub def_flags: u32,
    /// Palette name (decal_system tag basename, e.g.
    /// `riverworld_rockblend_decal`). Used by the
    /// `PROTOMORPH_DECAL_ONLY` filter at draw time to isolate
    /// which palette is producing visible artifacts.
    pub palette_name: String,
}

/// GPU vertex + index buffers for one BSP's packed decal mesh. Engine
/// equivalent: the per-BSP `c_vertex_allocator.m_buffer` +
/// `c_index_allocator.m_buffer` LRU pools bound by `c_decal::render`
/// via `set_vertex_stream` + `draw_indexed_primitive`.
pub struct GpuDecalBspBuffers {
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    /// First slot index this BSP's decals occupy in the global
    /// `DecalGpuState::decal_constants_buffer` ring. Each decal's
    /// dynamic offset is `(constants_slot_base + local_decal_idx) *
    /// stride`. Lets the orchestrator address every decal across
    /// all BSPs without re-numbering on every frame.
    pub constants_slot_base: u32,
}

/// GPU-side state for the whole decal subsystem — populated by
/// `bake_decals_gpu` after `LoadedScenario::load` finishes.
pub struct DecalGpuState {
    /// Aligned 1:1 with `scenario.decal_palette[]`. `None` for empty
    /// palette entries / load failures / definitions without a shader.
    pub per_palette: Vec<Option<GpuDecalPalette>>,
    /// Aligned 1:1 with `LoadedScenario::active_bsps[]` (and
    /// `DecalRenderer::per_bsp[]`).
    pub per_bsp: Vec<GpuDecalBspBuffers>,
    /// Per-decal `DecalConstants` ring buffer. One entry per decal in
    /// the global render order (same order as `DecalGpuState::iter`
    /// below). Sized to `total_decals × stride` where stride =
    /// `device.limits().min_uniform_buffer_offset_alignment` (typically
    /// 256 B on Metal/Vulkan PC). Rewritten each frame from the CPU
    /// by `update_decal_constants` before `record_render` dispatches.
    pub decal_constants_buffer: Option<wgpu::Buffer>,
    /// Slot stride in bytes for the ring buffer. Each decal occupies
    /// `[i*stride, i*stride + sizeof::<DecalConstantsGpu>())`. Cached
    /// from `device.limits().min_uniform_buffer_offset_alignment` at
    /// bake time.
    pub decal_constants_stride: u32,
    /// Total decals across all BSPs — drives the constants ring
    /// buffer's slot count.
    pub total_decals: u32,
    /// Per-palette `(start, end)` distance-fade range in world units
    /// (engine `c_decal_system_definition.distance_fade_range`). `None`
    /// when the palette entry didn't author one or didn't load.
    /// Consumed by `update_decal_constants` per frame to drive
    /// `c_decal_system::get_distance_fade @ 0x1803988F0`.
    pub palette_fade_ranges: Vec<Option<(f32, f32)>>,
    /// Per-palette baked lifespan fade — engine `c_decal::render @
    /// 0x18039B100` lines 87-95 multiplies the per-frame
    /// `distance_fade` by
    /// `clamp((m_lifespan - m_age) / m_decay_period, 0, 1)`. For
    /// preplaced decals `m_age = 0`, so the lifespan factor collapses
    /// to `clamp(m_lifespan / m_decay_period, 0, 1)` which we precompute
    /// once per palette at bake time. `m_lifespan` comes from
    /// `DecalDefinition.lifespan.upper`, `m_decay_period` from
    /// `decay_time.upper` (engine `c_decal_system::create_at_index @
    /// 0x180398730:~145` picks the upper bound when randomization is
    /// off — verified in the cyberdyne / shrine corpus). When `decay`
    /// is degenerate (<=0) we fall back to 1.0 to avoid div-by-zero;
    /// when `lifespan` is unset (`<= 0`) engine treats the decal as
    /// infinite-lived so we ship 1.0 as well. Runtime decals (none in
    /// scope yet) would require per-decal age tracking + age!=0 in the
    /// formula — that's a separate Phase 4 once the runtime decal
    /// pool lands.
    pub palette_lifespan_fades: Vec<f32>,
    /// Render-time toggle. When false, `record_render` is a no-op so
    /// you can A/B compare the scene with and without decals. Bound
    /// to KeyP via the main-loop input handler.
    pub render_enabled: bool,
}

impl Default for DecalGpuState {
    fn default() -> Self {
        Self {
            per_palette: Vec::new(),
            per_bsp: Vec::new(),
            decal_constants_buffer: None,
            decal_constants_stride: 0,
            total_decals: 0,
            palette_fade_ranges: Vec::new(),
            palette_lifespan_fades: Vec::new(),
            render_enabled: true,
        }
    }
}

impl DecalGpuState {
    pub fn is_empty(&self) -> bool {
        self.per_palette.iter().all(|p| p.is_none()) || self.per_bsp.is_empty()
    }

    /// Rewrite the `decal_constants` ring buffer with per-decal
    /// payloads for the current frame. Engine equivalent:
    /// `c_decal::render` calling `set_shader_constant(0x140000, fade)`
    /// + `set_shader_constant(0x130000, sprite_corner)` per decal —
    /// we batch all of them into one `queue.write_buffer` call.
    ///
    /// `decal_renderer` provides the CPU-side records (per-BSP decal
    /// list with palette indices); the orchestrator-supplied
    /// `per_decal_fade` slice (1.0 default, distance-fade override
    /// when D6 #4 ships) lets the caller drive runtime fade without
    /// this module reaching into camera state.
    pub fn update_decal_constants(
        &self,
        queue: &wgpu::Queue,
        decal_renderer: &super::render_decals::DecalRenderer,
        camera_position: [f32; 3],
    ) {
        let Some(buf) = self.decal_constants_buffer.as_ref() else { return };
        if self.total_decals == 0 { return };
        let stride = self.decal_constants_stride as usize;
        let mut bytes = vec![0u8; (self.total_decals as usize) * stride];
        for (bsp_idx, gpu_bsp) in self.per_bsp.iter().enumerate() {
            let Some(cpu_bsp) = decal_renderer.per_bsp.get(bsp_idx) else { continue };
            for (local_idx, cpu_decal) in cpu_bsp.decals.iter().enumerate() {
                let palette_idx = cpu_decal.palette_index as usize;
                // Per-decal fade — engine `c_decal::render @ 0x18039B100`
                // lines 87-95:
                //   if (g_decal_fade)
                //     fade = get_distance_fade(parent) *
                //            clamp((lifespan-age)/decay, 0, 1)
                //   else
                //     fade = 1.0
                // We bake lifespan_fade per-palette (preplaced age = 0)
                // and multiply by the per-frame distance_fade. Camera
                // position drives distance.
                let mut fade = 1.0_f32;
                if let Some(Some((start, end))) = self.palette_fade_ranges.get(palette_idx) {
                    if *end > *start && *end > 0.0 {
                        // Distance from camera to the decal's first
                        // vertex (approximation of decal center for
                        // v1 — engine uses m_projection.position).
                        let vs = cpu_decal.vertex_start as usize;
                        if let Some(v) = cpu_bsp.vertices.get(vs) {
                            let dx = v.position[0] - camera_position[0];
                            let dy = v.position[1] - camera_position[1];
                            let dz = v.position[2] - camera_position[2];
                            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                            // `c_decal_system::get_distance_fade`:
                            // before `start` → 1.0, after `end` → 0.0,
                            // linear ramp between.
                            fade = if dist <= *start {
                                1.0
                            } else if dist >= *end {
                                0.0
                            } else {
                                1.0 - (dist - *start) / (*end - *start)
                            };
                        }
                    }
                }
                // Multiply by per-palette lifespan fade (baked at GPU
                // state construction — see `palette_lifespan_fades`
                // doc-comment). Missing entry defaults to 1.0 so a
                // failed-load palette doesn't accidentally null out
                // its decals.
                let lifespan_fade = self
                    .palette_lifespan_fades
                    .get(palette_idx)
                    .copied()
                    .unwrap_or(1.0);
                fade *= lifespan_fade;
                let constants = DecalConstantsGpu {
                    fade,
                    ..Default::default()
                };
                let slot = (gpu_bsp.constants_slot_base as usize) + local_idx;
                let off = slot * stride;
                let payload = bytemuck::bytes_of(&constants);
                bytes[off..off + payload.len()].copy_from_slice(payload);
            }
        }
        queue.write_buffer(buf, 0, &bytes);
    }

    /// Mirror of `c_decal_system::render_all @ 0x180399310` +
    /// `c_decal::render_all @ 0x18039b040` + `c_decal::render @
    /// 0x18039b100`. Iterates the per-BSP decal records and issues one
    /// `draw_indexed` per decal whose palette's `m_pass` matches
    /// `pass`. The pipeline + bind groups come from the per-palette
    /// resolved artifacts; vertex/index buffers from the per-BSP
    /// packed pool.
    ///
    /// Engine state setup the caller is responsible for (matches the
    /// `_z_buffer_mode_decals` + `_cull_mode_off` +
    /// `begin_high_quality_blend` triple from
    /// `c_decal_system::render_all`):
    ///   - rpass with `lighting_base` + `hdr_dark` attached as
    ///     [color0, color1] (loaded, not cleared).
    ///   - depth attachment: load + store, depth-test only (the
    ///     pipeline's `depth_write_enabled=false` for transparent
    ///     subclasses already enforces this).
    ///   - camera bind group at @group(0) bound by the caller.
    pub fn record_render<'rp>(
        &'rp self,
        rpass: &mut wgpu::RenderPass<'rp>,
        decal_renderer: &'rp super::render_decals::DecalRenderer,
        pass: DecalPass,
    ) {
        if !self.render_enabled {
            return;
        }
        let mut drawn = 0usize;
        for (bsp_idx, gpu_bsp) in self.per_bsp.iter().enumerate() {
            let (Some(vbuf), Some(ibuf)) =
                (gpu_bsp.vertex_buffer.as_ref(), gpu_bsp.index_buffer.as_ref())
            else {
                continue;
            };
            let Some(cpu_bsp) = decal_renderer.per_bsp.get(bsp_idx) else {
                continue;
            };
            // Per-engine: bind the BSP-scoped vertex+index buffer
            // ONCE; loop over each placement issuing `draw_indexed`.
            // Engine: `c_decal::render` calls `set_vertex_stream(
            // get_vertex_allocator(this).m_buffer)` per decal, but the
            // allocator is per-BSP so the buffer is the same across
            // the whole loop — we hoist outside.
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint16);

            for (local_idx, cpu_decal) in cpu_bsp.decals.iter().enumerate() {
                let Some(palette) = self
                    .per_palette
                    .get(cpu_decal.palette_index as usize)
                    .and_then(|p| p.as_ref())
                else {
                    continue;
                };
                if palette.pass != pass {
                    continue;
                }
                // `PROTOMORPH_DECAL_ONLY=<substring>` — gate decal
                // draws to a single palette so we can isolate which
                // one is producing the visible artifact. Substring
                // match against the decal_system tag basename (e.g.
                // `riverworld_rockblend_decal`). Empty/unset = no
                // filter, draw all.
                if let Ok(want) = std::env::var("PROTOMORPH_DECAL_ONLY") {
                    if !want.is_empty() && !palette.palette_name.contains(&want) {
                        continue;
                    }
                }
                // Diagnostic counter — log how many placements per
                // palette pass the filter on the first frame. Gated
                // behind PROTOMORPH_DECAL_DRAW_DEBUG=1.
                {
                    if std::env::var("PROTOMORPH_DECAL_DRAW_DEBUG").ok().as_deref() == Some("1") {
                        eprintln!(
                            "[decal-draw] palette='{}' bsp={} local={} pass={:?}",
                            palette.palette_name, bsp_idx, local_idx, pass,
                        );
                    }
                }
                // Pick the pass-appropriate pipeline + bind group.
                // post_albedo: writes into albedo_view (G-buffer) via
                // `albedo_artifacts`. post_SL: writes into
                // lighting_base via `artifacts`.
                let (pipeline, mat_bg) = match pass {
                    DecalPass::PreLighting => (
                        &palette.albedo_artifacts.pipeline,
                        &palette.albedo_material_bind_group,
                    ),
                    DecalPass::PostLighting => (
                        &palette.artifacts.pipeline,
                        &palette.material_bind_group,
                    ),
                };
                rpass.set_pipeline(pipeline);
                // model_bgl + node_matrices_bgl both declare
                // `has_dynamic_offset = true`, so wgpu requires a
                // 1-element offset slice when binding. We allocated a
                // single uniform at offset 0, so pass 0.
                rpass.set_bind_group(1, &palette.identity_model_bg, &[0]);
                // material BGL also has a dynamic-offset binding for
                // the per-decal DecalConstants slot. Compute slot =
                // (bsp's base) + (decal's local index in that BSP) and
                // multiply by the ring-buffer stride.
                let constants_slot = gpu_bsp.constants_slot_base + local_idx as u32;
                let constants_offset = constants_slot * self.decal_constants_stride;
                // Path B: material BGL now has TWO dynamic-offset entries —
                // cbuffer (binding order: lower) + decal_constants. Cbuffer
                // offset 0 = static slot (decals aren't per-object animated).
                rpass.set_bind_group(2, mat_bg, &[0u32, constants_offset]);
                rpass.set_bind_group(3, &palette.identity_nm_bg, &[0]);
                // `append` already rebased each decal's indices by
                // `cpu_decal.vertex_start` before storing them, so the
                // index buffer values are absolute positions into
                // `self.vertices`. Pass `base_vertex = 0` — wgpu would
                // otherwise add `v_start` a second time, producing a
                // 2× offset that reads vertex data from the wrong
                // decal and renders stretched triangles.
                let i_start = cpu_decal.index_start;
                let i_end = i_start + cpu_decal.index_count;
                rpass.draw_indexed(i_start..i_end, 0, 0..1);
                drawn += 1;
            }
        }
        // Once-per-process diagnostic — confirms record_render fires
        // + how many decals reach `draw_indexed`. Throttled via
        // OnceLock so we don't spam per frame.
        if drawn > 0 {
            use std::sync::OnceLock;
            static LOGGED: OnceLock<()> = OnceLock::new();
            let _ = LOGGED.get_or_init(|| {
                eprintln!(
                    "[decal_render] first frame: {drawn} decals dispatched for pass={pass:?}",
                );
            });
        }
    }
}

/// Build all GPU-side decal state from `LoadedScenario`.
///
/// 1. For each palette entry's first definition: resolve the in-memory
///    `RenderMethod` (`load_shader_material_from_rm`), ensure the
///    HaloEntryPoint::Decal pipeline variant, upload textures,
///    serialize the material cbuffer, build the bind group.
/// 2. For each active BSP: upload the packed
///    `RasterizerVertexWorld` + `u16` index buffers.
///
/// Engine equivalent: the offline cache builder pre-resolves
/// `c_render_method_shader_decal` per palette entry; the runtime
/// `c_decal::build_mesh @ 0x18039B820` writes the indexed mesh into
/// the per-BSP `c_vertex_allocator` / `c_index_allocator` pools that
/// `c_decal::render` later binds.
pub fn bake_decals_gpu(
    shared: &SharedResources,
    halo_pipelines: &mut HaloPipelineCache,
    sampler_cache: &mut crate::halo::rasterizer::dx11::SamplerStateCache,
    tags_root: &Path,
    loaded: &crate::halo::scenario::LoadedScenario,
) -> DecalGpuState {
    // Phase 1: total-decals tally + constants ring-buffer allocation.
    // We allocate it FIRST so per-palette bind groups can reference
    // it directly when they're built in phase 2.
    let total_decals: u32 = loaded
        .decal_renderer
        .per_bsp
        .iter()
        .map(|b| b.decals.len() as u32)
        .sum();
    let stride = shared
        .device
        .limits()
        .min_uniform_buffer_offset_alignment
        .max(std::mem::size_of::<DecalConstantsGpu>() as u32);
    let decal_constants_buffer = if total_decals > 0 {
        Some(shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decal_constants_ring"),
            size: (total_decals as u64) * (stride as u64),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };

    let mut per_palette = Vec::with_capacity(loaded.decal_palette.len());
    let mut palette_fade_ranges: Vec<Option<(f32, f32)>> =
        Vec::with_capacity(loaded.decal_palette.len());
    let mut palette_lifespan_fades: Vec<f32> =
        Vec::with_capacity(loaded.decal_palette.len());
    let mut built_palettes = 0usize;
    let mut empty_palettes = 0usize;
    let mut failed_palettes = 0usize;
    for (i, entry) in loaded.decal_palette.iter().enumerate() {
        // Stash distance_fade_range so `update_decal_constants` per
        // frame can apply `c_decal_system::get_distance_fade`. `None`
        // when the palette didn't load or the authored range is
        // degenerate (engine's get_distance_fade returns 1.0 when
        // `end <= 0` so we filter equivalent inputs here).
        let fade_range = entry.as_ref().and_then(|sys| {
            let (start, end) = sys.distance_fade_range;
            (end > 0.0 && end > start).then_some((start, end))
        });
        palette_fade_ranges.push(fade_range);

        // Engine `c_decal::render` per-decal `fade` multiplies the
        // distance fade by `clamp((lifespan - age) / decay, 0, 1)`.
        // Preplaced age = 0, so the factor reduces to
        // `clamp(lifespan / decay, 0, 1)`. We bake once per palette
        // here so `update_decal_constants` is a multiply per decal.
        // Default = 1.0 when authored data is degenerate (engine path
        // does the same — `g_decal_fade` global disabled or v18 >= 1).
        let lifespan_fade = entry.as_ref()
            .and_then(|sys| sys.definitions.first())
            .map(|def| {
                let lifespan = def.lifespan.1.max(def.lifespan.0); // upper bound
                let decay = def.decay_time.1.max(def.decay_time.0);
                if decay <= 0.0 || lifespan <= 0.0 {
                    1.0  // infinite-lived (engine treats as fully visible)
                } else {
                    (lifespan / decay).clamp(0.0, 1.0)
                }
            })
            .unwrap_or(1.0);
        palette_lifespan_fades.push(lifespan_fade);

        // Pick the first definition's shader as the palette's rmd.
        // The decal_system tag can carry multiple definitions for
        // animated sprite atlases / variant sprites; v1 uses the first
        // one. Engine `c_decal::render` picks the definition by
        // `m_definition_block_index` — we mirror at the build_mesh
        // stage when we attach the definition index to each decal.
        let candidate = entry.as_ref()
            .and_then(|sys| sys.definitions.first())
            .and_then(|def| def.shader.as_ref().map(|rm| (def.pass, def.flags, rm)));
        let Some((pass, def_flags, rm)) = candidate else {
            empty_palettes += 1;
            per_palette.push(None);
            continue;
        };

        let mat_name = format!("decal_palette[{i}]");
        let Some(material) = crate::halo::loader::load_shader_material_from_rm(
            rm, tags_root, &mat_name,
        ) else {
            failed_palettes += 1;
            per_palette.push(None);
            continue;
        };

        // Ensure pipeline variant — Decal entry point keyed by the
        // rmd's category choices (albedo / blend_mode / bump_mapping /
        // tinting). Repeated palettes with the same option vector
        // dedupe through `HaloPipelineCache`.
        let choices = if material.category_choices.is_empty() {
            stub_choices()
        } else {
            material.category_choices.clone()
        };
        // Diagnostic dump — one line per palette so we can attribute the
        // 3 visible scene-decal artifacts (brown polygonal patches,
        // blue iridescent edges, overexposed white) to specific
        // albedo/blend_mode/bump_mapping/tinting combinations. Lists
        // bitmap params too so a missing alpha_map / palette is visible.
        // Gate behind `PROTOMORPH_DEBUG_DECAL_PALETTES=1` to keep the
        // default-on output quiet.
        if std::env::var("PROTOMORPH_DEBUG_DECAL_PALETTES").ok().as_deref() == Some("1") {
            let pal_name = loaded
                .scenario
                .decal_palette
                .get(i)
                .map(|p| p.decal_system.rsplit(['/', '\\']).next().unwrap_or(""))
                .unwrap_or("?");
            let choice_str: String = choices
                .pairs()
                .iter()
                .map(|(c, o)| format!("{c}={o}"))
                .collect::<Vec<_>>()
                .join(",");
            let tex_str: String = material
                .textures
                .iter()
                .map(|t| {
                    let path = rm
                        .parameters
                        .iter()
                        .find(|p| p.parameter_name == t.name)
                        .map(|p| p.bitmap_path.rsplit(['/', '\\']).next().unwrap_or("?"))
                        .unwrap_or("(stub)");
                    // Per-binding sampler config — for BFS decals,
                    // boundary-vertex UVs land at ~[0,1] so wrong address
                    // mode at the edges produces visible opaque/banding.
                    let sampler = crate::halo::rasterizer::dx11::SamplerKey
                        ::for_binding_in_resolved(&material.resolved, &t.name)
                        .map(|k| format!("{:?}/{:?}", k.address_u, k.filter))
                        .unwrap_or_else(|| "default".into());
                    format!("{}={}({})", t.name, path, sampler)
                })
                .collect::<Vec<_>>()
                .join(" ");
            let sys_flags = entry.as_ref().map(|sys| sys.flags).unwrap_or(0);
            eprintln!(
                "[decal-palette {i}] '{pal_name}' def_flags=0x{def_flags:x} sys_flags=0x{sys_flags:x} pass={pass:?} \
                 choices=[{choice_str}] textures=[{tex_str}]"
            );
        }
        let artifacts = halo_pipelines.ensure(
            shared,
            &material.resolved,
            &material.cbuffer,
            &choices,
            HaloEntryPoint::Decal,
            tags_root,
        );
        let albedo_artifacts = halo_pipelines.ensure(
            shared,
            &material.resolved,
            &material.cbuffer,
            &choices,
            HaloEntryPoint::DecalAlbedo,
            tags_root,
        );

        // Build material bind group — same shape as rmsh's: one entry
        // per `MaterialBindings::textures[i]`, sampler at `sampler_slot`,
        // cbuffer at `cbuffer_slot`, PLUS the decal_constants slot
        // pointing at the ring buffer (first 32 bytes; the
        // `set_bind_group` dynamic offset re-targets per draw).
        // Texture views come from the loaded `InlineTexture` (or a
        // per-name fallback when the material doesn't carry that
        // texture).
        let material_bind_group = build_decal_material_bind_group(
            shared, sampler_cache, &artifacts, &material, decal_constants_buffer.as_ref(),
        );
        // Same texture/cbuffer resources, addressed against the
        // DecalAlbedo pipeline's material_bgl (wgpu requires
        // per-bgl bind_group objects even when contents match).
        let albedo_material_bind_group = build_decal_material_bind_group(
            shared, sampler_cache, &albedo_artifacts, &material, decal_constants_buffer.as_ref(),
        );

        // Identity bind groups for groups 1 + 3 — decals are world-space
        // rigid: model matrix = I, no skinning.
        let identity_model_bg = build_identity_model_bg(shared);
        let identity_nm_bg = build_identity_nm_bg(shared);

        // Fail-loud detector for engine-faithful Phase 2 deferrals.
        // `def_flags` bits:
        //   bit 0 = specular_modulate → engine `c_decal::render` calls
        //           `set_separate_alpha_blend_to_constant` +
        //           `set_blend_factor(specular_multiplier)` per draw.
        //           Our path uses regular alpha_blend on RT0.alpha →
        //           spec_mask at decal pixels is decal_alpha² instead
        //           of `decal.a * spec_mult + rock.a * (1-spec_mult)`.
        //   bit 1 = bump_modulate → engine emits a SECOND draw with
        //           `_alpha_blend_alpha_blend` to write the bump
        //           independently of the shader's main blend_mode.
        //           Our single-pass RT1 alpha_blend approximates this
        //           for `def_flags=0` (per the doc-comment above).
        //   bit 2 = has_sprite → consumes `sprite_corner` already.
        // Per [[project_decal_phase1_2026_05_17]] Phase 2A/2B: no
        // observed scenario in our test corpus sets bits 0 or 1; the
        // visible-delta call is to defer the spec_modulate /
        // bump_modulate engine-faithful path until a scenario surfaces
        // it. The eprintln below catches the moment that happens so we
        // can stop deferring.
        if def_flags & 0b11 != 0 {
            eprintln!(
                "[decal] palette[{i}] '{}' has def_flags=0b{:b} — bit0 \
                 (specular_modulate) or bit1 (bump_modulate) set. \
                 Phase 2A/2B engine-faithful path is deferred; spec or \
                 bump output at this palette's decals will diverge \
                 from engine. See project_decal_phase1_2026_05_17.md.",
                mat_name,
                def_flags & 0b11,
            );
        }

        built_palettes += 1;
        let palette_name = loaded
            .scenario
            .decal_palette
            .get(i)
            .map(|p| p.decal_system.rsplit(['/', '\\']).next().unwrap_or("").to_string())
            .unwrap_or_default();
        per_palette.push(Some(GpuDecalPalette {
            artifacts,
            material_bind_group,
            albedo_artifacts,
            albedo_material_bind_group,
            identity_model_bg,
            identity_nm_bg,
            pass,
            def_flags,
            palette_name,
        }));
    }

    // Phase 3: per-BSP packed buffers + per-BSP constants slot bases.
    // Walk in order so each BSP gets a contiguous slot range in the
    // global constants ring buffer.
    let mut per_bsp: Vec<GpuDecalBspBuffers> = Vec::with_capacity(loaded.decal_renderer.per_bsp.len());
    let mut running_slot_base: u32 = 0;
    for (bsp_idx, cpu) in loaded.decal_renderer.per_bsp.iter().enumerate() {
        if cpu.vertices.is_empty() || cpu.indices.is_empty() {
            per_bsp.push(GpuDecalBspBuffers {
                vertex_buffer: None,
                index_buffer: None,
                constants_slot_base: running_slot_base,
            });
            continue;
        }
        let vertex_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                cpu.vertices.as_ptr() as *const u8,
                cpu.vertices.len() * std::mem::size_of::<RasterizerVertexWorld>(),
            )
        };
        let vertex_buffer = shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("decal_vertices_bsp{bsp_idx}")),
                contents: vertex_bytes,
                usage: wgpu::BufferUsages::VERTEX,
            },
        );
        let index_buffer = shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("decal_indices_bsp{bsp_idx}")),
                contents: bytemuck::cast_slice(&cpu.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        per_bsp.push(GpuDecalBspBuffers {
            vertex_buffer: Some(vertex_buffer),
            index_buffer: Some(index_buffer),
            constants_slot_base: running_slot_base,
        });
        running_slot_base += cpu.decals.len() as u32;
    }

    eprintln!(
        "[decal_gpu] palettes: built={built_palettes} empty={empty_palettes} failed={failed_palettes} | bsp_buffers={} | constants_slots={total_decals} stride={stride}",
        per_bsp.iter().filter(|b| b.vertex_buffer.is_some()).count(),
    );

    DecalGpuState {
        per_palette,
        per_bsp,
        decal_constants_buffer,
        decal_constants_stride: stride,
        total_decals,
        palette_fade_ranges,
        palette_lifespan_fades,
        render_enabled: true,
    }
}

/// Build the per-palette material bind group. Mirrors the same shape
/// the `Renderer::upload_model_data` material-bind-group path uses for
/// rmsh — textures by name, sampler at `sampler_slot`, cbuffer at
/// `cbuffer_slot`, plus the decal_constants ring buffer at
/// `decal_constants_slot` (with `has_dynamic_offset = true`).
fn build_decal_material_bind_group(
    shared: &SharedResources,
    sampler_cache: &mut crate::halo::rasterizer::dx11::SamplerStateCache,
    artifacts: &VariantArtifacts,
    material: &MaterialData,
    decal_constants_buffer: Option<&wgpu::Buffer>,
) -> wgpu::BindGroup {
    let mut texture_views: Vec<(u32, wgpu::TextureView)> =
        Vec::with_capacity(artifacts.bindings.textures.len());
    for tb in &artifacts.bindings.textures {
        let view = match material.find_texture_by_name(&tb.name) {
            Some(tex) => upload_inline_texture(shared, tex, is_linear_param_name(&tb.name)),
            None => fallback_view_for_param(shared, &tb.name, tb.is_cube),
        };
        texture_views.push((tb.slot, view));
    }
    let props_bytes = serialize_material(material, &artifacts.layout);
    let props_buffer = shared.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("decal_material_props"),
            contents: &props_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let mut entries: Vec<wgpu::BindGroupEntry> =
        Vec::with_capacity(texture_views.len() + 3);
    for (slot, view) in &texture_views {
        entries.push(wgpu::BindGroupEntry {
            binding: *slot,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    // Per-key samplers — one per unique `SamplerKey` in the variant's
    // dedup map. For decals these typically all collapse to a single
    // shared Linear/Repeat dedupe entry. Engine equivalent: D3D11
    // `d3d11_sampler_state_cache::get` per texture stage.
    // Collect samplers into a Vec held outside the loop so the borrows
    // in `entries` stay alive until after `create_bind_group`.
    let per_key_samplers: Vec<wgpu::Sampler> = artifacts
        .sampler_map
        .unique_keys
        .iter()
        .map(|&k| sampler_cache.get(&shared.device, k))
        .collect();
    for (idx, sampler) in per_key_samplers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: artifacts.bindings.per_binding_sampler_slot(idx),
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: artifacts
            .bindings
            .per_binding_cbuffer_slot(&artifacts.sampler_map),
        resource: props_buffer.as_entire_binding(),
    });
    // Decal-only: bind the ring buffer with `size = sizeof(struct)`
    // so the dynamic offset picks the right slice. wgpu requires the
    // bound size to match the BGL's `min_binding_size`.
    if let (Some(slot), Some(buf)) = (
        artifacts
            .bindings
            .per_binding_decal_constants_slot(&artifacts.sampler_map),
        decal_constants_buffer,
    ) {
        entries.push(wgpu::BindGroupEntry {
            binding: slot,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: std::num::NonZeroU64::new(
                    std::mem::size_of::<DecalConstantsGpu>() as u64,
                ),
            }),
        });
    }
    shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decal_material_bg"),
        layout: &artifacts.material_bgl,
        entries: &entries,
    })
}

/// Upload an `InlineTexture` to a wgpu texture + return a sampleable
/// view. Mirrors the inline-texture path from
/// `Renderer::upload_model_data`.
fn upload_inline_texture(
    shared: &SharedResources,
    tex: &InlineTexture,
    linear: bool,
) -> wgpu::TextureView {
    let wgpu_format = crate::halo::render::inline_to_wgpu_format(tex.format, linear);
    // BC formats require block-aligned (multiple of 4) create extents.
    // See `feedback_wgpu_bc_mip_alignment.md` — mirrors the same fix in
    // `Renderer::upload_model_data` and `bsp_gpu::upload_inline_texture`.
    let (alloc_w, alloc_h) = if tex.format.is_compressed() {
        ((tex.width + 3) & !3, (tex.height + 3) & !3)
    } else {
        (tex.width, tex.height)
    };
    let texture = shared.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("decal_inline_texture"),
        size: wgpu::Extent3d {
            width: alloc_w,
            height: alloc_h,
            depth_or_array_layers: tex.layers,
        },
        mip_level_count: tex.mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut offset = 0usize;
    for layer in 0..tex.layers {
        for mip in 0..tex.mip_count {
            let mw = (tex.width >> mip).max(1);
            let mh = (tex.height >> mip).max(1);
            let level_bytes = tex.format.level_bytes(mw, mh);
            let (extent_w, extent_h, rows, bytes_per_row);
            if tex.format.is_compressed() {
                let bw = ((mw + 3) / 4).max(1);
                let bh = ((mh + 3) / 4).max(1);
                extent_w = bw * 4;
                extent_h = bh * 4;
                rows = bh;
                bytes_per_row = bw * tex.format.block_bytes();
            } else {
                extent_w = mw;
                extent_h = mh;
                rows = mh;
                bytes_per_row = mw * tex.format.bytes_per_pixel();
            }
            shared.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &tex.data[offset..offset + level_bytes],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: extent_w,
                    height: extent_h,
                    depth_or_array_layers: 1,
                },
            );
            offset += level_bytes;
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(if tex.is_cube {
            wgpu::TextureViewDimension::Cube
        } else {
            wgpu::TextureViewDimension::D2
        }),
        ..Default::default()
    })
}

fn build_identity_model_bg(shared: &SharedResources) -> wgpu::BindGroup {
    use crate::halo::geometry::ModelUniforms;
    let uniforms = ModelUniforms {
        model: glam::Mat4::IDENTITY.to_cols_array_2d(),
    };
    let bytes = bytemuck::bytes_of(&uniforms);
    let mut padded = vec![0u8; shared.model_stride];
    padded[..bytes.len()].copy_from_slice(bytes);
    let buffer = shared.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("decal_identity_model_uniform"),
            contents: &padded,
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decal_identity_model_bg"),
        layout: &shared.model_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(bytes.len() as u64),
            }),
        }],
    })
}

fn build_identity_nm_bg(shared: &SharedResources) -> wgpu::BindGroup {
    let identity_mat = glam::Mat4::IDENTITY.to_cols_array();
    let bytes = bytemuck::cast_slice(&identity_mat);
    let mut padded = vec![0u8; shared.node_matrices_stride];
    padded[..bytes.len()].copy_from_slice(bytes);
    let buffer = shared.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("decal_identity_nm_buffer"),
            contents: &padded,
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decal_identity_nm_bg"),
        layout: &shared.node_matrices_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(shared.node_matrices_stride as u64),
            }),
        }],
    })
}
