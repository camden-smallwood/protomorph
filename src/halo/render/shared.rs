// LEGACY INFRASTRUCTURE — DO NOT EXTEND.
//
// Nothing in this file mirrors `Ares/source/render/`. It's the
// pre-rewrite protomorph shape:
//   - SharedResources — device/queue/config + camera bind group +
//     model bind group + node-matrices bind group + lighting cbuffer.
//     Halo splits these across c_rasterizer + c_lighting_interface +
//     c_render_method_data; we collapsed them for v1 expedience.
//   - GBuffer / IntermediateTargets — render-target table. Halo uses
//     c_rasterizer's SurfaceTable (ported in halo/rasterizer/).
//   - FrameContext — per-frame argument bundle. Halo passes
//     individual references; the bundle is a Rust-idiomatic shortcut.
//   - RenderPass trait — virtual dispatch over pass objects. Halo's
//     c_player_view::render is direct calls in canonical order; the
//     trait is our v1 escape hatch.
//
// All five are scheduled for retirement (Phase Z in
// RENDERER_REWRITE_PLAN.md). Each pass that migrates from this
// shape to its canonical Ares-faithful shape can drop its
// dependency on `shared` one piece at a time. Once every pass is
// migrated, this file deletes outright.
//
// Until then: existing fields/types remain stable so the rewrite
// can ship incrementally without big-bang breakage.

use crate::halo::camera::CameraUniforms;
use crate::game::GameState;
use crate::halo::geometry::MAXIMUM_NUMBER_OF_MODEL_NODES;
use crate::halo::geometry::ModelUniforms;
use crate::halo::lighting::{
    GpuEngineDominantLight, GpuEngineLightingPs, GpuSimpleLights,
};
use crate::halo::objects::ObjectIndex;
use crate::halo::render::{
    GpuModel, MAX_OBJECTS, create_1x1_texture, create_screen_texture, cube_tex_entry,
    env_probe_pass::{GpuAtmosphereData, GpuSkyParams},
    sampler_entry, tex_entry, uniform_entry,
};
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Engine lighting buffer layout
// ---------------------------------------------------------------------------

/// Per-entry stride in `engine_lighting_buffer`. Must be ≥
/// `min_uniform_buffer_offset_alignment`. Hard-coded to 256 (works on
/// every backend wgpu supports).
pub const ENGINE_LIGHTING_STRIDE: u32 = 256;

/// Number of entries (default + per-instance probes + per-cluster
/// probes + per-scenery-placement L4 probes) the buffer holds. Sized
/// for deadlock-class maps (43 clusters + 857 sbsp instances + 154
/// scenery → ~1054 slots used). Bumped from 512 → 4096 to absorb
/// future content without re-tuning.
pub const ENGINE_LIGHTING_ENTRIES: u32 = 4096;

/// Buffer offset for the frame-default lighting (set by
/// `setup_default_lighting` once per frame).
pub const ENGINE_LIGHTING_DEFAULT_OFFSET: u32 = 0;

/// Per-entry stride in `simple_lights_buffer`. `GpuSimpleLights` is
/// 656 bytes (16-byte count header + 8 × 80-byte light); the next
/// 256-byte multiple is 768. Same 256-byte alignment guarantee as
/// `ENGINE_LIGHTING_STRIDE`.
pub const SIMPLE_LIGHTS_STRIDE: u32 = 768;

/// Number of slots in `simple_lights_buffer`. Slot 0 is reserved as
/// the empty/no-light slot (count=0); slots 1+ are per-cluster /
/// per-instance simple-light packs computed at scenario load. Capped
/// generously so MCC's biggest maps fit (campaign BSPs hit ~600
/// clusters; multiplayer maps stay well under 100).
pub const SIMPLE_LIGHTS_ENTRIES: u32 = 1024;

/// Buffer offset for the empty simple-lights slot (count=0). Used by
/// any draw path that hasn't pre-computed its own per-cluster /
/// per-instance lights.
pub const SIMPLE_LIGHTS_DEFAULT_OFFSET: u32 = 0;

// ---------------------------------------------------------------------------
// Atmosphere table (per-render-method override)
// ---------------------------------------------------------------------------
//
// Engine `render_method_submit_volatile_per_shader @ 0x180685cf0` rewrites the
// single atmosphere cbuffer before EACH non-albedo/non-shadow draw, selected by
// the render_method's flags:
//   UseCustomSetting (bit1) → set_atmosphere_constants_by_index(custom_index)
//                              (= accumulate ONE setting, weight 1.0)
//   DontFogMe (bit0)        → disable_atmosphere (slot1.w = -1 sentinel)
//   else                    → restore_default (the cluster-weighted blend)
// We can't rewrite a uniform mid-pass, so `atmosphere_buffer` holds a per-frame
// TABLE and `camera_bgl @ binding 2` is rebound per draw with a dynamic offset
// — the same mechanism already used for per-instance lightprobes at binding 1.

/// 256-byte stride per atmosphere table entry (`GpuAtmosphereData` is 176 B;
/// dynamic-offset uniforms require 256-byte alignment).
pub const ATMOSPHERE_STRIDE: u32 = 256;

/// Offset of the frame-default (cluster-weighted) atmosphere. Used by every
/// render_method WITHOUT `UseCustomSetting`/`DontFogMe` — the vast majority.
pub const ATMOSPHERE_DEFAULT_OFFSET: u32 = 0;

/// Offset of the "atmosphere disabled" entry (extinction=1, inscatter=0).
/// Engine `disable_atmosphere`; selected by `DontFogMe`.
pub const ATMOSPHERE_DISABLED_OFFSET: u32 = ATMOSPHERE_STRIDE;

/// Offset of atmosphere setting 0 in the table. Setting `i` lives at
/// `ATMOSPHERE_SETTING_BASE_OFFSET + i * ATMOSPHERE_STRIDE`. Selected by
/// `UseCustomSetting` with `custom_fog_setting_index = i`.
pub const ATMOSPHERE_SETTING_BASE_OFFSET: u32 = 2 * ATMOSPHERE_STRIDE;

/// Max authored atmosphere settings the table holds (construct has 10).
/// Settings beyond this index resolve to the default entry.
pub const MAX_ATMOSPHERE_SETTINGS: u32 = 32;

/// Total table entries: default + disabled + `MAX_ATMOSPHERE_SETTINGS`.
pub const ATMOSPHERE_TABLE_ENTRIES: u32 = 2 + MAX_ATMOSPHERE_SETTINGS;

/// Resolve a render_method's atmosphere table byte-offset from its
/// `GlobalRenderMethodFlags` bits + `custom_fog_setting_index`. Mirrors the
/// `render_method_submit_volatile_per_shader` flag dispatch.
pub fn atmosphere_offset_for(flag_bits: u16, custom_fog_setting_index: i32) -> u32 {
    // GlobalRenderMethodFlags: DontFogMe = bit0 (0x1), UseCustomSetting = bit1 (0x2).
    if flag_bits & 0x2 != 0 {
        let idx = custom_fog_setting_index.clamp(0, MAX_ATMOSPHERE_SETTINGS as i32 - 1) as u32;
        ATMOSPHERE_SETTING_BASE_OFFSET + idx * ATMOSPHERE_STRIDE
    } else if flag_bits & 0x1 != 0 {
        ATMOSPHERE_DISABLED_OFFSET
    } else {
        ATMOSPHERE_DEFAULT_OFFSET
    }
}

// ---------------------------------------------------------------------------
// Fullscreen Quad
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuadVertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
}

/// `EngineUnderwater` cbuffer — global underwater-fog state plumbed to
/// every transparent PS at `@group(0) @binding(7)`. Mirrors the WGSL
/// struct in `engine_bindings.wgsl`. Phase B6.
///
/// `is_underwater_flags.x` carries the boolean as f32 (0.0 / 1.0) so
/// the cbuffer stays trivially std140-aligned (16-byte vec4 columns).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuEngineUnderwater {
    pub fog_color: [f32; 4],
    pub murkiness: [f32; 4],
    pub is_underwater_flags: [f32; 4],
}

/// `EngineExposure` cbuffer — `g_exposure` + `g_alt_exposure` PS regs
/// from engine `setup_render_target_globals_with_exposure @ 0x180670AD0`.
/// Mirrors the WGSL struct in `engine_bindings.wgsl`. Bound at
/// `@group(0) @binding(9)`.
///
/// Layout (per IDA decompile):
///   `g_exposure     = (view_exposure, pow(2, HDR_target_stops), 1.0, 1.0)`
///   `g_alt_exposure = (illum_scale, illum_scale × view_exposure, 0, 0)`
///
/// `HDR_target_stops` is engine-side render-target encoding; for
/// protomorph's Rg11b10Ufloat path it's effectively 0 → `pow(2,0) = 1`.
/// `illum_scale` comes from `c_camera_fx_values::self_illum_scale`
/// (camera_fx_settings.rs:344).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuEngineExposure {
    pub g_exposure: [f32; 4],
    pub g_alt_exposure: [f32; 4],
}

/// `MiscPS` cbuffer — engine `hlsl_constant_persist_fx.hlsl:101`.
/// Holds per-frame globals consumed across material shaders:
/// render-target `texture_size`, dynamic-env-probe `dynamic_environment_blend`
/// (formerly per-material baked at 0.5; now engine-uploaded for temporal
/// blending), render-debug mode pin, and a packed flags vec4 collapsing
/// the engine's 4 bool slots (pc_specular_enabled / pc_albedo_lighting /
/// LDR_gamma2 / HDR_gamma2 — engine slots have 12B pad each; we pack into
/// one vec4 since both halves of the bus are protomorph-internal).
/// Bound at `@group(0) @binding(12)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuEngineMiscPs {
    /// `.xy` = render-target dims (pixels). `.zw` pad.
    pub texture_size: [f32; 4],
    /// 3-channel blend between the two dynamic env probes
    /// (`reflection_0 * blend + reflection_1 * (1 - blend)` per
    /// `environment_mapping_fx_dynamic.wgsl`). Engine writes per-draw
    /// via `set_shader_constant_single(0x2F0001,
    /// m_cubemap_blend_factor)`; protomorph defaults to `(1, 1, 1, 0)`
    /// so static placements get 100% reflection_0 (the active probe).
    /// v1 doesn't crossfade — every object's `current` and `last` probe
    /// slots map to the same per-cluster probe view, so the blend is
    /// effectively a no-op.
    pub dynamic_environment_blend: [f32; 4],
    /// `p_render_debug_mode` — engine debug pin. Currently unused;
    /// kept for forward compat with the engine layout.
    pub render_debug_mode: [f32; 4],
    /// `.x` = `p_shader_pc_specular_enabled` (1.0 on / 0.0 off).
    /// `.y` = `p_shader_pc_albedo_lighting`.
    /// `.z` = `LDR_gamma2`.
    /// `.w` = `HDR_gamma2`.
    pub flags: [f32; 4],
    /// `.x` = `ps_total_time` (engine real-time seconds since boot).
    pub ps_total_time: [f32; 4],
    /// `.x` = `actually_calc_albedo` (1.0 = sample material textures,
    /// 0.0 = read cached albedo G-buffer). protomorph runs the
    /// "always calc" path so this is 1.0.
    pub actually_calc_albedo: [f32; 4],
}

impl GpuEngineMiscPs {
    pub fn defaults(width: u32, height: u32) -> Self {
        Self {
            texture_size: [width as f32, height as f32, 0.0, 0.0],
            dynamic_environment_blend: [1.0, 1.0, 1.0, 0.0],
            render_debug_mode: [0.0; 4],
            flags: [1.0, 1.0, 0.0, 0.0],
            ps_total_time: [0.0; 4],
            actually_calc_albedo: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl GpuEngineExposure {
    /// Build from view exposure + illum scale + the current frame's
    /// total time. `hdr_target_stops` is the engine's HDR render-target
    /// encoding exponent; pass 0.0 unless a specific path needs the
    /// encoding factor. `total_time` lands in `g_alt_exposure.w` (an
    /// otherwise-unused VS+FS-visible slot) so the foliage VS can read
    /// it for wind animation without giving MiscPS VERTEX visibility
    /// (which collides with the water pipeline's Metal slot layout).
    pub fn from_camera_fx_with_time(
        view_exposure: f32,
        illum_scale: f32,
        hdr_target_stops: f32,
        total_time: f32,
    ) -> Self {
        let mut out = Self::from_camera_fx(view_exposure, illum_scale, hdr_target_stops);
        out.g_alt_exposure[3] = total_time;
        out
    }

    /// Build from view exposure + illum scale. `hdr_target_stops` is the
    /// engine's HDR render-target encoding exponent; pass 0.0 unless a
    /// specific path needs the encoding factor.
    pub fn from_camera_fx(view_exposure: f32, illum_scale: f32, hdr_target_stops: f32) -> Self {
        let two_to_stops = (hdr_target_stops * std::f32::consts::LN_2).exp();
        Self {
            g_exposure: [view_exposure, two_to_stops, 1.0, 1.0],
            g_alt_exposure: [illum_scale, illum_scale * view_exposure, 0.0, 0.0],
        }
    }
}

impl QuadVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const QUAD_VERTICES: [QuadVertex; 6] = [
    QuadVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },
    QuadVertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] },
    QuadVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },
    QuadVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },
    QuadVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },
    QuadVertex { position: [ 1.0,  1.0], texcoord: [1.0, 0.0] },
];

// ---------------------------------------------------------------------------
// G-Buffer
// ---------------------------------------------------------------------------

/// Phase 1 frame depth buffer. The deferred G-buffer is gone — geometry
/// renders forward into `lighting_base` directly. This struct only owns
/// the depth attachment shared across passes that need it (geometry,
/// shadow_apply later, etc.).
pub struct GBuffer {
    /// Render-attachment view (full aspect — used by depth-stencil
    /// attachments).
    pub depth_view: wgpu::TextureView,
    /// Sampling view (depth-only aspect) for passes that read scene
    /// depth as a texture while also leaving it attached read-only —
    /// e.g. water's extern 3 binding (`_surface_depth_stencil` in
    /// `c_player_view::render_water @ 0x18068c120`).
    pub depth_sample_view: wgpu::TextureView,
    /// Held to keep the GPU resource alive when callers create
    /// additional views downstream.
    pub depth_texture: wgpu::Texture,
    /// World-space normal G-buffer (engine: `_surface_post_HDR` second
    /// MRT slot — Halo 3 stores per-pixel normal alongside scene HDR).
    /// `Rgba8Unorm` — encoded as `(n * 0.5 + 0.5, 1.0)`. Read by SSAO,
    /// future SSR / deferred decals.
    pub normal_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
}

impl GBuffer {
    pub fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_sample_view = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene_depth_sample"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let normal = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_normal"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            depth_view,
            depth_sample_view,
            depth_texture: depth,
            normal_view,
            normal_texture: normal,
        }
    }
}

// ---------------------------------------------------------------------------
// Intermediate targets
// ---------------------------------------------------------------------------

pub struct IntermediateTargets {
    pub lighting_base_view: wgpu::TextureView,
    pub lighting_base_texture: wgpu::Texture,
    /// Copy of `lighting_base` taken right before `render_transparents`
    /// runs. Used as `tex_ldr_buffer` by water shaders for refraction
    /// sampling — see `reference_water_fx_blueprint.md`. Same screen
    /// dimensions + format as `lighting_base`. Mirrors Halo's
    /// "scene LDR copy" Bungie made before the transparency pass.
    pub scene_ldr_snapshot_view: wgpu::TextureView,
    pub scene_ldr_snapshot_texture: wgpu::Texture,
    /// `_surface_accum_HDR` — engine's albedo-pass RT0 (Halo
    /// `_surface_accum_HDR`). Dual-purpose target per engine HLSL:
    ///   - rmsh / rmtr / foliage `albedo_ps`: writes RAW albedo (RGB).
    ///     SL pass loads from here via `get_albedo` and lights it.
    ///   - decorator `default_ps`: writes the LIT color directly
    ///     (sample + ambient + inscatter + exposure baked in).
    ///     There is NO decorator entry in the SL pass — the lit
    ///     color flows through via the albedo→SL copy-blit
    ///     (`c_screen_postprocess::copy(_surface_accum_HDR →
    ///     _surface_screenshot_composite_depth)` per
    ///     `setup_targets_static_lighting`).
    /// Format `Rg11b10Ufloat` — HDR-capable so decorator's
    /// post-exposure lit color survives, no alpha (spec_mask is
    /// recomputed in SL via `calc_specular_mask`; engine puts
    /// `albedo.w` in normal-target's `.w` for paths that need it).
    /// Same format as `lighting_base` so the copy-blit is a 1:1 copy.
    pub albedo_view: wgpu::TextureView,
    pub albedo_texture: wgpu::Texture,
    /// `_surface_screenshot_composite_cubemap` — engine's SL-pass RT1
    /// ("HDR/DARK target"), per `convert_to_render_target.hlsl:38`:
    /// `result.dark_color.rgb = color.rgb / DARK_COLOR_MULTIPLIER`.
    /// Sampled by `downsample_4x4_block_bloom` (the bloom-extract feed)
    /// and by auto-exposure. On MCC PC `g_HDR_target_stops = 0` →
    /// `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0` so
    /// `dark_color == lighting_base` verbatim — visually trivial today
    /// but architecturally required for HDR sun-disc / atmosphere
    /// values >1.0 and for engine-faithful bloom feed semantics.
    pub hdr_dark_view: wgpu::TextureView,
    pub hdr_dark_texture: wgpu::Texture,
    /// `_surface_shadow_1` — engine's per-object/per-light shadow depth
    /// target (`c_rasterizer::setup_targets_shadow_generate @
    /// 0x180671080`). Single 2D D32Float, default 512×512 in shipped
    /// builds. Same target is reused serially: each shadow generate
    /// pass clears it, renders depth, then the apply pass samples it,
    /// then the next caster overwrites. **No cubemap array, no CSM
    /// atlas — both per-object ground shadows and per-light shadows
    /// share this single target.**
    pub shadow_depth_view: wgpu::TextureView,
    pub shadow_depth_texture: wgpu::Texture,
    /// `_surface_accum_HDR` in its postprocess role — post-bloom
    /// composite buffer. Engine `c_screen_postprocess::copy_accumulation_target
    /// @ 0x1806B71E0` writes shader 18 (`final_composite_base`) output
    /// here (tonemap + grade + bloom/bling combine). Then
    /// `render_lightshafts(_surface_accum_HDR → _surface_accum_HDR)`
    /// additive-blits sun shafts. Finally `copy(51 or 93,
    /// _surface_accum_HDR → _surface_display)` blits to the swapchain
    /// with hardware sRGB encoding.
    ///
    /// Format `Rg11b10Ufloat` — same as engine. The shader-18 cubic
    /// tone curve outputs roughly in [0, 1.5]; lightshafts add
    /// additive HDR on top before the sRGB-encode-on-blit. Distinct
    /// from `albedo_view` (engine `_surface_accum_HDR` in its
    /// albedo-pass-RT0 role) — the engine reuses one surface for
    /// both purposes since the albedo RT0 is done with by post-process
    /// time, but our pipeline separates them to keep ownership clear.
    pub accum_hdr_view: wgpu::TextureView,
    pub accum_hdr_texture: wgpu::Texture,
}

/// Engine `c_rasterizer::render_globals.k_shadow_resolution` — single
/// global. 512² in shipped Halo 3 builds. Per-object resolution is
/// dynamic in [16, 512] (see plan section S1) but the underlying
/// _surface_shadow_1 target is sized at the maximum.
pub const SHADOW_DEPTH_RESOLUTION: u32 = 512;

impl IntermediateTargets {
    pub fn new(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        let lighting_base = create_screen_texture(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba16Float,
            "lighting_base",
        );
        let lighting_base_view = lighting_base.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_ldr_snapshot = create_screen_texture(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba16Float,
            "scene_ldr_snapshot",
        );
        let scene_ldr_snapshot_view =
            scene_ldr_snapshot.create_view(&wgpu::TextureViewDescriptor::default());
        // `_surface_accum_HDR` — see struct doc-comment for the
        // dual-purpose (raw albedo / lit color) rationale.
        let albedo = create_screen_texture(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba16Float,
            "_surface_accum_HDR",
        );
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());
        // `_surface_screenshot_composite_cubemap` — see struct doc-comment.
        let hdr_dark = create_screen_texture(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba16Float,
            "_surface_screenshot_composite_cubemap",
        );
        let hdr_dark_view = hdr_dark.create_view(&wgpu::TextureViewDescriptor::default());
        // `_surface_accum_HDR` postprocess role — see field doc-comment.
        let accum_hdr = create_screen_texture(
            device,
            w,
            h,
            wgpu::TextureFormat::Rgba16Float,
            "_surface_accum_HDR_postprocess",
        );
        let accum_hdr_view = accum_hdr.create_view(&wgpu::TextureViewDescriptor::default());
        // `_surface_shadow_1` — see field doc-comment.
        let shadow_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("_surface_shadow_1"),
            size: wgpu::Extent3d {
                width: SHADOW_DEPTH_RESOLUTION,
                height: SHADOW_DEPTH_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_depth_view = shadow_depth.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            lighting_base_view,
            lighting_base_texture: lighting_base,
            scene_ldr_snapshot_view,
            scene_ldr_snapshot_texture: scene_ldr_snapshot,
            albedo_view,
            albedo_texture: albedo,
            hdr_dark_view,
            hdr_dark_texture: hdr_dark,
            shadow_depth_view,
            shadow_depth_texture: shadow_depth,
            accum_hdr_view,
            accum_hdr_texture: accum_hdr,
        }
    }
}

pub struct FallbackTextures {
    pub white_view: wgpu::TextureView,
    pub default_normal_view: wgpu::TextureView,
    pub black_view: wgpu::TextureView,
    /// Detail-neutral gray (0.218 linear = 1/DETAIL_MULTIPLIER) for
    /// unbound detail maps — see creation site in `new`.
    pub detail_view: wgpu::TextureView,
    /// 1×1×6 black cubemap. Materials whose `environment_map` option
    /// is `none` bind this so `calc_environment_map_none_ps` reads 0.
    pub black_cube_view: wgpu::TextureView,
    /// Mirror of `c_rasterizer_globals.default_bitmaps[]` — the engine
    /// owns 12 default-bitmap slots loaded from `rasterizer_globals`
    /// and any extern texture slot that fails its per-cluster /
    /// per-material lookup falls through to
    /// `c_rasterizer_globals::get_default_texture_ref(rasg, slot)`.
    /// Indexed by [`blam_tags::rasterizer_globals::DefaultBitmap`].
    /// Empty until `Renderer::load_scenario` parses
    /// `rasterizer_globals.rasterizer_globals`; entries that fail to
    /// load fall back to the matching hand-rolled view above (black /
    /// white / cube). Lookup helper: [`Self::default_bitmap_view`].
    pub default_bitmap_views: Vec<wgpu::TextureView>,
    /// Mirror of `c_rasterizer_globals.material_textures[]` — the 3
    /// pre-integrated Cook-Torrance BRDF LUTs (`cc0236`, `c78d78`,
    /// `dd0236`). Sampled by `cook_torrance_fx.hlsl::sh_glossy_ct_3`
    /// at view/roughness UV to produce the specular distribution
    /// shape. Indexed by [`blam_tags::rasterizer_globals::MaterialTexture`].
    /// Empty until `Renderer::load_scenario` parses
    /// `rasterizer_globals.rasterizer_globals`; entries that fail to
    /// load fall back to `white_view` (visually wrong — uniform spec
    /// shape — but stable). Lookup helper: [`Self::material_texture_view`].
    pub material_texture_views: Vec<wgpu::TextureView>,
}

impl FallbackTextures {
    /// Mirror of `c_rasterizer_globals::get_default_texture_ref(rasg, slot)`.
    /// Returns the loaded view for the requested
    /// [`blam_tags::rasterizer_globals::DefaultBitmap`] slot, falling
    /// back to a sensible hand-rolled placeholder if the slot wasn't
    /// loaded (rasg parse failure / bitmap missing / pre-scenario init).
    pub fn default_bitmap_view(
        &self,
        slot: blam_tags::rasterizer_globals::DefaultBitmap,
    ) -> &wgpu::TextureView {
        use blam_tags::rasterizer_globals::DefaultBitmap;
        let idx = slot as usize;
        if let Some(view) = self.default_bitmap_views.get(idx) {
            return view;
        }
        // Pre-scenario fallback per slot semantics.
        match slot {
            DefaultBitmap::ColorWhite => &self.white_view,
            DefaultBitmap::DefaultVector => &self.default_normal_view,
            DefaultBitmap::DefaultDynamicCubeMap => &self.black_cube_view,
            DefaultBitmap::ColorBlack
            | DefaultBitmap::ColorBlackAlphaBlack => &self.black_view,
            _ => &self.black_view,
        }
    }

    /// Mirror of the engine's `c_rasterizer_globals` material-texture
    /// accessor for cook_torrance LUTs. Returns the loaded view for
    /// the requested [`blam_tags::rasterizer_globals::MaterialTexture`]
    /// slot. Panics if the slot isn't loaded: rasterizer_globals is always
    /// present before any material draw, so a miss is a real load failure
    /// we want to surface — not mask with a uniform-white LUT that gives
    /// cook_torrance specular the wrong (flat) distribution shape.
    pub fn material_texture_view(
        &self,
        slot: blam_tags::rasterizer_globals::MaterialTexture,
    ) -> &wgpu::TextureView {
        let idx = slot as usize;
        self.material_texture_views.get(idx).unwrap_or_else(|| {
            panic!(
                "rasterizer_globals material texture slot {slot:?} (idx {idx}) not loaded — \
                 cook_torrance LUTs must be present before drawing materials"
            )
        })
    }
}

// ---------------------------------------------------------------------------
// RenderPass trait + FrameContext
// ---------------------------------------------------------------------------

pub type SharedBindGroup = Rc<RefCell<wgpu::BindGroup>>;

pub struct FrameContext<'a> {
    pub shared: &'a SharedResources,
    pub gbuffer: &'a GBuffer,
    pub intermediates: &'a IntermediateTargets,
    pub surface_view: &'a wgpu::TextureView,
    pub models: &'a [GpuModel],
    pub render_list: &'a [(ObjectIndex, usize)],
    pub game: &'a GameState,
    pub frame_index: u32,
    pub prev_view_projection: glam::Mat4,
    /// `StructureRenderer` — owns the loaded BSPs, per-frame BSP draw
    /// lists, identity bind groups for cluster draws, and the
    /// per-instance lighting offset allocator. Engine equivalent:
    /// `g_render_structure_globals` + `c_structure_renderer`.
    pub structure_renderer: &'a crate::halo::render::structure_renderer::StructureRenderer,
    /// `c_object_renderer` reference — passes call its
    /// `render_albedo` / `render_static_lighting` helpers from inside
    /// their open render-pass scopes.
    pub object_renderer: &'a crate::halo::render::render_objects::ObjectRenderer,
    /// `c_structure_renderer::render_decorators @ 0x18068F1B0` —
    /// GPU-instanced foliage. Engine renders this INSIDE
    /// `render_albedo` between setup and `render_first_person_albedo`,
    /// so we plumb the renderer through `FrameContext` for in-rpass
    /// dispatch.
    pub decorator_renderer:
        &'a crate::halo::render::render_decorators::DecoratorRenderer,
    /// Sky `GpuModel` (Phase B17) — `Some` when scenario has a sky
    /// chain that resolved to a loaded render_model. Drawn early in
    /// `render_albedo` with camera-position-translated model matrix
    /// (parallax-locked).
    pub sky_model: Option<&'a GpuModel>,
    /// Dedicated single-slot model_u + node_matrices buffers for the
    /// sky pass. See `render_sky::SkyGpu`.
    pub sky_gpu: &'a crate::halo::render::render_sky::SkyGpu,
    /// `c_visibility_collection::get_region()` for the camera view
    /// — engine equivalent: `get_global_camera_collection()->m_input
    /// ->region`. `None` until the player view populates it (Phase
    /// I0); consumers (decorators, decals, particles, foliage, etc)
    /// fall back to ad-hoc culls when absent. When `Some`, consumers
    /// should use the engine-faithful path
    /// [`crate::halo::visibility::visibility_region_test_sphere`]
    /// against this region.
    pub camera_visibility_region:
        Option<&'a crate::halo::visibility::SVisibilityRegion>,
    /// Decal GPU state — per-palette pipelines + bind groups + per-BSP
    /// vertex/index buffers. Built once at scenario load; the player_view
    /// post-SL draw uses this to issue per-decal `draw_indexed`.
    /// `None` until `Renderer::load_scenario` has run.
    pub decal_gpu: Option<&'a crate::halo::render::render_decals::DecalGpuState>,
    /// CPU-side per-BSP decal records (LoadedDecal{vertex_start,
    /// vertex_count, index_start, index_count, palette_index}). Mirrors
    /// the engine's `x_preplaced_to_render_this_frame[]` arrays — the
    /// renderer iterates these to find which decals to draw.
    /// `None` until `Renderer::load_scenario` has run.
    pub decal_renderer:
        Option<&'a crate::halo::render::render_decals::DecalRenderer>,
    /// `ParticleGpu` — plumbed through for in-rpass dispatch of
    /// `TransparentDispatch::Particle` batches (F-full): particles draw
    /// inside the shared transparents rpass so they sort against
    /// glass/water. Engine `c_particle_emitter::render_callback`.
    pub particle_gpu: &'a crate::halo::gpu_particle::ParticleGpu,
    /// `PatchyFogPass` — plumbed through for in-rpass dispatch of the
    /// `TransparentDispatch::PatchyFog` fullscreen composite (engine
    /// `render_patchy_fog_callback` registered by `queue_patchy_fog`).
    /// `draw` is a no-op unless the player view `prepare`d it this frame.
    pub patchy_fog_pass: &'a crate::halo::render::patchy_fog_pass::PatchyFogPass,
}

pub trait RenderPass {
    fn prepare(&mut self, _ctx: &FrameContext) {}

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext);

    fn resize(
        &mut self,
        _shared: &SharedResources,
        _gbuffer: &GBuffer,
        _intermediates: &IntermediateTargets,
    ) {}

    fn post_submit(&mut self) {}

    fn is_enabled(&self, _ctx: &FrameContext) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Shared resources
// ---------------------------------------------------------------------------

pub struct SharedResources {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // Samplers
    pub filtering_sampler: wgpu::Sampler,
    pub bloom_sampler: wgpu::Sampler,
    pub nearest_sampler: wgpu::Sampler,
    pub shadow_comparison_sampler: wgpu::Sampler,

    // Fullscreen quad
    pub quad_vertex_buffer: wgpu::Buffer,

    // Camera
    pub camera_bgl: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    /// Sun + SH ambient probe — populated per-frame from sky config.
    /// Bound at `@group(0) @binding(1)` alongside camera with a
    /// dynamic offset so per-instance probes can override the
    /// frame-default. `set_bind_group(0, &cam_bg, &[offset])`:
    /// - offset 0 = frame-global default (set by `setup_default_lighting`)
    /// - offset N×256 = N-th per-instance probe (uploaded by BspGpu)
    pub engine_lighting_buffer: wgpu::Buffer,
    /// Companion to `engine_lighting_buffer` holding the per-cluster
    /// dominant-light direction + intensity. Mirrors dllcache cbuffer
    /// slot 0x24 (paired with slot 0x30 `LightingPS`). Indexed by the
    /// SAME `next_probe_slot` cursor as `engine_lighting_buffer` —
    /// the per-draw dynamic offset is shared between bindings 1 and 13.
    /// Bound at `@group(0) @binding(13)`.
    pub engine_dominant_light_buffer: wgpu::Buffer,
    /// Engine `Kernel5PS` cbuffer — pool slot 0x33 in the cbuffer
    /// inventory. Carries `actually_calc_albedo` (entry 5 byte 8) plus
    /// other PS bool flags. Written via
    /// `Rasterizer::set_shader_constant_bool(0x80330005, …)` (i.e.
    /// `set_using_albedo_sampler`).
    ///
    /// Initially a 256B placeholder owned by `SharedRenderState`; once
    /// `Renderer::new` creates the Rasterizer, `set_kernel5_ps_buffer`
    /// swaps in the rasterizer's `cbuffer_pool.slots[0x33].gpu` so
    /// runtime writes land in the buffer the SL shaders read.
    /// Bound at `@group(0) @binding(14)`.
    pub kernel5_ps_buffer: wgpu::Buffer,
    /// Per-BSP lightmap compression scalars (3 vec4 = 48 bytes) used
    /// to dequantize the SH atlas. Reset to zero between scenarios;
    /// populated from `LightmapBspData.compression_vectors` on load.
    /// Bound at `@group(0) @binding(3)`.
    pub lightmap_compress_buffer: wgpu::Buffer,
    /// 1024×1024 BC3 array view (8 slices) — `LightmapBspData
    /// .lightprobe_texture`. 1x1 placeholder until scenario load.
    /// We hold the `Texture` alongside the view so the GPU resource
    /// stays alive even though only the view is referenced from bind
    /// groups — dropping the texture handle while a view exists has
    /// produced sampling artifacts in some wgpu builds.
    pub lightprobe_atlas: wgpu::Texture,
    /// REAL `_surface_albedo` view used by `camera_bind_group_sl`.
    /// Replaced via `set_gbuffer_views` after `IntermediateTargets` is
    /// constructed; sampled by SL PSs at camera_bgl @ binding 10.
    pub gbuffer_albedo_view: wgpu::TextureView,
    /// REAL normal G-buffer view used by `camera_bind_group_sl`.
    /// Same lifecycle as `gbuffer_albedo_view`.
    pub gbuffer_normal_view: wgpu::TextureView,
    /// 1×1 placeholder texture kept alive so we can create placeholder
    /// views for `camera_bind_group` (used in passes that WRITE the
    /// real G-buffer — wgpu rejects sampling a texture that's also a
    /// COLOR_TARGET in the same usage scope).
    pub gbuffer_placeholder_texture: wgpu::Texture,
    pub lightprobe_atlas_view: wgpu::TextureView,
    /// 1024×1024 BC3 array view (2 slices) — dominant_light_intensity_texture.
    pub lightprobe_intensity_atlas: wgpu::Texture,
    pub lightprobe_intensity_atlas_view: wgpu::TextureView,
    /// `EngineUnderwater` cbuffer — global underwater-fog state read
    /// by every transparent PS via `apply_underwater_fog`. Uploaded
    /// each frame from `WaterRenderer.is_underwater` + scenario
    /// underwater murkiness/color (or zeroed when no water present).
    /// Bound at `@group(0) @binding(7)`.
    pub engine_underwater_buffer: wgpu::Buffer,
    /// `simple_lights` cbuffer — 8 dynamic point/spot lights + count.
    /// Engine: `c_lights_view::submit_simple_light_draw_list_to_shader
    /// @ 0x1806c7e60` writes count to slot 0x310000 + 8 lights × 5 vec4
    /// to slot 0x310001. We collapse both into a single 656-byte cbuffer.
    /// Updated per-frame from `LightsView::simple_lights[]`.
    /// Bound at `@group(0) @binding(8)`.
    pub simple_lights_buffer: wgpu::Buffer,
    /// `EngineExposure` cbuffer — `g_exposure` + `g_alt_exposure` PS
    /// reg slots written by engine `setup_render_target_globals_with_exposure`.
    /// Updated at scenario load (and on per-frame `update_exposure` calls
    /// once auto-exposure ships). Bound at `@group(0) @binding(9)`.
    pub engine_exposure_buffer: wgpu::Buffer,
    /// `MiscPS` cbuffer — render-target `texture_size`, dynamic-env-probe
    /// blend, flags (pc_specular_enabled etc.), ps_total_time, debug
    /// pins. See [`GpuEngineMiscPs`]. Per-frame uploaded with current
    /// dims; `dynamic_environment_blend` defaults to `(1, 1, 1, 0)` so
    /// static placements get 100% reflection_0 (the active probe). v1
    /// has no crossfade; the temporal blend driver will land alongside
    /// dynamic objects.
    /// Bound at `@group(0) @binding(12)`.
    pub engine_misc_ps_buffer: wgpu::Buffer,
    /// Default `camera_bind_group` — bindings 10/11 hold placeholder
    /// 1×1 textures so the albedo pass (which writes `_surface_albedo`
    /// as a color target) can bind it without violating wgpu's
    /// "RESOURCE + COLOR_TARGET in same usage scope" rule.
    pub camera_bind_group: wgpu::BindGroup,
    /// `camera_bind_group_sl` — bindings 10/11 hold the REAL G-buffer
    /// views (`_surface_albedo` + normal). Used by static_lighting +
    /// transparents passes (where the G-buffer is read-only).
    pub camera_bind_group_sl: wgpu::BindGroup,
    /// `camera_bind_group_transparent` — identical to `_sl` plus the REAL
    /// scene depth at binding 15. Used ONLY by the transparent pass
    /// (`render_transparents`), where the depth attachment is read-only
    /// (`depth_ops: None`) so the same depth texture can be both depth-
    /// tested and sampled. Drives the `soft_fade` use_soft_z fade. The
    /// default + `_sl` groups bind a 1×1 placeholder depth at binding 15.
    pub camera_bind_group_transparent: wgpu::BindGroup,
    /// REAL scene-depth (DepthOnly aspect) view used by
    /// `camera_bind_group_transparent` @ binding 15. Replaced via
    /// `set_gbuffer_views` after the `GBuffer` is (re)constructed.
    pub gbuffer_depth_view: wgpu::TextureView,
    /// 1×1 placeholder DEPTH texture kept alive for the binding-15
    /// placeholder views in the default + `_sl` camera bind groups
    /// (passes that WRITE depth — can't also sample the live buffer).
    pub depth_placeholder_texture: wgpu::Texture,

    // Model transforms
    pub model_bgl: wgpu::BindGroupLayout,
    pub model_buffer: wgpu::Buffer,
    pub model_stride: usize,
    pub model_bind_group: wgpu::BindGroup,

    // Node/bone matrices
    pub node_matrices_bgl: wgpu::BindGroupLayout,
    pub node_matrices_buffer: wgpu::Buffer,
    pub node_matrices_stride: usize,
    pub node_matrices_bind_group: wgpu::BindGroup,

    // Lighting
    pub atmosphere_buffer: wgpu::Buffer,
    pub sky_params_buffer: wgpu::Buffer,

    // Env probe data (buffers owned here, written by EnvProbePass each frame)

    // Material BGL
    pub material_bgl: wgpu::BindGroupLayout,

    // Fallback textures
    pub fallback_textures: FallbackTextures,
}

impl SharedResources {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        // --- Samplers ---
        let filtering_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("filtering_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let bloom_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shadow_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Fallback textures ---
        let white_tex = create_1x1_texture(
            &device,
            &queue,
            [255, 255, 255, 255],
            "white_fallback",
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let default_normal_tex = create_1x1_texture(
            &device,
            &queue,
            [128, 128, 255, 255],
            "default_normal",
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let default_normal_view =
            default_normal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let black_tex = create_1x1_texture(
            &device,
            &queue,
            [0, 0, 0, 255],
            "black",
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let black_view = black_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Detail-neutral fallback for UNBOUND detail maps. The engine's
        // `shaders\default_bitmaps\bitmaps\default_detail` is a 1×1
        // mid-gray (stored 0.502, sRGB → ~0.218 linear = 1/DETAIL_MULTIPLIER).
        // Albedo variants multiply detail maps by DETAIL_MULTIPLIER
        // (4.59479; two_detail_overlay uses DETAIL_MULTIPLIER² with TWO
        // detail factors), so an unbound detail map MUST be this neutral —
        // binding `white_view` (1.0) instead made albedo ~5-7× too bright
        // (blown white mottle on glass_activation, which leaves
        // detail_map_overlay unbound). Stored 128 in an sRGB texture
        // decodes to ~0.218 on sample, matching default_detail.
        let detail_tex = create_1x1_texture(
            &device,
            &queue,
            [128, 128, 128, 255],
            "detail_neutral_fallback",
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let detail_view = detail_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // 1×1×6 black cube — used for materials whose env_map slot is
        // intentionally empty (e.g. material with no env-mapping option).
        // The "no per-cluster probe" path uses
        // `default_dynamic_cube_view` (loaded from
        // `rasterizer_globals.default_bitmaps[DefaultDynamicCubeMap]` at
        // scenario load).
        let black_cube_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("black_cube_fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for face in 0..6 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &black_cube_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 255],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        let black_cube_view = black_cube_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let fallback_textures = FallbackTextures {
            white_view,
            default_normal_view,
            black_view,
            detail_view,
            black_cube_view,
            // Populated by `Renderer::load_default_bitmaps_from_rasg`
            // after `rasterizer_globals` parses.
            default_bitmap_views: Vec::new(),
            material_texture_views: Vec::new(),
        };

        // --- Fullscreen quad vertex buffer ---
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_vb"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // --- Bind group layouts ---
        let camera_bgl = Self::create_camera_bgl(&device);
        let model_bgl = Self::create_model_bgl(&device);
        let material_bgl = Self::create_material_bgl(&device);
        let node_matrices_bgl = Self::create_node_matrices_bgl(&device);
        // --- Uniform buffers ---
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment as usize;
        let min_storage_alignment = device.limits().min_storage_buffer_offset_alignment as usize;
        let model_size = size_of::<ModelUniforms>();
        let model_stride = ((model_size + min_alignment - 1) / min_alignment) * min_alignment;

        let node_matrix_raw_size = MAXIMUM_NUMBER_OF_MODEL_NODES * size_of::<[[f32; 4]; 4]>();
        let node_matrices_stride = ((node_matrix_raw_size + min_storage_alignment - 1)
            / min_storage_alignment)
            * min_storage_alignment;

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let engine_lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_lighting_buffer"),
            size: ENGINE_LIGHTING_STRIDE as u64 * ENGINE_LIGHTING_ENTRIES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Companion buffer for dllcache slot 0x24 (dominant-light
        // direction + intensity). Same stride (256B) and entry count
        // as `engine_lighting_buffer` — the two are indexed by the
        // same `next_probe_slot` cursor and share the per-draw
        // dynamic-offset value.
        let engine_dominant_light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_dominant_light_buffer"),
            size: ENGINE_LIGHTING_STRIDE as u64 * ENGINE_LIGHTING_ENTRIES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 256B placeholder for Kernel5PS / camera_bgl @ binding 14. Swapped
        // for `Rasterizer::cbuffer_pool.slots[0x33].gpu` via
        // `set_kernel5_ps_buffer` once the rasterizer is constructed.
        let kernel5_ps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel5_ps_placeholder"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model_buffer"),
            size: (MAX_OBJECTS * model_stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_matrices_buffer"),
            size: (MAX_OBJECTS * node_matrices_stride) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Atmosphere TABLE: ATMOSPHERE_TABLE_ENTRIES entries at 256-byte
        // stride (default + disabled + per-setting). Bound at camera_bgl
        // binding 2 with a per-draw dynamic offset (see atmosphere_offset_for).
        let atmosphere_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere_buffer"),
            size: (ATMOSPHERE_TABLE_ENTRIES * ATMOSPHERE_STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_params_buffer"),
            size: size_of::<GpuSkyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Lightmap compression cbuffer (3 vec4 = 48 bytes). Default to
        // zero — `LightmapBspData::upload_to` overwrites at scenario load.
        let lightmap_compress_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lightmap_compress_buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // EngineUnderwater cbuffer (Phase B6). Default = not-underwater
        // (is_underwater_flags = 0); per-frame upload from
        // `WaterRenderer.is_underwater` overwrites on scenarios that
        // have water.
        let engine_underwater_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_underwater_buffer"),
            size: size_of::<GpuEngineUnderwater>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // simple_lights cbuffer (L2). One slot per draw "owner" —
        // BSP cluster, scenery instance, decorator placement bucket.
        // Bound at `@group(0) @binding(8)` with a DYNAMIC uniform
        // offset; per draw, the offset selects the right slot. Slot
        // 0 is reserved as empty (count=0). All slots are populated
        // at scenario load by `compute_simple_lights_slots` —
        // contents are stable across camera motion (engine-faithful).
        let simple_lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simple_lights_buffer"),
            size: (SIMPLE_LIGHTS_STRIDE as u64) * (SIMPLE_LIGHTS_ENTRIES as u64),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Slot 0 = empty (count=0). Other slots filled in at scenario
        // load.
        queue.write_buffer(
            &simple_lights_buffer,
            0,
            bytemuck::bytes_of(&GpuSimpleLights::default()),
        );
        // Default fog color matches the WaterSharedPS placeholder
        // (`(0.05, 0.18, 0.30)`) and murkiness=0.5; unused when
        // is_underwater=0.
        queue.write_buffer(
            &engine_underwater_buffer,
            0,
            bytemuck::bytes_of(&GpuEngineUnderwater {
                fog_color: [0.05, 0.18, 0.30, 1.0],
                murkiness: [0.5, 0.0, 0.0, 0.0],
                is_underwater_flags: [0.0; 4],
            }),
        );

        // EngineExposure cbuffer (B1). Defaults to identity (view_exposure
        // = 1, illum_scale = 1) until scenario load supplies real values
        // from camera_fx_settings.
        let engine_exposure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_exposure_buffer"),
            size: size_of::<GpuEngineExposure>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &engine_exposure_buffer,
            0,
            bytemuck::bytes_of(&GpuEngineExposure::from_camera_fx(1.0, 1.0, 0.0)),
        );

        // MiscPS cbuffer — engine `hlsl_constant_persist_fx.hlsl:101`.
        // Initialized with the swapchain dims; rebuilt on resize.
        // dynamic_environment_blend defaults to (1, 1, 1, 0) — 100%
        // reflection_0 (the active probe). Engine writes this per-draw
        // via `set_shader_constant_single(0x2F0001, m_blend_factor)`;
        // v1 static placements use the same probe for `current` and
        // `last`, so the blend is a no-op.
        let engine_misc_ps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_misc_ps_buffer"),
            size: size_of::<GpuEngineMiscPs>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &engine_misc_ps_buffer,
            0,
            bytemuck::bytes_of(&GpuEngineMiscPs::defaults(config.width, config.height)),
        );

        // Lightprobe atlas placeholders — 1x1 BC3 array textures with
        // matching slice counts (8 + 2). Fragment-shader sampling on
        // these returns zero, which is the correct fallback for clusters
        // without a lightmap. Real atlases get swapped in on scenario load.
        let lightprobe_atlas = make_placeholder_bc3_array(&device, &queue, 8, "lightprobe_atlas_placeholder");
        let lightprobe_intensity_atlas = make_placeholder_bc3_array(&device, &queue, 2, "lightprobe_intensity_placeholder");
        let lightprobe_atlas_view = lightprobe_atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let lightprobe_intensity_atlas_view = lightprobe_intensity_atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Placeholder G-buffer texture — 1×1, kept alive in struct so
        // we can mint placeholder views from it whenever rebuilding
        // `camera_bind_group`. The albedo pass binds `camera_bind_group`
        // (with placeholder views at 10/11) so its color-target write
        // to the real `_surface_albedo` doesn't conflict with sampling
        // it from the same bind group. Static-lighting / transparents
        // bind `camera_bind_group_sl` with the REAL views.
        let gbuffer_placeholder_texture = create_screen_texture(
            &device, 1, 1, wgpu::TextureFormat::Rgba8Unorm, "gbuffer_placeholder",
        );
        let placeholder_albedo_view =
            gbuffer_placeholder_texture.create_view(&Default::default());
        let placeholder_normal_view =
            gbuffer_placeholder_texture.create_view(&Default::default());
        // Initial real views — placeholder until `set_gbuffer_views`
        // installs the real `_surface_albedo` + scene-normal views.
        let gbuffer_albedo_view =
            gbuffer_placeholder_texture.create_view(&Default::default());
        let gbuffer_normal_view =
            gbuffer_placeholder_texture.create_view(&Default::default());

        // 1×1 placeholder DEPTH texture for camera_bgl binding 15 in the
        // default + `_sl` groups (depth-writing passes never sample it).
        // `Depth32Float` matches the gbuffer depth format so the layout's
        // `Depth` sample type is satisfied. The real scene depth is wired
        // into `camera_bind_group_transparent` via `set_gbuffer_views`.
        let depth_placeholder_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_placeholder_view =
            depth_placeholder_texture.create_view(&Default::default());
        // Initial real scene-depth view — placeholder until
        // `set_gbuffer_views` installs the real `GBuffer.depth_sample_view`.
        let gbuffer_depth_view =
            depth_placeholder_texture.create_view(&Default::default());

        // --- Bind groups ---
        // `camera_bind_group` uses placeholder views at 10/11 — bound
        // by `render_albedo` while it writes `_surface_albedo`.
        let camera_bind_group = build_camera_bind_group(
            &device,
            &camera_bgl,
            &camera_buffer,
            &engine_lighting_buffer,
            &atmosphere_buffer,
            &lightmap_compress_buffer,
            &lightprobe_atlas_view,
            &lightprobe_intensity_atlas_view,
            &filtering_sampler,
            &engine_underwater_buffer,
            &simple_lights_buffer,
            &engine_exposure_buffer,
            &placeholder_albedo_view,
            &placeholder_normal_view,
            &engine_misc_ps_buffer,
            &engine_dominant_light_buffer,
            &kernel5_ps_buffer,
            &depth_placeholder_view,
            "camera_bg",
        );
        // `camera_bind_group_sl` uses real views at 10/11 — bound by
        // `render_static_lighting` / `render_transparents` while the
        // G-buffer is read-only. Initially carries placeholder views
        // until `set_gbuffer_views` runs.
        let camera_bind_group_sl = build_camera_bind_group(
            &device,
            &camera_bgl,
            &camera_buffer,
            &engine_lighting_buffer,
            &atmosphere_buffer,
            &lightmap_compress_buffer,
            &lightprobe_atlas_view,
            &lightprobe_intensity_atlas_view,
            &filtering_sampler,
            &engine_underwater_buffer,
            &simple_lights_buffer,
            &engine_exposure_buffer,
            &gbuffer_albedo_view,
            &gbuffer_normal_view,
            &engine_misc_ps_buffer,
            &engine_dominant_light_buffer,
            &kernel5_ps_buffer,
            &depth_placeholder_view,
            "camera_bg_sl",
        );
        // `camera_bind_group_transparent` — real G-buffer views + real
        // scene depth at 15. Initially placeholder depth (and placeholder
        // gbuffer) until `set_gbuffer_views` installs the real views.
        let camera_bind_group_transparent = build_camera_bind_group(
            &device,
            &camera_bgl,
            &camera_buffer,
            &engine_lighting_buffer,
            &atmosphere_buffer,
            &lightmap_compress_buffer,
            &lightprobe_atlas_view,
            &lightprobe_intensity_atlas_view,
            &filtering_sampler,
            &engine_underwater_buffer,
            &simple_lights_buffer,
            &engine_exposure_buffer,
            &gbuffer_albedo_view,
            &gbuffer_normal_view,
            &engine_misc_ps_buffer,
            &engine_dominant_light_buffer,
            &kernel5_ps_buffer,
            &gbuffer_depth_view,
            "camera_bg_transparent",
        );
        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_bg"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(model_size as u64),
                }),
            }],
        });
        let node_matrices_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("node_matrices_bg"),
            layout: &node_matrices_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &node_matrices_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(node_matrix_raw_size as u64),
                }),
            }],
        });
        Self {
            device,
            queue,
            config,
            filtering_sampler,
            bloom_sampler,
            nearest_sampler,
            shadow_comparison_sampler,
            quad_vertex_buffer,
            camera_bgl,
            camera_buffer,
            camera_bind_group,
            model_bgl,
            model_buffer,
            model_stride,
            model_bind_group,
            node_matrices_bgl,
            node_matrices_buffer,
            node_matrices_stride,
            node_matrices_bind_group,
            engine_lighting_buffer,
            atmosphere_buffer,
            sky_params_buffer,
            material_bgl,
            fallback_textures,
            lightmap_compress_buffer,
            lightprobe_atlas,
            lightprobe_atlas_view,
            lightprobe_intensity_atlas,
            lightprobe_intensity_atlas_view,
            engine_underwater_buffer,
            simple_lights_buffer,
            engine_exposure_buffer,
            engine_misc_ps_buffer,
            engine_dominant_light_buffer,
            kernel5_ps_buffer,
            gbuffer_albedo_view,
            gbuffer_normal_view,
            gbuffer_placeholder_texture,
            camera_bind_group_sl,
            camera_bind_group_transparent,
            gbuffer_depth_view,
            depth_placeholder_texture,
        }
    }

    /// Replace the placeholder G-buffer views with the real ones from
    /// `IntermediateTargets` + `GBuffer`. Must be called once after
    /// those structs are constructed (in `Renderer::new`); the SL pass
    /// PSs sample these via camera_bgl bindings 10 / 11. ONLY rebuilds
    /// `camera_bind_group_sl` — `camera_bind_group` keeps placeholder
    /// views so the albedo pass can write `_surface_albedo` while
    /// having the camera bind group bound.
    pub fn set_gbuffer_views(
        &mut self,
        albedo_view: wgpu::TextureView,
        normal_view: wgpu::TextureView,
        depth_view: wgpu::TextureView,
    ) {
        self.gbuffer_albedo_view = albedo_view;
        self.gbuffer_normal_view = normal_view;
        self.gbuffer_depth_view = depth_view;
        let placeholder_depth = self.depth_placeholder_texture.create_view(&Default::default());
        self.camera_bind_group_sl = build_camera_bind_group(
            &self.device,
            &self.camera_bgl,
            &self.camera_buffer,
            &self.engine_lighting_buffer,
            &self.atmosphere_buffer,
            &self.lightmap_compress_buffer,
            &self.lightprobe_atlas_view,
            &self.lightprobe_intensity_atlas_view,
            &self.filtering_sampler,
            &self.engine_underwater_buffer,
            &self.simple_lights_buffer,
            &self.engine_exposure_buffer,
            &self.gbuffer_albedo_view,
            &self.gbuffer_normal_view,
            &self.engine_misc_ps_buffer,
            &self.engine_dominant_light_buffer,
            &self.kernel5_ps_buffer,
            &placeholder_depth,
            "camera_bg_sl",
        );
        // The transparent group is the only one with the REAL scene depth.
        self.camera_bind_group_transparent = build_camera_bind_group(
            &self.device,
            &self.camera_bgl,
            &self.camera_buffer,
            &self.engine_lighting_buffer,
            &self.atmosphere_buffer,
            &self.lightmap_compress_buffer,
            &self.lightprobe_atlas_view,
            &self.lightprobe_intensity_atlas_view,
            &self.filtering_sampler,
            &self.engine_underwater_buffer,
            &self.simple_lights_buffer,
            &self.engine_exposure_buffer,
            &self.gbuffer_albedo_view,
            &self.gbuffer_normal_view,
            &self.engine_misc_ps_buffer,
            &self.engine_dominant_light_buffer,
            &self.kernel5_ps_buffer,
            &self.gbuffer_depth_view,
            "camera_bg_transparent",
        );
    }

    /// Swap the placeholder Kernel5PS buffer for the rasterizer-owned
    /// pool slot 0x33 buffer once `Renderer::new` has built the
    /// Rasterizer. Both camera bind groups are rebuilt so subsequent
    /// `set_shader_constant_bool(0x80330005, …)` writes land in the
    /// buffer the SL shaders bind. Call ONCE post-rasterizer-init; the
    /// placeholder is leaked (cheap — 256B).
    pub fn set_kernel5_ps_buffer(&mut self, buffer: wgpu::Buffer) {
        self.kernel5_ps_buffer = buffer;
        self.rebuild_camera_bind_group();
    }

    /// Rebuild both camera bind groups after the lightprobe atlas
    /// views are replaced with real per-BSP atlases. Call after
    /// `set_lightprobe_atlas`. `camera_bind_group` keeps placeholder
    /// G-buffer views; `camera_bind_group_sl` keeps the real ones.
    pub fn rebuild_camera_bind_group(&mut self) {
        let placeholder_albedo = self
            .gbuffer_placeholder_texture
            .create_view(&Default::default());
        let placeholder_normal = self
            .gbuffer_placeholder_texture
            .create_view(&Default::default());
        let placeholder_depth = self
            .depth_placeholder_texture
            .create_view(&Default::default());
        self.camera_bind_group = build_camera_bind_group(
            &self.device,
            &self.camera_bgl,
            &self.camera_buffer,
            &self.engine_lighting_buffer,
            &self.atmosphere_buffer,
            &self.lightmap_compress_buffer,
            &self.lightprobe_atlas_view,
            &self.lightprobe_intensity_atlas_view,
            &self.filtering_sampler,
            &self.engine_underwater_buffer,
            &self.simple_lights_buffer,
            &self.engine_exposure_buffer,
            &placeholder_albedo,
            &placeholder_normal,
            &self.engine_misc_ps_buffer,
            &self.engine_dominant_light_buffer,
            &self.kernel5_ps_buffer,
            &placeholder_depth,
            "camera_bg",
        );
        self.camera_bind_group_sl = build_camera_bind_group(
            &self.device,
            &self.camera_bgl,
            &self.camera_buffer,
            &self.engine_lighting_buffer,
            &self.atmosphere_buffer,
            &self.lightmap_compress_buffer,
            &self.lightprobe_atlas_view,
            &self.lightprobe_intensity_atlas_view,
            &self.filtering_sampler,
            &self.engine_underwater_buffer,
            &self.simple_lights_buffer,
            &self.engine_exposure_buffer,
            &self.gbuffer_albedo_view,
            &self.gbuffer_normal_view,
            &self.engine_misc_ps_buffer,
            &self.engine_dominant_light_buffer,
            &self.kernel5_ps_buffer,
            &placeholder_depth,
            "camera_bg_sl",
        );
        self.camera_bind_group_transparent = build_camera_bind_group(
            &self.device,
            &self.camera_bgl,
            &self.camera_buffer,
            &self.engine_lighting_buffer,
            &self.atmosphere_buffer,
            &self.lightmap_compress_buffer,
            &self.lightprobe_atlas_view,
            &self.lightprobe_intensity_atlas_view,
            &self.filtering_sampler,
            &self.engine_underwater_buffer,
            &self.simple_lights_buffer,
            &self.engine_exposure_buffer,
            &self.gbuffer_albedo_view,
            &self.gbuffer_normal_view,
            &self.engine_misc_ps_buffer,
            &self.engine_dominant_light_buffer,
            &self.kernel5_ps_buffer,
            &self.gbuffer_depth_view,
            "camera_bg_transparent",
        );
    }

    fn create_camera_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[
                // VERTEX_FRAGMENT: the VS uses `camera.projection`/`view`
                // for clip transform; the FS now reads `camera.projection`
                // in the `soft_fade` use_soft_z branch (`sf_linear_depth`
                // reconstructs view-space distance from device depth).
                uniform_entry(
                    0,
                    size_of::<CameraUniforms>() as u64,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    false,
                ),
                // Engine `LightingPS` cbuffer (slot 0x30 / 0x2E) —
                // ravi-packed SH3. Bound at binding 1; VS+FS visible.
                // **Dynamic offset**: structure_renderer overrides
                // per-SH-instance via `select_instance_entry_point`
                // (offset = probe_idx × 256). offset=0 holds the
                // frame-default from `setup_default_lighting`.
                uniform_entry(
                    1,
                    size_of::<GpuEngineLightingPs>() as u64,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    true,
                ),
                // Engine atmosphere (Rayleigh/Mie params from
                // scenario.atmospheric tag). Read by atmosphere.wgsl's
                // `compute_scattering` in BOTH stages — VS computes
                // per-vertex extinction/inscatter and PS reads them
                // for final composition.
                // **Dynamic offset**: each draw selects its atmosphere table
                // entry (default / disabled / per-setting) per the render_method
                // flags — engine `render_method_submit_volatile_per_shader`'s
                // per-shader atmosphere override. offset 0 = frame-default.
                uniform_entry(
                    2,
                    size_of::<GpuAtmosphereData>() as u64,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    true,
                ),
                // Lightmap compression scalars (BSP terrain SH unpack).
                // Halo `LightmapCompressPS` cbuffer — 3 vec4s of L-range
                // values built from `LightmapBspData.compression_vectors`.
                // VERTEX_FRAGMENT: terrain shaders read in PS, decorators
                // read in VS to evaluate per-placement SH at world_up.
                uniform_entry(
                    3,
                    48, // 3 vec4 = 48 bytes
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    false,
                ),
                // Lightprobe atlas (4 SH coefs encoded as L*UVW pairs
                // across 8 BC3 array slices). Sampled by BSP terrain
                // shaders in PS for per-pixel lighting; sampled by the
                // decorator VS for per-placement ambient (one ray per
                // placement → atlas UV → SH eval at world_up).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Lightprobe intensity atlas (1 vec3 encoded as L*UVW
                // pair across 2 BC3 array slices). Carries the
                // dominant-light radiance per terrain pixel. VS-visible
                // for the decorator path (currently unused there but
                // matches the atlas binding's stage set).
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Filtering sampler for both atlases.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // EngineUnderwater (Phase B6) — global underwater-fog
                // state (fog_color, murkiness, is_underwater flag) read
                // by every transparent PS via `apply_underwater_fog`.
                // Uploaded each frame from `WaterRenderer.is_underwater`
                // + the engine's underwater murkiness/color.
                uniform_entry(
                    7,
                    size_of::<GpuEngineUnderwater>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
                // simple_lights (L2) — 8 dynamic point/spot lights +
                // count, ONE slot per cluster/instance/placement.
                // Dynamic-offset: per draw, the offset selects the
                // owner's slot (engine-faithful per-cluster light
                // selection that mirrors `c_lights_view::add_simple_light_to_draw_list`
                // being called per cluster part in dllcache).
                // VERTEX_FRAGMENT: foliage VS reads for analytical
                // accumulation, material PS reads for diffuse/specular
                // accumulation.
                uniform_entry(
                    8,
                    size_of::<GpuSimpleLights>() as u64,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    true,
                ),
                // EngineExposure (B1) — `g_exposure` + `g_alt_exposure`
                // PS regs from `setup_render_target_globals_with_exposure`.
                // VERTEX_FRAGMENT: VS-side `g_exposure` reads from atmosphere
                // composition path (foliage VS uses scaled inscatter); PS
                // is the dominant consumer.
                uniform_entry(
                    9,
                    size_of::<GpuEngineExposure>() as u64,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    false,
                ),
                // `_surface_albedo` G-buffer texture — written by
                // `render_albedo`, sampled by `render_static_lighting`
                // PSs via `textureLoad(…, vec2<i32>(in.clip_position.xy), 0)`
                // to skip recomputing albedo from material textures.
                // Engine equivalent: `albedo_texture` extern in
                // entry_points_fx.hlsl::get_albedo_and_normal.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Normal G-buffer (`_surface_post_HDR` MRT[1]).
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // `MiscPS` cbuffer — engine `hlsl_constant_persist_fx.hlsl
                // :101`. Render-target dims, dynamic env-probe blend, flags
                // (pc_specular_enabled etc.). FRAGMENT-only — adding VS
                // visibility here hits a Metal MSL slot-allocator conflict
                // with `_mslBufferSizes` in the water pipeline. VS-side
                // time access (foliage wind) reuses `engine_exposure.g_alt_exposure.w`
                // which is already VERTEX_FRAGMENT-visible.
                uniform_entry(
                    12,
                    size_of::<GpuEngineMiscPs>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
                // Engine dominant-light cbuffer (slot 0x24) — paired
                // with `LightingPS` at binding 1. Per-cluster, shares
                // the same dynamic-offset cursor. FRAGMENT-only — adding
                // VS visibility here hits the same Metal MSL slot
                // allocator conflict with `_mslBufferSizes` we saw on
                // binding 12 (Phase 7). The one VS reader (decorator
                // sun_dir/sun_color in entry_decorator.wgsl) routes
                // through the already-VS-visible
                // `engine_atmosphere_raw.slot0_sun_dir_dist_bias` (the
                // engine analytical sun, which is what the decorator
                // VS actually wants — per-cluster dominant was a
                // structural mismatch).
                uniform_entry(
                    13,
                    size_of::<GpuEngineDominantLight>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    true,
                ),
                // Engine `Kernel5PS` cbuffer pool slot 0x33 — `actually_calc_albedo`
                // at byte 88, paired with other PS bool flags. Backed by
                // `Rasterizer::cbuffer_pool.slots[0x33].gpu` (pre-allocated
                // 256B at init). Written via `set_shader_constant_bool(0x80330005, …)`
                // i.e. `set_using_albedo_sampler`. FRAGMENT-only (same Metal
                // MSL constraint as bindings 12/13). P5.6 swaps the WGSL
                // `__SL_USE_GBUFFER__` compile-time substitution for a
                // runtime read of this cbuffer.
                //
                // min_binding_size = None: the actual HLSL Kernel5PS size is
                // unknown without inventorying the engine cbuffer enum; the
                // backing buffer is rounded up to 256B (wgpu UBO alignment),
                // which is plenty for the handful of bool flags this slot carries.
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scene depth (read-only) for the `soft_fade` use_soft_z
                // branch. `texture_depth_2d` in WGSL — point-loaded (no
                // sampler). Real only in `camera_bind_group_transparent`;
                // a 1×1 placeholder in the default + `_sl` groups (which
                // bind in depth-WRITING passes, where soft_z never runs).
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_model_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model_bgl"),
            entries: &[uniform_entry(
                0,
                size_of::<ModelUniforms>() as u64,
                wgpu::ShaderStages::VERTEX,
                true,
            )],
        })
    }

    fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let filterable = wgpu::TextureSampleType::Float {
            filterable: true,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
            entries: &[
                // 0: base_map  1: bump_map  2: self_illum_map
                // 3: change_color_map  4: sampler  5: material uniform
                // 6: detail_map  7: bump_detail_map  8: env_cubemap
                tex_entry(0, filterable),
                tex_entry(1, filterable),
                tex_entry(2, filterable),
                tex_entry(3, filterable),
                sampler_entry(4, wgpu::SamplerBindingType::Filtering),
                // Material uniform — size varies per VariantKey, so
                // we declare `min_binding_size = None` (any uniform
                // buffer goes; wgpu won't validate against a fixed
                // shader-struct size). The shader's declared struct
                // must be ≤ the buffer we bind, but that's already
                // satisfied because we serialize bytes from the same
                // layout we generate the WGSL struct from.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(6, filterable),
                tex_entry(7, filterable),
                cube_tex_entry(8, filterable),
            ],
        })
    }

    fn create_node_matrices_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("node_matrices_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        (MAXIMUM_NUMBER_OF_MODEL_NODES * size_of::<[[f32; 4]; 4]>()) as u64,
                    ),
                },
                count: None,
            }],
        })
    }

}

/// Build a `camera_bgl` bind group with the supplied resources. Used
/// by both `camera_bind_group` (placeholder G-buffer views at 10/11)
/// and `camera_bind_group_sl` (real G-buffer views at 10/11).
fn build_camera_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    engine_lighting_buffer: &wgpu::Buffer,
    atmosphere_buffer: &wgpu::Buffer,
    lightmap_compress_buffer: &wgpu::Buffer,
    lightprobe_atlas_view: &wgpu::TextureView,
    lightprobe_intensity_atlas_view: &wgpu::TextureView,
    filtering_sampler: &wgpu::Sampler,
    engine_underwater_buffer: &wgpu::Buffer,
    simple_lights_buffer: &wgpu::Buffer,
    engine_exposure_buffer: &wgpu::Buffer,
    albedo_view: &wgpu::TextureView,
    normal_view: &wgpu::TextureView,
    engine_misc_ps_buffer: &wgpu::Buffer,
    engine_dominant_light_buffer: &wgpu::Buffer,
    kernel5_ps_buffer: &wgpu::Buffer,
    scene_depth_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: engine_lighting_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        size_of::<GpuEngineLightingPs>() as u64,
                    ),
                }),
            },
            // Bind ONE table entry's worth (dynamic offset selects which);
            // the backing buffer holds ATMOSPHERE_TABLE_ENTRIES of them.
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: atmosphere_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(size_of::<GpuAtmosphereData>() as u64),
                }),
            },
            wgpu::BindGroupEntry { binding: 3, resource: lightmap_compress_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(lightprobe_atlas_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lightprobe_intensity_atlas_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(filtering_sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: engine_underwater_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: simple_lights_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<GpuSimpleLights>() as u64),
                }),
            },
            wgpu::BindGroupEntry { binding: 9, resource: engine_exposure_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(albedo_view) },
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(normal_view) },
            wgpu::BindGroupEntry { binding: 12, resource: engine_misc_ps_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: engine_dominant_light_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        size_of::<GpuEngineDominantLight>() as u64,
                    ),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: kernel5_ps_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::TextureView(scene_depth_view),
            },
        ],
    })
}

/// Build a 1×1 BC3 (DXT5) Texture2DArray with `slices` zeroed slices.
/// Each BC3 4×4 block is 16 bytes; we pad the 1×1 to one full block per
/// slice (`slices × 16` bytes). Sampling this returns transparent black,
/// which is the right "no lightmap" fallback for terrain shaders.
fn make_placeholder_bc3_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    slices: u32,
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: slices,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc3RgbaUnorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zero_block = [0u8; 16];
    for slice in 0..slices {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: slice },
                aspect: wgpu::TextureAspect::All,
            },
            &zero_block,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        );
    }
    texture
}
