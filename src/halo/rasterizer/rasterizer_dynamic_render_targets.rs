//! Mirror of `Ares/source/rasterizer/rasterizer_dynamic_render_targets.h`.
//! Plus our `SurfaceTable` — the runtime owner of every wgpu texture
//! that backs the `Surface` enum.
//!
//! Halo's rasterizer has two distinct surface concepts:
//!   - **Static surfaces** — the 56 entries of `e_surface`. These
//!     map to fixed-purpose render targets (display, accum_HDR,
//!     post_HDR, depth_stencil, aux_bloom, etc.). Their wgpu
//!     textures live in `SurfaceTable`.
//!   - **Dynamic surfaces** — runtime-allocated render targets for
//!     texture cameras / mirrors / refractions. Halo uses
//!     `c_dynamic_render_targets` (k_maximum_dynamic_render_targets
//!     = 6). Mirrored as `DynamicRenderTargets`.

use crate::halo::rasterizer::rasterizer_render_targets::{
    resolve_surface_size, SurfaceSpecializationFlags, SURFACE_DESCRIPTIONS,
};
use crate::halo::rasterizer::{Surface, K_NUMBER_OF_SURFACES};

/// Mirror of `e_render_target_type` (Ares
/// `rasterizer_dynamic_render_targets.h:13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DynamicRenderTargetType {
    None = 0,
    TextureCamera,
    Mirror,
    Refraction,
    Code,
}

pub const K_MAXIMUM_DYNAMIC_RENDER_TARGETS: usize = 6;

/// Mirror of `s_dynamic_render_target` (Ares 284 bytes).
#[derive(Debug, Clone)]
pub struct DynamicRenderTarget {
    pub identifier: i16,
    pub frame_allocated: i32,
    pub target_type: DynamicRenderTargetType,
    pub accumulation_surface: Surface,
    pub albedo_surface: Surface,
    pub normal_surface: Surface,
    pub depth_stencil_surface: Surface,
    pub name: String,
}

/// Owns the wgpu textures backing every `Surface`. Allocated on
/// init from the swapchain / window size; resized on window resize.
///
/// Mirrors Halo's static surface layout — the rasterizer never
/// gives out a `wgpu::Texture` directly; render code references
/// surfaces by `Surface` enum.
pub struct SurfaceTable {
    /// Per-Surface wgpu texture + view. `None` for surfaces we
    /// haven't allocated (e.g. Xbox-specific aux exposure slots).
    /// Indexed by `Surface as usize`.
    pub entries: Vec<Option<SurfaceEntry>>,
    /// Display-resolution width/height. Most surfaces match.
    pub display_width: u32,
    pub display_height: u32,
}

pub struct SurfaceEntry {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub size: wgpu::Extent3d,
    pub label: &'static str,
}

impl SurfaceTable {
    pub fn new(display_width: u32, display_height: u32) -> Self {
        let mut entries: Vec<Option<SurfaceEntry>> = Vec::with_capacity(K_NUMBER_OF_SURFACES);
        for _ in 0..K_NUMBER_OF_SURFACES {
            entries.push(None);
        }
        Self {
            entries,
            display_width,
            display_height,
        }
    }

    /// Resolve a `Surface` to its current wgpu view, or `None` if
    /// not allocated. Mirrors Halo's `get_surface` helper.
    pub fn get(&self, surface: Surface) -> Option<&SurfaceEntry> {
        self.entries
            .get(surface as usize)
            .and_then(|e| e.as_ref())
    }
    pub fn get_mut(&mut self, surface: Surface) -> Option<&mut SurfaceEntry> {
        self.entries
            .get_mut(surface as usize)
            .and_then(|e| e.as_mut())
    }

    /// Set the entry for a surface. Used during `allocate_*` calls
    /// at init / window resize.
    pub fn set(&mut self, surface: Surface, entry: SurfaceEntry) {
        let idx = surface as usize;
        if idx < self.entries.len() {
            self.entries[idx] = Some(entry);
        }
    }

    /// `c_rasterizer::surface_valid`.
    pub fn surface_valid(&self, surface: Surface) -> bool {
        self.get(surface).is_some()
    }

    /// Allocate every surface in `SURFACE_DESCRIPTIONS` at the current
    /// display resolution. Mirrors engine
    /// `c_render_surface_group::create` over `g_surface_group_descriptions`.
    ///
    /// Surfaces with `Unknown` format are skipped (Xbox-360 root surfaces
    /// like `_surface_screenshot_composite_depth` that the engine binds
    /// via per-pass type reinterpretation). For wgpu we don't need them
    /// as separate textures — children of an alias chain are allocated
    /// standalone with the alias's own format.
    ///
    /// `Surface::Display` is skipped — the swapchain owns its texture.
    pub fn allocate_default(&mut self, device: &wgpu::Device) {
        for (idx, desc) in SURFACE_DESCRIPTIONS.iter().enumerate() {
            let surface = match idx_to_surface(idx) {
                Some(s) => s,
                None => continue,
            };
            if surface == Surface::None || surface == Surface::Display {
                continue;
            }
            // Skip Xbox-360 EDRAM-only surfaces (TILED_SURFACE flag).
            // PC engine still allocates them as plain textures, but
            // they're unreferenced in our wgpu pipeline.
            if desc.specialization_flags.has(SurfaceSpecializationFlags::TILED_SURFACE) {
                continue;
            }
            // Skip alias surfaces — they're alternate-format VIEWS over
            // the parent surface's memory in engine. PC wgpu would need
            // view_formats compatibility (which most engine alias pairs
            // lack — Rgba8 vs R10G10B10A2 vs Rgba16Float across the same
            // root). When P3 lands proper rpass helpers, callers
            // resolve `Surface::AccumHdr` (an alias) by looking up the
            // parent + creating a typed view on demand.
            if desc.texture_alias_surface != Surface::None {
                continue;
            }
            // Root surfaces with Unknown format (e.g.
            // _surface_screenshot_composite_depth). Engine creates them
            // as TYPELESS_R8G8B8A8; aliased children carry their own
            // format. Skip and let the children carry their own
            // standalone textures.
            let wgpu_format = match desc.surface_format.to_wgpu() {
                Some(f) => f,
                None => continue,
            };
            let (width, height) = resolve_surface_size(desc, self.display_width, self.display_height);
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
            // RENDER_ATTACHMENT only when the format supports it without
            // a special feature. R16/R16G16/Rgba16 _UNORM_ variants
            // require `TEXTURE_FORMAT_*_RENDERABLE` features which we
            // don't enable. Aliased / screenshot surfaces using those
            // formats are never written-to in the protomorph path.
            let render_attach_ok = matches!(
                wgpu_format,
                wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Rgba8UnormSrgb
                    | wgpu::TextureFormat::Rgba16Float
                    | wgpu::TextureFormat::Rgba32Float
                    | wgpu::TextureFormat::Rgb10a2Unorm
                    | wgpu::TextureFormat::Rg16Float
                    | wgpu::TextureFormat::R32Float
                    | wgpu::TextureFormat::R8Unorm
                    | wgpu::TextureFormat::Depth32Float
                    | wgpu::TextureFormat::Depth24PlusStencil8
                    | wgpu::TextureFormat::Depth32FloatStencil8
            );
            if render_attach_ok {
                usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
            }
            let label = surface_label(surface);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: desc.surface_msaa,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                format: Some(wgpu_format),
                ..Default::default()
            });
            self.set(
                surface,
                SurfaceEntry {
                    texture,
                    view,
                    format: wgpu_format,
                    size,
                    label,
                },
            );
        }
    }

    /// Resize all owned surfaces to `(width, height)`. Re-runs
    /// `allocate_default` after clearing entries.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.display_width = width;
        self.display_height = height;
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
        self.allocate_default(device);
    }
}

/// `c_dynamic_render_targets` mirror. Owns the dynamic texture-camera
/// / mirror / refraction allocations.
pub struct DynamicRenderTargets {
    pub allocated_flags: i32,
    pub used_targets: [i32; K_MAXIMUM_DYNAMIC_RENDER_TARGETS],
    pub targets: Vec<Option<DynamicRenderTarget>>,
}

impl DynamicRenderTargets {
    pub fn new() -> Self {
        Self {
            allocated_flags: 0,
            used_targets: [0; K_MAXIMUM_DYNAMIC_RENDER_TARGETS],
            targets: (0..K_MAXIMUM_DYNAMIC_RENDER_TARGETS).map(|_| None).collect(),
        }
    }

    /// `c_dynamic_render_targets::allocate_target`.
    pub fn allocate_target(
        &mut self,
        _target_type: DynamicRenderTargetType,
        _width: i32,
        _height: i32,
        _target_handle: i32,
        _frame_index: i32,
        _name: &str,
    ) -> i32 {
        todo!("DynamicRenderTargets::allocate_target — phase 6 (texture cameras)")
    }

    /// `c_dynamic_render_targets::get_target`.
    pub fn get_target(&self, target_handle: i32) -> Option<&DynamicRenderTarget> {
        self.targets
            .get(target_handle as usize)
            .and_then(|t| t.as_ref())
    }

    /// `c_dynamic_render_targets::get_target_from_target_type`.
    pub fn get_target_from_target_type(
        &self,
        target_type: DynamicRenderTargetType,
    ) -> Option<&DynamicRenderTarget> {
        self.targets
            .iter()
            .filter_map(|t| t.as_ref())
            .find(|t| t.target_type == target_type)
    }

    /// `c_dynamic_render_targets::target_allocated`.
    pub fn target_allocated(&self, target_type: DynamicRenderTargetType) -> bool {
        self.get_target_from_target_type(target_type).is_some()
    }
}

/// Index → `Surface` enum reverse lookup. Returns `None` if the index
/// is out of range.
fn idx_to_surface(idx: usize) -> Option<Surface> {
    Some(match idx {
        0 => Surface::None,
        1 => Surface::Display,
        2 => Surface::Shadow1,
        3 => Surface::DepthStencil,
        4 => Surface::DepthStencilReadOnly,
        5 => Surface::DepthStencilMs,
        6 => Surface::DepthStencilTronEdram,
        7 => Surface::DepthStencilTronEdram8x3,
        8 => Surface::ScreenshotComposite16f,
        9 => Surface::ScreenshotComposite8bit,
        10 => Surface::ScreenshotDisplay,
        11 => Surface::ScreenshotComposite16i,
        12 => Surface::ScreenshotCompositeDepth,
        13 => Surface::ScreenshotCompositeCubemap,
        14 => Surface::ScreenshotSimpleResolve,
        15 => Surface::AccumLdr,
        16 => Surface::AccumHdr,
        17 => Surface::PostLdr,
        18 => Surface::PostHdr,
        19 => Surface::Albedo,
        20 => Surface::AlbedoDebug,
        21 => Surface::Normal,
        22 => Surface::FullscreenBlur,
        23 => Surface::AuxReflection,
        24 => Surface::AuxRefraction,
        25 => Surface::AuxBloom,
        26 => Surface::AuxChudOverlay,
        27 => Surface::AuxStar,
        28 => Surface::AuxSmall,
        29 => Surface::AuxTiny,
        30 => Surface::AuxMini,
        31 => Surface::AuxExposure0,
        32 => Surface::AuxExposure1,
        33 => Surface::AuxExposure2,
        34 => Surface::AuxExposure3,
        35 => Surface::AuxExposure4,
        36 => Surface::AuxExposure5,
        37 => Surface::AuxExposure6,
        38 => Surface::AuxExposure7,
        39 => Surface::AuxSmall2,
        40 => Surface::AuxTiny2,
        41 => Surface::AuxMini2,
        42 => Surface::AuxDepthOfFieldHighRez,
        43 => Surface::Distortion,
        44 => Surface::DepthCameraDepth,
        45 => Surface::DepthCameraStencil,
        46 => Surface::DepthCamera,
        47 => Surface::AuxWaterInteractionHeight,
        48 => Surface::AuxWaterInteractionSlope,
        49 => Surface::ChudTurbulence,
        50 => Surface::CortanaEffect,
        51 => Surface::DepthCameraTexture,
        52 => Surface::WeatherOcclusion,
        53 => Surface::DynamicAlbedo,
        54 => Surface::DynamicAccumulation,
        55 => Surface::DynamicNormal,
        _ => return None,
    })
}

/// Human-readable label for `wgpu::Texture` debugging.
fn surface_label(s: Surface) -> &'static str {
    match s {
        Surface::None => "surface_none",
        Surface::Display => "surface_display",
        Surface::Shadow1 => "surface_shadow_1",
        Surface::DepthStencil => "surface_depth_stencil",
        Surface::DepthStencilReadOnly => "surface_depth_stencil_read_only",
        Surface::DepthStencilMs => "surface_depth_stencil_ms",
        Surface::DepthStencilTronEdram => "surface_depth_stencil_tron_edram",
        Surface::DepthStencilTronEdram8x3 => "surface_depth_stencil_tron_edram_8x3",
        Surface::ScreenshotComposite16f => "surface_screenshot_composite_16f",
        Surface::ScreenshotComposite8bit => "surface_screenshot_composite_8bit",
        Surface::ScreenshotDisplay => "surface_screenshot_display",
        Surface::ScreenshotComposite16i => "surface_screenshot_composite_16i",
        Surface::ScreenshotCompositeDepth => "surface_screenshot_composite_depth",
        Surface::ScreenshotCompositeCubemap => "surface_screenshot_composite_cubemap",
        Surface::ScreenshotSimpleResolve => "surface_screenshot_simple_resolve",
        Surface::AccumLdr => "surface_accum_LDR",
        Surface::AccumHdr => "surface_accum_HDR",
        Surface::PostLdr => "surface_post_LDR",
        Surface::PostHdr => "surface_post_HDR",
        Surface::Albedo => "surface_albedo",
        Surface::AlbedoDebug => "surface_albedo_debug",
        Surface::Normal => "surface_normal",
        Surface::FullscreenBlur => "surface_fullscreen_blur",
        Surface::AuxReflection => "surface_aux_reflection",
        Surface::AuxRefraction => "surface_aux_refraction",
        Surface::AuxBloom => "surface_aux_bloom",
        Surface::AuxChudOverlay => "surface_aux_chud_overlay",
        Surface::AuxStar => "surface_aux_star",
        Surface::AuxSmall => "surface_aux_small",
        Surface::AuxTiny => "surface_aux_tiny",
        Surface::AuxMini => "surface_aux_mini",
        Surface::AuxExposure0 => "surface_aux_exposure_0",
        Surface::AuxExposure1 => "surface_aux_exposure_1",
        Surface::AuxExposure2 => "surface_aux_exposure_2",
        Surface::AuxExposure3 => "surface_aux_exposure_3",
        Surface::AuxExposure4 => "surface_aux_exposure_4",
        Surface::AuxExposure5 => "surface_aux_exposure_5",
        Surface::AuxExposure6 => "surface_aux_exposure_6",
        Surface::AuxExposure7 => "surface_aux_exposure_7",
        Surface::AuxSmall2 => "surface_aux_small2",
        Surface::AuxTiny2 => "surface_aux_tiny2",
        Surface::AuxMini2 => "surface_aux_mini2",
        Surface::AuxDepthOfFieldHighRez => "surface_aux_depth_of_field_high_rez",
        Surface::Distortion => "surface_distortion",
        Surface::DepthCameraDepth => "surface_depth_camera_depth",
        Surface::DepthCameraStencil => "surface_depth_camera_stencil",
        Surface::DepthCamera => "surface_depth_camera",
        Surface::AuxWaterInteractionHeight => "surface_aux_water_interaction_height",
        Surface::AuxWaterInteractionSlope => "surface_aux_water_interaction_slope",
        Surface::ChudTurbulence => "surface_chud_turbulence",
        Surface::CortanaEffect => "surface_cortana_effect",
        Surface::DepthCameraTexture => "surface_depth_camera_texture",
        Surface::WeatherOcclusion => "surface_weather_occlusion",
        Surface::DynamicAlbedo => "surface_dynamic_albedo",
        Surface::DynamicAccumulation => "surface_dynamic_accumulation",
        Surface::DynamicNormal => "surface_dynamic_normal",
    }
}
