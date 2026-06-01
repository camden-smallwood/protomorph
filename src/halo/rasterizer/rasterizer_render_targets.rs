//! Mirror of `Ares/source/rasterizer/rasterizer_render_targets.h` —
//! the `s_surface_group_description` schema + the
//! `g_surface_group_descriptions[56]` table that maps every `Surface`
//! enum value to its format/dimensions/aliases.
//!
//! Engine source: `g_surface_group_descriptions @ 0x181188570` in
//! halo3_dllcache_play.dll (56 entries × 72 bytes). Decoded in
//! `reference_engine_surface_descriptions_full.md`.
//!
//! Phase P1 of the engine-faithful pipeline plan. Surface allocation
//! (`allocate_default`) walks this table to create the wgpu textures.

use crate::halo::rasterizer::Surface;

/// `e_surface_specialization_flags` (Ares
/// `rasterizer_render_targets.h:16`). Bit-packed flags describing how
/// the surface is dimensioned and aliased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSpecializationFlags(pub u32);

impl SurfaceSpecializationFlags {
    /// `_surface_specialization_flag_screen_res` — width/height params
    /// are DIVISORS of the render resolution (1.0 = full, 4.0 = quarter).
    /// When clear, params are literal pixel counts.
    pub const SCREEN_RES: u32 = 0x01;
    /// `_surface_specialization_flag_tiled_surface` — Xbox-360 EDRAM
    /// tile-managed surface. Ignored on PC / wgpu.
    pub const TILED_SURFACE: u32 = 0x02;
    /// `_surface_specialization_flag_screen_sized_texture` — texture
    /// mips sized to screen.
    pub const SCREEN_SIZED_TEXTURE: u32 = 0x04;
    /// `_surface_specialization_flag_HDRformat` — TYPELESS root that
    /// gets aliased into multiple typed views.
    pub const HDR_FORMAT: u32 = 0x08;
    /// `_surface_specialization_flag_EDRAM_tile_width_aligned` — Xbox
    /// EDRAM tile alignment. PC: no-op.
    pub const EDRAM_TILE_WIDTH_ALIGNED: u32 = 0x10;
    /// `_surface_specialization_flag_splitscreen_res` — resize per
    /// splitscreen quadrant.
    pub const SPLITSCREEN_RES: u32 = 0x20;
    /// `_surface_specialization_flag_alias_to_buffer_end` — alias
    /// offset is computed from end-of-buffer instead of start.
    pub const ALIAS_TO_BUFFER_END: u32 = 0x40;
    /// `_surface_specialization_flag_read_only_alias` — view-only
    /// alias; no write path.
    pub const READ_ONLY_ALIAS: u32 = 0x80;
    /// `_surface_specialization_flag_stencil_shader_view` — extra SRV
    /// for stencil sampling.
    pub const STENCIL_SHADER_VIEW: u32 = 0x100;

    #[inline]
    pub const fn has(self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }
}

/// Mirror of `s_surface_group_description` (Ares
/// `rasterizer_render_targets.h:137`). Engine sizeof = 72 bytes; we
/// keep field order verbatim.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceDescription {
    pub specialization_flags: SurfaceSpecializationFlags,
    pub is_depth_stencil: bool,
    pub surface_width_param: f32,
    pub surface_height_param: f32,
    /// DXGI format value (decoded against wgpu in [`Self::wgpu_format`]).
    pub surface_format: DxgiFormat,
    pub surface_exp_bias: i32,
    pub surface_srgb: bool,
    /// MSAA sample count (1, 2, 4, 8). Engine PC almost always 1; only
    /// `_surface_aux_mini2` uses 4× for distortion accumulator.
    pub surface_msaa: u32,
    pub texture_width_param: f32,
    pub texture_height_param: f32,
    pub texture_format: DxgiFormat,
    pub texture_exp_bias: i32,
    pub texture_srgb: bool,
    pub resolve_exp_bias: i32,
    /// Surface this entry aliases over. `Surface::None` if standalone.
    /// Engine uses signed i16 (-1 = no alias) — we use `None`.
    pub texture_alias_surface: Surface,
    /// Byte offset within the aliased surface. 0 for plain aliases.
    pub texture_alias_byte_offset: u32,
}

/// DXGI format mirror — only the values that appear in
/// `g_surface_group_descriptions`. Numeric values match
/// `DXGI_FORMAT` enum from `dxgiformat.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DxgiFormat {
    Unknown = 0,
    R32G32B32A32Float = 2,
    R16G16B16A16Float = 10,
    R16G16B16A16Unorm = 11,
    R10G10B10A2Unorm = 24,
    R8G8B8A8Typeless = 27,
    R8G8B8A8Unorm = 28,
    R8G8B8A8UnormSrgb = 29,
    R16G16Float = 34,
    R16G16Unorm = 35,
    R32Float = 41,
    R8Unorm = 61,
    D24UnormS8Uint = 45,
    D32Float = 40,
    D32FloatS8X24Uint = 20,
}

impl DxgiFormat {
    /// Map to a wgpu format. Engine sRGB upgrade applied by caller via
    /// `c_render_surface::create` — see `SurfaceTable::allocate_default`.
    pub fn to_wgpu(self) -> Option<wgpu::TextureFormat> {
        Some(match self {
            DxgiFormat::Unknown => return None,
            DxgiFormat::R32G32B32A32Float => wgpu::TextureFormat::Rgba32Float,
            DxgiFormat::R16G16B16A16Float => wgpu::TextureFormat::Rgba16Float,
            DxgiFormat::R16G16B16A16Unorm => wgpu::TextureFormat::Rgba16Unorm,
            DxgiFormat::R10G10B10A2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
            DxgiFormat::R8G8B8A8Typeless => wgpu::TextureFormat::Rgba8Unorm,
            DxgiFormat::R8G8B8A8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            DxgiFormat::R8G8B8A8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            DxgiFormat::R16G16Float => wgpu::TextureFormat::Rg16Float,
            DxgiFormat::R16G16Unorm => wgpu::TextureFormat::Rg16Unorm,
            DxgiFormat::R32Float => wgpu::TextureFormat::R32Float,
            DxgiFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            DxgiFormat::D24UnormS8Uint => wgpu::TextureFormat::Depth24PlusStencil8,
            DxgiFormat::D32Float => wgpu::TextureFormat::Depth32Float,
            DxgiFormat::D32FloatS8X24Uint => wgpu::TextureFormat::Depth32FloatStencil8,
        })
    }
}

const fn desc(
    specialization_flags: u32,
    is_depth_stencil: bool,
    surface_width_param: f32,
    surface_height_param: f32,
    surface_format: DxgiFormat,
    surface_srgb: bool,
    surface_msaa: u32,
    texture_format: DxgiFormat,
    texture_srgb: bool,
    texture_alias_surface: Surface,
) -> SurfaceDescription {
    SurfaceDescription {
        specialization_flags: SurfaceSpecializationFlags(specialization_flags),
        is_depth_stencil,
        surface_width_param,
        surface_height_param,
        surface_format,
        surface_exp_bias: 0,
        surface_srgb,
        surface_msaa,
        texture_width_param: surface_width_param,
        texture_height_param: surface_height_param,
        texture_format,
        texture_exp_bias: 0,
        texture_srgb,
        resolve_exp_bias: 0,
        texture_alias_surface,
        texture_alias_byte_offset: 0,
    }
}

/// Mirror of `g_surface_group_descriptions @ 0x181188570`. 56 entries
/// indexed by `Surface as usize`.
///
/// Engine raw decode in `reference_engine_surface_descriptions_full.md`.
/// EDRAM/HiZ offsets dropped (Xbox-only). Alias byte offsets dropped
/// (all observed values are 0 in the engine table — the 0x4C0 / 0x720
/// etc. constants are EDRAM positioning, not byte offsets within an
/// alias chain).
pub const SURFACE_DESCRIPTIONS: [SurfaceDescription; 56] = {
    use DxgiFormat::*;
    use Surface as S;
    [
        // 0  _surface_none
        desc(0, false, 0.0, 0.0, Unknown, false, 1, Unknown, false, S::None),
        // 1  _surface_display
        desc(0x21, false, 1.0, 1.0, R8G8B8A8Unorm, true, 1, R8G8B8A8Unorm, true, S::None),
        // 2  _surface_shadow_1 (literal 512×512)
        desc(0, true, 512.0, 512.0, D24UnormS8Uint, false, 1, D24UnormS8Uint, false, S::None),
        // 3  _surface_depth_stencil (screen-res + stencil_shader_view)
        desc(0x121, true, 1.0, 1.0, D32FloatS8X24Uint, false, 1, D32FloatS8X24Uint, false, S::None),
        // 4  _surface_depth_stencil_read_only (alias over e3)
        desc(0xA1, true, 1.0, 1.0, D32FloatS8X24Uint, false, 1, D32FloatS8X24Uint, false, S::DepthStencil),
        // 5  _surface_depth_stencil_ms (Xbox EDRAM-aliased; PC: typed)
        desc(1, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 6  _surface_depth_stencil_tron_edram (alias over e5)
        desc(1, false, 1.0, 1.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::DepthStencilMs),
        // 7  _surface_depth_stencil_tron_edram_8x3 (alias over e3)
        desc(1, false, 1.0, 1.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::DepthStencil),
        // 8  _surface_screenshot_composite_16f (alias over e5)
        desc(1, false, 1.0, 1.0, R16G16B16A16Unorm, false, 1, R16G16B16A16Unorm, false, S::DepthStencilMs),
        // 9  _surface_screenshot_composite_8bit (alias over e5)
        desc(1, false, 1.0, 1.0, R32Float, false, 1, R32Float, false, S::DepthStencilMs),
        // 10 _surface_screenshot_display (literal 512×512)
        desc(0, false, 512.0, 512.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::DepthStencilMs),
        // 11 _surface_screenshot_composite_16i (alias over e5)
        desc(1, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::DepthStencilMs),
        // 12 _surface_screenshot_composite_depth (TYPELESS HDR root)
        desc(0x29, false, 0.0, 0.0, Unknown, false, 1, Unknown, false, S::None),
        // 13 _surface_screenshot_composite_cubemap (TYPELESS HDR root)
        desc(0x29, false, 0.0, 0.0, Unknown, false, 1, Unknown, false, S::None),
        // 14 _surface_screenshot_simple_resolve (alias over e12)
        desc(0x29, false, 0.0, 0.0, Unknown, false, 1, Unknown, false, S::ScreenshotCompositeDepth),
        // 15 _surface_accum_LDR (alias over e13)
        desc(0x29, false, 0.0, 0.0, Unknown, false, 1, Unknown, false, S::ScreenshotCompositeCubemap),
        // 16 _surface_accum_HDR (sRGB UNORM view over e12)
        desc(0x21, false, 1.0, 1.0, R8G8B8A8Unorm, true, 1, R8G8B8A8Unorm, true, S::ScreenshotCompositeDepth),
        // 17 _surface_post_LDR (plain UNORM view over e12)
        desc(0x21, false, 1.0, 1.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::ScreenshotCompositeDepth),
        // 18 _surface_post_HDR (R10G10B10A2 view over e13)
        desc(0x21, false, 1.0, 1.0, R10G10B10A2Unorm, false, 1, R10G10B10A2Unorm, false, S::ScreenshotCompositeCubemap),
        // 19 _surface_albedo (UNORM view over e12)
        desc(1, false, 1.0, 1.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::ScreenshotCompositeDepth),
        // 20 _surface_albedo_debug (quarter-res FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 21 _surface_normal (quarter-res FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 22 _surface_fullscreen_blur (quarter-res FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 23 _surface_aux_reflection (1/16 UNORM)
        desc(1, false, 16.0, 16.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::None),
        // 24 _surface_aux_refraction (quarter FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 25 _surface_aux_bloom (quarter FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 26 _surface_aux_chud_overlay (1/16 FLOAT)
        desc(1, false, 16.0, 16.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 27 _surface_aux_star (1/64 FLOAT)
        desc(1, false, 64.0, 64.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 28 _surface_aux_small (1×1 literal FLOAT — exposure metering)
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 29 _surface_aux_tiny (1×1 literal)
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 30 _surface_aux_mini (1×1 literal)
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 31-35 _surface_aux_exposure_0..4 (1×1 literal)
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        desc(0, false, 1.0, 1.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 36 _surface_aux_exposure_5 (quarter FLOAT)
        desc(1, false, 4.0, 4.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 37 _surface_aux_exposure_6 (1/16 FLOAT)
        desc(1, false, 16.0, 16.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 38 _surface_aux_exposure_7 (1/64 FLOAT)
        desc(1, false, 64.0, 64.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 39 _surface_aux_small2 (half-res sRGB UNORM, alias over e13)
        desc(0x21, false, 2.0, 2.0, R8G8B8A8Unorm, true, 1, R8G8B8A8Unorm, true, S::ScreenshotCompositeCubemap),
        // 40 _surface_aux_tiny2 (half-res sRGB UNORM, alias over e13)
        desc(0x21, false, 2.0, 2.0, R8G8B8A8Unorm, true, 1, R8G8B8A8Unorm, true, S::ScreenshotCompositeCubemap),
        // 41 _surface_aux_mini2 (half-res R16G16_FLOAT 4×MSAA — distortion accum)
        desc(0x21, false, 2.0, 2.0, R16G16Float, false, 4, R16G16Float, false, S::None),
        // 42 _surface_aux_depth_of_field_high_rez (literal 320×192)
        desc(0, true, 320.0, 192.0, D32FloatS8X24Uint, false, 1, D32FloatS8X24Uint, false, S::Shadow1),
        // 43 _surface_distortion (literal 320×192)
        desc(0, false, 320.0, 192.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::Shadow1),
        // 44 _surface_depth_camera_depth (full-res, sRGB UNORM — odd; engine srgb=1 on R16G16_FLOAT is no-op)
        desc(0x21, false, 1.0, 1.0, R16G16Float, true, 1, R16G16Float, true, S::None),
        // 45 _surface_depth_camera_stencil (literal 800×800)
        desc(0, false, 800.0, 800.0, R16G16Float, false, 1, R16G16Float, false, S::None),
        // 46 _surface_depth_camera (literal 800×800, alias over depth_camera_stencil)
        desc(0, false, 800.0, 800.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::DepthCameraStencil),
        // 47 _surface_aux_water_interaction_height (half-res FLOAT)
        desc(0x21, false, 2.0, 2.0, R16G16B16A16Float, false, 1, R16G16B16A16Float, false, S::None),
        // 48 _surface_aux_water_interaction_slope (full-res UNORM, alias over e13)
        desc(0x21, false, 1.0, 1.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::ScreenshotCompositeCubemap),
        // 49 _surface_chud_turbulence (literal 576×320, sRGB R16G16_UNORM no-op)
        desc(0, false, 576.0, 320.0, R16G16Unorm, true, 1, R16G16Unorm, true, S::None),
        // 50 _surface_cortana_effect (literal 512×512 depth)
        desc(0, true, 512.0, 512.0, D24UnormS8Uint, false, 1, D24UnormS8Uint, false, S::None),
        // 51 _surface_depth_camera_texture (literal 384×384)
        desc(0, false, 384.0, 384.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::None),
        // 52 _surface_weather_occlusion (literal 384×384)
        desc(0, false, 384.0, 384.0, R8G8B8A8Unorm, false, 1, R8G8B8A8Unorm, false, S::None),
        // 53 _surface_dynamic_albedo (literal 384×384, 10-bit)
        desc(0, false, 384.0, 384.0, R10G10B10A2Unorm, false, 1, R10G10B10A2Unorm, false, S::None),
        // 54 _surface_dynamic_accumulation (literal 384×384 depth)
        desc(0, true, 384.0, 384.0, D32FloatS8X24Uint, false, 1, D32FloatS8X24Uint, false, S::None),
        // 55 _surface_dynamic_normal (read-only alias over e54)
        desc(0x80, true, 384.0, 384.0, D32FloatS8X24Uint, false, 1, D32FloatS8X24Uint, false, S::DynamicAccumulation),
    ]
};

/// Resolve a surface's pixel dimensions at the current render
/// resolution. Engine `c_render_surface_group::create` logic:
/// - if `screen_res` flag set: `width = render_width / width_param`
/// - else: `width = width_param` (literal pixel count)
pub fn resolve_surface_size(desc: &SurfaceDescription, render_width: u32, render_height: u32) -> (u32, u32) {
    let resolve = |param: f32, render: u32| -> u32 {
        if desc.specialization_flags.has(SurfaceSpecializationFlags::SCREEN_RES) {
            ((render as f32 / param).ceil() as u32).max(1)
        } else {
            (param as u32).max(1)
        }
    };
    (
        resolve(desc.surface_width_param, render_width),
        resolve(desc.surface_height_param, render_height),
    )
}
