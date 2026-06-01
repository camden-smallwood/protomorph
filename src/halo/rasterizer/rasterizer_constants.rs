//! Mirror of `c_rasterizer`'s public enums from
//! `Ares/source/rasterizer/rasterizer.h`. Verbatim names + values
//! (without the `_` prefix Halo uses on enum members) so call sites
//! match Ares 1:1.
//!
//! Halo C++ has its enums as nested types (`c_rasterizer::e_alpha_blend_mode`
//! etc.); Rust idiom is to flatten under the rasterizer module so
//! call sites read `rasterizer::AlphaBlendMode::Opaque` instead of
//! `c_rasterizer::_alpha_blend_opaque`.

pub const K_NUMBER_OF_SAMPLERS: usize = 16;
pub const K_NUMBER_OF_TEXTURES: usize = 32;
pub const K_NUMBER_OF_VERTEX_SAMPLERS: usize = 4;
pub const K_NUMBER_OF_VERTEX_TEXTURES: usize = 4;
pub const K_NUMBER_OF_COMPUTE_SAMPLERS: usize = 4;
pub const K_NUMBER_OF_COMPUTE_TEXTURES: usize = 4;
pub const K_NUMBER_OF_COLOR_SURFACES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Platform {
    Xenon = 0,
    Dx9 = 1,
    Durango = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum PresentationInterval {
    Default = 0,
    One,
    Two,
    Three,
    Four,
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum AlphaBlendMode {
    #[default]
    Opaque = 0,
    Additive,
    Multiply,
    AlphaBlend,
    DoubleMultiply,
    PreMultipliedAlpha,
    Maximum,
    MultiplyAdd,
    AddSrcTimesDstAlpha,
    AddSrcTimesSrcAlpha,
    InvAlphaBlend,
    MotionBlurStatic,
    MotionBlurInhibit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SeparateAlphaBlendMode {
    #[default]
    Off = 0,
    Opaque,
    Additive,
    Multiply,
    ToConstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum FirstPersonMode {
    Never = 0,
    Sometimes,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum ZBufferMode {
    #[default]
    Write = 0,
    Read,
    Off,
    ShadowGenerate,
    ShadowGenerateDynamicLights,
    ShadowApply,
    Decals,
    DebugGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum StencilMode {
    #[default]
    Off = 0,
    OldStipple,
    VolumeBackPlanes,
    VolumeFrontPlanes,
    VolumeApply,
    VolumeApplyHiStencilOnly,
    VolumeApplyAndClear,
    VolumeClear,
    VolumeClearHiStencil,
    Decorators,
    Decals,
    TronWrite,
    TronRead,
    AmbientObjectClearShadowBit,
    GlassDecalsWrite,
    GlassDecalsRead,
    ObjectRendering,
    TronReadPc,
}

/// Bitflags for `set_color_write_enable`. Default = all channels.
pub mod color_write_enable {
    pub const NONE: u32 = 0;
    pub const RED: u32 = 1;
    pub const GREEN: u32 = 2;
    pub const BLUE: u32 = 4;
    pub const ALPHA: u32 = 8;
    pub const COLOR: u32 = 7; // RGB
    pub const ALL: u32 = 15;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum CullMode {
    Off = 1,
    #[default]
    Cw = 2,
    Ccw = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum FillMode {
    Point = 1,
    Wireframe = 2,
    #[default]
    Solid = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SamplerAddressMode {
    #[default]
    Wrap = 0,
    Clamp,
    Mirror,
    Border,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SamplerFilterMode {
    #[default]
    Trilinear = 0,
    Point,
    Bilinear,
    Anisotropic1,
    Anisotropic2,
    Anisotropic3,
    Anisotropic4,
    /// `_sampler_filter_lightprobe_texture_array` —
    /// `D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT` per dllcache's
    /// `d3d11_sampler_state_cache::get`.
    LightprobeTextureArray,
}

pub const SAMPLER_CLEAR_ALL_TEXTURES: u32 = 0xFFFF;
pub const SAMPLER_MIP_BIAS_NONE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GprAllocation {
    DefaultD3d = 0,
    Default,
    MaxToPixelShader,
    MaxToVertexShader,
    AllToVertexShader,
    DecoratorAllocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TessellationMode {
    Discrete = 0,
    Continues,
    PerEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SplitscreenRes {
    Default = 0,
    HalfWidth,
    HalfSize,
    EightByThreeHalfHeight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum RenderMode {
    Default = 0,
    Mode7e3,
    Gamma2,
    XRgb,
}

/// Mirror of `c_rasterizer::e_surface` (engine `g_surface_group_descriptions`
/// at `0x181188570`). All 56 entries kept verbatim — engine indices match
/// 1:1 so call sites and the SURFACE_DESCRIPTIONS table line up.
///
/// See `reference_engine_surface_descriptions_full.md` for the full per-entry
/// format/spec-flags/alias decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Surface {
    None = 0,
    Display = 1,
    /// Aliases `_surface_occlusion` (both `= 2` in engine).
    Shadow1 = 2,
    DepthStencil = 3,
    DepthStencilReadOnly = 4,
    DepthStencilMs = 5,
    DepthStencilTronEdram = 6,
    DepthStencilTronEdram8x3 = 7,
    ScreenshotComposite16f = 8,
    ScreenshotComposite8bit = 9,
    ScreenshotDisplay = 10,
    ScreenshotComposite16i = 11,
    ScreenshotCompositeDepth = 12,
    ScreenshotCompositeCubemap = 13,
    ScreenshotSimpleResolve = 14,
    AccumLdr = 15,
    AccumHdr = 16,
    PostLdr = 17,
    PostHdr = 18,
    Albedo = 19,
    AlbedoDebug = 20,
    Normal = 21,
    FullscreenBlur = 22,
    AuxReflection = 23,
    AuxRefraction = 24,
    AuxBloom = 25,
    AuxChudOverlay = 26,
    AuxStar = 27,
    AuxSmall = 28,
    AuxTiny = 29,
    AuxMini = 30,
    AuxExposure0 = 31,
    AuxExposure1 = 32,
    AuxExposure2 = 33,
    AuxExposure3 = 34,
    AuxExposure4 = 35,
    AuxExposure5 = 36,
    AuxExposure6 = 37,
    AuxExposure7 = 38,
    AuxSmall2 = 39,
    AuxTiny2 = 40,
    AuxMini2 = 41,
    AuxDepthOfFieldHighRez = 42,
    Distortion = 43,
    DepthCameraDepth = 44,
    DepthCameraStencil = 45,
    DepthCamera = 46,
    AuxWaterInteractionHeight = 47,
    AuxWaterInteractionSlope = 48,
    ChudTurbulence = 49,
    CortanaEffect = 50,
    DepthCameraTexture = 51,
    WeatherOcclusion = 52,
    DynamicAlbedo = 53,
    DynamicAccumulation = 54,
    DynamicNormal = 55,
}

pub const K_NUMBER_OF_SURFACES: usize = 56;
pub const K_SURFACE_AUX_EXPOSURE_COUNT: usize = 8;

impl Surface {
    /// Alias — mirrors engine's `_surface_occlusion = 2` (same value as
    /// `_surface_shadow_1`). Engine overloads the slot for both
    /// shadow generation and occlusion queries.
    pub const OCCLUSION: Surface = Surface::Shadow1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BufferGammaMode {
    TrueSrgb = 0,
    PiecewiseLinearSrgb,
    Mode7e3,
}
