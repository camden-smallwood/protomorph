//! Per-material data for the renderer. Halo-only — every material
//! comes from the rmsh tag pipeline (`halo::load_shader_material` →
//! `blam_tags::render_method::ResolvedRenderMethod`), so the shape
//! mirrors what Halo's shader system exposes plus a few derived
//! scalars the WGSL reads.

use blam_tags::render_method::{
    RenderMethod, RenderMethodOptionParameter, RenderMethodTemplate,
    ResolvedCbuffer, ResolvedRenderMethod,
};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Texture usage
// ---------------------------------------------------------------------------

/// Which logical slot a `MaterialTexture` binds to. Names are the Halo
/// rmop parameter names, mapped onto material_bgl slots in the renderer.
///
/// Halo's per-pixel specular mask comes from `base_map.alpha` under the
/// `specular_mask = from_diffuse` rmdf option — no separate texture
/// needed; the shader reads `albedo.a` after sampling base_map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialTextureUsage {
    /// `base_map` — albedo / diffuse color (sRGB).
    Diffuse,
    /// `self_illum_map` — emissive contribution.
    Emissive,
    /// `bump_map` — tangent-space normal map (CPU-decoded from BC5/DXN
    /// with Z reconstructed in the bitmap loader).
    Normal,
    /// `change_color_map` — 4-channel mask selecting per-region tint
    /// regions (R=primary, G=secondary, etc.).
    ChangeColorMap,
    /// `detail_map` — high-frequency surface variation multiplied
    /// into `base_map`.
    DetailMap,
    /// `bump_detail_map` — high-frequency tangent-space normal map
    /// added on top of `bump_map` (XY summed, then renormalized) for
    /// the `detail` bump_mapping option. Also BC5/DXN with Z
    /// reconstructed at load time, same as `bump_map`.
    BumpDetail,
    /// `environment_map` — reflection cubemap (6 faces). Sampled with
    /// the view-reflect direction; multiplied by `env_tint_color`,
    /// `environment_map_specular_contribution`, and the per-fragment
    /// spec_mask × specular_coefficient.
    Environment,
}

/// Pre-extracted texture from a Halo `bitm` tag with native format +
/// mip pyramid intact. The renderer hands the bytes directly to wgpu
/// without re-encoding.
///
/// `layers` = 1 for 2D images, 6 for cube maps. For cube maps, `data`
/// stores the layers back-to-back, each holding its own full mip
/// pyramid (face 0's mips, then face 1's, etc.) — same layout wgpu
/// expects when uploading with `depth_or_array_layers = 6`.
#[derive(Debug)]
pub struct InlineTexture {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub layers: u32,
    pub is_cube: bool,
    pub format: InlinePixelFormat,
    pub data: Vec<u8>,
}

/// Subset of wgpu-native formats we actually emit from the bitmap
/// loader. Halo formats with no clean wgpu equivalent (`DxnMonoAlpha`,
/// `A4r4g4b4`, etc.) get pre-decoded to one of these on the loader side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlinePixelFormat {
    Bc1RgbaUnormSrgb, Bc1RgbaUnorm,
    Bc2RgbaUnormSrgb, Bc2RgbaUnorm,
    Bc3RgbaUnormSrgb, Bc3RgbaUnorm,
    Bc4RUnorm, Bc4RSnorm,
    Bc5RgUnorm, Bc5RgSnorm,
    Rgba8UnormSrgb, Rgba8Unorm, Rgba8Snorm,
    Bgra8UnormSrgb, Bgra8Unorm,
    Rg8Unorm, Rg8Snorm,
    R8Unorm,
    Rgba16Float, Rgba16Unorm, Rgba16Snorm,
    Rgba32Float,
}

impl InlinePixelFormat {
    pub fn block_bytes(self) -> u32 {
        match self {
            Self::Bc1RgbaUnormSrgb | Self::Bc1RgbaUnorm | Self::Bc4RUnorm | Self::Bc4RSnorm => 8,
            Self::Bc2RgbaUnormSrgb | Self::Bc2RgbaUnorm
            | Self::Bc3RgbaUnormSrgb | Self::Bc3RgbaUnorm
            | Self::Bc5RgUnorm | Self::Bc5RgSnorm => 16,
            _ => 0,
        }
    }

    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::Rg8Snorm => 2,
            Self::Rgba8UnormSrgb | Self::Rgba8Unorm | Self::Rgba8Snorm
            | Self::Bgra8UnormSrgb | Self::Bgra8Unorm => 4,
            Self::Rgba16Float | Self::Rgba16Unorm | Self::Rgba16Snorm => 8,
            Self::Rgba32Float => 16,
            _ => 0,
        }
    }

    pub fn is_compressed(self) -> bool { self.block_bytes() > 0 }

    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        if self.is_compressed() {
            let bw = ((width + 3) / 4).max(1);
            let bh = ((height + 3) / 4).max(1);
            (bw * bh * self.block_bytes()) as usize
        } else {
            (width * height * self.bytes_per_pixel()) as usize
        }
    }
}

#[derive(Debug)]
pub struct MaterialTexture {
    /// Halo parameter name as declared in the rmop (e.g. `"base_map"`,
    /// `"base_map_m_0"`, `"blend_map"`). Used by per-variant binding
    /// lookup — `MaterialBindings::textures[i].name` maps to one of
    /// these. The legacy `usage` enum is for rmsh-style texture-decode
    /// hints (linear vs srgb); per-name binding is what wgpu sees.
    pub name: String,
    pub usage: MaterialTextureUsage,
    pub texture: InlineTexture,
}

// ---------------------------------------------------------------------------
// MaterialData — flat Halo-shaped material
// ---------------------------------------------------------------------------

/// Halo material. All scalar/color fields default to Halo-typical values
/// (matching the rmop defaults for the most-common shader options); the
/// rmsh walker overrides them in `halo::load_shader_material` whenever
/// the tag carries an explicit value.
#[derive(Debug)]
pub struct MaterialData {
    pub name: String,
    pub textures: Vec<MaterialTexture>,

    // Halo two-lobe-phong scalars (read by halo_shader.wgsl)
    pub diffuse_coefficient: f32,
    pub specular_coefficient: f32,
    pub specular_power: f32,            // normal_specular_power
    pub glancing_specular_power: f32,
    pub fresnel_curve_steepness: f32,
    pub self_illum_intensity: f32,

    // Two-lobe Phong tints (rmsh `normal_specular_tint` / `glancing_specular_tint`)
    pub normal_specular_tint: Vec3,
    pub glancing_specular_tint: Vec3,
    pub albedo_specular_tint_blend: f32,
    /// `env_tint_color` from the environment_mapping rmop. Multiplied
    /// into the cubemap reflection along with `env_map_specular_contribution`.
    pub env_tint_color: Vec3,

    // Lighting-loop weights from rmsh `*_specular_contribution`. Engine
    // sums analytic + simple-lights, then weights by analytical, then
    // adds area × area_contribution, then env-map × env_contribution.
    pub analytical_specular_contribution: f32,
    pub area_specular_contribution: f32,
    pub environment_map_specular_contribution: f32,

    /// Strength of `bump_detail_map` when summed onto `bump_map.xy`
    /// (Halo `calc_bumpmap_detail_ps` uses this scalar to scale the
    /// detail before adding). Ignored unless the rmdf bump_mapping
    /// option is `detail`.
    pub bump_detail_coefficient: f32,

    // Change colors (engine externs in production; defaults stand in
    // until protomorph wires a biped-variant pipeline)
    pub primary_change_color: Vec3,
    pub secondary_change_color: Vec3,

    /// Resolved (category_name → option_name) map from `rmsh.options`
    /// against the rmdf — what Halo's offline shader compiler used to
    /// pick HLSL macro expansions. Drives the assembler's `pick_*`
    /// dispatch and the pipeline-cache `VariantKey`. Empty for stub
    /// materials (load failures fall back to a default pipeline).
    pub category_choices: crate::halo::render_methods::CategoryChoices,

    /// Walker output for this rmsh — drives the per-variant cbuffer
    /// layout + texture binding generation. Empty for stub materials.
    pub resolved: ResolvedRenderMethod,

    /// rmt2-routed pixel cbuffer for the Albedo entry point: layout
    /// (slot names + offsets) + pre-filled bytes ready to upload.
    /// Computed at material-load time from `(rmsh, rmt2, entry_point)`
    /// — see `halo::resolve_render_method`. Empty `slots` for stub
    /// materials; the renderer falls back to a default upload.
    pub cbuffer: ResolvedCbuffer,

    /// Source tag data needed to re-resolve the cbuffer at a
    /// per-frame `current_time`. `None` for stub materials and
    /// materials whose rmsh/rmt2/rmop chain failed to load.
    /// Engine equivalent: the runtime keeps `c_render_method`
    /// instances + their rmop chain alive for as long as the
    /// material is used so `update_constants @ 0x180685300` can
    /// re-evaluate animated parameters each frame.
    pub render_method_source: Option<RenderMethodSource>,

    /// True when at least one cbuffer slot's resolved vec4 differs
    /// between `eval_time = 0.0` and `eval_time = 1.0` — i.e., the
    /// material has a time-bearing TagFunction (scrolling textures,
    /// pulsing colors, etc.). Computed once at load via
    /// `blam_tags::is_cbuffer_animated`.
    ///
    /// **P6.2 ships detection only.** Per-frame re-resolve + GPU write
    /// is a P6.5 follow-up — the renderer-side material BG flow needs
    /// surgery to keep `wgpu::Buffer` references reachable per material
    /// (currently created inside `resolve_material_resources` and only
    /// reachable through bind-group ref counts). Until that lands,
    /// animated materials still freeze at their t=0 values.
    pub is_animated: bool,
}

/// Snapshot of the (rmsh, rmt2, rmop_chain) source data the cbuffer
/// merge consumes. Cached on `MaterialData` so animated parameters
/// can be re-evaluated per frame via
/// `blam_tags::render_method::cbuffer::rebuild_cbuffer_bytes_at_time`.
/// Heavy enough that we only carry it on materials that benefit
/// (currently only rmw — animated `time_warp` etc. drive wave
/// motion). Other rm** subclasses can opt in as their animation use
/// cases land.
#[derive(Debug, Clone)]
pub struct RenderMethodSource {
    pub rmsh: RenderMethod,
    /// `None` when the rmsh ships with an empty `postprocess_definition`
    /// (author-format tags — e.g., `vahalla_waterfall.shader`). In that
    /// case the per-frame re-resolve walks `rmop_params` directly
    /// instead of going through `rmt2.float_constants[]`. See
    /// `loader.rs::shader_from_render_method` for the load-time
    /// equivalent (`cbuffer_from_rmt2.unwrap_or_else(|| ...)`).
    pub rmt2: Option<RenderMethodTemplate>,
    /// Flat rmop parameter list — equivalent to
    /// `c_render_method_definition::build_parameter_list @ 0x826E31D8`.
    pub rmop_params: Vec<RenderMethodOptionParameter>,
}

impl MaterialData {
    pub fn stub_for_shader(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            textures: Vec::new(),
            diffuse_coefficient: 1.0,
            specular_coefficient: 0.0,
            specular_power: 10.0,
            glancing_specular_power: 10.0,
            fresnel_curve_steepness: 5.0,
            self_illum_intensity: 0.0,
            normal_specular_tint: Vec3::ONE,
            glancing_specular_tint: Vec3::ONE,
            albedo_specular_tint_blend: 0.0,
            env_tint_color: Vec3::ONE,
            analytical_specular_contribution: 1.0,
            area_specular_contribution: 0.0,
            environment_map_specular_contribution: 0.0,
            bump_detail_coefficient: 1.0,
            // Match engine zero-init: dllcache's `m_change_colors[4]` is
            // `unsigned long[4]` (packed pixel32), zero-init at scenario
            // load. `render_object_update_change_colors` only writes
            // when `object_get_change_color` returns true (per-object
            // customization). For scenery without customization the
            // slots stay zero → pixel32_to_real_rgb_color(0) = (0,0,0).
            // Per-object customization (scenario palette, player team
            // color, etc.) is a future phase — for now match engine.
            primary_change_color: Vec3::ZERO,
            secondary_change_color: Vec3::ZERO,
            category_choices: crate::halo::render_methods::CategoryChoices::default(),
            resolved: ResolvedRenderMethod::default(),
            cbuffer: stub_cbuffer(),
            render_method_source: None,
            is_animated: false,
        }
    }

    pub fn find_texture(&self, usage: MaterialTextureUsage) -> Option<&InlineTexture> {
        self.textures.iter().find(|t| t.usage == usage).map(|t| &t.texture)
    }

    /// Look up a loaded texture by Halo parameter name. Used by the
    /// per-variant material bind group builder — `MaterialBindings`
    /// gives a slot per parameter name; this resolves the actual
    /// texture data to bind there.
    pub fn find_texture_by_name(&self, name: &str) -> Option<&InlineTexture> {
        self.textures.iter().find(|t| t.name == name).map(|t| &t.texture)
    }
}

/// Stub cbuffer pre-populated with the WGSL-required slots so a stub
/// material still produces a layout that all our fragments can compile
/// against. Defaults match `crate::halo::loader::WGSL_REQUIRED_SLOTS`.
fn stub_cbuffer() -> ResolvedCbuffer {
    use blam_tags::render_method::CbufferSlot;
    let mut cbuffer = ResolvedCbuffer { slots: Vec::new(), total_bytes: 0, bytes: Vec::new() };
    let mut byte_offset = 0u32;
    for (name, default, is_xform) in crate::halo::loader::WGSL_REQUIRED_SLOTS {
        cbuffer.slots.push(CbufferSlot {
            source_name: name.to_string(),
            is_xform: *is_xform,
            byte_offset,
            value: *default,
        });
        for c in default.iter() {
            cbuffer.bytes.extend_from_slice(&c.to_le_bytes());
        }
        byte_offset += 16;
    }
    cbuffer.total_bytes = byte_offset.max(16);
    cbuffer
}

// (GpuMaterialProps deleted — material uniforms are now serialized
// per-VariantKey by `halo_shader::serializer::serialize` using the
// layout `halo_shader::cbuffer::CbufferLayout` builds from each rmsh's
// resolved parameter list. The Rust side no longer hardcodes a
// material-cbuffer struct.)
