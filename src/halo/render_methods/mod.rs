//! Halo WGSL shader assembler. Mirrors Halo's offline shader pipeline
//! at runtime: given a `ResolvedRenderMethod` (rmsh + rmdf + rmop chain)
//! plus a target entry point, picks the right shading-fragment WGSL
//! files for each rmdf category, concatenates them, and returns a
//! complete WGSL source string ready for naga.
//!
//! ## Design
//!
//! Each shader category (albedo, bump_mapping, material_model, ...) has
//! one WGSL fragment per rmdf option. Every variant exports the SAME
//! function name (`calc_albedo`, `calc_bumpmap`, `calc_material_lighting`)
//! so the entry-point umbrella's call sites don't change. This is the
//! WGSL equivalent of Halo's HLSL `#define calc_albedo_ps
//! calc_albedo_two_change_color_ps` macro injection.
//!
//! ## Phase 2A scope
//!
//! - Single `albedo` entry point
//! - `albedo` category: only `two_change_color` variant
//! - `bump_mapping` category: `default` (standard) regardless of option
//!   (grunt's `detail` variant needs an extra texture slot we haven't
//!   plumbed yet)
//! - `material_model` category: real two_lobe_phong analytic + Fresnel
//!   tint blend (area lobe stubbed to 0 until SH lands; no PRT, no
//!   simple lights, no env-map reflection)
//!
//! Phase 2C expands: detail bump, SH ambient, simple-lights loop,
//! shadow_generate entry, more albedo variants, etc.

pub mod animated;
pub mod cbuffer;
pub mod externs;
pub mod material_bindings;
pub mod material_samplers;
pub mod materials;
pub mod pipeline_cache;
pub mod postprocess_sampler;
pub mod serializer;

use blam_tags::render_method::{
    ParameterSource, RenderMethod, RenderMethodDefinition, ResolvedCbuffer, ResolvedRenderMethod,
    ResolvedValue,
};

/// Per-category option choices for one render_method, sourced from the
/// rmdf by name — the same shape Halo's offline shader compiler keys on.
///
/// `rmsh.options[i]` is just an integer index into `rmdf.categories[i].options[]`.
/// At runtime, dllcache never inspects `options[i]` directly: the
/// offline compiler resolved the category-name → option-name → pixel
/// function mapping when it baked the pixl tag. We mirror that
/// resolution here and dispatch by **name**, not by hardcoded position.
///
/// Stored as a sorted vec so the type is `Hash + Eq` for use as a
/// `VariantKey` field (cache key for compiled pipelines).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct CategoryChoices {
    /// `(category_name, chosen_option_name)` pairs sorted by category_name.
    /// Empty for stub materials (load failures).
    pairs: Vec<(String, String)>,
    /// Category names whose chosen option is NON-DEFAULT (rmsh option
    /// index != 0 — i.e. not the rmdf's neutral `none`/`off`/`default`
    /// option[0]). Recorded at `resolve` time (the only place the rmdf
    /// option ordering is known). Drives the fail-loud guard in
    /// `assemble`: a non-default option in a category the chosen subclass
    /// path doesn't render would silently produce a wrong shader, so we
    /// panic instead. Sorted; subset of `pairs`' category names.
    non_default: Vec<String>,
}

impl CategoryChoices {
    /// Build from a parsed rmsh + its rmdf. For each category in the
    /// rmdf, pick the option named at `rmsh.options[i]` and record
    /// `(category.category_name, option.option_name)`.
    pub fn resolve(rm: &RenderMethod, rmdf: &RenderMethodDefinition) -> Self {
        let mut non_default: Vec<String> = Vec::new();
        let mut pairs: Vec<(String, String)> = rmdf.categories.iter().enumerate()
            .filter_map(|(cat_idx, category)| {
                if category.category_name.is_empty() { return None; }
                let opt_idx = match rm.options.get(cat_idx).copied() {
                    Some(idx) => idx.max(0) as usize,
                    None => {
                        eprintln!(
                            "[render_method] rmsh has no option for category '{}' (#{cat_idx}) \
                             — rmsh.options ({}) shorter than rmdf categories; using option 0",
                            category.category_name, rm.options.len(),
                        );
                        0
                    }
                };
                // Option index 0 is the rmdf's neutral default (none/off/
                // default/the category's identity); anything else is a
                // meaningful choice the assembly MUST consume.
                if opt_idx != 0 {
                    non_default.push(category.category_name.clone());
                }
                let option_name = match category.options.get(opt_idx) {
                    Some(o) => o.option_name.clone(),
                    None => {
                        eprintln!(
                            "[render_method] category '{}' option index {opt_idx} out of range \
                             ({} options) — empty option name will fail the downstream option pick",
                            category.category_name, category.options.len(),
                        );
                        String::new()
                    }
                };
                Some((category.category_name.clone(), option_name))
            })
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        non_default.sort();
        Self { pairs, non_default }
    }

    /// Category names whose chosen option is non-default (rmdf option index
    /// != 0). The assemble guard panics on any of these its subclass path
    /// doesn't consume.
    pub fn non_default(&self) -> &[String] {
        &self.non_default
    }

    /// Look up the chosen option_name for a category. Returns `None`
    /// when the rmdf doesn't declare that category — caller decides
    /// whether to use a default or panic.
    pub fn get(&self, category: &str) -> Option<&str> {
        self.pairs.iter().find(|(c, _)| c == category).map(|(_, o)| o.as_str())
    }

    /// Look up with a fallback default. Use when "category absent" is
    /// equivalent to "category set to its first option" (e.g. a sparse
    /// rmsh where missing categories should behave as `"none"` /
    /// `"off"`).
    pub fn get_or<'a>(&'a self, category: &str, default: &'a str) -> &'a str {
        self.get(category).unwrap_or(default)
    }

    pub fn is_empty(&self) -> bool { self.pairs.is_empty() }
    pub fn pairs(&self) -> &[(String, String)] { &self.pairs }

    /// Build directly from sorted pairs. Caller is responsible for
    /// sorting by `category_name` (keeps `Hash + Eq` deterministic).
    pub fn from_pairs(pairs: Vec<(String, String)>) -> Self {
        // Manual construction (no rmdf) → no non-default tracking; the
        // assemble guard is a no-op for these.
        Self { pairs, non_default: Vec::new() }
    }
}

/// Halo entry point. Maps 1:1 to `EntryPoint` in blam-tags.
///
/// Naming honesty: dllcache exposes 16 entry points (see `entry_fx.hlsl`).
/// The four most relevant for our v1 BSP path:
///
/// - `_entry_point_albedo` (HLSL `albedo_ps`, idx=1): G-buffer fill
///   only — writes albedo+specmask to RT0 and encoded normal to RT1.
///   No lighting math. Used in dllcache's albedo pass.
/// - `_entry_point_static_default` (HLSL `static_default_ps`, idx=2):
///   Also G-buffer fill (just calls `albedo_ps` underneath). Used
///   when a static-lit BSP cluster has no per-pixel/per-vertex/probe
///   lightmap data — picked by `select_instance_entry_point` as the
///   minimal albedo fill.
/// - `_entry_point_static_sh` (HLSL `static_sh_ps`, idx=5): forward
///   shaded with default SH probe — the picker's fallback when the
///   lightmap chain misses entirely. Reads default-lighting cbuffer
///   (slot 0x24 + 0x30) populated by `setup_default_lighting`. Calls
///   `calc_output_color_with_explicit_light_quadratic` umbrella with
///   `ravi_order_3` SH evaluation, material_model dispatch, env_map,
///   self_illum, scattering, exposure.
/// - `_entry_point_static_per_pixel` (HLSL `static_per_pixel_ps`, idx=3):
///   forward shaded with per-pixel lightmap atlas — the proper lit
///   path. Phase F1.
///
/// Our v1 single-pass forward uses `static_sh_ps`'s body (not
/// `static_default_ps` — that's just G-buffer fill).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HaloEntryPoint {
    /// HLSL `albedo_ps` / `albedo_pass_fx::convert_to_albedo_target`.
    /// First geometry pass — outputs MRT[0] = albedo color (RGB) +
    /// specular_mask (A) to `_surface_albedo`, MRT[1] = world normal
    /// (encoded `n*0.5+0.5`) + albedo.w to `_surface_post_HDR`. NO
    /// lighting, NO atmosphere, NO exposure. Engine: called via
    /// `c_object_renderer::render_albedo(1)` + `c_structure_renderer
    /// ::render_albedo()` inside `c_player_view::render_albedo`.
    Albedo,
    /// HLSL `static_sh_ps`. Single-probe SH cbuffer (10×vec4 ravi) +
    /// `ravi_order_3` + `_quadratic` umbrella. The path for OBJECTS
    /// (vehicles/scenery objects) and the fallback when a surface has
    /// no lightmap chart.
    StaticSh,
    /// HLSL `static_per_pixel_ps`. Per-pixel atlas SH (4 vec4 order-2)
    /// + `ravi_order_2_with_dominant_light` + `_linear_with_dominant_light`
    /// umbrella. The path for BSP cluster + structure-instanced rmsh
    /// surfaces — anything with a per-pixel lightmap chart.
    StaticPerPixel,
    /// `static_sh` body but the SH coefficients come from a SECONDARY
    /// VERTEX BUFFER (one probe per vertex, interpolated to fragment)
    /// rather than the cbuffer-bound averaged probe. Engine-faithful
    /// path for `pervertex_block_index >= 0` BSP instances and
    /// clusters: gives spatially-varying ambient across the mesh
    /// instead of the per-cluster/per-instance averaging bridge.
    StaticShPerVertex,
    /// HLSL `static_lighting_prt_ambient_ps`. `static_sh` lighting math
    /// with the addition that each vertex carries a UNORM scalar
    /// transfer coefficient (per-vertex ambient visibility) baked into
    /// a secondary vertex buffer (`R32_FLOAT BLENDWEIGHT1` slot 2 in
    /// MCC PC; Ares `rasterizer_resource_definitions.cpp:46`). The
    /// fragment multiplies the SH probe's L0 (ambient) band by this
    /// coefficient — i.e. ambient occlusion baked at lightmap time and
    /// stored per-vertex. Engine: `_entry_point_static_lighting_prt_quadratic`
    /// is the upstream caller; `render_mesh_part_default @
    /// 0x18069EBC0` remaps to the actual ambient/linear/quadratic
    /// variant via `entry_point_remapping_0[mesh.transfer_vector_vertex_type]`.
    /// Activated only for instances with `lightmapping_policy == 2
    /// (single-probe)` AND meshes with `vertex_buffer_indices[3] !=
    /// 0xFFFF`. See `project_research_per_mesh_prt_2026_05_11.md`.
    StaticPrtAmbient,
    /// HLSL `static_per_vertex_color_ps` (entry_points_fx.hlsl:836) —
    /// engine `_entry_point_vertex_color_lighting`. `static_sh` math but
    /// the diffuse LIGHTING term is the per-vertex baked `vert_color`
    /// (interpolated from a SECONDARY VERTEX BUFFER, `SkyVertColorVertex`
    /// at slot 1 / location 12) instead of the SH probe. Engine routes
    /// sky `.render_model` mesh parts whose mesh has the
    /// `_mesh_has_vertex_color_bit` (`mesh->flags & 1`) here
    /// (`render_mesh_part_default @ 0x18069EBC0:64-82`). Diffuse-only
    /// (no bump/spec/envmap/PRT); albedo + self-illum + fog + exposure +
    /// per-blend-mode convert all identical to `static_sh`.
    StaticVertexColor,
    /// HLSL `decal_fx.hlsl::default_ps`. Per-decal indexed draw, called
    /// by `c_decal::render @ 0x18039B100`. Vertex stream is
    /// `rasterizer_vertex_world` (44 B), NOT the standard ModelVertex.
    /// PS branches on the rmd's 6 categories (albedo / blend_mode /
    /// render_pass / specular / bump_mapping / tinting).
    ///
    /// `Decal` (this variant) targets the post-SL composition:
    /// `[lighting_base, hdr_dark]` MRTs — used when
    /// `c_decal_definition::m_pass == _pass_post_static_lighting`
    /// (additive overlays, laser scorches). Pipeline blend uses the
    /// rmd's authored `blend_mode` option.
    Decal,
    /// Same WGSL as `Decal`, but targets the albedo-pass MRT shape
    /// `[albedo_view (Rg11b10Ufloat), normal_view (Rgba8Unorm)]` for
    /// `c_decal_definition::m_pass == _pass_post_albedo`. Engine call
    /// site: `c_decal_system::render_all(_pass_post_albedo)` inside
    /// `c_player_view::render_albedo`. RT0 receives the blended decal
    /// color with engine `color_write_enable(0, 7)` (RGB only); RT1
    /// (normal buffer / `_surface_post_HDR`) has writes masked off.
    /// Decals dispatched here modify the albedo G-buffer that
    /// render_static_lighting re-samples + lights — engine-faithful
    /// "decal participates in lighting" pathway.
    DecalAlbedo,
    /// Object-attached rmd decals — engine `c_object_renderer::
    /// render_albedo_decals @ 0x1806E4A60`. SAME PS body as
    /// `DecalAlbedo` (G-buffer fill via the rmd `default_ps`), but the
    /// VS takes the OBJECT skinned `ModelVertex` layout instead of
    /// `rasterizer_vertex_world`. Engine compiles the rmd shader for
    /// BOTH vertex types; `render_method_submit(material, vertex_type
    /// = ObjectVertex, _entry_point_default)` picks this variant for
    /// object mesh parts whose material `c_render_method::get_is_decal
    /// @ 0x1806C0D60` returns true (group_tag == 'rmd '). Visibility
    /// submit (`c_object_renderer::submit_object_mesh_parts @
    /// 0x1806E1BF0`) sets the `_render_object_mesh_part_decal_bit`
    /// (= 1 << 5) on those parts so they dispatch under
    /// `render_albedo_decals` only. RT shape matches `DecalAlbedo`
    /// (albedo + normal G-buffer write).
    DecalObject,
}

/// Cache key for compiled shader variants. Two materials sharing the
/// same `(entry_point, group_tag, category_choices)` resolve to the
/// SAME WGSL program — that's exactly Halo's offline-compiler equivalence
/// class. group_tag in the key keeps rmsh and rmtr (with possibly
/// overlapping category names) on separate pipelines.
///
/// The choices ALONE don't fully determine the WGSL + BGL, though: two
/// per-material facts that the assembler reads also shape the output and
/// so must be in the key, or distinct-layout materials would collide on
/// a cache hit and reuse a mismatched binding layout:
///   - `cube_mask`: per-texture-slot cube-vs-2D classification. The
///     assembler emits `texture_cube<f32>` vs `texture_2d<f32>` (and
///     the BGL view-dimension Cube vs D2) per slot based on
///     `MaterialBindings::from_resolved` (force-cube names + the bitm
///     tag's `is_cube()`). Two materials with identical choices but a
///     different cube classification on some slot produce different WGSL.
///   - `sampler_signature`: the per-material `MaterialSamplerMap`
///     signature (a hash over the unique `SamplerKey` set — address /
///     filter modes). The assembler emits one `s_dedupe_K` sampler per
///     unique key and rewrites `<tex>_sampler` references against the
///     per-binding dedup map; a different sampler-mode set → different
///     unique-key count / WGSL + BGL.
///   - `layout_signature`: a hash of the cbuffer's ordered slot names
///     and the ordered texture-binding names. The `choices` fix the
///     shader LOGIC, but not the cbuffer STRUCT layout: two materials
///     with identical choices can still resolve their cbuffers in
///     different orders (rmt2-baked `float_constants` order vs author-
///     format rmop-param order, which may interleave extra scalars such
///     as `no_dynamic_lights`). They assemble different WGSL structs and
///     the renderer uploads bytes positionally, so sharing one cached
///     variant reads the second material's bytes through the first's
///     offsets. See `VariantKey::layout_signature`.
/// All components are necessary (each changes WGSL/BGL or byte layout)
/// and, together with the choices, sufficient: any two materials with
/// the same full key assemble byte-identical WGSL + an identical BGL
/// over an identical cbuffer layout, so they correctly share one cache
/// entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantKey {
    pub entry_point: HaloEntryPoint,
    /// FOURCC of source rm** tag (`'rmsh'`, `'rmtr'`, etc.). The
    /// shader assembler dispatches on this to pick terrain/water/
    /// foliage WGSL bodies. See `reference_rmtr_runtime_distinction.md`.
    pub group_tag: u32,
    /// Resolved (category_name → option_name) pairs from the rmdf —
    /// what the offline compiler used to pick HLSL macro expansions.
    pub choices: CategoryChoices,
    /// Bitmask of cube-classified texture slots: bit `i` set iff
    /// `bindings.textures[i].is_cube`. Distinguishes materials whose
    /// per-slot `texture_cube` vs `texture_2d` declarations differ.
    /// Slot counts past 64 fold into the high bit (extremely unlikely;
    /// max baseline is ~22) — see [`VariantKey::cube_mask_of`].
    pub cube_mask: u64,
    /// Stable hash of the material's deduped sampler-key set
    /// (`MaterialSamplerMap::signature`). Distinguishes materials whose
    /// authored address/filter sampler modes differ.
    pub sampler_signature: u64,
    /// Stable hash of the cbuffer's ordered slot names plus the ordered
    /// texture-binding names. The assembled WGSL declares its material
    /// cbuffer struct in `cbuffer.slots` order and its texture bindings
    /// in `bindings.textures` order; the renderer then uploads per-
    /// material bytes positionally and binds textures by those same
    /// slots. Two materials can share identical `choices` yet compile
    /// to DIFFERENT layouts — e.g. an rmt2-baked shader (slots in
    /// `rmt2.float_constants` order) vs an author-format shader with no
    /// baked rmt2 (slots in rmop-param order, which can interleave extra
    /// scalars like `no_dynamic_lights`). Without this component they
    /// collide on one cached variant, and the second material's bytes
    /// are read through the first's struct offsets — scrambling colors
    /// and reading an xform scale out of a color slot (observed: the
    /// snowbound battery plasma core, sharing the airlock_field variant,
    /// rendered red with ~9× over-tiling). See `cov_battery_illum_core`
    /// vs `airlock_field`.
    pub layout_signature: u64,
}

impl VariantKey {
    /// Build the key from a material's resolved render method plus its
    /// already-computed `bindings` (for the cube mask) and `sampler_map`
    /// (for the sampler signature). Threading these in avoids a second
    /// bitm-tag read at key time — `ensure` computes them once and reuses
    /// the same instances when it assembles the variant on a cache miss.
    pub fn from_resolved(
        rm: &ResolvedRenderMethod,
        entry_point: HaloEntryPoint,
        choices: &CategoryChoices,
        cbuffer: &blam_tags::render_method::ResolvedCbuffer,
        bindings: &material_bindings::MaterialBindings,
        sampler_map: &material_samplers::MaterialSamplerMap,
    ) -> Self {
        Self {
            entry_point,
            group_tag: rm.group_tag,
            choices: choices.clone(),
            cube_mask: Self::cube_mask_of(bindings),
            sampler_signature: sampler_map.signature,
            layout_signature: Self::layout_signature_of(cbuffer, bindings),
        }
    }

    /// Hash the assembled material's layout — the ordered cbuffer slot
    /// names and the ordered texture-binding names. The WGSL struct +
    /// texture bindings are emitted in exactly these orders, and the
    /// renderer uploads/binds positionally against them, so any change
    /// to the order or membership produces an incompatible cached
    /// variant. Two shaders with identical `choices` but different
    /// layouts (rmt2-baked vs rmop-param order, extra interleaved
    /// scalars, …) hash differently here and get separate variants.
    pub fn layout_signature_of(
        cbuffer: &blam_tags::render_method::ResolvedCbuffer,
        bindings: &material_bindings::MaterialBindings,
    ) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        cbuffer.slots.len().hash(&mut h);
        for s in &cbuffer.slots {
            s.source_name.hash(&mut h);
            s.is_xform.hash(&mut h);
        }
        bindings.textures.len().hash(&mut h);
        for tb in &bindings.textures {
            tb.name.hash(&mut h);
            tb.slot.hash(&mut h);
        }
        h.finish()
    }

    /// Fold the per-slot cube classification into a single bitmask.
    /// Bit `i` set iff `bindings.textures[i]` is a cube. Slots `>= 64`
    /// wrap (`i % 64`); the realistic max slot count (~22) never reaches
    /// the wrap, so no two distinct-layout materials collide in practice.
    pub fn cube_mask_of(bindings: &material_bindings::MaterialBindings) -> u64 {
        let mut mask = 0u64;
        for (i, tb) in bindings.textures.iter().enumerate() {
            if tb.is_cube {
                mask |= 1u64 << (i % 64);
            }
        }
        mask
    }
}

/// Result of assembling a variant: the WGSL source string ready for
/// naga, plus the cbuffer layout the renderer needs to serialize
/// per-material values into the correct byte offsets, plus the
/// per-variant texture binding map (slot indices for each Bitmap-typed
/// rmop parameter — used by both the WGSL declarations AND the
/// material bind group builder).
pub struct AssembledVariant {
    pub wgsl: String,
    pub cbuffer: cbuffer::CbufferLayout,
    pub bindings: material_bindings::MaterialBindings,
    /// Per-binding sampler dedup map — drives the per-binding sampler
    /// BGL entries + binds in `bsp_gpu::resolve_material_resources` and
    /// `render::mod::material_bind_group`. Engine-faithful: each
    /// `BitmapBinding`'s authored `BitmapAddressMode` / `BitmapFilterMode`
    /// gets honored via dedup'd sampler bindings.
    pub sampler_map: material_samplers::MaterialSamplerMap,
}

/// Build a complete WGSL source string for the given material + entry
/// point. The returned string is ready for `wgpu::Device::create_shader_module`.
///
/// `choices` is the (category_name → option_name) map resolved from the
/// rmsh's options against the rmdf's categories — the same mapping
/// Halo's offline shader compiler used to pick HLSL macro expansions.
/// Dispatching by name (not by hardcoded `options[N]` position) is
/// what lets this same path serve rmsh, rmtr, water, etc. — every
/// subclass has its own rmdf with its own category names.
pub fn assemble(
    rm: &ResolvedRenderMethod,
    cb: &ResolvedCbuffer,
    choices: &CategoryChoices,
    entry_point: HaloEntryPoint,
    tags_root: &std::path::Path,
) -> AssembledVariant {
    let bindings = material_bindings::MaterialBindings::from_resolved(rm, tags_root);
    let sampler_map = material_samplers::MaterialSamplerMap::from_resolved(rm, &bindings);
    assemble_with(rm, cb, choices, entry_point, bindings, sampler_map)
}

/// Same as [`assemble`] but takes the already-computed `bindings` +
/// `sampler_map` (so it needs no `tags_root` — the only consumer of it
/// was the bindings construction, which the caller now owns). `ensure`
/// computes these once to derive the [`VariantKey`] (cube mask + sampler
/// signature) and hands the same instances back here on a cache miss, so
/// the bitm-tag reads inside `MaterialBindings::from_resolved` happen
/// exactly once per build.
pub fn assemble_with(
    rm: &ResolvedRenderMethod,
    cb: &ResolvedCbuffer,
    choices: &CategoryChoices,
    entry_point: HaloEntryPoint,
    bindings: material_bindings::MaterialBindings,
    sampler_map: material_samplers::MaterialSamplerMap,
) -> AssembledVariant {
    let mut wgsl = String::new();

    // 1. Engine bindings (camera, default_lighting, per-model uniforms,
    //    bone matrices — groups 0, 1, 3). Static across all variants.
    wgsl.push_str(include_str!("../../../assets/halo_shaders/engine_bindings.wgsl"));
    wgsl.push('\n');

    // 1b. Spherical harmonics evaluation — `ravi_order_2`/`ravi_order_3`
    //     ports of spherical_harmonics_fx.hlsl + the
    //     `build_default_sh_array` adapter that maps our 5-vec4
    //     `default_lighting` cbuf into the 10-vec4 ravi packing.
    wgsl.push_str(include_str!("../../../assets/halo_shaders/spherical_harmonics_fx.wgsl"));
    wgsl.push('\n');

    // 1c. Atmospheric scattering — `compute_scattering` port from
    //     atmosphere_fx.hlsl with hardcoded riverworld haze_skydome
    //     params + g_exposure constant.
    //
    //     `BLEND_FOG_INSCATTER_SCALE` is per-variant: the engine sets it
    //     via `#define` per blend_mode in `blend_fx.hlsl`. Mirror that
    //     here by substituting before the WGSL is parsed. additive +
    //     (multiply/double_multiply, deferred) → 0.0; everything else → 1.0.
    let inscatter_scale = match choices.get_or("blend_mode", "opaque") {
        "additive" => "0.0",
        "multiply" | "double_multiply" => "0.0",
        _ => "1.0",
    };
    let atmosphere_fx = include_str!("../../../assets/halo_shaders/atmosphere_fx.wgsl")
        .replace("__BLEND_FOG_INSCATTER_SCALE__", inscatter_scale);
    wgsl.push_str(&atmosphere_fx);
    wgsl.push('\n');

    // Per-variant `blend_fx.hlsl` placeholder values for entry_static_sh /
    // entry_static_per_pixel. Computed once and applied via `apply_blend_fx_substitutions`
    // when the entry-shader fragment is appended below.
    let blend_fx = blend_fx_substitutions(choices.get_or("blend_mode", "opaque"));

    // 1d. Lightmap atlas decode + ravi packing — port of
    //     lightmap_sampling_fx.hlsl. Used by every per-pixel-lightmap
    //     entry point (terrain + rmsh static_sh + ...).
    wgsl.push_str(include_str!("../../../assets/halo_shaders/lightmap_sampling_fx.wgsl"));
    wgsl.push('\n');

    // 2. Per-variant texture+sampler bindings (group 2). One slot per
    //    Bitmap-typed parameter in the resolved rmop chain — exactly
    //    what dllcache binds at sampler register assignment time.
    //    Sampler dedup: each `BitmapBinding`'s authored
    //    `BitmapAddressMode` / `BitmapFilterMode` is honored via a
    //    per-material deduped sampler set (typically 2-4 unique keys).
    //    `bindings` + `sampler_map` arrive precomputed (see `assemble_with`).
    wgsl.push_str(&bindings.emit_wgsl_per_binding(&sampler_map));
    wgsl.push('\n');

    // 3. Per-variant user-parameter cbuffer struct generated from the
    //    rmt2's routing — every slot is a `vec4<f32>` matching the
    //    pixl shader's actual cb13 layout. Names mirror Halo's HLSL
    //    PARAM declarations (xforms get a `_xform` suffix).
    //
    //    For terrain (rmtr): pad with the 4-layer slots the terrain
    //    WGSL entry references unconditionally. Some riverworld
    //    terrain rmt2s declare only 3 layers (m_0..m_2); the WGSL is
    //    a fixed 4-layer body, so missing slots get identity defaults.
    // Pad now happens at load time in `resolve_render_method` so the
    // cbuffer bytes that `MaterialData.cbuffer` carries already include
    // every padded slot. Use the cbuffer here as-is — its layout matches
    // what's uploaded at runtime, so assemble's WGSL struct stays in sync.
    let cbuffer_layout = cbuffer::CbufferLayout::from_resolved(cb);
    wgsl.push_str(&cbuffer::emit_wgsl(
        &cbuffer_layout,
        "MaterialParameters",
        2,
        bindings.per_binding_cbuffer_slot(&sampler_map),
    ));
    wgsl.push('\n');

    // 4. Shared utilities (transform_texcoord, sample_bumpmap, DETAIL_MULTIPLIER)
    wgsl.push_str(include_str!("../../../assets/halo_shaders/utilities.wgsl"));
    wgsl.push('\n');

    // 5. Per-rm**-subclass dispatch. dllcache's `render_method_submit`
    //    chain is class-blind; the difference between rmsh, rmtr, etc.
    //    lives ENTIRELY in the rmt2 sampler/parameter list and the
    //    shader bytecode. We mirror the shader-side dispatch via the
    //    source group_tag — terrain emits the 4-layer terrain_fx body,
    //    rmsh emits the per-category rmsh fragments + static_sh entry.
    //    See `reference_rmtr_runtime_distinction.md`.
    // Shared simple-lights helper — port of `simple_lights_fx.hlsl`
    // (modified with smooth-quadratic distance falloff, see comments
    // in the WGSL file). Bound at `@group(0) @binding(8)`. Adds the
    // `calc_simple_lights_analytical` and `_diffuse_translucent`
    // helpers that material PS code calls below for the dynamic
    // point/spot light contribution.
    const SIMPLE_LIGHTS_FX: &str = include_str!(
        "../../../assets/halo_shaders/simple_lights_fx.wgsl"
    );

    let group_fourcc = rm.group_tag.to_be_bytes();
    match &group_fourcc {
        b"rmtr" => {
            // Terrain: single self-contained entry that does
            // `sample_blend_normalized` + `ACCUMULATE_MATERIAL_*` × 4
            // + ravi_order_3. Doesn't use the rmsh per-category
            // fragments — its umbrella is hand-inlined.
            //
            // Substitutions mirror HLSL's compile-time gating:
            //   __MATERIAL_N_ACTIVE__       — `#if ACTIVE_MATERIAL(material_N_type)`
            //   __DETAIL_BUMP_ENABLED_N__   — `#if DETAIL_BUMP_ENABLED` (terrain_fx.hlsl:116):
            //                                  active_material_count < 4 globally
            //                                  AND no `(four_material_shaders_disable_detail_bump)`
            //                                  suffix on this layer's variant.
            // Pick the entry-point variant: Albedo writes the
            // G-buffer, StaticPerPixel/StaticSh writes the FINAL lit
            // color. Both share the 4-layer blend / per-layer bump
            // accumulation logic; differ in PS output (and, for the
            // SL variant, lighting/atmospheric/exposure terms).
            let mut terrain = String::from(match entry_point {
                HaloEntryPoint::Albedo => include_str!(
                    "../../../assets/halo_shaders/entry_albedo_terrain.wgsl"
                ),
                // StaticSh and StaticPerPixel share one WGSL — engine
                // `terrain_fx.hlsl:1196/1284` both `static_per_pixel_ps`
                // and `static_sh_ps` delegate to the same
                // `static_lighting_shared_ps`. Our WGSL branches at
                // runtime on `lightmap_texcoord != (0,0)`: per-pixel
                // samples the lightprobe atlas, SH path uses the
                // single-probe cbuffer. So one file covers both.
                HaloEntryPoint::StaticSh | HaloEntryPoint::StaticPerPixel => include_str!(
                    "../../../assets/halo_shaders/entry_terrain_static_sh.wgsl"
                ),
                // Per-vertex SH variant: secondary VB carries 4
                // SH coefs per channel; PS uses interpolated stream
                // instead of atlas/cbuffer. Engine `static_per_vertex_vs`
                // + `static_lighting_shared_ps` path
                // (terrain_fx.hlsl:1228-1284, 845-853).
                HaloEntryPoint::StaticShPerVertex => include_str!(
                    "../../../assets/halo_shaders/entry_terrain_static_sh_per_vertex.wgsl"
                ),
                // PRT-Ambient variant: per-vertex AO scalar; engine
                // VS (terrain_new_fx.hlsl:1364) computes prt_ravi_diff
                // but BUILD_ENTRY_POINT_DATA drops it before
                // `static_lighting_shared_ps_quadratic` (entry_point_data
                // for prt_ambient is just `unused` — line 445-449).
                // So PS output equals static_sh; only the VS input
                // layout differs (PrtAmbientVertex secondary VB).
                HaloEntryPoint::StaticPrtAmbient => include_str!(
                    "../../../assets/halo_shaders/entry_terrain_static_prt_ambient.wgsl"
                ),
                other => panic!(
                    "rmtr (terrain) entry point not ported: {other:?}. \
                     Add an HLSL port (terrain_fx.hlsl:885+ / terrain_new_fx.hlsl) \
                     and a match arm here instead of silently falling through.",
                ),
            });

            // Read each material_N choice, decide active + detail-bump
            // gate + SPECULAR / SELF_ILLUM gating per HLSL macros at
            // terrain_new_fx.hlsl:93-119. material_N_type options:
            //   off
            //   diffuse_only
            //   diffuse_plus_specular
            //   diffuse_only_plus_self_illum
            //   diffuse_plus_specular_plus_self_illum
            let mut layer_opts: [&str; 4] = [""; 4];
            let mut active_count = 0;
            let mut specular_count = 0;
            let mut self_illum_count = 0;
            for n in 0..4 {
                layer_opts[n] = choices.get_or(&["material_0","material_1","material_2","material_3"][n], "off");
                if layer_opts[n] != "off" { active_count += 1; }
                if layer_opts[n].contains("specular") { specular_count += 1; }
                // Engine bug mirror (terrain_new_fx.hlsl:753-765):
                // self_illum loop has no material_3_type branch. Layer
                // 3 never self-illuminates even when authored.
                if n < 3 && layer_opts[n].contains("self_illum") { self_illum_count += 1; }
            }
            let global_detail_bump = active_count < 4;

            for n in 0..4 {
                let opt = layer_opts[n];
                let active = if opt == "off" { "0.0" } else { "1.0" };
                terrain = terrain.replace(&format!("__MATERIAL_{n}_ACTIVE__"), active);

                // SPECULAR_MATERIAL — 1.0 when this layer's type carries
                // a `specular` component (diffuse_plus_specular*).
                let spec = if opt.contains("specular") { "1.0" } else { "0.0" };
                terrain = terrain.replace(&format!("__SPECULAR_MATERIAL_{n}__"), spec);

                // SELF_ILLUM_MATERIAL — engine HLSL only checks layers
                // 0/1/2 so layer 3 always gets 0.0 (mirror engine bug).
                let illum = if n < 3 && opt.contains("self_illum") { "1.0" } else { "0.0" };
                terrain = terrain.replace(&format!("__SELF_ILLUM_MATERIAL_{n}__"), illum);

                // Detail bump enabled when:
                //   (a) layer is active AND
                //   (b) global_detail_bump (active_count < 4) AND
                //   (c) variant name doesn't carry the disable suffix
                let layer_disables = opt.contains("disable_detail_bump");
                let dbe = if opt != "off" && global_detail_bump && !layer_disables {
                    "1.0"
                } else {
                    "0.0"
                };
                terrain = terrain.replace(&format!("__DETAIL_BUMP_ENABLED_{n}__"), dbe);
            }

            // Count tokens (used to gate the entire specular branch
            // and self_illum accumulation at compile-time, mirroring
            // HLSL `#if SPECULAR_MATERIAL_COUNT > 0`).
            terrain = terrain.replace(
                "__SPECULAR_MATERIAL_COUNT__",
                &format!("{specular_count}.0"),
            );
            terrain = terrain.replace(
                "__SELF_ILLUM_MATERIAL_COUNT__",
                &format!("{self_illum_count}.0"),
            );
            wgsl.push_str(SIMPLE_LIGHTS_FX);
            wgsl.push('\n');
            // Engine `terrain_new_fx.hlsl:993-998` —
            //   envmap_light = CALC_ENVMAP(envmap_type)(view_dir, bump_normal,
            //                                            view_reflect_dir, ...)
            // Reuse the rmsh envmap WGSL fragment based on the rmtr's
            // environment_map choice. Default `none` → calc_environment_map
            // returns 0 (matches the previous hardcoded zero).
            let terrain_envmap = choices.get_or("environment_mapping", "none");
            wgsl.push_str(pick_env_mapping(terrain_envmap));
            wgsl.push('\n');
            wgsl.push_str(&terrain);
        }
        b"rmfl" => {
            // Foliage: alpha-tested leaf cards, SH3 diffuse only,
            // no specular. Per-vertex SH eval + interpolated to PS.
            // Per `reference_foliage_fx_blueprint.md`.
            //
            // Two entry-point variants per material — Albedo writes
            // the G-buffer, StaticSh writes the FINAL lit color.
            //
            // Prepend the rmt2-chosen `calc_albedo_*_ps` variant so
            // `entry_albedo_foliage.wgsl::calc_albedo(...)` resolves.
            // Defaults to `default` (= base × detail × albedo_color)
            // per `foliage_material_fx.hlsl`. Other valid options
            // (two_detail / detail_blend / ...) inherit the rmsh
            // implementations from `pick_albedo`.
            wgsl.push_str(pick_albedo(choices.get_or("albedo", "default")));
            wgsl.push('\n');
            wgsl.push_str(SIMPLE_LIGHTS_FX);
            wgsl.push('\n');
            let foliage_entry = match entry_point {
                HaloEntryPoint::Albedo => include_str!(
                    "../../../assets/halo_shaders/entry_albedo_foliage.wgsl"
                ),
                HaloEntryPoint::StaticSh => include_str!(
                    "../../../assets/halo_shaders/entry_foliage_static_sh.wgsl"
                ),
                // Per-pixel: engine warning path (foliage_fx.hlsl:336)
                // — same SH3 lighting as static_sh but RED-tinted as a
                // content-author warning ("no one should be using the
                // foliage shader and per pixel lighting").
                HaloEntryPoint::StaticPerPixel => include_str!(
                    "../../../assets/halo_shaders/entry_foliage_static_per_pixel.wgsl"
                ),
                // Per-vertex SH: secondary VB carries 4 SH coefs per
                // channel + dominant intensity. Engine `static_per_vertex_vs`
                // (foliage_fx.hlsl:347) evaluates ravi_order_2 with
                // dominant-light extraction at the vertex normal.
                HaloEntryPoint::StaticShPerVertex => include_str!(
                    "../../../assets/halo_shaders/entry_foliage_static_sh_per_vertex.wgsl"
                ),
                // PRT-Ambient: cbuffer SH3 lighting modulated by
                // per-vertex AO scalar (`prt_c0_c3` BLENDWEIGHT1).
                // Engine `static_prt_ambient_vs` (foliage_fx.hlsl:432-479).
                HaloEntryPoint::StaticPrtAmbient => include_str!(
                    "../../../assets/halo_shaders/entry_foliage_static_prt_ambient.wgsl"
                ),
                other => panic!(
                    "rmfl (foliage) entry point not ported: {other:?}. \
                     Add the WGSL port and a match arm here instead of \
                     silently routing.",
                ),
            };
            wgsl.push_str(foliage_entry);
        }
        b"rmsh" | b"rmcs" | b"rmhg" => {
            // rmsh / rmcs / rmhg share the rmsh rmdf body. rmhg is a
            // thin alias per `halogram_fx.hlsl:1-12`:
            //   "halogram is basically the same as .shader, except it
            //    has several hardcoded categories: bump_mapping NONE,
            //    alpha_test NONE, specular_mask NONE, material_model
            //    NONE, environment_mapping NONE, parallax NONE."
            // It then `#define`s `calc_bumpmap_ps calc_bumpmap_off_ps`
            // etc. and includes the SAME `entry_points.fx` as rmsh.
            // We mirror by overriding those choices for `b"rmhg"`.
            let is_halogram = matches!(&group_fourcc, b"rmhg");
            let bump = if is_halogram { "off" } else { choices.get_or("bump_mapping", "off") };
            let alpha_test = if is_halogram { "none" } else { choices.get_or("alpha_test", "none") };
            let spec_mask = if is_halogram { "no_specular_mask" } else { choices.get_or("specular_mask", "no_specular_mask") };
            let mat_model = if is_halogram { "none" } else { choices.get_or("material_model", "diffuse_only") };
            let env_map_choice = if is_halogram { "none" } else { choices.get_or("environment_mapping", "none") };
            // When the rmop's flat_environment_map resolves to a cube
            // bitmap, route the `from_flat_texture` option to the
            // `from_flat_texture_as_cubemap` HLSL function — the
            // binding's actual dimension is source-of-truth for which
            // sample path to use (engine PARAM_SAMPLER_2D vs _CUBE
            // declaration switches per envmap_type, but the bitm tag
            // ultimately drives the runtime register binding type).
            let flat_env_is_cube = bindings
                .textures
                .iter()
                .any(|t| t.name == "flat_environment_map" && t.is_cube);
            let env_map = if env_map_choice == "from_flat_texture" && flat_env_is_cube {
                "from_flat_texture_as_cubemap"
            } else {
                env_map_choice
            };
            let parallax = if is_halogram { "off" } else { choices.get_or("parallax", "off") };

            wgsl.push_str(pick_albedo(choices.get_or("albedo", "default")));
            wgsl.push('\n');
            wgsl.push_str(pick_bump(bump));
            wgsl.push('\n');
            // `warp` and `parallax` both define `calc_parallax` (the working-
            // texcoord producer); they're mutually exclusive. The engine
            // routes warp into the parallax slot — so emit the warp fragment
            // when warp != none, else the parallax fragment.
            let warp = choices.get_or("warp", "none");
            if warp != "none" {
                wgsl.push_str(pick_warp(warp));
            } else {
                wgsl.push_str(pick_parallax(parallax));
            }
            wgsl.push('\n');
            wgsl.push_str(pick_alpha_test(alpha_test));
            wgsl.push('\n');
            wgsl.push_str(pick_specular_mask(spec_mask));
            wgsl.push('\n');
            wgsl.push_str(pick_env_mapping(env_map));
            wgsl.push('\n');
            wgsl.push_str(pick_material_model(mat_model));
            wgsl.push('\n');
            wgsl.push_str(pick_self_illum(choices.get_or("self_illumination", "none")));
            wgsl.push('\n');
            // overlay + edge_fade compose onto the final lit color (engine
            // APPLY_OVERLAYS; entry point calls calc_overlay_ps then
            // calc_edge_fade_ps). Always included so the entry's calls
            // resolve; `none` variants are no-ops.
            wgsl.push_str(pick_overlay(choices.get_or("overlay", "none")));
            wgsl.push('\n');
            wgsl.push_str(pick_edge_fade(choices.get_or("edge_fade", "none")));
            wgsl.push('\n');
            // soft_fade (common_fx.hlsl apply_soft_fade) — fades the
            // material by a fresnel and/or soft-z depth term, applied to
            // albedo right after it's computed in the entry point. Always
            // emits an `apply_soft_fade` definition (the entry calls it
            // unconditionally); `off`/absent → no-op. `use_soft_fresnel`/
            // `use_soft_z` are per-material PARAM(bool)s resolved here.
            let alpha_blend = choices.get_or("blend_mode", "opaque") == "alpha_blend";
            wgsl.push_str(&pick_soft_fade(choices.get_or("soft_fade", "off"), rm, alpha_blend));
            wgsl.push('\n');

            // Fail-loud on a silently-dropped render-method category.
            // Project policy: no silent shader fallbacks. This path
            // consumes the categories below (rmhg additionally FORCES
            // bump/alpha_test/specular_mask/material_model/environment_
            // mapping/parallax off per halogram_fx.hlsl, so they count as
            // handled). Any OTHER category set to a non-default option
            // (rmdf option index != 0) is not rendered — it would produce
            // a wrong shader without warning. The guardian hologram
            // exposed two: `overlay = multiply_and_additive_detail` (the
            // cell-like inner layer) and `edge_fade = simple` (silhouette
            // fade), both silently lost. Panic so they get implemented or
            // explicitly overridden, not quietly mis-rendered.
            const RMSH_HANDLED: &[&str] = &[
                "albedo", "bump_mapping", "alpha_test", "specular_mask",
                "material_model", "environment_mapping", "parallax",
                "self_illumination", "blend_mode", "overlay", "edge_fade",
                "warp", "soft_fade",
                // `distortion` is ADDITIVE in the engine: the shader's
                // primary entry point (static_sh_ps etc.) renders its
                // colour identically whether or not distortion is on; the
                // category only enables a SEPARATE displacement-accumulation
                // entry (displacement_hlsl.hlsl) drawn into the warp buffer
                // that the full-screen distortion pass applies. So letting
                // the colour path render the surface normally is faithful —
                // the only deferred piece is the heat-haze ripple (the
                // existing particle-distortion accumulate→warp infra can be
                // reused for it; tracked as a follow-up). These s3d distortion
                // shaders are alpha_blend/additive water/energy surfaces; they
                // render correctly minus the ripple.
                "distortion",
            ];
            // Behavioral categories that change WHEN/WHETHER a material
            // draws, not its shaded output — intentionally ignored for
            // protomorph's static world view (no first-person arm, no
            // per-vertex CPU attr animation path). Dropping these is
            // correct, not a silent mis-render, so they don't panic.
            // `misc` = first_person_{never,sometimes,always}; the FP-only
            // behavior is irrelevant when we render the world view.
            const RMSH_IGNORED: &[&str] = &["misc", "misc_attr_animation"];
            for cat in choices.non_default() {
                let c = cat.as_str();
                if !RMSH_HANDLED.contains(&c) && !RMSH_IGNORED.contains(&c) {
                    panic!(
                        "render_method ({}) category '{cat}' = '{}' is a non-default option that \
                         the rmsh/rmcs/rmhg shader path does NOT render — it would be silently \
                         dropped, producing a wrong shader. Implement the category or override it. \
                         (Found auditing the guardian halogram: overlay=multiply_and_additive_detail \
                         + edge_fade=simple.)",
                        std::str::from_utf8(&group_fourcc).unwrap_or("?"),
                        choices.get_or(cat, "?"),
                    );
                }
            }

            // Entry-point umbrella (vs_main + fs_main). Apply per-variant
            // blend_fx.hlsl substitutions (multiplicative branch enable +
            // factor, alpha-channel output, premultiply-alpha switch) so the
            // PS body matches what HLSL's offline preprocessor would emit.
            let entry = match entry_point {
                HaloEntryPoint::Albedo => include_str!(
                    "../../../assets/halo_shaders/entry_albedo.wgsl"
                ),
                HaloEntryPoint::StaticSh => include_str!(
                    "../../../assets/halo_shaders/entry_static_sh.wgsl"
                ),
                HaloEntryPoint::StaticPerPixel => include_str!(
                    "../../../assets/halo_shaders/entry_static_per_pixel.wgsl"
                ),
                HaloEntryPoint::StaticShPerVertex => include_str!(
                    "../../../assets/halo_shaders/entry_static_sh_per_vertex.wgsl"
                ),
                HaloEntryPoint::StaticPrtAmbient => include_str!(
                    "../../../assets/halo_shaders/entry_static_prt_ambient.wgsl"
                ),
                // Vertex-color lighting (sky meshes) — diffuse term is
                // the per-vertex baked color from slot 1, not the SH probe.
                HaloEntryPoint::StaticVertexColor => include_str!(
                    "../../../assets/halo_shaders/entry_static_per_vertex_color.wgsl"
                ),
                HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo | HaloEntryPoint::DecalObject => unreachable!(
                    "decal entry points are for rmd, not rmsh/rmcs/rmhg"
                ),
            };
            // simple_lights_fx prepended for ALL entry-points — the
            // material_model wgsl files (two_lobe_phong, cook_torrance, …)
            // reference `calc_simple_lights_analytical` in their bodies
            // even though the Albedo entry-point's PS doesn't call them.
            // wgpu/naga rejects shaders with unresolved identifiers
            // anywhere in the module, so we always include the helper.
            wgsl.push_str(SIMPLE_LIGHTS_FX);
            wgsl.push('\n');
            // BLEND_FRESNEL — engine defines it IFF `material_type == glass`
            // (material_models_fx.hlsl:178-180):
            //   out_rgb = diffuse·albedo·albedo.w + self_illum + env + specular
            //   alpha   = saturate(specular.w/*fresnel*/ + albedo.w)
            // Kept ON (faithful). The earlier "fresnel makes glass opaque" was
            // a back-facing normal: guardian_glass_platform is a two-sided
            // transparent surface whose authored normal points away from a
            // top viewer, so N·V<0 → fresnel pinned to ~1.15 → opaque. The
            // fix is the engine's two-sided flip (decorators_hlsl.hlsl:265,
            // `world_normal * sign(dot(world_normal, frag_to_cam))`) applied in
            // the transparent branch of the static entry shaders — NOT
            // disabling fresnel. See 2026-06-18 glass investigation.
            let fresnel_enabled = if mat_model == "glass" { "1.0" } else { "0.0" };
            let entry_sub = apply_blend_fx_substitutions(entry, &blend_fx)
                .replace("__BLEND_FRESNEL_ENABLED__", fresnel_enabled);
            wgsl.push_str(&entry_sub);
        }
        b"rmd " => {
            // Decals — port of `decal_fx.hlsl` per the rmd rmdf's 6
            // categories (albedo / blend_mode / render_pass /
            // specular / bump_mapping / tinting). Each option maps to
            // a small WGSL fragment that defines the engine helper
            // function for its branch; the entry shader concatenates
            // them and calls them from `default_vs` / `default_ps`.
            //
            // Specular/render_pass don't produce helper fragments —
            // specular_modulate fires a pipeline-state change in
            // `c_decal::render` (handled at draw time), and
            // render_pass routes the PS output target (handled by
            // pipeline_cache when wiring the targets).
            wgsl.push_str(pick_decal_albedo(choices.get_or("albedo", "diffuse_only")));
            wgsl.push('\n');
            wgsl.push_str(pick_decal_blend_mode(choices.get_or("blend_mode", "alpha_blend")));
            wgsl.push('\n');
            wgsl.push_str(pick_decal_bump_mapping(choices.get_or("bump_mapping", "leave")));
            wgsl.push('\n');
            wgsl.push_str(pick_decal_tinting(choices.get_or("tinting", "none")));
            wgsl.push('\n');
            // Decal entry VS choice — `DecalObject` (object-attached
            // rmd via `c_object_renderer::render_albedo_decals`) needs
            // the skinned ModelVertex VS body; `Decal` / `DecalAlbedo`
            // (scene decals via `c_decal_system::render_all`) use the
            // rasterizer_vertex_world VS body. Both variants share the
            // identical `fs_main` PS body so the PS-side substitutions
            // (sample_diffuse / fade_out / sample_bump / etc.) apply
            // uniformly to both.
            if matches!(entry_point, HaloEntryPoint::DecalObject) {
                wgsl.push_str(include_str!(
                    "../../../assets/halo_shaders/entry_decal_object.wgsl"
                ));
            } else {
                wgsl.push_str(include_str!(
                    "../../../assets/halo_shaders/entry_decal.wgsl"
                ));
            }
        }
        other => {
            // Render_method subclass not yet ported. Per
            // `feedback_wgsl_must_mirror_hlsl.md`: silent fallbacks
            // hide what needs porting. Panic with a pointer to the
            // HLSL file so the next step is obvious.
            let group = std::str::from_utf8(other).unwrap_or("????");
            let blueprint = match other {
                b"rmd " | b"rmd\0" => "decal_fx.hlsl",
                b"rmhg" => "halogram_fx.hlsl",
                b"rmsk" => "skin_fx.hlsl",
                b"rmct" => "contrail_fx.hlsl",
                b"rmp " | b"rmp\0" => "particle_fx.hlsl",
                b"rmb " | b"rmb\0" => "beam_fx.hlsl",
                b"rmco" => "cortana_fx.hlsl",
                b"rmlv" => "light_volume_fx.hlsl",
                b"rmw " | b"rmw\0" => "water_fx.hlsl",
                _ => "(unknown — search /Users/camden/Halo/halo3_mcc_hlsl_extracted/)",
            };
            panic!(
                "[protomorph] render_method subclass '{group}' has no WGSL port. \
                 Port the HLSL stack starting from `{blueprint}` to a new \
                 `assets/halo_shaders/entry_{group}_static_sh.wgsl`, add a \
                 `b\"{group}\" => {{ ... }}` arm in `render_methods::assemble()`, \
                 and wire any new bind-group entries in `material_bindings.rs`. \
                 No silent fallbacks — see `feedback_wgsl_must_mirror_hlsl.md`."
            );
        }
    }

    // Per-variant placeholder substitution: bool/string values pulled
    // from the resolved rm** chain and baked as WGSL `const`s. Applied
    // once over the full assembled string so any included fragment can
    // reference the placeholder without per-branch wiring.
    //
    // `__ORDER3_AREA_SPECULAR__` — engine `material_models_fx.hlsl:70`.
    // Material models (`single_lobe_phong`, `cook_torrance`,
    // `two_lobe_phong`) branch between order-3 and order-2 SH specular
    // evaluation on this bool. Resolved per-rmt2 from the rmop chain.
    let order3 = resolve_order3_area_specular(rm);
    let wgsl = wgsl.replace("__ORDER3_AREA_SPECULAR__", order3);

    // `__NO_DYNAMIC_LIGHTS__` — engine `material_models_fx.hlsl:71`
    // `PARAM(bool, no_dynamic_lights)`. See `resolve_no_dynamic_lights`.
    let ndl = resolve_no_dynamic_lights(rm);
    let wgsl = wgsl.replace("__NO_DYNAMIC_LIGHTS__", ndl);

    // `__SL_USE_GBUFFER__` — engine `entry_points_fx.hlsl:26` —
    // `#ifdef maybe_calc_albedo` gate. Opaque shaders take the
    // `else` branch (load from G-buffer); transparent shaders go
    // through `actually_calc_albedo == true` (recompute from textures)
    // because they don't have an albedo pass writing the G-buffer.
    //
    // **Engine-faithful (opaque)**: substitute a runtime expression
    // reading `Kernel5PS.entries[5].z` (byte 88, the `actually_calc_albedo`
    // bool written via `set_using_albedo_sampler`). When the frame loop
    // sets the sampler TRUE before SL passes, `actually_calc_albedo`
    // is 0 → `SL_USE_GBUFFER == true` → sample G-buffer (engine match).
    // **Transparent**: substitute the literal `false` because transparent
    // passes have no G-buffer to sample (the runtime flip to `false`
    // before transparent draws is a P5-followup; pinning here keeps
    // the current bandaid behavior for transparents).
    // Classification mirrors `pipeline_cache::is_transparent_subclass`:
    // rmd/rmhg always transparent; rmsh/rmcs transparent iff
    // blend_mode != opaque.
    let blend_mode = choices.get_or("blend_mode", "opaque");
    let is_always_transparent = matches!(&group_fourcc, b"rmd " | b"rmhg");
    let is_rmsh_transparent =
        matches!(&group_fourcc, b"rmsh" | b"rmcs") && blend_mode != "opaque";
    let sl_use_gbuffer = if is_always_transparent || is_rmsh_transparent {
        "false"
    } else {
        // Runtime read of `actually_calc_albedo` cbuffer bit. The bool
        // is the LOGICAL NEGATION of `using_albedo_sampler`, so
        // `entries[5].z == 0u` means "use G-buffer".
        "(misc_bool_ps.entries[5].z == 0u)"
    };
    let wgsl = wgsl.replace("__SL_USE_GBUFFER__", sl_use_gbuffer);

    // Decal-only substitutions for `tint_and_modulate`'s post_lighting
    // exposure block (engine `decal_fx.hlsl:364-380`). The block runs
    // regardless of tinting branch but is gated on render_pass and
    // skipped for multiply / double_multiply blend modes.
    let decal_post_lighting =
        if choices.get_or("render_pass", "pre_lighting") == "post_lighting" {
            "true"
        } else {
            "false"
        };
    let decal_multiplicative_blend =
        if matches!(blend_mode, "multiply" | "double_multiply") {
            "true"
        } else {
            "false"
        };
    let wgsl = wgsl.replace("__DECAL_POST_LIGHTING__", decal_post_lighting);
    let wgsl = wgsl.replace("__DECAL_MULTIPLICATIVE_BLEND__", decal_multiplicative_blend);
    // Diagnostic toggle for the dark-hard-rectangle decal symptom
    // (riverworld/snowbound + every other map, 2026-05-17). Modes:
    //   0 = normal output (default)
    //   1 = bitmap alpha as grayscale
    //   2 = bitmap RGB only (alpha=1, blend math ignored)
    //   3 = solid magenta (geometry-only visibility check)
    // Set via `PROTOMORPH_DECAL_VIZ=N` at process start.
    let decal_viz_mode = std::env::var("PROTOMORPH_DECAL_VIZ")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|m| *m <= 3)
        .map(|m| format!("{m}u"))
        .unwrap_or_else(|| "0u".to_string());
    let wgsl = wgsl.replace("__DECAL_VIZ_MODE__", &decal_viz_mode);
    // DIAGNOSTIC: `PROTOMORPH_ENV_GAIN=N` scales the dynamic env-reflection
    // term (environment_mapping_fx_dynamic.wgsl), default 1.0 = off. Used
    // to A/B the env-reflection magnitude on glass/metal floors against MCC
    // and localize the pre-existing reflection/exposure deficit.
    let env_gain = std::env::var("PROTOMORPH_ENV_GAIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(1.0);
    let wgsl = wgsl.replace("__ENV_REFLECTION_GAIN__", &format!("{env_gain:?}"));
    // Empty-lightmap-atlas fallback: substitute the cluster default sky-probe
    // SH when the per-pixel atlas DC is ~0. TRANSPARENT BSP surfaces (glass,
    // etc.) have NO baked per-pixel lightmap chart in the loose `.bitmap` tag
    // (atlas DC ≈ 0) → zero SH → zero diffuse → the surface shows only its env
    // reflection (the guardian glass's dark-green-instead-of-lit-gray bug,
    // measured via PROTOMORPH_GLASS_VIZ: albedo present, diffuse term black,
    // ambient ~0). MCC renders on tool.exe's compiled cache atlas where those
    // charts are baked; we approximate that with the cluster ambient probe.
    //
    // Gated ON for transparent materials (blend_mode != opaque): their empty
    // chart is a KNOWN data gap, not a real baked shadow, so the probe
    // substitution is safe. OPAQUE surfaces keep it OFF — their DC≈0 is a
    // legitimately-baked shadow that must not be lifted. `PROTOMORPH_ATLAS_FALLBACK=1`
    // forces it on everywhere (diagnostic).
    // NOTE: the cluster-probe substitution is GREEN for guardian (its frame-
    // default sky probe is teal), so it can't stand in for the glass's LOCAL
    // gray ambient — it greens the reflection (env_tint is already green) and
    // the diffuse. Diagnostic-only now (PROTOMORPH_ATLAS_FALLBACK=1); the real
    // fix routes transparent glass to the cluster's baked SH (see below).
    let atlas_fallback = std::env::var("PROTOMORPH_ATLAS_FALLBACK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let wgsl = wgsl.replace("__LIGHTMAP_EMPTY_ATLAS_FALLBACK__", if atlas_fallback { "true" } else { "false" });
    let wgsl = wgsl.replace("__ATLAS_FALLBACK_TRANSPARENT__", "false");
    // DIAGNOSTIC: `PROTOMORPH_GLASS_VIZ=N` isolates one glass lighting term
    // (entry_static_per_pixel BLEND_FRESNEL branch) so we can see which term
    // is wrong vs MCC. 0 = off (normal).
    let glass_viz = std::env::var("PROTOMORPH_GLASS_VIZ")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|m| *m <= 9)
        .unwrap_or(0);
    let wgsl = wgsl.replace("__GLASS_VIZ_MODE__", &format!("{glass_viz}u"));
    let areaspec_dc = std::env::var("PROTOMORPH_GLASS_AREASPEC_DC").is_ok();
    let wgsl = wgsl.replace("__GLASS_AREASPEC_DC__", if areaspec_dc { "1u" } else { "0u" });
    // entry_decal::fs_main picks the RT1 payload — bump-packed for
    // pre_lighting (G-buffer normal), duplicated color for
    // post_lighting (hdr_dark accumulator). Mirrors engine
    // `convert_to_decal_target` RENDER_TARGET_TYPE switch
    // (decal_fx.hlsl:425-432).
    let decal_pre_lighting = if decal_post_lighting == "true" { "false" } else { "true" };
    let wgsl = wgsl.replace("__DECAL_RENDER_PASS_PRE_LIGHTING__", decal_pre_lighting);

    // Decal pre_multiplied_alpha branch (engine `decal_fx.hlsl:405-420`) skips
    // the `color.xyz *= color.w` premultiply when the albedo is already a
    // pre-multiplied vector_alpha variant. The HLSL guards this via the
    // category-option-defined preprocessor; we mirror with a build-time bool.
    let decal_albedo = choices.get_or("albedo", "diffuse_only");
    let decal_albedo_is_vector_alpha =
        if matches!(decal_albedo, "vector_alpha" | "vector_alpha_drop_shadow") {
            "true"
        } else {
            "false"
        };
    let wgsl = wgsl.replace("__DECAL_ALBEDO_IS_VECTOR_ALPHA__", decal_albedo_is_vector_alpha);

    // Per-binding sampler substitution. Fragments use the readable
    // `<tex>_sampler` syntax (`textureSample(base_map, base_map_sampler, ...)`)
    // and the deduped emit declared bindings as `s_dedupe_K`. Substitute
    // each `<tex>_sampler` reference with its corresponding dedupe name
    // based on the per-material sampler map. Done LAST so substitution
    // sees the final WGSL text.
    //
    // Single-pass implementation: build a placeholder → dedupe map once,
    // then walk the WGSL text once looking for `_sampler` suffixes and
    // checking the preceding identifier against the map. O(text length)
    // total instead of O(text × textures) — material shaders are ~100 KB
    // and a material can have 22 baseline texture bindings.
    let mut sampler_substitutions: std::collections::HashMap<String, String> =
        std::collections::HashMap::with_capacity(bindings.textures.len());
    for tb in &bindings.textures {
        let placeholder = format!("{}_sampler", tb.name);
        let Some(dedupe) = sampler_map.lookup_sampler_name(&tb.name) else {
            continue;
        };
        if placeholder == dedupe {
            continue;
        }
        sampler_substitutions.insert(placeholder, dedupe);
    }
    let wgsl = if sampler_substitutions.is_empty() {
        wgsl
    } else {
        substitute_sampler_refs(&wgsl, &sampler_substitutions)
    };

    AssembledVariant {
        wgsl,
        cbuffer: cbuffer_layout,
        bindings,
        sampler_map,
    }
}

/// Per-blend-mode placeholder values, mirroring `blend_fx.hlsl`'s
/// `#define` table. Each entry shader (entry_static_sh / entry_static_per_pixel)
/// uses these as compile-time constants — substitution happens during WGSL
/// assembly and is recorded in the variant cache key (since `choices`
/// contains blend_mode), so each variant gets a fresh shader with the
/// right values baked in. See `feedback_blend_fx_define_per_variant.md`.
struct BlendFxSubstitutions {
    /// `1.0` if blend_mode is `multiply` or `double_multiply` (use the
    /// HLSL `#ifdef BLEND_MULTIPLICATIVE` branch); `0.0` otherwise.
    multiplicative_enabled: &'static str,
    /// HLSL `#define BLEND_MULTIPLICATIVE`: `1.0` for multiply, `2.0` for
    /// double_multiply. Default `1.0` (no-op when ENABLED is 0).
    multiplicative_factor: &'static str,
    /// HLSL `ALPHA_CHANNEL_OUTPUT`. The WGSL substitution is the WGSL
    /// expression that evaluates to the right alpha at fs_main return.
    alpha_channel_output: &'static str,
    /// `albedo.a` only for `pre_multiplied_alpha` (engine's
    /// `convert_to_render_target_premultiplied_alpha` does
    /// `color.xyz *= color.w` before the standard render-target convert).
    /// `1.0` for every other mode (no-op multiply).
    alpha_premultiply: &'static str,
}

/// Single-pass per-binding sampler substitution. Scans `text` once for
/// `_sampler` suffixes at identifier boundaries, walks back to the
/// identifier start, and looks up the full `<tex>_sampler` in `map` for
/// replacement. O(text length) total regardless of map size — replaces
/// what would otherwise be an O(text × textures) inner loop. WGSL source
/// is ASCII so byte indexing is safe.
fn substitute_sampler_refs(
    text: &str,
    map: &std::collections::HashMap<String, String>,
) -> String {
    fn is_ident_char(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }
    const SAMPLER_SUFFIX: &[u8] = b"_sampler";
    let bytes = text.as_bytes();
    let suffix_len = SAMPLER_SUFFIX.len();
    let mut result = String::with_capacity(text.len());
    let mut last_emit = 0;
    let mut i = 0;
    while i + suffix_len <= bytes.len() {
        if &bytes[i..i + suffix_len] != SAMPLER_SUFFIX {
            i += 1;
            continue;
        }
        // Right boundary: char after `_sampler` must NOT be an ident char.
        let right_ok =
            i + suffix_len == bytes.len() || !is_ident_char(bytes[i + suffix_len]);
        if !right_ok {
            i += 1;
            continue;
        }
        // Walk back to find the start of the identifier preceding `_sampler`.
        let mut ident_start = i;
        while ident_start > 0 && is_ident_char(bytes[ident_start - 1]) {
            ident_start -= 1;
        }
        if ident_start == i {
            // `_sampler` with no preceding identifier (e.g. literal leading underscore).
            i += suffix_len;
            continue;
        }
        let placeholder = &text[ident_start..i + suffix_len];
        if let Some(replacement) = map.get(placeholder) {
            result.push_str(&text[last_emit..ident_start]);
            result.push_str(replacement);
            last_emit = i + suffix_len;
            i = last_emit;
        } else {
            i += suffix_len;
        }
    }
    result.push_str(&text[last_emit..]);
    result
}

fn blend_fx_substitutions(blend_mode: &str) -> BlendFxSubstitutions {
    match blend_mode {
        // blend_fx.hlsl:22-27 — opaque
        "opaque" => BlendFxSubstitutions {
            multiplicative_enabled: "0.0",
            multiplicative_factor: "1.0",
            alpha_channel_output: "output_alpha",
            alpha_premultiply: "1.0",
        },
        // blend_fx.hlsl:29-35 — additive
        "additive" => BlendFxSubstitutions {
            multiplicative_enabled: "0.0",
            multiplicative_factor: "1.0",
            alpha_channel_output: "0.0",
            alpha_premultiply: "1.0",
        },
        // blend_fx.hlsl:37-43 — multiply
        "multiply" => BlendFxSubstitutions {
            multiplicative_enabled: "1.0",
            multiplicative_factor: "1.0",
            alpha_channel_output: "1.0",
            alpha_premultiply: "1.0",
        },
        // blend_fx.hlsl:45-51 — alpha_blend
        "alpha_blend" => BlendFxSubstitutions {
            multiplicative_enabled: "0.0",
            multiplicative_factor: "1.0",
            alpha_channel_output: "albedo.a",
            alpha_premultiply: "1.0",
        },
        // blend_fx.hlsl:53-59 — double_multiply
        "double_multiply" => BlendFxSubstitutions {
            multiplicative_enabled: "1.0",
            multiplicative_factor: "2.0",
            alpha_channel_output: "1.0",
            alpha_premultiply: "1.0",
        },
        // blend_fx.hlsl:61-67 — pre_multiplied_alpha
        "pre_multiplied_alpha" => BlendFxSubstitutions {
            multiplicative_enabled: "0.0",
            multiplicative_factor: "1.0",
            alpha_channel_output: "albedo.a",
            // Engine `convert_to_render_target_premultiplied_alpha`
            // (render_target_fx.hlsl:63) does `color.xyz *= color.w` — it
            // premultiplies by the FINAL output alpha (`out_color.w`), NOT a
            // separate `albedo.a`. For the BLEND_FRESNEL (glass) path
            // `out_color.w = saturate(fresnel + albedo.w)`, which is much
            // larger than `albedo.a` at grazing angles. Using `albedo.a` here
            // under-multiplied the RGB while the device blend (One,
            // 1-srcAlpha) still occluded the background by the high fresnel
            // alpha → black/opaque glass (construct, guardian channels).
            // Mirror the engine: premultiply by the computed `alpha_out`
            // (which equals `albedo.a` for non-fresnel pre_mult materials, so
            // those are unchanged).
            alpha_premultiply: "alpha_out",
        },
        // Unknown mode — fall back to alpha_blend's table. Walker should
        // never produce a name outside the blend_fx.hlsl set; keep it
        // a fail-loud path with eprintln so we notice if it does.
        other => {
            eprintln!(
                "[blend_fx] unknown blend_mode '{other}' — defaulting to alpha_blend table. \
                 Add it to `blend_fx_substitutions` if it's a real Halo mode."
            );
            BlendFxSubstitutions {
                multiplicative_enabled: "0.0",
                multiplicative_factor: "1.0",
                alpha_channel_output: "albedo.a",
                alpha_premultiply: "1.0",
            }
        }
    }
}

fn apply_blend_fx_substitutions(src: &str, sub: &BlendFxSubstitutions) -> String {
    src.replace("__BLEND_MULTIPLICATIVE_ENABLED__", sub.multiplicative_enabled)
        .replace("__BLEND_MULTIPLICATIVE_FACTOR__", sub.multiplicative_factor)
        .replace("__ALPHA_CHANNEL_OUTPUT__", sub.alpha_channel_output)
        .replace("__ALPHA_PREMULTIPLY__", sub.alpha_premultiply)
}

/// Resolve `order3_area_specular` (HLSL `material_models_fx.hlsl:70`)
/// from the rmt2's bool constants. The engine's offline compiler bakes
/// this per-rmt2 — `true` for the cook_torrance / phong material models
/// that branch between `calculate_area_specular_phong_order_3` and
/// `_phong_order_2`; `false` is rare (notably glass, which hardcodes it
/// in the HLSL body).
///
/// Default `true` matches the engine default for materials that consume
/// it. Returns the literal `"true"` / `"false"` for substitution into
/// WGSL `const` declarations.
fn resolve_order3_area_specular(rm: &ResolvedRenderMethod) -> &'static str {
    match rm.find("order3_area_specular").map(|p| &p.source) {
        Some(ParameterSource::Inline(ResolvedValue::Bool(false))) => "false",
        _ => "true",
    }
}

/// Resolve `no_dynamic_lights` — engine HLSL `material_models_fx.hlsl:71`
/// `PARAM(bool, no_dynamic_lights)`. Each material model's PS body gates
/// the `calc_simple_lights_analytical` accumulation on `if (!no_dynamic_lights)`
/// (diffuse_only_fx.hlsl:74, single_lobe_phong_fx.hlsl:114,
/// glass_material_fx.hlsl:104, foliage_material_fx.hlsl:75,
/// cook_torrance_fx.hlsl:1087/1357/1600, two_lobe_phong_fx.hlsl:346/465).
/// rmsh authors override per-material via their rmop's `no_dynamic_lights`
/// bool param (default `false` — dynamic lights enabled).
///
/// Returns the WGSL literal `"true"` / `"false"` for substitution into
/// `const NO_DYNAMIC_LIGHTS: bool` declarations.
fn resolve_no_dynamic_lights(rm: &ResolvedRenderMethod) -> &'static str {
    match rm.find("no_dynamic_lights").map(|p| &p.source) {
        Some(ParameterSource::Inline(ResolvedValue::Bool(true))) => "true",
        _ => "false",
    }
}

/// Format a panic message for an option_name that doesn't have a
/// ported WGSL fragment yet. The name comes straight from the rmdf
/// (`category.options[i].option_name`) — no hand-typed table to drift.
fn unsupported_option(category: &str, option_name: &str, hlsl_file: &str) -> ! {
    panic!(
        "[protomorph] unsupported {category} option '{option_name}'. \
         Port the HLSL function `calc_{category}_{option_name}_ps` from \
         `/Users/camden/Halo/halo3_mcc_hlsl_extracted/{hlsl_file}` to \
         `assets/halo_shaders/...{option_name}.wgsl` and add a `pick_{category}` \
         match arm. (Per `feedback_wgsl_must_mirror_hlsl.md`: silent fallbacks \
         hide what needs porting — fix the dispatcher rather than degrade.)",
    )
}

fn pick_albedo(option_name: &str) -> &'static str {
    match option_name {
        "default"                  => include_str!("../../../assets/halo_shaders/albedo_fx_default.wgsl"),
        "detail_blend"             => include_str!("../../../assets/halo_shaders/albedo_fx_detail_blend.wgsl"),
        "three_detail_blend"       => include_str!("../../../assets/halo_shaders/albedo_fx_three_detail_blend.wgsl"),
        "constant_color"           => include_str!("../../../assets/halo_shaders/albedo_fx_constant_color.wgsl"),
        "two_change_color"         => include_str!("../../../assets/halo_shaders/albedo_fx_two_change_color.wgsl"),
        "two_detail_overlay"       => include_str!("../../../assets/halo_shaders/albedo_fx_two_detail_overlay.wgsl"),
        "two_detail"               => include_str!("../../../assets/halo_shaders/albedo_fx_two_detail.wgsl"),
        "two_detail_black_point"   => include_str!("../../../assets/halo_shaders/albedo_fx_two_detail_black_point.wgsl"),
        "color_mask"               => include_str!("../../../assets/halo_shaders/albedo_fx_color_mask.wgsl"),
        "waterfall"                => include_str!("../../../assets/halo_shaders/albedo_fx_waterfall.wgsl"),
        "chameleon"                => include_str!("../../../assets/halo_shaders/albedo_fx_chameleon.wgsl"),
        "chameleon_masked"         => include_str!("../../../assets/halo_shaders/albedo_fx_chameleon_masked.wgsl"),
        "chameleon_albedo_masked"  => include_str!("../../../assets/halo_shaders/albedo_fx_chameleon_albedo_masked.wgsl"),
        n                          => unsupported_option("albedo", n, "albedo_fx.hlsl"),
    }
}

fn pick_bump(option_name: &str) -> &'static str {
    match option_name {
        "off"           => include_str!("../../../assets/halo_shaders/bump_mapping_fx_off.wgsl"),
        "standard"      => include_str!("../../../assets/halo_shaders/bump_mapping_fx_default.wgsl"),
        "default"       => include_str!("../../../assets/halo_shaders/bump_mapping_fx_default.wgsl"),
        "detail"        => include_str!("../../../assets/halo_shaders/bump_mapping_fx_detail.wgsl"),
        "detail_unorm"  => include_str!("../../../assets/halo_shaders/bump_mapping_fx_detail_unorm.wgsl"),
        "detail_masked" => include_str!("../../../assets/halo_shaders/bump_mapping_fx_detail_masked.wgsl"),
        "detail_plus_detail_masked" => include_str!("../../../assets/halo_shaders/bump_mapping_fx_detail_plus_detail_masked.wgsl"),
        n               => unsupported_option("bumpmap", n, "bump_mapping_fx.hlsl"),
    }
}

fn pick_parallax(option_name: &str) -> &'static str {
    match option_name {
        "off"          => include_str!("../../../assets/halo_shaders/parallax_fx_off.wgsl"),
        "simple"       => include_str!("../../../assets/halo_shaders/parallax_fx_simple.wgsl"),
        "interpolated" => include_str!("../../../assets/halo_shaders/parallax_fx_interpolated.wgsl"),
        n              => unsupported_option("parallax", n, "parallax_fx.hlsl"),
    }
}

fn pick_alpha_test(option_name: &str) -> &'static str {
    match option_name {
        "none"         => include_str!("../../../assets/halo_shaders/alpha_test_fx_off.wgsl"),
        "off"          => include_str!("../../../assets/halo_shaders/alpha_test_fx_off.wgsl"),
        "simple"       => include_str!("../../../assets/halo_shaders/alpha_test_fx_on.wgsl"),
        // shader_custom (rmcs) — engine `alpha_test_custom_fx.hlsl`.
        // Samples both `alpha_test_map` and `multiply_map`, clips on
        // `alpha_test_map.a * multiply_map.a`. Used by deadlock nets
        // and similar chain-link/canopy materials.
        "multiply_map" => include_str!("../../../assets/halo_shaders/alpha_test_fx_multiply_map.wgsl"),
        n              => unsupported_option("alpha_test", n, "alpha_test_fx.hlsl + alpha_test_custom_fx.hlsl"),
    }
}

fn pick_specular_mask(option_name: &str) -> &'static str {
    match option_name {
        "no_specular_mask"            => include_str!("../../../assets/halo_shaders/specular_mask_fx_no_specular_mask.wgsl"),
        "specular_mask_from_diffuse"  => include_str!("../../../assets/halo_shaders/specular_mask_fx_from_diffuse.wgsl"),
        "specular_mask_from_texture"  => include_str!("../../../assets/halo_shaders/specular_mask_fx_specular_mask_from_texture.wgsl"),
        "specular_mask_from_color_texture" => include_str!("../../../assets/halo_shaders/specular_mask_fx_specular_mask_from_color_texture.wgsl"),
        n                             => unsupported_option("specular_mask", n, "specular_mask_fx.hlsl"),
    }
}

fn pick_env_mapping(option_name: &str) -> &'static str {
    match option_name {
        "none"               => include_str!("../../../assets/halo_shaders/environment_mapping_fx_none.wgsl"),
        "per_pixel"          => include_str!("../../../assets/halo_shaders/environment_mapping_fx_per_pixel.wgsl"),
        "dynamic"            => include_str!("../../../assets/halo_shaders/environment_mapping_fx_dynamic.wgsl"),
        "from_flat_texture"  => include_str!("../../../assets/halo_shaders/environment_mapping_fx_from_flat_texture.wgsl"),
        "from_flat_texture_as_cubemap" => include_str!("../../../assets/halo_shaders/environment_mapping_fx_from_flat_texture_as_cubemap.wgsl"),
        n                    => unsupported_option("environment_map", n, "environment_mapping_fx.hlsl"),
    }
}

fn pick_self_illum(option_name: &str) -> &'static str {
    match option_name {
        "none" | "off"             => include_str!("../../../assets/halo_shaders/self_illumination_fx_none.wgsl"),
        "simple"                   => include_str!("../../../assets/halo_shaders/self_illumination_fx_simple.wgsl"),
        "simple_with_alpha_mask"   => include_str!("../../../assets/halo_shaders/self_illumination_fx_simple_with_alpha_mask.wgsl"),
        // `3_channel_self_illum` is an MCC schema-drift alias for the same
        // engine HLSL `calc_self_illumination_three_channel_ps`.
        "3_channel" | "3_channel_self_illum" => include_str!("../../../assets/halo_shaders/self_illumination_fx_three_channel.wgsl"),
        "from_diffuse"             => include_str!("../../../assets/halo_shaders/self_illumination_fx_from_albedo.wgsl"),
        "illum_detail"             => include_str!("../../../assets/halo_shaders/self_illumination_fx_detail.wgsl"),
        "self_illum_times_diffuse" => include_str!("../../../assets/halo_shaders/self_illumination_fx_times_diffuse.wgsl"),
        // ⚠️  Reconstructed — no HLSL ships in MCC/Reach for this variant.
        // See assets/halo_shaders/self_illumination_fx_simple_four_change_color.wgsl
        // header for the full caveat. Selected by 343i-ported maps
        // (s3d_powerhouse and similar).
        "simple_four_change_color" => include_str!("../../../assets/halo_shaders/self_illumination_fx_simple_four_change_color.wgsl"),
        "plasma"                   => include_str!("../../../assets/halo_shaders/self_illumination_fx_plasma.wgsl"),
        "meter"                    => include_str!("../../../assets/halo_shaders/self_illumination_fx_meter.wgsl"),
        // Halogram rmdf self-illum options that engine routes to
        // `calc_self_illumination_multilayer_ps`. The halogram rmdf
        // exposes 3 distinct option names that all call the same
        // pixel function:
        //   - multilayer_additive
        //   - ml_add_four_change_color
        //   - ml_add_five_change_color
        // (`_multilayer_depth_ps` / `_multilayer_cheap_ps` are defined
        // in `self_illumination_halogram_fx.hlsl` but no rmdf option
        // calls them — dead engine code.)
        "multilayer_additive"
        | "ml_add_four_change_color"
        | "ml_add_five_change_color" => include_str!("../../../assets/halo_shaders/self_illumination_halogram_fx_multilayer.wgsl"),
        "scope_blur"               => include_str!("../../../assets/halo_shaders/self_illumination_halogram_fx_scope_blur.wgsl"),
        n                          => unsupported_option("self_illumination", n, "self_illumination_fx.hlsl"),
    }
}

/// `overlay` category — `overlays_fx.hlsl`. Composited onto the final lit
/// color (after radiance, before extinction/exposure) via `calc_overlay_ps`
/// in the entry point, mirroring the engine `APPLY_OVERLAYS` macro.
fn pick_overlay(option_name: &str) -> &'static str {
    match option_name {
        "none"                          => include_str!("../../../assets/halo_shaders/overlays_fx_none.wgsl"),
        "additive"                      => include_str!("../../../assets/halo_shaders/overlays_fx_additive.wgsl"),
        "additive_detail"               => include_str!("../../../assets/halo_shaders/overlays_fx_additive_detail.wgsl"),
        "multiply"                      => include_str!("../../../assets/halo_shaders/overlays_fx_multiply.wgsl"),
        "multiply_and_additive_detail"  => include_str!("../../../assets/halo_shaders/overlays_fx_multiply_and_additive_detail.wgsl"),
        n                               => unsupported_option("overlay", n, "overlays_fx.hlsl"),
    }
}

/// `warp` category — `warp_fx.hlsl`. Shares the `calc_parallax` texcoord-
/// producer slot (mutually exclusive with parallax — the engine #defines
/// `calc_parallax_ps` to the warp function when warp is set). The `none`
/// case is handled by the assembler falling back to `pick_parallax`.
fn pick_warp(option_name: &str) -> &'static str {
    match option_name {
        "from_texture" => include_str!("../../../assets/halo_shaders/warp_fx_from_texture.wgsl"),
        n              => unsupported_option("warp", n, "warp_fx.hlsl"),
    }
}

/// `edge_fade` category — `overlays_fx.hlsl`. Applied after the overlay via
/// `calc_edge_fade_ps(color, view_dot_normal)` in the entry point.
fn pick_edge_fade(option_name: &str) -> &'static str {
    match option_name {
        "none"   => include_str!("../../../assets/halo_shaders/edge_fade_fx_none.wgsl"),
        "simple" => include_str!("../../../assets/halo_shaders/edge_fade_fx_simple.wgsl"),
        n        => unsupported_option("edge_fade", n, "overlays_fx.hlsl"),
    }
}

/// `soft_fade` category — common_fx.hlsl `apply_soft_fade_{off,on}`.
/// Always returns an `apply_soft_fade` definition (the entry point calls
/// it unconditionally); `off`/absent → no-op. The per-material
/// `use_soft_fresnel` / `use_soft_z` PARAM(bool)s are resolved here (like
/// `no_dynamic_lights`) so each HLSL branch is emitted only when set.
///
/// `use_soft_z` (the scene-depth soft-particle fade) is fail-loud
/// DEFERRED: it needs `scene_depth` bound into the transparent material
/// group, and every shipped material that sets it is in a level that also
/// requires the (unported) distortion displacement subsystem — so it
/// clears no level yet and is best landed alongside distortion. The
/// `apply_soft_fade` signature already carries `frag_device_z`/`vpos` so
/// adding the branch later is localized.
fn pick_soft_fade(option_name: &str, rm: &ResolvedRenderMethod, alpha_blend: bool) -> String {
    let resolve_bool = |n: &str| {
        matches!(
            rm.find(n).map(|p| &p.source),
            Some(ParameterSource::Inline(ResolvedValue::Bool(true)))
        )
    };
    match option_name {
        "off" | "none" => {
            include_str!("../../../assets/halo_shaders/soft_fade_fx_off.wgsl").to_string()
        }
        "on" => {
            // use_soft_fresnel branch (common_fx.hlsl:73-76). When unset,
            // `val` stays 1 and apply_soft_fade is an effective no-op.
            let fresnel = if resolve_bool("use_soft_fresnel") {
                "    let fresnel_dp = sf_calc_fresnel_dp(wnorm, wview);\n\
                 \x20   val *= pow(fresnel_dp, material.soft_fresnel_power.x);"
            } else {
                "    // use_soft_fresnel = false → fresnel term skipped"
            };
            // use_soft_z branch (common_fx.hlsl:77-82). Fades the surface
            // as it nears scene geometry behind it (soft-particle effect):
            //   val *= get_softness(z_to_w(depth_map.r), linearDepth, soft_z_range)
            //        = saturate((scene_dist - frag_dist) * soft_z_range)
            // `scene_depth_tex` is the read-only scene depth bound at
            // camera_bgl binding 15 (only real in the transparent pass).
            // `vpos` (= clip_position.xy) is the fragment's pixel coord →
            // point-load the scene depth at the same pixel. Both distances
            // use `sf_linear_depth` (camera-projection reconstruction).
            let soft_z = if resolve_bool("use_soft_z") {
                "    let sz_scene = sf_linear_depth(textureLoad(scene_depth_tex, vec2<i32>(vpos), 0));\n\
                 \x20   let sz_frag = sf_linear_depth(frag_device_z);\n\
                 \x20   val *= saturate((sz_scene - sz_frag) * material.soft_z_range.x);"
            } else {
                "    // use_soft_z = false → scene-depth fade skipped"
            };
            // BLEND_MODE(alpha_blend) → fade alpha; else fade rgb
            // (common_fx.hlsl:83-87).
            let apply = if alpha_blend {
                "    (*albedo).w = (*albedo).w * val;"
            } else {
                "    (*albedo) = vec4<f32>((*albedo).xyz * val, (*albedo).w);"
            };
            include_str!("../../../assets/halo_shaders/soft_fade_fx_on.wgsl")
                .replace("__SOFT_FADE_FRESNEL__", fresnel)
                .replace("__SOFT_FADE_SOFTZ__", soft_z)
                .replace("__SOFT_FADE_APPLY__", apply)
        }
        n => unsupported_option("soft_fade", n, "common_fx.hlsl"),
    }
}

/// Note: rmfl-group (foliage shader) materials don't pass through here
/// — they have their own self-contained entry point.
fn pick_material_model(option_name: &str) -> &'static str {
    match option_name {
        "diffuse_only"      => include_str!("../../../assets/halo_shaders/diffuse_only_fx.wgsl"),
        "cook_torrance"     => include_str!("../../../assets/halo_shaders/cook_torrance_fx.wgsl"),
        "cook_torrance_pbr_maps" => include_str!("../../../assets/halo_shaders/cook_torrance_pbr_maps_fx.wgsl"),
        // `cook_torrance_from_albedo` (s3d_turf etc.) reads PBR control
        // values (metallic / roughness) from the diffuse texture's
        // alpha + extra channels instead of dedicated maps. No engine
        // HLSL function exists in our extracted corpus — alias to
        // standard cook_torrance as a stand-in (specular shape +
        // material model match; per-pixel PBR weighting falls back to
        // the rmt2's flat coefficients). Refine when the HLSL surfaces.
        "cook_torrance_from_albedo" => include_str!("../../../assets/halo_shaders/cook_torrance_fx.wgsl"),
        // `cook_torrance_rim_fresnel` (s3d_waterfall etc.) adds a
        // view-angle rim term on top of standard cook_torrance. No
        // engine HLSL in our extracted corpus — alias to vanilla
        // cook_torrance as a stand-in; the base specular shape is
        // correct, the rim accent is missing until the HLSL surfaces.
        "cook_torrance_rim_fresnel" => include_str!("../../../assets/halo_shaders/cook_torrance_fx.wgsl"),
        // `two_lobe_phong_tint_map` is the same engine HLSL with
        // `normal_specular_tint_map` / `glancing_specular_tint_map`
        // sampled per-texel instead of using the flat tint values
        // (`two_lobe_phong_fx.hlsl:47-68`). v1 alias to flat-tint —
        // texture sampling deferred until binding plumbing lands.
        "two_lobe_phong" | "two_lobe_phong_tint_map" => include_str!("../../../assets/halo_shaders/two_lobe_phong_fx.wgsl"),
        "foliage"           => include_str!("../../../assets/halo_shaders/foliage_material_fx.wgsl"),
        // Engine `material_model_none_fx.hlsl::calc_material_none_ps` —
        // ALL outputs zeroed. Used by halogram materials so only the
        // self_illumination contribution is visible (no diffuse base
        // color leaking from the planes the holograms render on).
        "none"              => include_str!("../../../assets/halo_shaders/material_model_none_fx.wgsl"),
        "glass"             => include_str!("../../../assets/halo_shaders/glass_material_fx.wgsl"),
        "single_lobe_phong" => include_str!("../../../assets/halo_shaders/single_lobe_phong_fx.wgsl"),
        "organism"          => include_str!("../../../assets/halo_shaders/organism_material_fx.wgsl"),
        n                   => unsupported_option("material_model", n, "material_models_fx.hlsl"),
    }
}

// =============================================================================
// Per-category WGSL pickers for the rmd (`c_render_method_shader_decal`)
// subclass. Each defines the engine helper function (`sample_diffuse`,
// `fade_out`, `sample_bump`, `tint_and_modulate`) for the option's
// branch in `decal_fx.hlsl`. Concatenated by `assemble`'s `b"rmd "`
// arm before `entry_decal.wgsl`.
// =============================================================================

fn pick_decal_albedo(option_name: &str) -> &'static str {
    match option_name {
        "diffuse_only" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_diffuse_only.wgsl"
        ),
        "palettized" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_palettized.wgsl"
        ),
        "palettized_plus_alpha" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_palettized_plus_alpha.wgsl"
        ),
        "diffuse_plus_alpha" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_diffuse_plus_alpha.wgsl"
        ),
        "diffuse_plus_alpha_mask" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_diffuse_plus_alpha_mask.wgsl"
        ),
        "vector_alpha" => include_str!(
            "../../../assets/halo_shaders/decal_albedo_vector_alpha.wgsl"
        ),
        n => unsupported_option("decal_albedo", n, "decal_fx.hlsl::sample_diffuse"),
    }
}

fn pick_decal_blend_mode(option_name: &str) -> &'static str {
    match option_name {
        "alpha_blend" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_alpha_blend.wgsl"
        ),
        "additive" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_additive.wgsl"
        ),
        "multiply" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_multiply.wgsl"
        ),
        "double_multiply" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_double_multiply.wgsl"
        ),
        "pre_multiplied_alpha" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_pre_multiplied_alpha.wgsl"
        ),
        "opaque" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_opaque.wgsl"
        ),
        "inv_alpha_blend" => include_str!(
            "../../../assets/halo_shaders/decal_blend_mode_inv_alpha_blend.wgsl"
        ),
        n => unsupported_option("decal_blend_mode", n, "decal_fx.hlsl::fade_out"),
    }
}

fn pick_decal_bump_mapping(option_name: &str) -> &'static str {
    match option_name {
        "leave" => include_str!(
            "../../../assets/halo_shaders/decal_bump_mapping_leave.wgsl"
        ),
        "standard" => include_str!(
            "../../../assets/halo_shaders/decal_bump_mapping_standard.wgsl"
        ),
        "standard_mask" => include_str!(
            "../../../assets/halo_shaders/decal_bump_mapping_standard_mask.wgsl"
        ),
        n => unsupported_option("decal_bump_mapping", n, "decal_fx.hlsl::sample_bump"),
    }
}

fn pick_decal_tinting(option_name: &str) -> &'static str {
    match option_name {
        "none" => include_str!(
            "../../../assets/halo_shaders/decal_tinting_none.wgsl"
        ),
        "unmodulated" => include_str!(
            "../../../assets/halo_shaders/decal_tinting_unmodulated.wgsl"
        ),
        "fully_modulated" => include_str!(
            "../../../assets/halo_shaders/decal_tinting_fully_modulated.wgsl"
        ),
        "partially_modulated" => include_str!(
            "../../../assets/halo_shaders/decal_tinting_partially_modulated.wgsl"
        ),
        n => unsupported_option("decal_tinting", n, "decal_fx.hlsl::tint_and_modulate"),
    }
}
