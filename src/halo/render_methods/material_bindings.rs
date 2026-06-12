//! Per-variant material texture binding generator.
//!
//! Walks `ResolvedRenderMethod.parameters` for `Bitmap`-typed entries
//! and produces:
//! - A `wgpu::BindGroupLayout` with one entry per texture + a sampler
//!   + the user-parameter cbuffer.
//! - Generated WGSL declarations naming each texture by its Halo
//!   parameter name (`base_map`, `bump_map`, `base_map_m_0`, etc.).
//! - A name→slot map so the material bind group builder can route
//!   textures into the right slots at material-load time.
//!
//! The hardcoded 8-slot rmsh-style BGL in `shared.rs` is replaced by
//! this — terrain (rmtr) needs ~17 texture slots (4× per-layer base
//! + detail + bump + detail_bump + blend_map), water has its own
//! sampler list, etc. dllcache mirrors the rmt2's actual sampler
//! register table; we mirror that by walking the resolved parameter
//! list.

use blam_tags::render_method::{
    ParameterSource, ResolvedParameter, ResolvedRenderMethod, ResolvedValue,
};

/// One texture binding slot for a variant — its Halo parameter name
/// + its `@binding(N)` slot in the material bind group.
#[derive(Debug, Clone)]
pub struct TextureBindingSlot {
    /// Halo parameter name (e.g. `"base_map"`, `"base_map_m_0"`).
    pub name: String,
    /// `@binding(N)` slot in @group(2).
    pub slot: u32,
    /// `texture_2d<f32>` vs `texture_cube<f32>`. Detected by name —
    /// only `environment_map` and obvious cube names get cube; rest
    /// default to 2D. Faithful per-bitmap-tag detection deferred
    /// until we plumb the bitm tag's usage field through.
    pub is_cube: bool,
    /// `Some(extern)` when the rmt2 declared this parameter as an
    /// engine extern (`ParameterSource::Extern`). Lets the bind-group
    /// builder route engine-managed textures (cook_torrance LUTs,
    /// dynamic env probes, etc.) to the correct view by extern enum
    /// match, rather than falling through to white/black fallbacks.
    pub source_extern: Option<blam_tags::render_method::RenderMethodExtern>,
}

/// Complete binding map for a variant's @group(2).
#[derive(Debug, Clone)]
pub struct MaterialBindings {
    pub textures: Vec<TextureBindingSlot>,
    /// Slot for the shared filtering sampler. Always last after textures.
    pub sampler_slot: u32,
    /// Slot for the user-parameter cbuffer. Always last after sampler.
    pub cbuffer_slot: u32,
    /// Slot for the per-draw decal constants UBO (`DecalConstants`).
    /// `Some` only when the material is rmd — mirrors engine
    /// `c_decal::render @ 0x18039B100` issuing `set_shader_constant
    /// (0x140000, fade_vec4)` + `set_shader_constant(0x130000,
    /// m_sprite_corner_vec4)` per draw. Bound with `has_dynamic_offset
    /// = true` so a single buffer can ring-buffer all decals + each
    /// `draw_indexed` picks its slice via a per-decal offset.
    pub decal_constants_slot: Option<u32>,
}

/// Standard texture names the included rmsh-style WGSL fragments
/// (`albedo_default.wgsl`, `bump_default.wgsl`, ...) reference
/// unconditionally. The MaterialBindings generator unions these with
/// the rmop's actual declared textures so fragments always have their
/// declarations available — sparse rmops (e.g. `z_invis_lightquad`)
/// would otherwise produce WGSL with undefined identifiers.
///
/// Once per-fragment texture declaration is in place (each fragment
/// declares which textures it needs and the assembler emits exactly
/// that subset), this baseline can shrink to zero.
const RMSH_BASELINE_TEXTURES: &[(&str, bool)] = &[
    ("base_map", false),
    ("bump_map", false),
    ("self_illum_map", false),
    ("change_color_map", false),
    ("detail_map", false),
    ("bump_detail_map", false),
    ("environment_map", true),
    ("alpha_test_map", false),
    // `calc_alpha_test_multiply_map_ps` (alpha_test_custom_fx.hlsl) —
    // shader_custom `alpha_test = multiply_map` option samples a SECOND
    // alpha bitmap and clips on `alpha_test_map.a * multiply_map.a`.
    // Used by e.g. `levels/multi/deadlock/shaders/deadlock_net.shader_custom`.
    ("multiply_map", false),
    ("specular_mask_texture", false),
    ("self_illum_detail_map", false),
    ("detail_map2", false),
    ("detail_map3", false),
    ("detail_map_overlay", false),
    // `overlay` category (overlays_fx.hlsl) — the halogram's cell-like
    // inner layer (multiply_and_additive_detail). overlay_multiply_map
    // multiplies the lit color; overlay_map × overlay_detail_map add.
    ("overlay_map", false),
    ("overlay_detail_map", false),
    ("overlay_multiply_map", false),
    // `warp` category (warp_fx.hlsl) — offsets the working texcoord by
    // warp_map.xy × (warp_amount_x, warp_amount_y).
    ("warp_map", false),
    ("height_map", false),
    // env_mapping/dynamic — two cubemaps blended per-frame. Until
    // Phase G2 (per-cluster dynamic env probes) lands, these resolve
    // to the 1×1 black-cube fallback so reflections render as 0.
    ("dynamic_environment_map_0", true),
    ("dynamic_environment_map_1", true),
    // `plasma` self_illumination textures — only used by the plasma
    // option's WGSL but the assembler always emits the declarations.
    // Non-plasma variants get the runtime-default fallback bound here.
    ("alpha_mask_map", false),
    ("noise_map_a", false),
    ("noise_map_b", false),
    // `cook_torrance_pbr_maps` per-pixel material textures.
    // `material_texture.r` = specular_coefficient; `.g` = roughness
    // (HLSL `cook_torrance_fx.hlsl:406` `mp = sample(material_texture).xxyy`).
    // `spec_tint_map` replaces cbuffer `specular_tint` per-pixel
    // (HLSL line 1555).
    ("material_texture", false),
    ("spec_tint_map", false),
];

/// Texture names the terrain WGSL entry (`entry_terrain_static_sh.wgsl`)
/// references unconditionally for all 4 blend-map layers + the blend
/// mask. Riverworld terrain rmt2s declare 3 layers (m_0..m_2) for
/// some shaders and 4 (m_0..m_3) for others; the WGSL is a single
/// 4-layer body, so missing layer textures get bound to neutral
/// fallbacks (white for albedo/detail, default-normal for bumps).
const RMTR_BASELINE_TEXTURES: &[(&str, bool)] = &[
    ("blend_map", false),
    ("base_map_m_0", false),
    ("base_map_m_1", false),
    ("base_map_m_2", false),
    ("base_map_m_3", false),
    ("detail_map_m_0", false),
    ("detail_map_m_1", false),
    ("detail_map_m_2", false),
    ("detail_map_m_3", false),
    ("bump_map_m_0", false),
    ("bump_map_m_1", false),
    ("bump_map_m_2", false),
    ("bump_map_m_3", false),
    ("detail_bump_m_0", false),
    ("detail_bump_m_1", false),
    ("detail_bump_m_2", false),
    ("detail_bump_m_3", false),
    // Self-illum (only layers 0-2 per terrain_new_fx.hlsl:753-765 —
    // engine `blend_surface_parameters` has no material_3_type branch).
    ("self_illum_map_m_0", false),
    ("self_illum_map_m_1", false),
    ("self_illum_map_m_2", false),
    ("self_illum_detail_map_m_0", false),
    ("self_illum_detail_map_m_1", false),
    ("self_illum_detail_map_m_2", false),
];

impl MaterialBindings {
    /// Build the per-variant texture binding map. `tags_root` lets us
    /// read each Bitmap parameter's bitm tag to query its actual
    /// dimension (`type_name() == "cube map"`) — Halo's source-of-truth
    /// for cube vs 2D classification. The runtime
    /// (`render_method_submit_set_texture @ 0x180684490`) trusts the
    /// bitm tag's type and lets D3D enforce sampler-bytecode matching
    /// against the pixl shader. We don't have pixl bytecode, so we
    /// generate the WGSL declaration from the bitmap tag directly.
    ///
    /// Engine externs (extern_texture_mode != 0) and our own baseline
    /// entries can't be classified from a tag — for those we fall back
    /// to the name heuristic (`environment_map`, `dynamic_environment_map_*`).
    /// Halo gets the right answer there from the pixl shader bytecode's
    /// sampler register declarations.
    pub fn from_resolved(rm: &ResolvedRenderMethod, tags_root: &std::path::Path) -> Self {
        let mut textures = Vec::new();
        let mut next_slot = 0u32;

        // Iterate ALL Bitmap-typed parameters in resolved order.
        // Insertion order = rmdf category + rmop parameter order, which
        // matches dllcache's runtime register assignment.
        for p in &rm.parameters {
            if !is_bitmap_param(p) {
                continue;
            }
            // Skip duplicates — rmdf can declare the same parameter in
            // multiple categories; first declaration wins (matches
            // `c_render_method::find_parameter_index` runtime semantics).
            if textures.iter().any(|t: &TextureBindingSlot| t.name == p.name) {
                continue;
            }
            // Force-cube names — engine's pixl bytecode binds these
            // sampler registers as cube regardless of the bitm tag's
            // actual dimension (the HLSL declares PARAM_SAMPLER_CUBE).
            // Some rmop authors target an `environment_map` slot with a
            // 2D bitmap; the runtime register binding still cubes, so
            // mirror that. Without this override, our WGSL `texture_cube`
            // declaration would mismatch the binding type → naga
            // "Image coordinate type does not match dimension D2" panic
            // (seen on warehouse / lockout).
            let force_cube = matches!(
                p.name.as_str(),
                "environment_map"
                    | "dynamic_environment_map_0"
                    | "dynamic_environment_map_1",
            );
            let is_cube = force_cube || classify_bitmap_param(p, tags_root);
            // Capture the engine extern enum when the rmt2 declares
            // this parameter as `ParameterSource::Extern`. The
            // bind-group builder uses this to route engine-managed
            // textures (cook_torrance LUTs etc.) to the right view.
            let source_extern = match &p.source {
                ParameterSource::Extern(ext) => Some(*ext),
                ParameterSource::Inline(_) => None,
            };
            textures.push(TextureBindingSlot {
                name: p.name.clone(),
                slot: next_slot,
                is_cube,
                source_extern,
            });
            next_slot += 1;
        }

        // Union with the per-subclass baseline texture set the
        // assembled WGSL fragments reference unconditionally. Skip
        // names already declared by the rmop walk (preserves rmop
        // ordering for shared names). Baseline entries' `is_cube` is
        // hardcoded since these are engine externs we add — Halo's
        // pixl bytecode would dictate the same answer.
        let baseline: &[(&str, bool)] = match &rm.group_tag.to_be_bytes() {
            b"rmtr" => RMTR_BASELINE_TEXTURES,
            _ => RMSH_BASELINE_TEXTURES,
        };
        for &(name, is_cube) in baseline {
            if textures.iter().any(|t| t.name == name) {
                continue;
            }
            textures.push(TextureBindingSlot {
                name: name.to_string(),
                slot: next_slot,
                is_cube,
                // Baseline entries are author-defined fallbacks the
                // assembled WGSL references unconditionally; they're
                // not engine externs themselves.
                source_extern: None,
            });
            next_slot += 1;
        }

        let sampler_slot = next_slot;
        let cbuffer_slot = next_slot + 1;
        // rmd materials get an EXTRA per-draw UBO for engine
        // `set_shader_constant(0x140000, fade_vec4)` +
        // `set_shader_constant(0x130000, m_sprite_corner_vec4)`.
        // Other rm** subclasses don't need it.
        let decal_constants_slot = if &rm.group_tag.to_be_bytes() == b"rmd " {
            Some(next_slot + 2)
        } else {
            None
        };
        Self {
            textures,
            sampler_slot,
            cbuffer_slot,
            decal_constants_slot,
        }
    }

    /// Emit the WGSL declarations for @group(2) — one `var <name>:
    /// texture_*<f32>;` per texture + the sampler at `sampler_slot`.
    /// The cbuffer struct itself is emitted separately by
    /// `cbuffer::emit_wgsl` at `cbuffer_slot`.
    ///
    /// Legacy single-sampler path — emits one `s_material` binding.
    /// New call sites should use [`Self::emit_wgsl_per_binding`] with
    /// a [`MaterialSamplerMap`] to honor the rmt2's authored per-binding
    /// sampler config.
    pub fn emit_wgsl(&self) -> String {
        let mut out = String::new();
        for tb in &self.textures {
            let kind = if tb.is_cube {
                "texture_cube<f32>"
            } else {
                "texture_2d<f32>"
            };
            out.push_str(&format!(
                "@group(2) @binding({}) var {}: {};\n",
                tb.slot, tb.name, kind,
            ));
        }
        out.push_str(&format!(
            "@group(2) @binding({}) var s_material: sampler;\n",
            self.sampler_slot,
        ));
        // Per-draw decal constants — engine `set_shader_constant
        // (0x140000, fade)` + `(0x130000, sprite_corner)` ports.
        // Layout matches `DecalConstantsGpu` in
        // `src/halo/render/render_decals.rs`.
        if let Some(slot) = self.decal_constants_slot {
            out.push_str(
                "struct DecalConstants {\n\
                    fade: f32,\n\
                    pixel_kill_enabled: u32,\n\
                    u_tiles: f32,\n\
                    v_tiles: f32,\n\
                    sprite_corner: vec4<f32>,\n\
                };\n",
            );
            out.push_str(&format!(
                "@group(2) @binding({}) var<uniform> decal_constants: DecalConstants;\n",
                slot,
            ));
        }
        out
    }

    /// Per-binding-sampler emit. Lays out @group(2) as:
    /// - `[0..N)` — texture bindings (same as legacy)
    /// - `[N..N+K)` — one sampler per unique `SamplerKey` in
    ///   `sampler_map`, named `s_dedupe_0..s_dedupe_{K-1}`
    /// - `N+K` — `MaterialParameters` cbuffer
    /// - `N+K+1` — `DecalConstants` (rmd only)
    ///
    /// Engine-faithful: each rmt2-routed bitmap parameter's authored
    /// `BitmapAddressMode` + `BitmapFilterMode` drives sampler config.
    /// Baseline textures with no `BitmapBinding` fall back to
    /// `FILTERING_FALLBACK` and join the shared default dedupe slot.
    pub fn emit_wgsl_per_binding(
        &self,
        sampler_map: &crate::halo::render_methods::material_samplers::MaterialSamplerMap,
    ) -> String {
        let mut out = String::new();
        for tb in &self.textures {
            let kind = if tb.is_cube {
                "texture_cube<f32>"
            } else {
                "texture_2d<f32>"
            };
            out.push_str(&format!(
                "@group(2) @binding({}) var {}: {};\n",
                tb.slot, tb.name, kind,
            ));
        }
        for (idx, _key) in sampler_map.unique_keys.iter().enumerate() {
            let slot = self.per_binding_sampler_slot(idx);
            let name =
                crate::halo::render_methods::material_samplers::MaterialSamplerMap::dedupe_name(idx);
            out.push_str(&format!(
                "@group(2) @binding({}) var {}: sampler;\n",
                slot, name,
            ));
        }
        // DecalConstants emitted via the per-binding decal slot.
        if let Some(slot) = self.per_binding_decal_constants_slot(sampler_map) {
            out.push_str(
                "struct DecalConstants {\n\
                    fade: f32,\n\
                    pixel_kill_enabled: u32,\n\
                    u_tiles: f32,\n\
                    v_tiles: f32,\n\
                    sprite_corner: vec4<f32>,\n\
                };\n",
            );
            out.push_str(&format!(
                "@group(2) @binding({}) var<uniform> decal_constants: DecalConstants;\n",
                slot,
            ));
        }
        out
    }

    /// Per-binding-sampler slot index for `unique_keys[idx]`.
    /// Samplers occupy `[textures.len()..textures.len()+K)`.
    pub fn per_binding_sampler_slot(&self, idx: usize) -> u32 {
        self.textures.len() as u32 + idx as u32
    }

    /// Per-binding-sampler cbuffer slot — sits right after the samplers.
    pub fn per_binding_cbuffer_slot(
        &self,
        sampler_map: &crate::halo::render_methods::material_samplers::MaterialSamplerMap,
    ) -> u32 {
        self.textures.len() as u32 + sampler_map.unique_keys.len() as u32
    }

    /// Per-binding-sampler decal-constants slot — present only for rmd.
    /// Sits right after the cbuffer.
    pub fn per_binding_decal_constants_slot(
        &self,
        sampler_map: &crate::halo::render_methods::material_samplers::MaterialSamplerMap,
    ) -> Option<u32> {
        self.decal_constants_slot.map(|_| {
            self.per_binding_cbuffer_slot(sampler_map) + 1
        })
    }

    /// Build the matching `wgpu::BindGroupLayout`.
    pub fn build_bgl(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let filterable = wgpu::TextureSampleType::Float { filterable: true };
        let mut entries = Vec::with_capacity(self.textures.len() + 2);
        for tb in &self.textures {
            let view_dimension = if tb.is_cube {
                wgpu::TextureViewDimension::Cube
            } else {
                wgpu::TextureViewDimension::D2
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: tb.slot,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension,
                    sample_type: filterable,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.sampler_slot,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.cbuffer_slot,
            // VERTEX_FRAGMENT — the foliage VS reads
            // `material.animation_amplitude_horizontal.x` for the wind
            // displacement (foliage_fx.hlsl:38 PARAM). Texture / sampler
            // entries remain FRAGMENT-only.
            //
            // **Path B per-object cbuffer pool (2026-05-20):**
            // `has_dynamic_offset: true` so a single material props
            // buffer can host N per-(object, material) slots, picked
            // via per-draw `set_bind_group(2, bg, &[slot * slot_size])`.
            // Engine analog: `g_ps_parameters_data` / `g_vs_parameters_data`
            // global pools (the engine writes per-draw into the slot
            // indicated by `destination_indices` from
            // `render_method_submit_volatile_per_node`).
            //
            // Static materials use offset 0 (single slot, never
            // overwritten with per-object data). Animated materials
            // get one slot per draw instance.
            //
            // **Engine fidelity note (P6.3 — 2026-05-13):** the engine
            // technically has TWO separate cbuffers — `_ParametersPS`
            // (pool 0x6F) and `_ParametersVS` (pool 0x6E), each sized
            // independently from `pass.pixel_parameters_size` /
            // `pass.vertex_parameters_size`. Protomorph merges them
            // into this single binding with all of `rmt2.float_constants`
            // in source order.
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });
        // Per-draw decal constants UBO (DecalConstants). Has dynamic
        // offset so a single buffer can host ring-buffered per-decal
        // payloads + the per-`draw_indexed` offset picks each decal's
        // slice. VERTEX_FRAGMENT — sprite_corner is read in the VS.
        if let Some(slot) = self.decal_constants_slot {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: slot,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::halo::render::render_decals::DecalConstantsGpu>() as u64,
                    ),
                },
                count: None,
            });
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("halo_material_bgl_per_variant"),
            entries: &entries,
        })
    }

    /// Per-binding-sampler BGL — one sampler entry per unique
    /// `SamplerKey` in `sampler_map`, plus the texture and cbuffer
    /// entries. Engine-faithful sampler routing per [`emit_wgsl_per_binding`].
    pub fn build_bgl_per_binding(
        &self,
        device: &wgpu::Device,
        sampler_map: &crate::halo::render_methods::material_samplers::MaterialSamplerMap,
    ) -> wgpu::BindGroupLayout {
        let filterable = wgpu::TextureSampleType::Float { filterable: true };
        let mut entries = Vec::with_capacity(
            self.textures.len() + sampler_map.unique_keys.len() + 2,
        );
        // Texture bindings (same as legacy path).
        for tb in &self.textures {
            let view_dimension = if tb.is_cube {
                wgpu::TextureViewDimension::Cube
            } else {
                wgpu::TextureViewDimension::D2
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: tb.slot,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension,
                    sample_type: filterable,
                },
                count: None,
            });
        }
        // Per-unique-key samplers — typically 2-4 per material.
        for idx in 0..sampler_map.unique_keys.len() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: self.per_binding_sampler_slot(idx),
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        // Material cbuffer at `cbuffer_slot`. VERTEX_FRAGMENT — VS
        // shaders (foliage wind etc.) read this too.
        // `has_dynamic_offset: true` — Path B per-object cbuffer pool.
        // See `build_bgl_per_variant`'s cbuffer entry for the engine
        // analog (`g_ps_parameters_data` global pool).
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.per_binding_cbuffer_slot(sampler_map),
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });
        // Per-draw decal constants UBO — rmd only.
        if let Some(slot) = self.per_binding_decal_constants_slot(sampler_map) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: slot,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::halo::render::render_decals::DecalConstantsGpu>() as u64,
                    ),
                },
                count: None,
            });
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("halo_material_bgl_per_binding"),
            entries: &entries,
        })
    }

    /// Look up a texture binding slot by Halo parameter name.
    pub fn find(&self, name: &str) -> Option<&TextureBindingSlot> {
        self.textures.iter().find(|t| t.name == name)
    }
}

fn is_bitmap_param(p: &ResolvedParameter) -> bool {
    use blam_tags::render_method::RenderMethodParameterType;
    if !matches!(p.parameter_type, RenderMethodParameterType::Bitmap) {
        return false;
    }
    // Filter out parameters whose ParameterSource isn't actually
    // bitmap-shaped — defensive against schema drift.
    matches!(
        &p.source,
        ParameterSource::Inline(ResolvedValue::Bitmap(_)) | ParameterSource::Extern(_),
    )
}

/// Classify a Bitmap parameter as cube vs 2D, matching how Halo's
/// runtime decides the same question:
///
/// - **Authored bitmap** (`ParameterSource::Inline`): the bitm tag's
///   `type` field is source-of-truth (`bitmap_group_get_bitmap_hardware_format
///   @ 0x18041af20`). Read the tag and check `is_cube()`.
/// - **Engine extern** (`ParameterSource::Extern`): match on the
///   `RenderMethodExtern` enum value — the same routing Halo's runtime
///   uses (`render_method_submit_dynamic_cubemap` only fires for the
///   two `TextureDynamicEnvironmentMap*` variants).
///
/// No string compare on parameter names — the rmop's typed extern
/// enum is structurally meaningful, the user-facing param name is not.
fn classify_bitmap_param(
    p: &ResolvedParameter,
    tags_root: &std::path::Path,
) -> bool {
    use blam_tags::file::TagFile;
    use blam_tags::render_method::RenderMethodExtern as Ext;
    match &p.source {
        ParameterSource::Inline(ResolvedValue::Bitmap(b)) => {
            if b.bitmap_path.is_empty() {
                return false;
            }
            let bitm_path = blam_tags::paths::resolve_tag_path(tags_root, &b.bitmap_path, "bitmap");
            if let Ok(tag) = TagFile::read(&bitm_path) {
                if let Ok(bitmap) = blam_tags::bitmap::Bitmap::new(&tag) {
                    if let Some(image) = bitmap.image(b.bitmap_index.max(0) as usize) {
                        return image.is_cube();
                    }
                }
            }
            false
        }
        ParameterSource::Extern(ext) => matches!(
            ext,
            Ext::TextureDynamicEnvironmentMap0 | Ext::TextureDynamicEnvironmentMap1
        ),
        _ => false,
    }
}

/// Sampling intent per Halo parameter name. Bump/normal data is authored
/// as Bc5 SNORM (linear) and must NOT be sRGB-decoded → `ForceLinear`;
/// every other param follows the bitmap's authored curve (`FollowCurve`).
/// The bump override is a safety net for the rare mis-tagged bump map whose
/// curve says gamma.
pub fn sample_intent_for_param(name: &str) -> crate::halo::render::SampleIntent {
    use crate::halo::render::SampleIntent;
    let n = name.to_ascii_lowercase();
    if n.starts_with("bump_") || n.contains("_bump_") || n.ends_with("_bump") {
        SampleIntent::ForceLinear
    } else {
        SampleIntent::FollowCurve
    }
}

/// Pick a fallback texture view per parameter name when the material
/// doesn't carry the bitmap (optional rmop slot, missing override,
/// etc.). White for diffuse-like, default-normal for bump-like, black
/// for emissive/changecolor, black-cube for environment.
pub fn fallback_view_for_param(
    shared: &crate::halo::render::shared::SharedResources,
    name: &str,
    is_cube: bool,
) -> wgpu::TextureView {
    if is_cube {
        // Temporarily reverted to 1×1×6 black: routing cube extern
        // slots through rasg's `DefaultDynamicCubeMap` is engine-
        // correct in principle, but our spec / fresnel / env-intensity
        // plumbing hasn't caught up — binding the real sky cube blew
        // out structures and made dark rocks look wet-shiny. Restore
        // the rasg path once the dominant-light + exposure setup is
        // fully wired (see `feedback_riverworld_darkness_is_lighting_setup.md`).
        return shared.fallback_textures.black_cube_view.clone();
    }
    let n = name.to_ascii_lowercase();
    if n.starts_with("bump_") || n.contains("_bump_") || n.ends_with("_bump") {
        shared.fallback_textures.default_normal_view.clone()
    } else if n.starts_with("self_illum") || n == "change_color_map" {
        shared.fallback_textures.black_view.clone()
    } else if n.contains("detail") {
        // Unbound detail maps are multiplied by DETAIL_MULTIPLIER
        // (4.59479) in the albedo combine — white (1.0) blows albedo
        // 5-7× too bright. Use the engine's detail-neutral default
        // (~0.218 linear = 1/DETAIL_MULTIPLIER); see `default_detail`.
        shared.fallback_textures.detail_view.clone()
    } else {
        shared.fallback_textures.white_view.clone()
    }
}
