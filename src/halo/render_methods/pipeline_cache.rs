//! Per-VariantKey pipeline cache for the Halo forward pipeline.
//!
//! Each rmsh's `options` vector picks one variant per rmdf category
//! (albedo, bump_mapping, …). The assembler turns `(resolved, options,
//! entry_point)` into a complete WGSL string + cbuffer layout; this
//! module wraps the compile + RenderPipeline creation so passes can
//! share pipelines across all materials that resolve to the same
//! variant. Same VariantKey → same WGSL → same pipeline AND same
//! cbuffer layout (the rmop chain is fully determined by the option
//! vector, so layouts deduplicate cleanly).

use std::collections::HashMap;
use std::rc::Rc;

use blam_tags::render_method::{ResolvedCbuffer, ResolvedRenderMethod};

use crate::halo::render_methods::{
    assemble, cbuffer::CbufferLayout, material_bindings::MaterialBindings, CategoryChoices,
    HaloEntryPoint, VariantKey,
};
use crate::halo::geometry::ModelVertex;
use crate::halo::render::shared::SharedResources;

/// What the renderer needs per material at draw + upload time. The
/// pipeline drives `set_pipeline`; the layout drives material-uniform
/// serialization. The bindings map drives material bind group
/// construction (slot lookup by parameter name). The bgl drives bind
/// group creation. Refcounted so all materials sharing a VariantKey
/// also share these resources.
pub struct VariantArtifacts {
    pub pipeline: wgpu::RenderPipeline,
    pub layout: CbufferLayout,
    pub bindings: MaterialBindings,
    /// Per-material sampler dedup map — parallel-indexed unique
    /// `SamplerKey`s + binding-name → key-index map. Drives bind group
    /// construction so each `BitmapBinding`'s authored
    /// `BitmapAddressMode` / `BitmapFilterMode` is honored.
    pub sampler_map:
        crate::halo::render_methods::material_samplers::MaterialSamplerMap,
    pub material_bgl: wgpu::BindGroupLayout,
}

/// wgpu's `min_uniform_buffer_offset_alignment` worst case (the spec
/// guarantees 256; device-specific values may be smaller but 256 is
/// always valid). Each material cbuffer slot is padded up to a
/// multiple of this value so dynamic offsets `(slot_idx * slot_size)`
/// satisfy wgpu validation.
pub const MATERIAL_SLOT_ALIGNMENT: usize = 256;

/// Maximum number of placement instances per model that share an
/// animated material's cbuffer pool. Each placement gets one slot
/// (`instance_within_model` indexes here). Sized to cover the
/// highest-instance-count single-model placement in shipping
/// scenarios (shrine has 209 scenery placements split across many
/// models; per-model max is well under 256). If a scenario exceeds
/// this, the load path panics loudly — bump the constant.
pub const MAX_INSTANCES_PER_MODEL: usize = 256;

/// Lazily-populated map from `VariantKey` → artifacts. Owned by
/// [`crate::halo::render::Renderer`].
pub struct HaloPipelineCache {
    artifacts: HashMap<VariantKey, Rc<VariantArtifacts>>,
}

impl HaloPipelineCache {
    pub fn new() -> Self {
        Self { artifacts: HashMap::new() }
    }

    /// Get-or-build the (pipeline, layout) pair for this `(resolved,
    /// options, entry_point)`. Repeated lookups with the same key hit
    /// the cache (no recompile). The first call's `resolved` shapes
    /// the layout; subsequent calls' `resolved` is ignored on the
    /// cache-hit path (different rmshes with the same option vector
    /// resolve to the same rmop chain → same parameters → same layout).
    pub fn ensure(
        &mut self,
        shared: &SharedResources,
        resolved: &ResolvedRenderMethod,
        cbuffer: &ResolvedCbuffer,
        choices: &CategoryChoices,
        entry_point: HaloEntryPoint,
        tags_root: &std::path::Path,
    ) -> Rc<VariantArtifacts> {
        let key = VariantKey::from_resolved(resolved, entry_point, choices);
        if let Some(a) = self.artifacts.get(&key) {
            return a.clone();
        }
        eprintln!(
            "[halo_pipeline] new variant {:?} group={} choices={:?} (cache size → {})",
            entry_point,
            std::str::from_utf8(&resolved.group_tag.to_be_bytes()).unwrap_or("????"),
            choices.pairs(),
            self.artifacts.len() + 1,
        );
        let artifacts = build_variant(shared, resolved, cbuffer, choices, entry_point, tags_root);
        let rc = Rc::new(artifacts);
        self.artifacts.insert(key, rc.clone());
        rc
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }
}

fn build_variant(
    shared: &SharedResources,
    resolved: &ResolvedRenderMethod,
    cbuffer: &ResolvedCbuffer,
    choices: &CategoryChoices,
    entry_point: HaloEntryPoint,
    tags_root: &std::path::Path,
) -> VariantArtifacts {
    let assembled = assemble(resolved, cbuffer, choices, entry_point, tags_root);
    let shader = shared.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("halo_albedo_variant"),
        source: wgpu::ShaderSource::Wgsl(assembled.wgsl.into()),
    });

    // Build per-variant material BGL from the rmop's bitmap parameter
    // list. Replaces `shared.material_bgl` (the v1 hardcoded 8-slot
    // rmsh-only layout). Faithful to dllcache's per-rmt2 sampler
    // register assignment.
    let material_bgl = assembled
        .bindings
        .build_bgl_per_binding(&shared.device, &assembled.sampler_map);

    let pipeline_layout = shared.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("halo_forward_layout"),
        bind_group_layouts: &[
            &shared.camera_bgl,
            &shared.model_bgl,
            &material_bgl,
            &shared.node_matrices_bgl,
        ],
        immediate_size: 0,
    });

    // Pipeline state for transparent subclasses. Mirrors the engine
    // `c_rasterizer::set_alpha_blend_mode_no_cache @ 0x18066E930` 11-mode
    // table, keyed by the rmdf `blend_mode` option NAME (the same name
    // the engine's `e_alpha_blend_mode` enum slots align with). Documented
    // in `reference_halo_blend_modes.md`.
    // Determine whether this variant needs transparent pipeline state.
    // Engine semantics: a rmsh material with `blend_mode != opaque`
    // classifies into the transparency pass (`get_is_transparent`);
    // own-pipeline subclasses (rmd today; rmhg/rmlv/etc. later) are
    // always transparent. The blend_mode option NAME drives the
    // blend-state translator regardless of subclass.
    let group_fourcc = resolved.group_tag.to_be_bytes();
    let blend_mode = choices.get_or("blend_mode", "opaque");
    // rmd/rmhg are always transparent (rmhg per its alias of rmsh, but
    // halograms exclusively render in the transparency pass — they're
    // additively/alpha-blended HUD/UI overlays).
    let is_always_transparent = matches!(&group_fourcc, b"rmd " | b"rmhg");
    let is_rmsh_transparent =
        matches!(&group_fourcc, b"rmsh" | b"rmcs") && blend_mode != "opaque";
    let is_transparent_subclass = is_always_transparent || is_rmsh_transparent;
    let blend = if is_transparent_subclass {
        let mode_for_state = if is_always_transparent && blend_mode == "opaque" {
            // Always-transparent subclass with no blend_mode choice
            // (or default "opaque" from stub). Fall back to alpha_blend.
            "alpha_blend"
        } else {
            blend_mode
        };
        Some(blend_state_for_mode(mode_for_state))
    } else {
        None
    };

    // MRT layouts per pass / entry-point:
    //   Albedo entry-point        → MRT[0] = `_surface_accum_HDR` (Rg11b10Ufloat),
    //                                MRT[1] = normal G-buffer    (Rgba8Unorm)
    //   Static-lighting entries   → MRT[0] = `lighting_base`  (Rg11b10Ufloat,
    //                                engine `_surface_screenshot_composite_depth`),
    //                                MRT[1] = `hdr_dark`      (Rg11b10Ufloat,
    //                                engine `_surface_screenshot_composite_cubemap`,
    //                                bloom-feed source). Engine
    //                                `convert_to_render_target.hlsl:38` writes
    //                                LDR + DARK per fragment; on MCC PC
    //                                `DARK_COLOR_MULTIPLIER = 1.0` so RT1 == RT0
    //                                content-wise.
    //   Transparent subclasses    → MRT[0] = `lighting_base`, single target.
    let color_target_lighting = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba16Float,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    });
    // RT1 in the SL pass — same format/blend as RT0 so the engine's
    // RT0/RT1-equal-on-PC invariant holds when we copy the same color
    // out of the fragment to both locations.
    let color_target_lighting_dark = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba16Float,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let color_target_albedo = Some(wgpu::ColorTargetState {
        // engine `_surface_accum_HDR` — same format as `lighting_base`
        // so the albedo→SL copy-blit is a 1:1 texel copy. See
        // `IntermediateTargets::albedo_view` doc-comment.
        format: wgpu::TextureFormat::Rgba16Float,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let color_target_normal = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let albedo_targets = [color_target_albedo, color_target_normal];
    let static_lighting_targets =
        [color_target_lighting.clone(), color_target_lighting_dark.clone()];
    // Transparency rpasses (player_view::render_transparents,
    // render_water) attach `lighting_base` + `hdr_dark` as 2 MRTs to
    // match the 2-output AccumPixel WGSL emitted from the entry-point
    // shaders. wgpu requires pipeline targets length == rpass color
    // attachments length. On PC `DARK_MULT=1.0` so the 2nd target's
    // content matches the 1st verbatim — engine-faithful per
    // `convert_to_render_target_premultiplied_alpha`.
    let transparent_targets = [color_target_lighting, color_target_lighting_dark];
    // `_pass_post_albedo` decals — engine `c_decal_system::render_all`
    // attaches `_surface_post_HDR` (normal buffer in our setup) as
    // RT1 and sets `color_write_enable(0, 7)` so the decal PS writes
    // RT0 RGB only. We approximate by:
    //   RT0 = albedo_view (Rg11b10Ufloat), alpha-blended from rmd's
    //         `blend_mode` choice, write_mask = RGB (engine's
    //         color_write 0=7 — the alpha channel is reserved for
    //         spec_mask reads in SL).
    //   RT1 = normal_view (Rgba8Unorm), write_mask = empty (the
    //         basic decal pass doesn't touch the normal buffer;
    //         bump_modulate's second pass is deferred).
    //
    // RT0 write_mask — engine `c_decal::render @ 0x18039B100` gates on
    // the decal-DEFINITION's `def_flags & 1` (NOT the rmt2's specular
    // option):
    //   if (definition->m_flags & 1) {
    //       set_color_write_enable(0, 15);                  // RT0 RGBA
    //       set_separate_alpha_blend_mode(_separate_alpha_blend_to_constant);
    //       set_blend_factor(specular_multiplier);
    //   } else {
    //       set_color_write_enable(0, 7);                   // RT0 RGB only
    //   }
    // The rmt2's `specular=modulate` option just makes the SHADER
    // capable of producing spec-modulated alpha; the runtime gate is
    // the per-decal-definition flag. Per the fail-loud detector at
    // `render_decals.rs:681`, no palette in our cyberdyne/riverworld/
    // shrine/snowbound corpus sets `def_flags & 1`, so the engine
    // always takes the RGB-only branch. Use RGB-only here too — the
    // previous `specular=modulate` gate wrote `faded.w` (bitmap alpha)
    // into RT0.w, overwriting the rock's spec_mask and producing
    // blown-out specular at decal pixels (riverworld rockblend
    // observed 2026-05-17). When a future scenario surfaces
    // `def_flags & 1`, the fail-loud eprintln fires and we add the
    // per-decal spec_modulate variant. See [[project_decal_phase1_2026_05_17]].
    let decal_albedo_write_mask = wgpu::ColorWrites::COLOR;
    let decal_albedo_color_rt0 = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba16Float,
        blend,
        write_mask: decal_albedo_write_mask,
    });
    // RT1 (normal G-buffer) — engine `c_decal::render @ 0x18039B100`
    // ONLY masks RT1 (`set_color_write_enable(1, 0)`) when the palette
    // has `def_flags & 2` (bump_modulate) set, which triggers a separate
    // 2nd draw with `_alpha_blend_alpha_blend` to write the bump
    // independently from the shader's main blend_mode. For ALL OTHER
    // pre_lighting decals (the common case — shrine + snowbound
    // observed with `def_flags=0`), RT1's mask is INHERITED from prior
    // state (default RGBA-enabled), and the PS's
    // `convert_to_albedo_target_no_srgb` writes the bump-packed normal
    // alpha-blended against the rock's normal via the shader's blend
    // state.
    //
    // We approximate the engine's single-pass-with-RT1 path for the
    // common bumped variants. For `bump_mapping=leave` (scorch decals,
    // sand mounds — bump=0 → packed (0.5,0.5,0.5)), writing to RT1
    // would corrupt the underlying rock normal, so we mask RT1 entirely
    // there. The bump_modulate-specific 2nd draw isn't ported (no map
    // in our scope hits `def_flags & 2`).
    //
    // RT1 always uses `_alpha_blend_alpha_blend` regardless of the
    // shader's blend_mode — normals must NOT be multiplied (would
    // halve the encoded direction) or additively summed. This is what
    // engine's bump_modulate 2nd pass enforces explicitly via
    // `set_alpha_blend_mode(_alpha_blend_alpha_blend)`; we apply it
    // unconditionally to RT1 in the single-pass path.
    let decal_bump_writes_rt1 = matches!(
        choices.get("bump_mapping"),
        Some("standard") | Some("standard_mask"),
    );
    let decal_albedo_color_rt1 = if decal_bump_writes_rt1 {
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: Some(blend_state_for_mode("alpha_blend")),
            // RGB only — preserve RT1.w (rock's albedo.w companion
            // unused by our SL path but kept for engine-faithfulness).
            write_mask: wgpu::ColorWrites::COLOR,
        })
    } else {
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        })
    };
    let decal_albedo_targets = [decal_albedo_color_rt0, decal_albedo_color_rt1];
    let targets: &[Option<wgpu::ColorTargetState>] = match entry_point {
        // DecalObject mirrors DecalAlbedo's MRT shape — object-attached
        // rmd materials write to the G-buffer via the rmd PS body, same
        // as scene pre-lighting decals. Engine path: `render_albedo_decals`
        // → object's skinned ModelVertex VS → rmd default_ps →
        // convert_to_albedo_target_no_srgb (RT0=albedo, RT1=bump).
        HaloEntryPoint::DecalAlbedo | HaloEntryPoint::DecalObject => &decal_albedo_targets,
        // Transparent subclasses (rmd/rmhg, alpha-blended rmsh/rmcs) skip
        // the G-buffer + write directly to accum_HDR with blend.
        HaloEntryPoint::Decal if is_transparent_subclass => &transparent_targets,
        HaloEntryPoint::StaticSh
        | HaloEntryPoint::StaticPerPixel
        | HaloEntryPoint::StaticShPerVertex
        | HaloEntryPoint::StaticPrtAmbient
        | HaloEntryPoint::StaticVertexColor
            if is_transparent_subclass => &transparent_targets,
        HaloEntryPoint::Albedo => &albedo_targets,
        // Standard SL entries write the lit color to accum_HDR.
        HaloEntryPoint::StaticSh
        | HaloEntryPoint::StaticPerPixel
        | HaloEntryPoint::StaticShPerVertex
        | HaloEntryPoint::StaticPrtAmbient
        | HaloEntryPoint::StaticVertexColor => &static_lighting_targets,
        // Decal entry without transparency is unsupported in this branch
        // (handled above when transparent). Panic so we surface any new
        // dispatch we forgot to map.
        HaloEntryPoint::Decal => panic!(
            "rmd Decal entry reached MRT picker as non-transparent — \
             unexpected combination; check classify_transparency.",
        ),
    };

    // Per-vertex SH entry needs a SECOND vertex buffer carrying the
    // interpolated SH probe stream (`PerVertexShVertex`). Other
    // entries use only the primary `ModelVertex` buffer.
    //
    // Decals use `rasterizer_vertex_world` (44 B) — the engine
    // `c_decal::write_mesh_fragment` output. Completely different
    // attribute layout from ModelVertex (no skinning/lightmap UVs,
    // explicit normal/tangent/binormal as Snorm16x4).
    use crate::halo::decal::writer::RasterizerVertexWorld;
    use crate::halo::geometry::{PerVertexShVertex, PrtAmbientVertex, SkyVertColorVertex};
    let vertex_buffers_pv: [wgpu::VertexBufferLayout<'_>; 2] =
        [ModelVertex::layout(), PerVertexShVertex::layout()];
    // PRT Ambient mirrors the engine's slot-2 binding: `ModelVertex`
    // primary stream + a single-f32-per-vertex transfer stream.
    let vertex_buffers_prt: [wgpu::VertexBufferLayout<'_>; 2] =
        [ModelVertex::layout(), PrtAmbientVertex::layout()];
    // Vertex-color lighting (sky): primary `ModelVertex` + the
    // `SkyVertColorVertex` stream (slot 1, location 12) carrying the
    // per-vertex baked sky color.
    let vertex_buffers_vc: [wgpu::VertexBufferLayout<'_>; 2] =
        [ModelVertex::layout(), SkyVertColorVertex::layout()];
    let vertex_buffers_default: [wgpu::VertexBufferLayout<'_>; 1] =
        [ModelVertex::layout()];
    let vertex_buffers_decal: [wgpu::VertexBufferLayout<'_>; 1] =
        [RasterizerVertexWorld::layout()];
    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = match entry_point {
        HaloEntryPoint::StaticShPerVertex => &vertex_buffers_pv,
        HaloEntryPoint::StaticPrtAmbient => &vertex_buffers_prt,
        HaloEntryPoint::StaticVertexColor => &vertex_buffers_vc,
        HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo => &vertex_buffers_decal,
        // DecalObject = rmd PS body + OBJECT skinned ModelVertex VS.
        // Engine dispatches via `render_object_contexts(_default, ...)`
        // which uses the object's normal vertex stream (same layout as
        // every other object pass).
        HaloEntryPoint::DecalObject => &vertex_buffers_default,
        HaloEntryPoint::Albedo
        | HaloEntryPoint::StaticSh
        | HaloEntryPoint::StaticPerPixel => &vertex_buffers_default,
    };

    let pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("halo_forward_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            // Decal entry points emit indexed TRIANGLE STRIP per the
            // engine — `c_decal::build_mesh_fragment_recursive` writes
            // a strip with degenerate-start `[ss, ss, ss+1, ...]` join
            // pattern (mesh_builder.rs:205-235), and engine
            // `c_decal::render @ 0x18039B100` calls
            // `draw_indexed_primitive` whose `c_rasterizer_index_buffer
            // ::m_primitive_type` is 5 (= triangle strip). Floating
            // decals use the same topology via
            // `draw_primitive_up(_primitive_type_triangle_strip, ...)`.
            // All other entry points keep TriangleList.
            // Scene decals (`Decal`, `DecalAlbedo`) emit TriangleStrip
            // — engine `c_decal::build_mesh_fragment_recursive` writes
            // degenerate-strip joins. Object decals (`DecalObject`)
            // share the object mesh's index buffer which is always
            // TriangleList. All other entries: TriangleList.
            topology: if matches!(entry_point, HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo) {
                wgpu::PrimitiveTopology::TriangleStrip
            } else {
                wgpu::PrimitiveTopology::TriangleList
            },
            // wgpu requires `strip_index_format` set when topology is
            // a strip variant (so primitive-restart values match the
            // index buffer's int width).
            strip_index_format: if matches!(entry_point, HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo) {
                Some(wgpu::IndexFormat::Uint16)
            } else {
                None
            },
            front_face: wgpu::FrontFace::Ccw,
            // Per-subclass cull mode mirrors dllcache `set_cull_mode` per pass:
            //   structure render_albedo / render_static_lighting @ 0x18068ff90 / 0x180690060
            //     → _cull_mode_cw (back-face cull)
            //   c_decal_system::render_all @ 0x180399310 → _cull_mode_off
            //   c_transparency_renderer::render @ 0x1806cdb70 → _cull_mode_off
            //   c_water_renderer::render_shading @ 0x180693e90 → _cull_mode_cw
            //   foliage leaf cards → 2-sided per reference_foliage_fx_blueprint
            // Transparent subclasses (rmd, rmhg, or rmsh blend_mode!=opaque)
            // drop culling. Foliage is 2-sided. Everything else cuts back faces.
            cull_mode: if is_transparent_subclass || matches!(&group_fourcc, b"rmfl") {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            // Depth-write mirrors engine z_buffer_mode per pass:
            //   render_albedo @ 0x18068ff90 → _z_buffer_mode_write
            //   render_static_lighting @ 0x180690060 → _z_buffer_mode_read
            //   c_transparency_renderer::render @ 0x1806cdb70 → _z_buffer_mode_read
            // Only the Albedo pre-pass writes depth; SL reads the depth
            // the albedo pass already wrote (depth_compare GreaterEqual under
            // reverse-Z = engine's EQUAL semantic). Writing depth from SL
            // would Z-fight against itself across the 2-pass split AND
            // corrupt depth for downstream transparents / decals.
            // Decal entry points MUST NOT write depth. Engine
            // `c_player_view::render_albedo @ 0x18068ad36` flips
            // `set_z_buffer_mode(_z_buffer_mode_read)` BEFORE calling
            // `render_albedo_decals` + `c_decal_system::render_all` —
            // test only, no write. If decals write their (depth-biased)
            // depth into the buffer, the subsequent SL pass's
            // `GreaterEqual` test rejects every rock-SL fragment at
            // decal pixels (rock's true depth < biased decal depth in
            // reverse-Z). The pixel keeps the pre-SL seed blit
            // (decal-modified albedo) with NO lighting, NO bumps, NO
            // spec — exactly the "flat gray blot" symptom from
            // shrine/snowbound 2026-05-17.
            depth_write_enabled: matches!(entry_point, HaloEntryPoint::Albedo),
            // GreaterEqual (reverse-Z) so the static_lighting pass passes
            // its depth test against the IDENTICAL depth values written
            // by the preceding albedo pass. Greater would reject every
            // fragment (X > X is false) and leave lighting_base black.
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: Default::default(),
            // Decal pipelines need a depth-bias toward the camera so
            // their fragments win the z-test against the BSP surface
            // they project onto. Engine `c_rasterizer::set_z_buffer_mode
            // (_z_buffer_mode_decals)` @ 0x18066ec60 reads:
            //   g_decal_depth_bias        @ 0x181186e08 = -5e-6
            //   g_decal_slope_depth_bias  @ 0x181186e04 = -0.5
            // then applies for FP z-buffer:
            //   d3d_depth_bias_int       = -bias * -1.0 * 2^24 = (int)83.886 = 83
            //   d3d_slope_scale_depth_bias = -slope * -1.0 = 0.5
            // (the `* -1.0` is the engine's FP/reverse-Z sign flip;
            // POSITIVE D3D11 DepthBias under reverse-Z pushes depth
            // larger = closer to camera = wins GreaterEqual). Match
            // engine exactly so decal fragments consistently win the
            // depth test against the underlying BSP surface — without
            // this, ULP-level interpolation mismatches between the
            // decal mesh and the rock cause some triangles to win and
            // others to lose ("some look ok, some look bad"; observed
            // 2026-05-17 user A/B on riverworld BFS-mesh decals).
            bias: if matches!(
                entry_point,
                HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo | HaloEntryPoint::DecalObject,
            ) {
                wgpu::DepthBiasState {
                    constant: 83,
                    slope_scale: 0.5,
                    clamp: 0.0,
                }
            } else {
                wgpu::DepthBiasState::default()
            },
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    VariantArtifacts {
        pipeline,
        layout: assembled.cbuffer,
        bindings: assembled.bindings,
        sampler_map: assembled.sampler_map,
        material_bgl,
    }
}

/// Map a Halo rmdf `blend_mode` option name to a `wgpu::BlendState`,
/// mirroring `c_rasterizer::set_alpha_blend_mode_no_cache @ 0x18066E930`
/// (port 13372). The engine `e_alpha_blend_mode` enum slots align 1:1
/// with the rmdf options[] order; protomorph uses the resolved option
/// NAME so dispatch is name-keyed rather than enum-slot-keyed.
///
/// See `reference_halo_blend_modes.md` for the full mapping with engine
/// provenance. Modes not surfaced by any rmdf (motion_blur_static /
/// motion_blur_inhibit) are not handled — those are invoked by
/// motion-blur post-process passes, not render methods.
///
/// Color-write masking (color_write_enable=7 / RGB-only) is set at
/// rpass time by the transparency renderer, not in the pipeline state.
pub fn blend_state_for_mode(mode_name: &str) -> wgpu::BlendState {
    use wgpu::{BlendComponent, BlendFactor as F, BlendOperation as O, BlendState};
    let color = match mode_name {
        // Engine `_alpha_blend_opaque` — no blending; the PS output
        // overwrites the color target. Engine would never reach this
        // branch for a transparent-classified material (opaque rmsh's
        // classify into the opaque pass), but if we do see it, REPLACE
        // (= One/Zero/Add) matches engine's "BlendEnable=0" semantics.
        "opaque" => BlendComponent::REPLACE,
        // SRC=ONE, DST=ONE, ADD — straight additive.
        "additive" => BlendComponent {
            src_factor: F::One, dst_factor: F::One, operation: O::Add,
        },
        // SRC=DST_COLOR, DST=ZERO — output is multiplied with what's
        // there, the framebuffer side contributes nothing else.
        "multiply" => BlendComponent {
            src_factor: F::Dst, dst_factor: F::Zero, operation: O::Add,
        },
        // SRC=SRC_ALPHA, DST=ONE_MINUS_SRC_ALPHA — standard alpha blend.
        "alpha_blend" => BlendComponent {
            src_factor: F::SrcAlpha, dst_factor: F::OneMinusSrcAlpha,
            operation: O::Add,
        },
        // SRC=DST_COLOR, DST=SRC_COLOR — 2× multiply.
        "double_multiply" => BlendComponent {
            src_factor: F::Dst, dst_factor: F::Src, operation: O::Add,
        },
        // SRC=ONE, DST=ONE_MINUS_SRC_ALPHA — premultiplied alpha.
        // Engine collapses pre_multiplied_alpha and multiply_add to
        // the same D3D state; the PS handles the multiply_add semantic
        // by encoding `(color * dst, color)` in its output.
        "pre_multiplied_alpha" | "multiply_add" => BlendComponent {
            src_factor: F::One, dst_factor: F::OneMinusSrcAlpha,
            operation: O::Add,
        },
        // SRC=ONE, DST=ONE, MAX.
        "maximum" => BlendComponent {
            src_factor: F::One, dst_factor: F::One, operation: O::Max,
        },
        // SRC=DST_ALPHA, DST=ONE.
        "add_src_times_dstalpha" => BlendComponent {
            src_factor: F::DstAlpha, dst_factor: F::One, operation: O::Add,
        },
        // SRC=SRC_ALPHA, DST=ONE.
        "add_src_times_srcalpha" => BlendComponent {
            src_factor: F::SrcAlpha, dst_factor: F::One, operation: O::Add,
        },
        // SRC=ONE_MINUS_SRC_ALPHA, DST=SRC_ALPHA — inverse alpha blend.
        "inv_alpha_blend" => BlendComponent {
            src_factor: F::OneMinusSrcAlpha, dst_factor: F::SrcAlpha,
            operation: O::Add,
        },
        other => panic!(
            "[protomorph] unsupported blend_mode '{other}' — see \
             `reference_halo_blend_modes.md` for the canonical 11-mode \
             list. Add the mapping to `blend_state_for_mode` if it's a \
             real Halo mode, or audit the rmsh's `blend_mode` choice \
             resolution if the name looks wrong."
        ),
    };
    // Engine `set_alpha_blend_mode_no_cache @ 0x18066E930` packs each
    // mode as a single 32-bit D3D11_RENDER_TARGET_BLEND_DESC; decoding
    // the table (additive, multiply, alpha_blend, double_multiply,
    // pre_multiplied/multiply_add, maximum, add_src_times_dstalpha,
    // add_src_times_srcalpha, inv_alpha_blend) shows the alpha channel
    // ALWAYS uses the same src/dst/op as RGB. The only exception is
    // `motion_blur_static`, which isn't reachable from a render method.
    // Mirror that here.
    BlendState { color, alpha: color }
}

/// Build a minimal-shape `CategoryChoices` for stub materials (load
/// failures). All categories pick the no-op / simplest variant so the
/// assembler always finds a ported fragment. Used when an rmsh fails
/// to load — the renderer falls back to this.
pub fn stub_choices() -> CategoryChoices {
    let pairs: &[(&str, &str)] = &[
        ("albedo",              "default"),
        ("bump_mapping",        "off"),
        ("alpha_test",          "none"),
        ("specular_mask",       "no_specular_mask"),
        ("material_model",      "diffuse_only"),
        ("environment_mapping", "none"),
        ("self_illumination",   "none"),
        ("parallax",            "off"),
    ];
    let mut sorted: Vec<(String, String)> = pairs.iter().map(|(c, o)| (c.to_string(), o.to_string())).collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    CategoryChoices::from_pairs(sorted)
}
