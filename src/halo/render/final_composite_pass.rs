//! Engine `c_screen_postprocess::copy_accumulation_target @ 0x1806B71E0`
//! + `c_screen_postprocess::copy(51, _surface_accum_HDR → _surface_display)`.
//!
//! Two-stage final pipeline:
//!
//!   1. `record()` — runs the engine shader-18 (`final_composite_base`)
//!      port: reads `lighting_base` (LDR scene composite) + `aux_bloom`,
//!      applies hue/sat + contrast + cubic tone curve + 3D-LUT color
//!      grading + DoF blend, writes the result into
//!      `intermediates.accum_hdr_view` (Rg11b10Ufloat — engine
//!      `_surface_accum_HDR` in its post-composite role).
//!   2. (caller injects `render_lightshafts(accum_hdr ↔ accum_hdr)` here)
//!   3. `record_blit_to_swapchain()` — engine `copy(51)` equivalent:
//!      plain passthrough blit from `accum_hdr_view` to the swapchain.
//!      Hardware applies sRGB encoding on swapchain write. Glyphon HUD
//!      text is rendered on the swapchain afterwards so text colors
//!      stay in sRGB space.

use crate::halo::render::{
    sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    tex_entry,
};
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as TextColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;

/// Mirrors `final_composite.wgsl`'s `CompositeParams` cbuffer.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CompositeParams {
    /// `ps_postprocess_hue_saturation_matrix` — column-major 4×4.
    /// Identity = no color shift.
    hue_saturation_matrix: [[f32; 4]; 4],
    /// `(contrast.x, _, _, _)`. Halo default = 1.0 (no contrast change).
    contrast: [f32; 4],
    /// `(cg_blend_factor.x, _, _, _)`. 0 = use cg0 only.
    cg_blend_factor: [f32; 4],
    /// E1 DoF — `(focus_depth_ndc, half_width_ndc, max_blur, _)`.
    /// `max_blur ≤ 0` disables (sharp passthrough). Engine cbuffers
    /// 0x1E0000 (depth_constants) + 0x1E0001 (depth_constants2) folded
    /// into a single vec4 here.
    dof_focus: [f32; 4],
    /// Engine `tone_curve_constants` cbuffer (slot 0x1D0001), uploaded
    /// per-frame by `copy_accumulation_target` from
    /// `c_screen_postprocess::x_settings_internal.m_tone_curve` /
    /// `.m_tone_curve_white_point`. HLSL convention per
    /// `final_composite_registers_fx.hlsl`:
    /// `(.x = max, .y = linear, .z = quadratic, .w = cubic)` — the
    /// 4 coefficients of the Horner-form polynomial
    /// `result = ((c·.w + .z)·c + .y)·c` evaluated on the saturation-
    /// clamped input `clamped = min(blend, .x)`.
    ///
    /// Engine computation per `copy_accumulation_target` (with
    /// `WHITE = m_tone_curve_white_point`):
    /// ```
    /// k = 1.4938016 / WHITE
    /// if m_tone_curve:
    ///     (.x, .y, .z, .w) = (WHITE, k * 1.0041494, 0.0, -0.15 * k³)
    /// else:
    ///     (.x, .y, .z, .w) = (65535, k, 0.0, 0.0)   // brightness scale only
    /// ```
    /// IDA labels these writes split across `tone_curve_constants.n[2]`
    /// / `.n[3]` + `depth_constants.n[0]` / `.n[1]` because the C source
    /// uses two stack-local var names that the upload-time
    /// `set_shader_constant(0x1D0001, &tone_curve_constants)` reads as
    /// one contiguous 16-byte block. The ONLY ordering of the 4 written
    /// values that yields a valid tonemap polynomial (`result(WHITE) =
    /// 1.0` saturation) is `(.x = WHITE, .y = scaled_linear, .z = 0, .w
    /// = scaled_cubic)` — confirmed by plugging defaults `WHITE=1` into
    /// Horner: `-0.5·1³ + 0 + 1.5·1 = 1.0`. See
    /// `/tmp/divergence_tracker.md` D-415 for the full reverse-engineer
    /// trail.
    tone_curve_constants: [f32; 4],
    /// Engine `intensity` cbuffer (slot 0x1D0000) = `(natural, bloom,
    /// bling, persist)` — the shader-18 (`copy_target.hlsl`) weighted
    /// combine of the separate scene/bloom/bling/persist buffers. Engine
    /// caller passes `(1.0, m_bloom_intensity, m_bling_intensity, 0.5)`;
    /// protomorph bakes bloom_intensity into the bloom pyramid (so bloom
    /// weight is 1.0 here) and omits persist (no persist buffer ported),
    /// leaving `(1.0, 1.0, bling_scale, 0.0)`.
    intensity: [f32; 4],
}

impl CompositeParams {
    /// Identity defaults — no hue/sat shift, no contrast change, DoF off,
    /// tone curve disabled (pure brightness scale at default WHITE=1.0).
    /// Replaced per-frame by `set_composite_params` from real settings.
    pub(crate) const IDENTITY: Self = Self {
        hue_saturation_matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        contrast: [1.0, 0.0, 0.0, 0.0],
        cg_blend_factor: [0.0, 0.0, 0.0, 0.0],
        dof_focus: [0.0, 0.0, 0.0, 0.0],
        // Engine `copy_accumulation_target` ELSE branch for WHITE=1.0:
        // (65535, 1.4938016, 0, 0) — saturation effectively unbounded,
        // linear-only brightness scale of 1.4938016.
        tone_curve_constants: [65535.0, 1.4938016, 0.0, 0.0],
        // natural=1, bloom=1 (intensity baked in pyramid), bling=0
        // (inactive), persist=0 (not ported).
        intensity: [1.0, 1.0, 0.0, 0.0],
    };
}

/// Compute the engine `tone_curve_constants` cbuffer payload.
/// Mirrors the IF/ELSE branch in `copy_accumulation_target` keyed on
/// `c_screen_postprocess::x_settings_internal.m_tone_curve`.
///
/// At engine default `WHITE = 1.0, m_tone_curve = true` this returns
/// `(1.0, 1.5000, 0.0, -0.5000)` — the canonical Halo 3 cubic tonemap.
pub(crate) fn compute_tone_curve_constants(
    tone_curve_enabled: bool,
    white_point: f32,
) -> [f32; 4] {
    // Guard against zero white-point (would div-by-zero in `k` and
    // produce inf cubic). Engine doesn't bother but the static init we
    // ship with is `1.0` so this is purely defensive.
    let safe_white = if white_point > 1.0e-6 { white_point } else { 1.0e-6 };
    let k = 1.4938016 / safe_white;
    if tone_curve_enabled {
        // Engine IF branch (m_tone_curve = true):
        //   tone_curve_constants[2] = WHITE              → .x = max clamp
        //   tone_curve_constants[3] = k * 1.0041494      → .y = scaled linear
        //   depth_constants[0]      = 0                  → .z = quadratic
        //   depth_constants[1]      = -0.15 * k³         → .w = scaled cubic
        // Mapping rationale: only this ordering yields a saturating
        // tonemap (`result(WHITE) = 1.0`). See struct doc above.
        [safe_white, k * 1.0041494, 0.0, -0.15 * k * k * k]
    } else {
        // Engine ELSE branch (m_tone_curve = false): no curve, just
        // linear brightness scaling.
        //   tone_curve_constants[2] = 65535              → .x = no clamp
        //   tone_curve_constants[3] = 1.4938016/WHITE    → .y = linear
        //   depth_constants[0]      = 0                  → .z = 0
        //   depth_constants[1]      = 0                  → .w = 0
        [65535.0, k, 0.0, 0.0]
    }
}

// Color grading 3D LUT lifecycle moved to `ScreenPostprocess` —
// matches Ares's `c_screen_postprocess::color_grading_init / _release /
// update_color_grading` ownership. This pass borrows the active +
// back views via `set_color_grading_views()`.

pub struct FinalCompositePass {
    /// Stage-1 pipeline: engine shader 18 (`final_composite_base`) →
    /// writes into `accum_hdr_view` (Rg11b10Ufloat).
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    /// Cached bind group, rebuilt by `set_bloom_view` whenever the
    /// `c_screen_postprocess` bloom output texture changes
    /// (currently: at startup + on resize).
    blit_bind_group: wgpu::BindGroup,
    /// `final_composite.wgsl` uniform — hue/saturation matrix +
    /// contrast scalar + cg_blend_factor. Uploaded once at init with
    /// identity defaults; future per-scenario `c_camera_fx_values`
    /// plumbing will write here each frame.
    composite_uniform_buffer: wgpu::Buffer,

    /// Stage-3 pipeline: engine `copy(51)` → reads `accum_hdr_view`,
    /// writes the swapchain. `postprocess_copy.wgsl` port; sRGB encoding
    /// on the swapchain happens via hardware on write.
    swapchain_blit_pipeline: wgpu::RenderPipeline,
    swapchain_blit_bgl: wgpu::BindGroupLayout,
    swapchain_blit_bind_group: wgpu::BindGroup,
    /// PostprocessParams uniform for the swapchain blit — only `scale`
    /// is read (other slots ignored). Filled with `scale = (1, 1, 1, 1)`
    /// at init; never modified afterwards.
    swapchain_blit_uniform_buffer: wgpu::Buffer,

    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    text_buffer: TextBuffer,
    hud_string: String,
}

impl FinalCompositePass {
    pub fn new(
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        gbuffer: &GBuffer,
        surface_format: wgpu::TextureFormat,
        bloom_view: &wgpu::TextureView,
        cg_active_view: &wgpu::TextureView,
        cg_back_view: &wgpu::TextureView,
        dof_blur_view: &wgpu::TextureView,
        star_view: &wgpu::TextureView,
    ) -> Self {
        let device = &shared.device;
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        let cg_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                tex_entry(0, filterable),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                tex_entry(2, filterable),
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                cg_tex_entry(5),
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),
                cg_tex_entry(7),
                sampler_entry(8, wgpu::SamplerBindingType::Filtering),
                // E1 DoF — half-res blurred HDR + scene depth.
                tex_entry(9, filterable),
                sampler_entry(10, wgpu::SamplerBindingType::Filtering),
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(12, wgpu::SamplerBindingType::Filtering),
                // Bling/star buffer (engine copy_target.hlsl sampler 3).
                tex_entry(13, filterable),
                sampler_entry(14, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let composite_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite_uniforms"),
            contents: bytemuck::bytes_of(&CompositeParams::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Tone-curve coefficients now live in the CompositeParams
        // cbuffer (uniform binding 4) and are recomputed each frame
        // from `screen_postprocess.settings_internal.m_tone_curve` /
        // `.m_tone_curve_white_point` via `compute_tone_curve_constants`.
        // The shader reads them at runtime — no compile-time bake.
        // Engine-faithful per `copy_accumulation_target`.
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../../assets/halo_shaders/final_composite.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_layout"),
            bind_group_layouts: &[&blit_bgl],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // Engine `_surface_accum_HDR` format. Tonemap output
                    // (cubic curve → ~[0, 1.5]) stays in linear HDR here
                    // so lightshafts can additive-blit on top before the
                    // sRGB encode on swapchain write.
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Stage 3: passthrough blit accum_hdr → swapchain.
        // Engine `c_screen_postprocess::copy(51, _surface_accum_HDR,
        // _surface_display, point, clamp, 1, 1, 1, 1)`. Shader 51 =
        // `postprocess_copy` (engine `copy.hlsl::default_ps`).
        let swapchain_blit_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("swapchain_blit_bgl"),
                entries: &[
                    tex_entry(0, filterable),
                    sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        // `postprocess_copy.wgsl` reads the shared postprocess uniform's
        // scale.xy. We pass scale = (1, 1, 1, 1) — plain passthrough. Reuse
        // `PostprocessUniforms` (same `postprocess_copy.wgsl` cbuffer) rather
        // than redeclaring an identical local struct.
        let swapchain_blit_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("swapchain_blit_uniforms"),
                contents: bytemuck::bytes_of(
                    &crate::halo::render::screen_postprocess::PostprocessUniforms {
                        pixel_size: [0.0; 4],
                        scale: [1.0, 1.0, 1.0, 1.0],
                        intensity_vector: [0.0; 4],
                        dark_color_multiplier: [0.0; 4],
                    },
                ),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let swapchain_blit_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../../assets/halo_shaders/postprocess_copy.wgsl"
        ));
        let swapchain_blit_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("swapchain_blit_layout"),
                bind_group_layouts: &[&swapchain_blit_bgl],
                immediate_size: 0,
            });
        let swapchain_blit_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("swapchain_blit_pipeline"),
                layout: Some(&swapchain_blit_layout),
                vertex: wgpu::VertexState {
                    module: &swapchain_blit_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &swapchain_blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let swapchain_blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("swapchain_blit_bg"),
            layout: &swapchain_blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intermediates.accum_hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: swapchain_blit_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intermediates.lighting_base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: composite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(cg_active_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(cg_back_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(dof_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(star_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
            ],
        });

        // --- Text (glyphon) ---
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut text_atlas = TextAtlas::new(device, &shared.queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut text_buffer = TextBuffer::new(&mut font_system, Metrics::new(16.0, 18.0));
        text_buffer.set_size(&mut font_system, Some(480.0), Some(200.0));

        Self {
            blit_pipeline,
            blit_bgl,
            blit_bind_group,
            composite_uniform_buffer,
            swapchain_blit_pipeline,
            swapchain_blit_bgl,
            swapchain_blit_bind_group,
            swapchain_blit_uniform_buffer,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            text_buffer,
            hud_string: String::new(),
        }
    }

    /// Rebuild the blit bind group when intermediate textures get
    /// recreated on resize. Note: only repoints `lighting_base`; the
    /// bloom_view is rebuilt separately via `set_bloom_view` after
    /// `ScreenPostprocess::resize`.
    pub fn rebind(
        &mut self,
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        gbuffer: &GBuffer,
        bloom_view: &wgpu::TextureView,
        cg_active_view: &wgpu::TextureView,
        cg_back_view: &wgpu::TextureView,
        dof_blur_view: &wgpu::TextureView,
        star_view: &wgpu::TextureView,
    ) {
        self.blit_bind_group = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intermediates.lighting_base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.composite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(cg_active_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(cg_back_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(dof_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(star_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                },
            ],
        });

        // Stage-3 swapchain blit reads `accum_hdr` — repoint on resize.
        self.swapchain_blit_bind_group =
            shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("swapchain_blit_bg"),
                layout: &self.swapchain_blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &intermediates.accum_hdr_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.swapchain_blit_uniform_buffer.as_entire_binding(),
                    },
                ],
            });
    }
}

impl RenderPass for FinalCompositePass {
    fn prepare(&mut self, ctx: &FrameContext) {
        let game = ctx.game;
        let p = game.camera.position;
        let f = game.camera.forward;
        self.hud_string = format!(
            "FPS: {:.1}\npos: ({:.3}, {:.3}, {:.3})\nyaw/pitch: ({:.1}°, {:.1}°)\nfwd: ({:.3}, {:.3}, {:.3})",
            game.fps_counter.display_fps,
            p.x, p.y, p.z,
            game.camera.rotation.x, game.camera.rotation.y,
            f.x, f.y, f.z,
        );
        self.text_buffer.set_text(
            &mut self.font_system,
            &self.hud_string,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.text_buffer.shape_until_scroll(&mut self.font_system, false);

        self.viewport.update(
            &ctx.shared.queue,
            Resolution {
                width: ctx.shared.config.width,
                height: ctx.shared.config.height,
            },
        );

        let text_area = TextArea {
            buffer: &self.text_buffer,
            left: 12.0,
            top: 8.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: ctx.shared.config.width as i32,
                bottom: ctx.shared.config.height as i32,
            },
            default_color: TextColor::rgb(255, 240, 0),
            custom_glyphs: &[],
        };

        self.text_renderer
            .prepare(
                &ctx.shared.device,
                &ctx.shared.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.viewport,
                [text_area],
                &mut self.swash_cache,
            )
            .unwrap();
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        // Stage 1 — `copy_accumulation_target` body: shader-18 port
        // composites lighting_base + bloom + cg + DoF into accum_hdr
        // (Rg11b10Ufloat). Lightshafts injection + swapchain blit are
        // handled by the caller via `record_blit_to_swapchain()`.
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("copy_accumulation_target"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.accum_hdr_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(&self.blit_pipeline);
        rpass.set_bind_group(0, &self.blit_bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    fn post_submit(&mut self) {
        self.text_atlas.trim();
    }
}

impl FinalCompositePass {
    /// Stage 3 — engine `c_screen_postprocess::copy(51,
    /// _surface_accum_HDR → _surface_display)`. Passthrough blit of
    /// `accum_hdr_view` to the swapchain; hardware sRGB encoding on
    /// write. Glyphon HUD text renders afterwards on the same rpass so
    /// it lives in sRGB space (NOT linear) and reads as authored.
    pub fn record_blit_to_swapchain(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &FrameContext,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("accum_hdr_to_swapchain"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(&self.swapchain_blit_pipeline);
        rpass.set_bind_group(0, &self.swapchain_blit_bind_group, &[]);
        rpass.draw(0..3, 0..1);

        // Text overlay on the swapchain (sRGB) — must run AFTER the
        // blit so it sits on top of the scene.
        self.text_renderer
            .render(&self.text_atlas, &self.viewport, &mut rpass)
            .ok();
    }
}

impl FinalCompositePass {
    /// Rebuild the blit bind group when surface views change. Called
    /// from `Renderer::resize` (bloom view) and `Renderer::load_scenario`
    /// (color grading swap pulls the new active/back views from
    /// `ScreenPostprocess`).
    pub fn set_bound_views(
        &mut self,
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        gbuffer: &GBuffer,
        bloom_view: &wgpu::TextureView,
        cg_active_view: &wgpu::TextureView,
        cg_back_view: &wgpu::TextureView,
        dof_blur_view: &wgpu::TextureView,
        star_view: &wgpu::TextureView,
    ) {
        self.rebind(
            shared, intermediates, gbuffer, bloom_view, cg_active_view, cg_back_view, dof_blur_view,
            star_view,
        );
    }

    /// Update the full `CompositeParams` cbuffer in one call. Mirrors
    /// the engine's per-frame `c_screen_postprocess::update_constants`
    /// upload — color-grading + DoF + hue/sat + tone curve all flow
    /// through one slot.
    ///
    /// `tone_curve_enabled` + `tone_curve_white_point` come from
    /// `c_screen_postprocess::x_settings_internal.m_tone_curve` /
    /// `.m_tone_curve_white_point`; this method threads them through
    /// `compute_tone_curve_constants` so the cbuffer always agrees with
    /// the engine `copy_accumulation_target` IF/ELSE math.
    pub fn set_composite_params(
        &self,
        queue: &wgpu::Queue,
        cg_blend_factor: f32,
        dof_focus: [f32; 4],
        tone_curve_enabled: bool,
        tone_curve_white_point: f32,
        bling_scale: f32,
        screen_grade: crate::halo::render::screen_effect::ScreenEffectGrade,
    ) {
        let mut params = CompositeParams::IDENTITY;
        // Screen-effect (sefc) color grade. `hue_saturation_matrix` = engine
        // `ps_postprocess_hue_saturation_matrix` (const 0x420002: hue/sat/
        // desaturate/color-filter, contrast_enhance folded in). `contrast.x` =
        // engine `ps_postprocess_contrast` (const 0x420005) which actually
        // carries the GAMMA — the shader applies `pow(luminance, contrast.x)`.
        // IDENTITY (matrix=I, gamma=1) when no global screen effect is active.
        params.hue_saturation_matrix = screen_grade.matrix;
        params.contrast[0] = screen_grade.gamma;
        params.cg_blend_factor[0] = cg_blend_factor;
        params.dof_focus = dof_focus;
        params.tone_curve_constants =
            compute_tone_curve_constants(tone_curve_enabled, tone_curve_white_point);
        // Engine `intensity` = (natural, bloom, bling, persist). bloom is
        // pre-baked in the pyramid (weight 1); bling = m_bling_intensity
        // (0 when inactive); persist not ported.
        params.intensity = [1.0, 1.0, bling_scale, 0.0];
        queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

}

/// Convert a world-space focus distance + half-width into the
/// `(focus_depth_ndc, half_width_ndc)` pair the composite shader's
/// DoF blend expects.
///
/// The renderer uses a REVERSE-Z infinite-far projection
/// (`Mat4::perspective_infinite_reverse_rh`, near=1), so the depth buffer
/// the composite shader samples stores `depth = near / z` (1 at the near
/// plane, →0 at infinity) — the same mapping the particle/water shaders
/// linearize as `near_clip / depth`. The DoF focus value MUST be in this
/// encoding to be comparable to the sampled depth (a prior forward-Z
/// `F*(z-N)/(z*(F-N))` form here was incompatible with the reverse-Z buffer).
/// `far` is unused for an infinite far plane; kept for caller symmetry.
pub fn dof_world_to_ndc(
    focus_world: f32,
    half_width_world: f32,
    near: f32,
    _far: f32,
) -> (f32, f32) {
    let focus_z = focus_world.max(near + 1e-3);
    // Reverse-Z infinite-far: depth_ndc = near / z.
    let focus_ndc = near / focus_z;
    // |d(depth_ndc)/dz| = near / z² → half-focus-band width in depth units.
    let dz = near / (focus_z * focus_z);
    let half_ndc = (half_width_world * dz).max(1e-6);
    (focus_ndc, half_ndc)
}
