//! `c_player_view` — the main game viewpoint. Mirrors Ares
//! `render/views/render_view.h:181-260` and dllcache `c_player_view::*
//! @ 0x180688ba0` etc.
//!
//! Class chain: `c_player_view : c_world_view : c_view`. We
//! implement both `View` and `WorldView` traits on the concrete
//! `PlayerView` struct, plus a `PlayerView` trait that mirrors
//! `c_player_view`'s OWN protected/public methods.

use crate::halo::camera::CameraUniforms;
use crate::game::GameState;
use crate::halo::geometry::{ModelUniforms, MAXIMUM_NUMBER_OF_MODEL_NODES};
use crate::halo::rasterizer::{SplitscreenRes, Surface};
use crate::halo::render::env_probe_pass::GpuSkyParams;
use crate::halo::render::render_cameras::RealRectangle2d;
use crate::halo::render::render_patchy_fog::PatchyFog;
use crate::halo::render::shared::{FrameContext, RenderPass};
use crate::halo::render::views::lights_view::LightsView;
use glam::{Mat4, Vec3};

/// Parse `PROTOMORPH_DOF=<focus_distance_world_units>` once.
/// `None` if unset / "0" / unparseable; `Some(distance)` enables the
/// E1 DoF debug toggle.
fn parse_dof_env() -> Option<f32> {
    use std::sync::OnceLock;
    static DOF: OnceLock<Option<f32>> = OnceLock::new();
    *DOF.get_or_init(|| match std::env::var("PROTOMORPH_DOF").ok().as_deref() {
        Some("0") | None => None,
        Some(s) => s.trim().parse::<f32>().ok().filter(|&v| v > 0.0),
    })
}


/// `PROTOMORPH_PATCHY_FOG_DEBUG=N` selects a debug mode for the shader
/// (1..6, see `patchy_fog.wgsl` for the table). 0 = normal output.
fn patchy_fog_debug_mode() -> f32 {
    use std::sync::OnceLock;
    static MODE: OnceLock<f32> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("PROTOMORPH_PATCHY_FOG_DEBUG")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
    })
}

/// Mirrors `c_player_view::s_camera_user_data` (Ares 28 bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraUserData {
    pub player_window_index: i32,
    pub player_window_count: i32,
    pub player_window_arrangement: i32,
    pub user_index: i32,
    pub controller_index: i32,
    pub splitscreen_res_index: SplitscreenRes,
    pub splitscreen_resolve_surface: Surface,
}

impl Default for SplitscreenRes {
    fn default() -> Self {
        SplitscreenRes::Default
    }
}
impl Default for Surface {
    fn default() -> Self {
        Surface::None
    }
}

/// Concrete struct mirroring `c_player_view`. Engine inherits
/// `c_world_view` → `c_view`; the trait scaffolding for those was
/// removed in the 2026-05-09 audit (it was a stub-only pass-through
/// no other type used). The actively-used fields below are
/// `c_player_view`'s own non-inherited data.
#[derive(Debug, Clone, Default)]
pub struct PlayerView {
    pub window_game_state: PlayerWindowGameState,
    pub render_exposure: f32,
    pub illum_render_scale: f32,
    pub observer_depth_of_field: ObserverDepthOfField,
    pub patchy_fog: PatchyFog,
    pub last_frame_motion_blur_state: MotionBlurState,

    /// Per-frame simple-light list. Engine: `c_player_view` composes
    /// `c_lights_view` as a member; we keep just that subview since
    /// the others (first-person, lightmap-shadows, occlusion,
    /// reflection, UI) were stub-only in protomorph.
    pub lights_view: LightsView,

    pub final_surface_window_rect: RealRectangle2d,
    pub camera_user_data: CameraUserData,
    pub stall_cpu_to_wait_for_gpu: bool,
}

/// Inherent methods of `c_player_view` — the protected and public
/// non-virtual methods declared on the class itself. These don't
/// participate in the View/WorldView trait hierarchy; they're
/// concrete to PlayerView. Bodies come from dllcache (addresses
/// noted per-method) in phases 4-12.
impl PlayerView {
    /// `c_player_view::render @ 0x180688ba0`. Per-frame orchestration
    /// — uploads camera/lighting/atmosphere uniforms, builds visibility
    /// draw lists, encodes shadow + env_probe + albedo + transparents +
    /// postprocess + composite passes, presents.
    ///
    /// Takes `&mut Renderer` for the per-pass renderers + shared
    /// resources. The PlayerView itself is split out via
    /// `mem::take` in `Renderer::render` (PlayerView's `Default` impl
    /// is the swap stand-in). This is the engine-faithful entry point;
    /// `Renderer::render` is now a thin shell.
    ///
    /// # Engine call sequence reference (P4 anchor)
    ///
    /// The engine sequence from `c_player_view::render @ 0x180688BA0`
    /// is decoded in `reference_engine_frame_loop_full.md`. Numbered
    /// op order:
    ///
    /// ```text
    ///  1. set_current_splitscreen_res
    ///  2. esram_push(_surface_depth_stencil, _surface_depth_stencil_read_only)
    ///     IF !suppress_render_scene:
    ///  3. render_method_clear_externs()                              // P5
    ///  4. get_starting_cluster + compute_cluster_weights
    ///  5. submit_attachments(objects/lights)                         // stub
    ///  6. set_depth_stencil_surface(DepthStencil)
    ///  7. update_water_part_list / animate_water
    ///  8. chud_draw_turbulence                                       // stub (P10)
    ///  9. screen_effect_sample                                       // stub (P10)
    /// 10. setup_camera_fx_parameters(exposure_boost)                 // stub
    /// 11. setup_cinematic_clip_planes                                // stub
    /// 12. clear_simple_light_draw_list / build_simple_light_draw_list
    /// 13. populate_atmosphere_parameters + set_constant
    /// 14. rasterizer_stipple_initialize
    /// 15. c_decal_system::submit_all
    /// 16. esram_push(accum_HDR, post_HDR)
    /// 17. render_albedo → returns bool                               // G-buffer pass
    /// 18. render_ssao
    ///     IF render_albedo returned true:
    /// 19. copy_to_dram + esram_pop / esram_push composite chain
    /// 20. set_using_albedo_sampler(1)                                // cb 0x33 entry 5
    /// 21. submit_distortions                                         // stub (P10)
    /// 22. setup_occlusion_state + lens_flares_submit_occlusions(conditional) // stub
    /// 23. render_static_lighting (if debug toggle)
    /// 24. render_lightmap_shadows + c_lights_view::render
    ///     IF render_default_sfx:
    /// 25. setup_targets_static_lighting → render_first_person(0)     // FP opaque (stub)
    /// 26. tron_effect::resolve (if tiling) → setup again
    /// 27. render_water + queue_patchy_fog
    /// 28. render_transparents → game_engine_render → apply_distortions  // stub
    /// 29. setup_targets_static_lighting_alpha_blend → render_first_person(1) // FP transparent (stub)
    /// 30. render_lens_flares + occlusion submit                      // stub (P10)
    /// 31. chud_generate_damage_flash_texture → chud_draw_screen_LDR  // stub
    /// 32. render_setup_window → render_weather_occlusion             // stub (P10)
    /// 33. postprocess_player_view (bloom + bling + composite)
    /// 34. UI / HUD / debug overlays                                  // partial
    /// 35. esram_pop + c_rasterizer::end
    /// ```
    ///
    /// Today's protomorph body interleaves many of these ops in a
    /// different order; the structural reordering to match engine
    /// flow is incremental, tracked in
    /// `project_engine_faithful_pipeline_plan_2026_05_12.md` Phases
    /// P5–P11.
    pub fn render_player_view(
        &mut self,
        renderer: &mut crate::halo::render::Renderer,
        game: &crate::game::GameState,
    ) {
        // Engine `render_method_clear_externs()` — first call inside
        // `c_player_view::render @ 0x180688BA0`. Engine PC behavior is
        // just an assertion walk (no actual clearing). Mirror as a true
        // no-op via the ExternState helper.
        renderer.rasterizer.extern_state.clear_externs();

        // P5.6 — establish the `actually_calc_albedo` cbuffer bit for
        // opaque SL passes. Engine flips this via `set_using_albedo_sampler`
        // calls scattered through the frame loop:
        //   - TRUE (use G-buffer) before opaque SL passes
        //   - FALSE (recompute) before transparent passes
        // Protomorph today calls TRUE once per frame here; opaque material
        // PSs read `Kernel5PS.entries[5].z == 0u` and use the G-buffer.
        // The per-pass flip to FALSE before transparents is a P5-followup
        // — transparent variants keep their WGSL-assembly-time pin to
        // `SL_USE_GBUFFER = false` for now.
        renderer.rasterizer.set_using_albedo_sampler(1);

        // P5.4 — seed `_ExposureVS/_ExposurePS/_MiscPS` cbuffers via
        // engine `setup_render_target_globals_with_exposure @
        // 0x180670AD0`. Writes 5 entries (see method docstring) — the
        // internal 0x280000 / 0x2D0000 / 0x280001 / 0x2D0001 / 0x2F0005
        // engine cbuffer slots. Protomorph shaders read the SEPARATE
        // `engine_exposure_buffer` (`@group(0) @binding(9)`) populated
        // by `upload_view_exposure` at frame top with the same cfxs
        // values, so visible delta is zero — but threading real values
        // here keeps the rasterizer state machine consistent with the
        // engine for any future shader that does sample slot 0x280000
        // directly. `HDR_target_stops = 0` since the Rg11b10Ufloat
        // accum target needs no encoding scale (engine's Rgba16F path
        // uses non-zero stops on Xbox).
        renderer.rasterizer.setup_render_target_globals_with_exposure(
            renderer.render_exposure,
            renderer.illum_render_scale,
            0.0,
            false,
        );

        renderer.rasterizer.flush_cbuffers();

        // P6.5 — per-frame animated cb13 (user) cbuffer re-upload.
        // Materials flagged `is_animated` (TagFunction with a time
        // term) get their cbuffer re-resolved at `game.total_time` and
        // the bytes written through to each variant's props_buffer.
        // Documented divergence from engine `update_constants @
        // 0x180685300`: engine walks `m_overlays` and updates only the
        // animated slot byte ranges; protomorph re-resolves and
        // rewrites the whole cbuffer. Same final bytes, more work. See
        // `BspGpu::update_animated_cbuffers` doc for the tradeoff.
        for bsp_gpu in &renderer.structure_renderer.bsps {
            bsp_gpu.update_animated_cbuffers(&renderer.shared.queue, game.total_time);
        }
        // Object-side animated cbuffers — engine path is the same
        // `update_constants @ 0x180685300` walker; the BSP/object split
        // is a protomorph storage detail (BSPs hold their materials in
        // `BspGpu`, objects in `GpuModel`).
        //
        // Path B per-instance: walk every placement. For each placement
        // with a model, find that model's animated materials and write
        // the placement's slot (`instance_within_model`) using an
        // `ObjectBoundContext` bound to the placement's
        // `header_index`. State-driven inputs (`battery_empty` etc.)
        // resolve through `object_get_function_value`, panicking in the
        // un-ported `<type>_compute_function_value` stub — the Phase 5
        // smoke-port trigger.
        //
        // For placements without `header_index` (test objects, sky),
        // fall back to `DefaultEvalContext` so named inputs return 0.0
        // (engine LABEL_82 fallback).
        use crate::halo::render_methods::animated::ObjectBoundContext;
        for (_obj_idx, obj) in game.objects.iter() {
            let Some(model_index) = obj.model_index else { continue };
            let Some(model) = renderer.models.get(model_index) else { continue };
            if model.animated_materials.is_empty() { continue; }
            let object_index = obj.header_index.unwrap_or(u32::MAX);
            let ctx = ObjectBoundContext {
                eval_time: game.total_time,
                object_index,
                owner_tag_index: u32::MAX,
            };
            for animated in &model.animated_materials {
                animated.write_to_slot(
                    &renderer.shared.queue,
                    obj.instance_within_model,
                    &ctx,
                );
            }
        }
        if let Some(sky) = renderer.sky_model.as_ref() {
            sky.update_animated_cbuffers(&renderer.shared.queue, game.total_time);
        }

        // Text prepare happens later with frame_ctx

        // Build sorted render list. Excludes `game.sky_object` — sky
        // dispatches through `c_object_renderer::submit_and_render_sky`
        // (with parallax-lock + per-mesh filtering), NOT through the
        // generic-object render_list which renders at authored world
        // coords. Engine equivalent: sky is added to the visibility
        // tree by `expand_sky_object` only when sky-passes are active,
        // not by the regular object-visibility submit.
        //
        // Frustum cull objects up-front. Each ObjectData has position +
        // scale; the model's `bounding_sphere.xyz` is the local-space
        // center, `.w` is the radius. Transform sphere center by object's
        // position + scale, then test against the camera frustum (5 planes,
        // no far). Engine equivalent: `c_visible_items` walker prunes
        // objects via region/cluster visibility — we don't have that yet,
        // so frustum is the safe minimum. On large MP maps with 200+ static
        // scenery objects, this drops 50-80% of per-frame draw calls.
        let render_list_frustum: [glam::Vec4; 5] = {
            let vp = game.camera.projection * game.camera.view;
            let row = |n: usize| {
                glam::Vec4::new(vp.x_axis[n], vp.y_axis[n], vp.z_axis[n], vp.w_axis[n])
            };
            let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
            [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2].map(|p| {
                let n = p.truncate().length().max(1e-6);
                p / n
            })
        };
        let object_in_frustum =
            |obj: &crate::halo::objects::ObjectData, model_idx: usize| -> bool {
                let Some(model) = renderer.models.get(model_idx) else { return true };
                // Never view-frustum-cull an object that carries transparent
                // parts. The engine submits object transparents via PVS-region
                // visibility (`submit_object_mesh_parts @ 0x1806E1BF0`), not the
                // view frustum — culling here would pop glass / FX scenery
                // (waterfalls, man cannons) as the bounding sphere center
                // crosses a frustum plane while geometry is still on screen.
                if model.materials.iter().any(|m| m.is_transparent) {
                    return true;
                }
                // World transform from the DATUM (engine `object_get_world_matrix`),
                // NOT the stale authored `obj.position`/`obj.scale`. For
                // parent-marker-attached objects (sky-attached scenery — the
                // bunkerworld dish, etc.) `set_world_transform` writes the
                // datum but never `obj.position`, so the store transform is the
                // pre-attachment placement and culling on it pops the object.
                // Fall back to the placement compose only for datum-less objects
                // (sky/test). Mirrors the model-uniform + shadow-caster paths.
                let world = obj
                    .header_index
                    .and_then(crate::halo::objects::object_header_data::world_matrix)
                    .unwrap_or_else(|| obj.model_matrix());
                let local_center = model.bounding_sphere.truncate();
                let world_center = world.transform_point3(local_center);
                let scale_max = world
                    .x_axis
                    .truncate()
                    .length()
                    .max(world.y_axis.truncate().length())
                    .max(world.z_axis.truncate().length())
                    .max(1e-3);
                let world_radius = model.bounding_sphere.w * scale_max;
                for plane in &render_list_frustum {
                    let s = plane.x * world_center.x
                        + plane.y * world_center.y
                        + plane.z * world_center.z
                        + plane.w;
                    if s < -world_radius {
                        return false;
                    }
                }
                true
            };
        renderer.render_list.clear();
        renderer.render_list.extend(
            game.objects
                .iter()
                .filter(|(idx, _)| game.sky_object.map(|s| s.0) != Some(idx.0))
                .filter_map(|(idx, obj)| {
                    let model_idx = obj.model_index?;
                    if !object_in_frustum(obj, model_idx) {
                        return None;
                    }
                    Some((idx, model_idx))
                }),
        );
        let mut render_list = std::mem::take(&mut renderer.render_list);

        // Upload camera uniforms
        let cam_uniforms = CameraUniforms {
            view: game.camera.view.to_cols_array_2d(),
            projection: game.camera.projection.to_cols_array_2d(),
        };
        renderer.shared.queue.write_buffer(
            &renderer.shared.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_uniforms),
        );


        // ============================================================
        // Phase I0c — engine-faithful per-frame visibility computation.
        //
        // Mirrors `c_player_view::compute_visibility @ 0x180689770` →
        // `render_visibility_camera_collection_compute @ 0x1806AB740`.
        //
        // Drives the full pipeline:
        //   1. scenario_location_from_point → camera cluster reference
        //   2. CVisibilityCollection::compute → projections walker
        //   3. CVisibleItems populated for downstream consumers
        //   4. Region pointer threaded through FrameContext so
        //      decorators (Phase I1) and other consumers can use
        //      `visibility_region_test_sphere` instead of ad-hoc culls
        // ============================================================
        if let Some(scenario_arc) =
            renderer.loaded_scenario.as_ref().map(std::sync::Arc::clone)
        {
            // Build observer via the engine-faithful free fn (mirrors
            // `observer_build_result_from_point_and_vectors @ 0x1803577F0`
            // followed by `observer_result_compute_parameters @ 0x18035B960`).
            let observer = game.camera.to_observer_result();

            let projection =
                crate::halo::render::render_visibility::RenderProjection::from_observer(
                    &observer,
                );

            // Camera cluster lookup (cached; recomputes only on
            // position change). Engine: `c_player_view`'s per-frame
            // `scenario_location_from_point` call.
            let camera_loc = renderer.camera_cluster_cache.update(
                &scenario_arc,
                blam_tags::math::RealPoint3d {
                    x: game.camera.position.x,
                    y: game.camera.position.y,
                    z: game.camera.position.z,
                },
            );

            // Engine's far is conventionally `camera.z_far`; protomorph
            // uses an infinite-reverse-RH projection so we pick a
            // scenario-scale value (riverworld is ~few hundred meters).
            const VISIBILITY_Z_FAR: f32 = 1024.0;

            crate::halo::render::render_visibility::render_visibility_camera_collection_compute(
                &observer,
                game.camera.near_clip,
                VISIBILITY_Z_FAR,
                &projection,
                camera_loc.cluster_reference,
                /* user_index */ 0,
                /* player_window_index */ 0,
                /* single_cluster_only */ false,
                /* debug_pvs_render_all */ false,
                &scenario_arc,
                &renderer.portal_activation,
                &mut renderer.camera_visibility,
                &mut renderer.visible_items,
            );

            let _ = camera_loc;
        }

        // Refresh `c_lighting_interface` from the scenario's sky tag
        // — same chain dllcache walks via `get_sun_constants_from_sky
        // @ 0x1803adcb0` + `setup_default_lighting`. The walker pulls
        // `sky.render_model.sky_lights[count-1]` (sun) and
        // `sky.render_model.default_lightprobe` (SH3 probe). Both come
        // straight from the tag — no analytical reconstruction.

        // Upload engine lighting cbuffers — dllcache slots 0x30
        // (`LightingPS` = ravi-packed SH3, binding 1) and 0x24
        // (dominant-light direction + intensity, binding 13). Bound at
        // `@group(0)` and read by every assembled Halo shader variant.
        let engine_lighting_ps =
            crate::halo::lighting::GpuEngineLightingPs::from_lighting_interface(
                &renderer.lighting_interface,
            );
        let dominant_light =
            crate::halo::lighting::GpuEngineDominantLight::from_lighting_interface(
                &renderer.lighting_interface,
            );
        renderer.shared.queue.write_buffer(
            &renderer.shared.engine_lighting_buffer,
            0,
            bytemuck::bytes_of(&engine_lighting_ps),
        );
        renderer.shared.queue.write_buffer(
            &renderer.shared.engine_dominant_light_buffer,
            0,
            bytemuck::bytes_of(&dominant_light),
        );

        // Upload model uniforms + bone matrices
        for (i, &(obj_idx, _model_idx)) in render_list.iter().enumerate() {
            let obj = game.objects.get(obj_idx);
            // Datum is the transform authority (engine object_get_world_matrix);
            // fall back to the placement compose for any header-less object.
            let world = obj
                .header_index
                .and_then(crate::halo::objects::object_header_data::world_matrix)
                .unwrap_or_else(|| obj.model_matrix());
            let model_u = ModelUniforms {
                model: world.to_cols_array_2d(),
            };
            let offset = (i * renderer.shared.model_stride) as u64;
            renderer.shared.queue.write_buffer(
                &renderer.shared.model_buffer,
                offset,
                bytemuck::bytes_of(&model_u),
            );

            // Per-object node (bone) bind-pose matrices come from the
            // model spec now (no animation system). Pack the model's
            // `node_local_matrices` (object-local node-world bind pose)
            // into the scratch staging buffer, then upload into this
            // object's slot in `node_matrices_buffer`. Skinned vertices
            // sample these via `model_u × node × vertex`.
            //
            // Borrow note: `renderer.models` (immut) and
            // `renderer.node_matrix_staging` (mut) are disjoint fields, so
            // the pack loop accesses both via the field paths directly.
            let mut nm_byte_len = 0usize;
            if let Some(mi) = obj.model_index {
                if let Some(gm) = renderer.models.get(mi) {
                    let nm = &gm.node_local_matrices;
                    nm_byte_len = nm.len().min(MAXIMUM_NUMBER_OF_MODEL_NODES) * 64;
                    for (j, mat) in nm.iter().take(MAXIMUM_NUMBER_OF_MODEL_NODES).enumerate() {
                        let off = j * 64;
                        renderer.node_matrix_staging[off..off + 64]
                            .copy_from_slice(bytemuck::cast_slice(&mat.to_cols_array()));
                    }
                }
            }
            if nm_byte_len > 0 {
                let nm_offset = (i * renderer.shared.node_matrices_stride) as u64;
                renderer.shared.queue.write_buffer(
                    &renderer.shared.node_matrices_buffer,
                    nm_offset,
                    &renderer.node_matrix_staging[..nm_byte_len],
                );
            }
        }

        // Atmosphere uniform — Phase 5 engine-faithful pipeline.
        //
        // Engine per-frame flow (`c_atmosphere_fog_interface`):
        //   compute_cluster_weights(starting, eye)
        //     → mutates per-setting `m_weight` / `m_effect_weight`
        //   <reset m_default_parameters>
        //   populate_atmosphere_parameters(starting, &m_default_parameters)
        //     → iterates settings × weight → accumulator
        //   set_constant(&m_default_parameters, fogged=1, vs=1)
        //     → writes 7-slot cbuffer at 0x290000 (VS) / 0x3F0005 (PS)
        //
        // Our `GpuAtmosphereData::from_weighted_parameters` consumes the
        // accumulator into slots 0-6 verbatim; slots 7-10 (sky-pass sun
        // disc / tint / horizon) source from the dominant setting until
        // Phase 5.3 separates them onto a dedicated sky cbuffer per
        // engine.
        //
        // For single-palette MP maps (every shipping MP scenario) the
        // MP fast path in compute_cluster_weights yields weight=1.0 on
        // exactly the setting the prior `resolve_atmosphere_setting_for_eye`
        // path picked → byte-identical cbuffer output. Multi-palette SP
        // maps light up once Phase 3b ships the seam-aware walker.
        let eye_pos: glam::Vec3 = game.camera.position.into();
        let atmosphere_payload = {
            let palette = &renderer.scenario_active_bsp_atmosphere_palette;
            let clusters = &renderer.scenario_active_bsp_clusters;
            let interface = &mut renderer.atmosphere_fog_interface;
            let sky_sun = renderer.scenario_sky_sun;
            let view_exposure = renderer.scenario_view_exposure;

            match renderer.scenario_sky_atmosphere.as_mut() {
                Some(sky_atm) if !sky_atm.atmosphere_settings.is_empty() => {
                    // Resolve eye cluster (engine `observer_build_result_from_point_and_vectors`
                    // → `c_observer::m_cluster_reference`; we use the cached
                    // active-BSP cluster bounds containment test).
                    let cluster_idx = crate::halo::render::env_probe_pass::find_cluster_for_eye(
                        clusters, eye_pos,
                    );
                    let cluster_atm_idx = cluster_idx
                        .and_then(|i| clusters.get(i))
                        .map(|c| c.atmosphere_index)
                        .unwrap_or(-1);
                    let starting_cluster = crate::halo::structures::clusters::ClusterReference {
                        bsp_index: 0, // single-active-BSP — multi-BSP needs scenario seam graph
                        cluster_index: cluster_idx.map(|i| i as i16).unwrap_or(-1),
                    };

                    // Phase 3c — per-setting weights.
                    interface.compute_cluster_weights(
                        sky_atm,
                        palette,
                        starting_cluster,
                        cluster_atm_idx,
                    );
                    // Phase 4 — accumulator (reset + populate). Engine reuses
                    // the same buffer frame-to-frame; we zero each frame
                    // so a stale prior-frame component (e.g., from a
                    // setting whose weight just dropped to 0) doesn't
                    // leak. Build into a local first to avoid simultaneous
                    // borrows of `interface.{populate,default_parameters}`.
                    let mut accum = crate::halo::render::atmosphere_fog_interface::WeightedAtmosphereParameters::default();
                    interface.populate_atmosphere_parameters(sky_atm, &mut accum, sky_sun);
                    interface.default_parameters = accum;

                    // Phase 5 — derive cbuffer. Sky-pass extras still
                    // sourced from the picked setting (Phase 5.3).
                    let picked_setting =
                        crate::halo::render::atmosphere_fog_interface::resolve_atmosphere_setting_index(
                            cluster_atm_idx,
                            palette,
                            sky_atm.atmosphere_settings.len(),
                        )
                        .and_then(|i| sky_atm.atmosphere_settings.get(i));
                    crate::halo::render::env_probe_pass::GpuAtmosphereData::from_weighted_parameters(
                        &interface.default_parameters,
                        picked_setting,
                        view_exposure,
                    )
                }
                _ if renderer.scenario_atmosphere.slot1_sun_int_norm_thickness[3] > 0.0 => {
                    // Fallback — pre-loaded primary-setting snapshot. Used
                    // when scenario load completed `scenario_atmosphere`
                    // but `scenario_sky_atmosphere` was dropped (shouldn't
                    // happen in normal flow).
                    renderer.scenario_atmosphere
                }
                _ => crate::halo::render::env_probe_pass::GpuAtmosphereData::neutral(),
            }
        };
        // Build the per-render-method atmosphere TABLE and upload it in one
        // write. Entry layout (256-B stride):
        //   [0]    default — the cluster-weighted blend computed above.
        //   [1]    disabled — slot1.w<0 sentinel (extinction=1, inscatter=0);
        //          engine `disable_atmosphere`, selected by `DontFogMe`.
        //   [2+i]  setting i = engine `set_atmosphere_constants_by_index(i)`:
        //          accumulate ONLY that setting at weight 1.0, derive cbuffer.
        // Sky meshes with `UseCustomSetting`/`DontFogMe` select entries via the
        // per-draw dynamic offset at camera_bgl binding 2 (see
        // `shared::atmosphere_offset_for`). This mirrors the engine's per-shader
        // atmosphere override in `render_method_submit_volatile_per_shader`.
        {
            use crate::halo::render::atmosphere_fog_interface::WeightedAtmosphereParameters;
            use crate::halo::render::env_probe_pass::GpuAtmosphereData;
            use crate::halo::render::shared::{
                ATMOSPHERE_DISABLED_OFFSET, ATMOSPHERE_SETTING_BASE_OFFSET, ATMOSPHERE_STRIDE,
                ATMOSPHERE_TABLE_ENTRIES, MAX_ATMOSPHERE_SETTINGS,
            };

            let n = std::mem::size_of::<GpuAtmosphereData>();
            let mut table = vec![0u8; (ATMOSPHERE_TABLE_ENTRIES * ATMOSPHERE_STRIDE) as usize];

            // [0] default.
            table[0..n].copy_from_slice(bytemuck::bytes_of(&atmosphere_payload));
            // [1] disabled.
            let mut disabled = atmosphere_payload;
            disabled.slot1_sun_int_norm_thickness[3] = -1.0;
            let d = ATMOSPHERE_DISABLED_OFFSET as usize;
            table[d..d + n].copy_from_slice(bytemuck::bytes_of(&disabled));

            // [2+i] per-setting. Out-of-range / absent → default.
            let view_exposure = renderer.scenario_view_exposure;
            let sky_sun = renderer.scenario_sky_sun;
            for i in 0..MAX_ATMOSPHERE_SETTINGS {
                let entry = renderer
                    .scenario_sky_atmosphere
                    .as_ref()
                    .and_then(|sky_atm| sky_atm.atmosphere_settings.get(i as usize))
                    .map(|setting| {
                        let mut accum = WeightedAtmosphereParameters::default();
                        renderer.atmosphere_fog_interface.accumulate_atmosphere_settings(
                            setting,
                            &mut accum,
                            1.0,
                            sky_sun,
                        );
                        GpuAtmosphereData::from_weighted_parameters(
                            &accum,
                            Some(setting),
                            view_exposure,
                        )
                    })
                    .unwrap_or(atmosphere_payload);
                let o = (ATMOSPHERE_SETTING_BASE_OFFSET + i * ATMOSPHERE_STRIDE) as usize;
                table[o..o + n].copy_from_slice(bytemuck::bytes_of(&entry));
            }

            renderer
                .shared
                .queue
                .write_buffer(&renderer.shared.atmosphere_buffer, 0, &table);
        }

        // Upload sky params for merged lighting+sky pass
        {
            let inverse_vp = (game.camera.projection * game.camera.view).inverse();
            let sky_params = GpuSkyParams {
                inverse_view_projection: inverse_vp.to_cols_array_2d(),
                camera_position: game.camera.position.into(),
                _pad: 0.0,
            };
            renderer.shared.queue.write_buffer(
                &renderer.shared.sky_params_buffer,
                0,
                bytemuck::bytes_of(&sky_params),
            );
        }

        // Acquire surface
        let output = match renderer.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                renderer.surface
                    .configure(&renderer.shared.device, &renderer.shared.config);
                renderer.render_list = render_list;
                return;
            }
            Err(e) => {
                eprintln!("Surface error: {e}");
                renderer.render_list = render_list;
                return;
            }
        };
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Build the per-frame BSP cluster_parts draw list. Mirrors
        // `c_structure_renderer::submit_visibility @ 0x18068E860`.
        // Engine uses `visibility_region_test_sphere` against the PVS
        // walker's output, but the walker (Phase E4) drops some
        // visible neighbour clusters until the BFS bug is traced —
        // fall back to the 5-plane Gribb-Hartmann camera frustum for
        // BSP clusters until then. Decorators stay on the region path
        // (their AABBs are tight, the region cull is conservative
        // there in our favour). Transparent BSP parts route into
        // `transparency_renderer`'s global pool.
        let view_proj_cull = game.camera.projection * game.camera.view;
        let row = |n: usize| {
            glam::Vec4::new(
                view_proj_cull.x_axis[n],
                view_proj_cull.y_axis[n],
                view_proj_cull.z_axis[n],
                view_proj_cull.w_axis[n],
            )
        };
        let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
        let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2];
        let bsp_frustum: [glam::Vec4; 5] = raw.map(|p| {
            let n = p.truncate().length().max(1e-6);
            p / n
        });
        renderer.structure_renderer
            .submit_visibility(&mut renderer.transparency_renderer, Some(&bsp_frustum));

        // `c_object_renderer::submit_visibility @ 0x1806e16d0`. v1
        // builds object_render_contexts from the legacy `render_list`
        // (one context per object, one mesh-part per (mesh, part)).
        // Full `c_visible_items` walker lands when F12 visibility
        // collection comes online.
        crate::halo::render::render_objects::submit_visibility_from_render_list(
            &mut renderer.object_renderer,
            &mut renderer.transparency_renderer,
            &render_list,
            &renderer.models,
            &game.objects,
        );


        // Sky transparents are NOT queued here — engine-faithful path
        // queues them inside `c_player_view::render_transparents` via
        // `c_object_renderer::submit_and_render_sky(2, …)` so they
        // render in their OWN sort batch (push_marker / sort / render /
        // pop_marker). See `render_objects::submit_and_render_sky`.

        let frame_ctx = FrameContext {
            shared: &renderer.shared,
            gbuffer: &renderer.gbuffer,
            intermediates: &renderer.intermediates,
            surface_view: &surface_view,
            models: &renderer.models,
            render_list: &render_list,
            game,
            frame_index: renderer.frame_counter,
            prev_view_projection: renderer.prev_vp,
            structure_renderer: &renderer.structure_renderer,
            object_renderer: &renderer.object_renderer,
            decorator_renderer: &renderer.decorator_renderer,
            sky_model: renderer.sky_model.as_ref(),
            sky_gpu: &renderer.sky_gpu,
            // Phase I0c: live engine-faithful visibility region built
            // above by `render_visibility_camera_collection_compute`.
            // Consumers (decorators, BSP cull, etc.) call
            // `visibility_region_test_*` instead of ad-hoc frustum culls.
            camera_visibility_region: renderer
                .loaded_scenario
                .as_ref()
                .map(|_| &renderer.camera_visibility.m_input.region),
            decal_gpu: if renderer.decal_gpu.is_empty() {
                None
            } else {
                Some(&renderer.decal_gpu)
            },
            decal_renderer: renderer
                .loaded_scenario
                .as_ref()
                .map(|s| &s.decal_renderer),
        };

        // Prepare phase — mutable pass access for uniform uploads.
        // Mirrors `c_player_view::render` setup; explicit per-pass
        // calls instead of the old generic scheduler.
        //
        // EnvProbePass disabled: its 6-face × scene + 5-mip cubemap
        // render runs every 8 frames (~30 render passes) but the
        // resulting cubemap is consumed by NOTHING — final_composite_pass
        // takes face_views as `_face_views` (unused), water has its own
        // env_cubemap path, rmsh materials fall back to black_cube_view.
        // MCC engine equivalent: cluster_cubemaps[] is empty so
        // c_dynamic_cubemap_sample::get_cluster_cubemap_index returns -1
        // and the path falls through to default_dynamic_cube_map.bitmap.
        // Recoups the 30+ render passes / 8 frames as pure perf win.
        // (no prepare for the albedo path — `PlayerView::render_albedo`
        // is self-contained: it opens the rpass + dispatches subsystems
        // inline. Engine equivalent: `c_player_view::render_albedo
        // @ 0x18068ABC0` similarly opens its rpass on entry.)
        if renderer.final_composite_pass.is_enabled(&frame_ctx) {
            renderer.final_composite_pass.prepare(&frame_ctx);
        }

        // Record phase — encode GPU commands. Order matches Halo's
        // `c_player_view::render @ 0x180688ba0`:
        //   shadow_generate → albedo → static_lighting → dynamic_lighting
        //   → ... → postprocess. Today our env_probe_pass covers
        //   sky, geometry_pass covers albedo+static_lighting+sky-pass-
        //   for-clusters, final_composite_pass does the postprocess
        //   blit. Phases 7c/8/9 split each into the canonical
        //   sub-passes.
        let mut encoder =
            renderer.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        // EnvProbePass record disabled — see prepare-phase comment above.
        // Per-frame wind globals upload (must run BEFORE the albedo
        // rpass opens — queue.write_buffer cannot run inside a pass).
        // Engine: equivalent to `s_wind_definition::update @ 0x180907F80`
        // called during `c_player_view::render` setup. Mutable access
        // to `decorator_renderer` for the ring-buffer state — done
        // here BEFORE frame_ctx aliasing reads kick in (frame_ctx
        // borrows `&renderer.decorator_renderer` immutably).
        //
        // NOTE: this looks redundant with the earlier &frame_ctx
        // construction at line ~370 — we drop frame_ctx, run the wind
        // update, and rebuild it. Cheap (it's just a struct of refs).
        drop(frame_ctx);
        renderer.decorator_renderer.update_per_frame_uniforms(
            &renderer.shared.queue,
            game.total_time,
        );
        // Rewrite the per-decal `DecalConstants` ring buffer for this
        // frame — engine `c_decal::render` writes `set_shader_constant
        // (0x140000, fade)` per draw; we batch them into one
        // `queue.write_buffer`. Distance fade evaluated from
        // `palette.distance_fade_range` against camera position.
        if let Some(loaded) = renderer.loaded_scenario.as_ref() {
            renderer.decal_gpu.update_decal_constants(
                &renderer.shared.queue,
                &loaded.decal_renderer,
                game.camera.position.into(),
            );
        }
        let frame_ctx = FrameContext {
            shared: &renderer.shared,
            gbuffer: &renderer.gbuffer,
            intermediates: &renderer.intermediates,
            surface_view: &surface_view,
            models: &renderer.models,
            render_list: &render_list,
            game,
            frame_index: renderer.frame_counter,
            prev_view_projection: renderer.prev_vp,
            structure_renderer: &renderer.structure_renderer,
            object_renderer: &renderer.object_renderer,
            decorator_renderer: &renderer.decorator_renderer,
            sky_model: renderer.sky_model.as_ref(),
            sky_gpu: &renderer.sky_gpu,
            // Phase I0c — see first FrameContext construction.
            camera_visibility_region: renderer
                .loaded_scenario
                .as_ref()
                .map(|_| &renderer.camera_visibility.m_input.region),
            decal_gpu: if renderer.decal_gpu.is_empty() {
                None
            } else {
                Some(&renderer.decal_gpu)
            },
            decal_renderer: renderer
                .loaded_scenario
                .as_ref()
                .map(|s| &s.decal_renderer),
        };

        // Sky model matrix is identity — engine-faithful for MCC, which
        // strips `render_sky_modify_node_matrices` to `return 1;`. Sky
        // meshes render at their authored world coords; nothing to upload.

        encoder.push_debug_group("render_albedo");
        self.render_albedo(
            &mut encoder,
            &frame_ctx,
            &mut renderer.transparency_renderer,
        );
        encoder.pop_debug_group();

        // `c_player_view::render_static_lighting @ 0x18068aea0` —
        // re-walks the geometry sorted by render_albedo, computes the
        // FINAL lit color into `lighting_base`. Engine: gated on
        // `render_debug_toggle_default_static_lighting`; we always run.
        encoder.push_debug_group("render_static_lighting");
        self.render_static_lighting(
            &mut encoder,
            &frame_ctx,
            &mut renderer.transparency_renderer,
        );
        encoder.pop_debug_group();

        // S1 — engine-faithful per-object shadow generate + apply.
        //
        // Engine path: `c_lightmap_shadows_view::submit_visibility_and_render
        // @ 0x1806BB7A0` walks each shadow-cast object, builds an OBB
        // ortho per caster, generates a tight 16..512 px shadow into
        // `_surface_shadow_1`, then runs `shadow_apply` to stamp the
        // world. Casters are objects ONLY — BSP clusters / instanced
        // geometry are baked into the lightmap statically, NOT into
        // the per-frame shadow target.
        //
        // S1-final-a: the unified-world ortho stop-gap is replaced by
        // a per-caster loop. Each object gets its own ortho.
        //
        // S1-final-b (this turn): per-caster bounds come from the model's
        // own bounding sphere (engine-faithful equivalent of
        // `object_get_bounding_sphere` — for objects without authored
        // bounds this is the AABB-derived `autogen` sphere computed at
        // upload time from `ModelMesh::vertices`). Center is transformed
        // through the object's world matrix so scaling + rotation are
        // applied. Attachment-tree expansion (engine
        // `render_lightmap_shadow_calculate_radius @ 0x82735870`) is
        // still deferred — pulls from `attached_bounds_radius` which
        // isn't populated yet.
        //
        // Resolution is still the full 512² (S1-final-c plumbs the
        // `(diameter_pixels × shadow_quality_lod / k_shadow_resolution)`
        // clamp).
        //
        // Sun direction — prefer authored sun from the active atmosphere
        // setting; fall back to the SH-derived
        // `lighting_interface.dominant_light_dir`. (Per-object cached
        // `object_get_cached_render_lighting()` deferred — needs an
        // object lighting cache that doesn't exist yet.)
        // Floor of 1.0 prevents depth-ortho self-acne for tiny objects:
        // a 0.05-radius caster with 3× expansion packs depth precision
        // into a 0.15-unit cube, well below f32 precision for the
        // depth-compare PCF — produces visible stripes on the casting
        // geometry itself. Engine effectively has the same floor via
        // the resolution-clamp's 16-pixel minimum, but our depth ortho
        // is built from the world-space radius directly.
        const CASTER_RADIUS_FALLBACK: f32 = 1.0;
        let light_dir = renderer
            .scenario_sky_atmosphere
            .as_ref()
            .and_then(|atm| atm.primary_setting())
            .map(|s| {
                let d = s.sun_direction();
                glam::Vec3::new(d.x, d.y, d.z)
            })
            .unwrap_or_else(|| {
                let dom = renderer.lighting_interface.dominant_light_dir;
                glam::Vec3::new(dom.x, dom.y, dom.z)
            });
        // Diagnostic toggle — `PROTOMORPH_NO_SHADOWS=1` skips all
        // generate + apply passes, leaving _surface_shadow_1
        // uninitialized + lighting_base unstamped. Useful for A/B
        // comparing the effect of shadow stamping on overall scene
        // brightness. Cached — env::var allocates a String per call.
        let shadows_enabled = {
            use std::sync::OnceLock;
            static ENABLED: OnceLock<bool> = OnceLock::new();
            *ENABLED.get_or_init(|| {
                std::env::var("PROTOMORPH_NO_SHADOWS")
                    .map(|v| v.is_empty() || v == "0")
                    .unwrap_or(true)
            })
        };
        if shadows_enabled && light_dir.length_squared() > 1e-6 && !render_list.is_empty() {
            // One-shot diagnostic — logs the shadow direction + per-caster
            // radius + caster counts on the FIRST frame after scenario
            // load so we can diagnose direction/coverage from the run log.
            // Load-then-swap: the load is a hot-cache READ on every
            // frame after the first, vs. swap which is always a write.
            static SHADOW_DIAG_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !SHADOW_DIAG_LOGGED.load(std::sync::atomic::Ordering::Relaxed)
                && !SHADOW_DIAG_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                let from_atmosphere = renderer.scenario_sky_atmosphere.is_some();
                eprintln!(
                    "[shadow] per-caster dir=({:.3},{:.3},{:.3}) source={} casters={}",
                    light_dir.x, light_dir.y, light_dir.z,
                    if from_atmosphere { "atmosphere.sun" } else { "lighting_interface.dominant" },
                    render_list.len(),
                );
            }

            // Camera-frustum cull — extract 5 lateral planes from
            // (proj × view) once, test each caster's bounding sphere
            // (radius × 3.0 to mirror `object_shadow_get_potential_bounds`)
            // before any per-caster matrix work. Engine equivalent:
            // `c_visibility_collection::sphere_is_visible @ 0x180392980`
            // called per object inside the camera pre-pass; survivors
            // populate `s_visible_items->root_objects`. We compress that
            // pre-pass into an inline filter at the shadow loop. Skips
            // the far plane — `perspective_infinite_reverse_rh` puts it
            // at infinity.
            let frustum_planes: [glam::Vec4; 5] = {
                let vp = game.camera.projection * game.camera.view;
                let row = |n: usize| {
                    glam::Vec4::new(
                        vp.x_axis[n],
                        vp.y_axis[n],
                        vp.z_axis[n],
                        vp.w_axis[n],
                    )
                };
                let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
                let raw = [
                    r3 + r0,  // left
                    r3 - r0,  // right
                    r3 + r1,  // bottom
                    r3 - r1,  // top
                    r3 + r2,  // near (or far for std-RH; doesn't matter — both lateral)
                ];
                raw.map(|p| {
                    let n = p.truncate().length().max(1e-6);
                    p / n
                })
            };
            let sphere_in_frustum = |center: glam::Vec3, radius: f32| -> bool {
                for plane in &frustum_planes {
                    let signed =
                        plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
                    if signed < -radius {
                        return false;
                    }
                }
                true
            };

            // Per-caster loop — one generate + one apply per object,
            // serialized through the shared `_surface_shadow_1` 512²
            // depth target. Engine equivalent: `c_lightmap_shadows_view::
            // submit_visibility_and_render` walks `root_objects` with
            // `flags & 2` and runs this exact serialized pattern.
            //
            // Filters mirror the engine cull chain (see
            // `project_research_shadow_visibility_2026_05_09.md`):
            //   STEP 0 (load-time): `model.casts_shadow` — gates on the
            //     obje tag flags + lightmap_shadow_mode + type matrix
            //     per `render_object_has_lightmap_shadow @ 0x180696EE0`.
            //   STEP 1 (per-frame): camera-frustum sphere cull on
            //     `(caster_center, caster_radius × 3)` — mirrors
            //     `object_shadow_get_potential_bounds @ 0x1806BAAD0`
            //     followed by `sphere_is_visible @ 0x180392980`.
            //
            // STEP 4 (per-OBB camera-frustum test via `obb_visible @
            // 0x1806BDF50`) and STEP 5 (parent_object_index / hidden
            // checks) are deferred; the sphere-×3 cull is conservative
            // enough that the OBB refinement is a future tightening.
            let mut culled_no_cast = 0usize;
            let mut culled_frustum = 0usize;
            let mut casters_rendered = 0usize;
            for (slot, &(obj_idx, model_idx)) in render_list.iter().enumerate() {
                let Some(model) = renderer.models.get(model_idx) else { continue };
                // STEP 0 — drop placements whose obje tag isn't shadow-
                // eligible. Most static scenery (lightmap_shadow_mode ==
                // default + type == scenery) lands here.
                if !model.casts_shadow {
                    culled_no_cast += 1;
                    continue;
                }
                // Engine `object_get_bounding_sphere` returns a world-
                // space sphere built from `render_model`'s bounds + the
                // object's transform. Our model bounding sphere is
                // model-space (offset, radius); transform the center
                // through the world matrix and scale the radius by the
                // matrix's max-axis scale. For unscaled objects (the
                // overwhelming majority), this collapses to (M·offset,
                // radius).
                let obj = game.objects.get(obj_idx);
                let model_matrix = obj
                    .header_index
                    .and_then(crate::halo::objects::object_header_data::world_matrix)
                    .unwrap_or_else(|| obj.model_matrix());
                let bs_local = model.bounding_sphere;
                let local_center = glam::Vec3::new(bs_local.x, bs_local.y, bs_local.z);
                let caster_center =
                    model_matrix.transform_point3(local_center);
                // World-scale extracted from the matrix's column lengths
                // (engine: `obj.scale` × `render_model_scale`). Use max
                // axis for the resolution-clamp's effective radius.
                let world_scale = model_matrix
                    .x_axis
                    .truncate()
                    .length()
                    .max(model_matrix.y_axis.truncate().length())
                    .max(model_matrix.z_axis.truncate().length())
                    .max(1e-3);
                let caster_radius =
                    (bs_local.w * world_scale).max(CASTER_RADIUS_FALLBACK);
                // Engine `object_shadow_get_potential_bounds @
                // 0x1806BAAD0`: shadow_bounds_radius = bounding_sphere_radius
                // × 3.0, same center. The 3× expansion captures shadow
                // receivers below/around the caster (ground beneath
                // objects on the floor) — without it, the OBB tightly
                // bounds caster geometry only and ground fragments fall
                // OUTSIDE the frustum. (Earlier attempt with model-space
                // AABB was correct in principle but most protomorph
                // models have degenerate uncompressed-position AABBs;
                // the engine's bounding-sphere path sidesteps that.)
                const SHADOW_RADIUS_FACTOR: f32 = 3.0;
                let shadow_radius = caster_radius * SHADOW_RADIUS_FACTOR;

                // STEP 1 — camera-frustum sphere cull on the inflated
                // shadow bounds. Cheaper than the engine's per-OBB test
                // but conservative (slightly more passes through than
                // strictly necessary). Drops the bulk of off-screen
                // casters before the per-pass setup cost.
                if !sphere_in_frustum(caster_center, shadow_radius) {
                    culled_frustum += 1;
                    continue;
                }
                casters_rendered += 1;

                // One-shot diagnostic: print first 10 casters' world centers
                // + radii so we can see whether matrices are bunching at
                // origin, scattered correctly, or have weird scales.
                // Load-then-fetch_add so we stop the atomic write after
                // the first 10 are logged (was running fetch_add per
                // caster forever).
                {
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static LOGGED: AtomicU32 = AtomicU32::new(0);
                    if LOGGED.load(Ordering::Relaxed) < 10
                        && let n = LOGGED.fetch_add(1, Ordering::Relaxed)
                        && n < 10
                    {
                        let pos = obj.position;
                        eprintln!(
                            "[shadow caster {}] obj_idx={:?} model_idx={} obj_pos=({:.2},{:.2},{:.2}) bs_local=(c=({:.3},{:.3},{:.3}), r={:.3}) world_scale={:.3} caster_center=({:.2},{:.2},{:.2}) caster_radius={:.3} shadow_radius={:.3}",
                            n, obj_idx, model_idx,
                            pos.x, pos.y, pos.z,
                            bs_local.x, bs_local.y, bs_local.z, bs_local.w,
                            world_scale,
                            caster_center.x, caster_center.y, caster_center.z,
                            caster_radius, shadow_radius,
                        );
                    }
                }
                let world_to_shadow =
                    crate::halo::render::shadow_apply_pass::build_caster_ortho(
                        caster_center,
                        glam::Vec3::splat(shadow_radius),
                        light_dir,
                    );
                let model_offset = (slot * renderer.shared.model_stride) as u32;

                // S1-final-c — per-caster resolution clamp. Distant /
                // small objects render into a small sub-rect of
                // _surface_shadow_1; near / large objects fill the full
                // 512². Engine `clamp(diameter_pixels × shadow_quality_lod,
                // 16, 512)`.
                //
                // DIAGNOSTIC: env-gated. The dynamic clamp produces a
                // sub_res that varies frame-to-frame as the camera moves
                // (distance ↔ pixel size), which makes the pillar's depth
                // silhouette get re-rendered at different resolutions
                // each frame — visually reads as the shadow "sliding into
                // place" when panning. Pinning sub_res=SHADOW_DEPTH_RESOLUTION
                // makes every caster fill the full 512² target, removing
                // the wobble at the cost of more shadow pixels rasterized
                // per caster (still cheap because the 132 casters/frame
                // cap is dominated by render-pass setup, not pixel work).
                let _cam_pos: glam::Vec3 = game.camera.position.into();
                // Cached — this is INSIDE the per-caster loop, so without
                // OnceLock we'd run env::var 5-15× per frame.
                let dynamic_res = {
                    use std::sync::OnceLock;
                    static ENABLED: OnceLock<bool> = OnceLock::new();
                    *ENABLED.get_or_init(|| {
                        std::env::var("PROTOMORPH_SHADOW_DYNAMIC_RES")
                            .map(|v| v == "1")
                            .unwrap_or(false)
                    })
                };
                let sub_res = if dynamic_res {
                    crate::halo::render::shadow_apply_pass::shadow_resolution_for_caster(
                        caster_center,
                        caster_radius,
                        _cam_pos,
                        game.camera.field_of_view,
                        renderer.shared.config.height,
                    )
                } else {
                    crate::halo::render::shared::SHADOW_DEPTH_RESOLUTION
                };

                // Generate — depth-only pass writes THIS object into
                // _surface_shadow_1 at its per-caster ortho. Other
                // casters' depth from prior iterations is overwritten by
                // the LoadOp::Clear inside `begin_generate_pass`. The
                // apply that immediately follows samples this caster's
                // depth before the next caster clears it.
                {
                    let mut rpass = renderer.shadow_apply.begin_generate_pass(
                        &renderer.shared,
                        &mut encoder,
                        &renderer.intermediates.shadow_depth_view,
                        world_to_shadow,
                        sub_res,
                    );
                    rpass.set_bind_group(1, &renderer.shared.model_bind_group, &[model_offset]);
                    for mesh in &model.meshes {
                        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        for part in &mesh.parts {
                            rpass.draw_indexed(
                                part.index_start..part.index_start + part.index_count,
                                0,
                                0..1,
                            );
                        }
                    }
                }

                // Apply — screen-space PCF stamp. Only fragments inside
                // this caster's shadow ortho clip volume get darkened;
                // everything else samples cleared depth and reads
                // `unshadowed=1` → stamp alpha=0 → no-op. Receiver normal
                // drives slope-bias + cosine falloff per engine.
                renderer.shadow_apply.apply_caster(
                    &renderer.shared,
                    &mut encoder,
                    &renderer.intermediates.lighting_base_view,
                    &renderer.intermediates.hdr_dark_view,
                    &renderer.gbuffer.depth_sample_view,
                    &renderer.intermediates.shadow_depth_view,
                    &renderer.gbuffer.normal_view,
                    frame_ctx.game.camera.view,
                    frame_ctx.game.camera.projection,
                    world_to_shadow,
                    light_dir,
                    /* opacity = */ 0.85,
                    sub_res,
                );
            }

            // One-shot summary: log how many of the ~150 placements
            // actually enter the generate+apply loop after culling.
            // Compare against the per-caster `[shadow] per-caster
            // casters=N` line — N is pre-cull, this is post-cull.
            static CULL_DIAG_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !CULL_DIAG_LOGGED.load(std::sync::atomic::Ordering::Relaxed)
                && !CULL_DIAG_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "[shadow] cull: total={} no_cast={} frustum={} rendered={}",
                    render_list.len(),
                    culled_no_cast,
                    culled_frustum,
                    casters_rendered,
                );
            }
        }

        // `c_screen_postprocess::render_ssao @ 0x1806b50e0` — modulates
        // `lighting_base` by ambient occlusion before water/transparents
        // sample the LDR snapshot. Reads depth + normal MRT (written by
        // render_albedo). No-op if the active cfxs has SSAO disabled.
        renderer.screen_postprocess.render_ssao(
            &renderer.shared,
            &mut encoder,
            renderer.scenario_camera_fx.as_ref().and_then(|c| c.ssao.as_ref()),
            frame_ctx.game.camera.view,
            frame_ctx.game.camera.field_of_view,
            frame_ctx.game.camera.aspect_ratio,
            frame_ctx.game.camera.near_clip,
            &renderer.gbuffer.depth_sample_view,
            &renderer.gbuffer.normal_view,
            &renderer.intermediates.lighting_base_view,
        );

        // `c_player_view::render_lights` — dynamic-lights pass.
        // Sits between render_static_lighting and render_water in
        // the canonical pass order. The player_view's lights_view
        // owns the simple-light array (`m_simple_lights[8]`); the
        // pass uploads the count + array to shader constant slot
        // 0x310000/0x310001 (Halo `submit_light_draw_list_to_shader
        // @ 0x1806c7e60`) and adds each light's contribution to
        // `lighting_base` via per-light proxy draws.
        //
        // v1 no-op: `LightsView` has the API + state but no game
        // system populates dynamic lights yet. Once a scenario sets
        // `lights_view.simple_light_count > 0` (Phase 12+), the
        // per-light draws plug in here.
        if self.lights_view.simple_light_count > 0 {
            encoder.push_debug_group("render_lights");
            // TODO: per-light proxy draws (sphere/cone) added when a
            // light shader / cbuffer slot land. Halo body in
            // `c_lights_view::render @ 0x1806c6040`.
            encoder.pop_debug_group();
        }

        // Snapshot the opaque-pass HDR target before water + transparency
        // mutate it. Engine equivalent: `_surface_screenshot_simple_resolve`
        // populated via `c_rasterizer::stretch_rect(_surface_screenshot_composite_depth,
        // _surface_screenshot_simple_resolve)` inside
        // `c_player_view::render_water @ 0x18068c120`. Water shaders
        // sample this as `tex_ldr_buffer` for refraction (extern 45 in
        // the DX9 path, per `reference_dllcache_water_pipeline.md`).
        // For now the snapshot is bound at `camera_bgl` slot 7 — Phase
        // A5 routes it through the rm** extern system.
        encoder.push_debug_group("scene_ldr_snapshot");
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &renderer.intermediates.lighting_base_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &renderer.intermediates.scene_ldr_snapshot_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: renderer.shared.config.width,
                height: renderer.shared.config.height,
                depth_or_array_layers: 1,
            },
        );
        encoder.pop_debug_group();

        // `c_player_view::render_water @ 0x18068c120` — water-shading
        // pass. Per dllcache flow, this sits between
        // `render_first_person(0)` and `render_transparents`, gated on
        // `c_water_renderer::update_water_part_list() || g_is_underwater`.
        // Two sub-passes inside: shading (`_entry_point_static_lighting_per_vertex`,
        // depth-test on, color RGBA) + shadow_generate (depth-write on,
        // color off). Phase 0: empty body — call site wired so future
        // phases can fill in (`reference_dllcache_water_pipeline.md`).
        encoder.push_debug_group("render_water");
        // `c_water_renderer::render_underwater_fog @ 0x180694080` runs
        // BEFORE `render_shading` per the dllcache `render_water`
        // sequence. Fullscreen quad fogs the scene LDR snapshot when
        // `g_is_underwater`; water surfaces draw on top.
        renderer.water_renderer.update_underwater_state(&frame_ctx);
        renderer.water_renderer.render_underwater_fog(&mut encoder, &frame_ctx);
        renderer.water_renderer.render_shading(&mut encoder, &frame_ctx);
        encoder.pop_debug_group();

        // A1 — Patchy fog. Engine `c_player_view::queue_patchy_fog @
        // 0x18068BFB0` enqueues a transparency-sort element that
        // dispatches `c_patchy_fog::render_patchy_fog` between water
        // and transparents. We dispatch directly (TransparencyRenderer
        // routing deferred). Gated on the active atmosphere setting
        // having the "Patchy Fog" flag (bit 2 / mask 4) and authored
        // density > 0.
        // Engine `c_patchy_fog::render_patchy_fog @ 0x1806B2160` gates on
        // `c_atmosphere_fog_interface::m_default_parameters.patchy_fog_density
        // <= 0.001f`. The accumulator's density is 0 when no contributing
        // setting has the Patchy Fog flag (bit 2) set — engine
        // `accumulate_atmosphere_settings` gates the patchy-fog field
        // copy on that bit. So this single threshold subsumes the prior
        // `has_patchy_fog() && density > 0.001` two-step check.
        if let Some(sky_atm) = renderer.scenario_sky_atmosphere.as_ref() {
            let interface_params = &renderer.atmosphere_fog_interface.default_parameters;
            if interface_params.patchy_fog_density > 0.001 {
                // A1 phase 6 — advance per-frame fog state from
                // camera delta + wind direction. Engine
                // `c_patchy_fog::render_patchy_fog @ 0x1806B2160`
                // lines 218-260. `dt` mirrors engine
                // `c_patchy_fog::ms_dt` — current per-frame delta.
                let dt: f32 = game.delta_time;
                let wind = glam::Vec3::new(
                    interface_params.wind_direction.i,
                    interface_params.wind_direction.j,
                    interface_params.wind_direction.k,
                );
                self.patchy_fog.update_per_frame(
                    game.camera.position.into(),
                    game.camera.forward,
                    game.camera.up,
                    sky_atm.distance_between_sheets,
                    game.camera.near_clip,
                    wind,
                    dt,
                );
                let params = crate::halo::render::patchy_fog_pass::build_patchy_fog_params(
                    interface_params,
                    sky_atm,
                    &self.patchy_fog,
                    game.camera.position.into(),
                    game.camera.projection,
                    (renderer.shared.config.width, renderer.shared.config.height),
                    game.camera.near_clip,
                );
                let atm = crate::halo::render::patchy_fog_pass::build_patchy_fog_atmosphere(
                    interface_params,
                );
                let view_proj = game.camera.projection * game.camera.view;
                // One-shot dump of the cbuffer inputs. Helps catch
                // saturated extinction / degenerate UV transforms
                // without staring at the shader.
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static LOGGED: AtomicBool = AtomicBool::new(false);
                    if !LOGGED.load(Ordering::Relaxed)
                        && !LOGGED.swap(true, Ordering::Relaxed)
                    {
                        eprintln!(
                            "[patchy_fog dump]\n  iz={:?}\n  atten={:?} (full={:.2}+sea_level={:.2}=eff_full={:.2} rate={:.4} dff={:.4})\n  eye={:?}\n  basis={:?} (roll={:.3})\n  density={} sep={} repeat={}\n  sheet_d0={:?}\n  sheet_d1={:?}\n  uv_xform0={:?}\n  uv_xform4={:?}\n  uv_xform7={:?}\n  atm.sun_dir={:?}\n  atm.sun_int_over_tm={:?}\n  atm.tm_log2e={:?}\n  atm.mie_theta={:?}\n  atm.hg_const+1={} hg_2g={}",
                            params.inverse_z_transform,
                            params.attenuation_data,
                            interface_params.full_intensity_height,
                            interface_params.reference_datum_plane,
                            interface_params.full_intensity_height + interface_params.reference_datum_plane,
                            params.attenuation_data[1],
                            sky_atm.depth_fade_factor,
                            params.eye_position,
                            params.texcoord_basis,
                            self.patchy_fog.roll,
                            interface_params.patchy_fog_density,
                            sky_atm.distance_between_sheets,
                            sky_atm.texture_repeat_rate,
                            params.sheet_depths0,
                            params.sheet_depths1,
                            params.tex_coord_transform0,
                            params.tex_coord_transform4,
                            params.tex_coord_transform7,
                            atm.sun_dir,
                            atm.sun_intensity_over_tm,
                            atm.total_mie_log2e,
                            atm.mie_theta_prefix_hgc,
                            atm.constant_2[3],
                            atm.extra[0],
                        );
                    }
                }
                renderer.patchy_fog.render(
                    &renderer.shared,
                    &mut encoder,
                    &renderer.intermediates.lighting_base_view,
                    &renderer.intermediates.hdr_dark_view,
                    &renderer.gbuffer.depth_sample_view,
                    &renderer.shared.nearest_sampler,
                    // Phase 5: prefer the scenario-loaded noise
                    // bitmap (`sky.patchy_fog_texture`). Falls back
                    // to `default_normal_view` (constant ~0.5) when
                    // the bitmap isn't authored or failed to load —
                    // visible result there is a flat tint, useful
                    // for verifying the wiring.
                    renderer.patchy_fog.noise_view().unwrap_or(
                        &renderer.shared.fallback_textures.default_normal_view,
                    ),
                    &renderer.shared.filtering_sampler,
                    &params,
                    &atm,
                    view_proj,
                    renderer.scenario_view_exposure,
                    patchy_fog_debug_mode(),
                );
            }
        }
        // `c_player_view::render_transparents @ 0x18068b3b0` —
        // sort + dispatch the transparency pool. Sits AFTER
        // `render_water` in the canonical pass order (water writes
        // depth in its second sub-pass so transparency draws know
        // about water occlusion). Engine-faithful sub-structure:
        // sky-only sort batch first, then outer regular batch.
        encoder.push_debug_group("render_transparents");
        self.render_transparents(&mut encoder, &frame_ctx, &mut renderer.transparency_renderer);
        encoder.pop_debug_group();

        // Step 19 of `c_player_view::render` — bloom + tonemap.
        // `c_screen_postprocess::postprocess_player_view` writes
        // its output into `screen_postprocess.aux_bloom`; the final
        // composite blit then reads both `lighting_base` (HDR) +
        // `aux_bloom` and adds them before applying the cubic tone
        // curve.
        //
        // L5: bloom-extract feed is `hdr_dark` (engine
        // `_surface_screenshot_composite_cubemap` — the DARK target
        // RT1 from the SL pass). Engine `convert_to_render_target`
        // writes `dark_color = color / DARK_COLOR_MULTIPLIER` per
        // fragment; bloom + auto-exposure read from this HDR/DARK
        // target. On MCC PC `DARK_MULT = 1.0` so `hdr_dark == lighting_base`
        // content-wise — but routing through the engine-named target
        // is engine-faithful and lays the groundwork for HDR sun-disc /
        // atmosphere with `g_HDR_target_stops > 0`.
        // `auto_exposure_sensitivity` lerps between mean(log2(intensity))
        // and log2(mean(intensity)) inside shader 35. Pulled from the
        // scenario cfxs (instant parameter, no per-frame blend).
        // Defaults to 0 (pure mean(log) / geometric mean) when no
        // scenario is loaded — engine cfxs `set_defaults` also uses 0.
        let auto_exposure_sensitivity = renderer
            .scenario_camera_fx
            .as_ref()
            .map(|cfx| cfx.auto_exposure_sensitivity.value)
            .unwrap_or(0.0);
        let scripted_globals = renderer.scripted_exposure_game_globals;
        let g_exposure_stops = renderer.g_exposure_stops;
        let exposure_slot = renderer.screen_postprocess.postprocess_player_view(
            &renderer.shared,
            &mut encoder,
            &renderer.intermediates.hdr_dark_view,
            &renderer.gbuffer.depth_sample_view,
            renderer.scenario_camera_fx.as_ref(),
            frame_ctx.game.camera.view,
            frame_ctx.game.camera.projection,
            frame_ctx.game.camera.field_of_view,
            frame_ctx.game.camera.aspect_ratio,
            frame_ctx.game.camera.near_clip,
            auto_exposure_sensitivity,
            &scripted_globals,
            g_exposure_stops,
        );
        // Engine `update_sampled_exposure(camera->m_exposure, v8)` —
        // push the freshly-scheduled ring slot into the FIFO so the
        // next `setup_camera_fx_parameters` reads it (after the 3-frame
        // async-map delay). Inlined here because postprocess_player_view
        // doesn't own the camera_fx_values mutable borrow.
        if let Some(slot) = exposure_slot {
            // Engine `update_sampled_exposure(camera->m_exposure, v8)` where
            // `v8 = m_next_render_target + 28` — the FIFO stores SURFACE
            // INDICES (28..35), not raw slot indices. Consumer side
            // (`setup_camera_fx_parameters`) uses
            // `Exposure::slot_from_surface_index` to translate back.
            let surface_index = crate::halo::render::camera_fx_settings::Exposure::surface_index_from_slot(slot);
            renderer
                .render_game_state
                .player_window[0]
                .camera_fx_values
                .exposure
                .update_sampled_exposure(surface_index);
        }

        // E1 — DoF test knob. `PROTOMORPH_DOF=<focus_world_units>`
        // enables the depth-of-field blend with focus at the given
        // distance (e.g. PROTOMORPH_DOF=8 for a 8-wu focus). Half-width
        // = focus × 0.1, max_blur = 0.8. When unset, leaves DoF
        // disabled (max_blur=0 → sharp passthrough). Engine-faithful
        // path is HSC `render_set_depth_of_field` + per-observer
        // override; this is a stop-gap until either lands.
        //
        // CompositeParams uploaded every frame (whether DoF or not) so
        // tone curve constants are always in sync with
        // `screen_postprocess.settings_internal.m_tone_curve` /
        // `.m_tone_curve_white_point`. Engine `copy_accumulation_target`
        // recomputes them per call; we mirror that.
        let dof_focus = if let Some(focus_world) = parse_dof_env() {
            let (focus_ndc, half_ndc) =
                crate::halo::render::final_composite_pass::dof_world_to_ndc(
                    focus_world,
                    focus_world * 0.1,
                    frame_ctx.game.camera.near_clip,
                    frame_ctx.game.camera.near_clip + 1024.0,
                );
            [focus_ndc, half_ndc, 0.8, 0.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        renderer.final_composite_pass.set_composite_params(
            &renderer.shared.queue,
            renderer.screen_postprocess.color_grading_blend_factor(),
            dof_focus,
            renderer.screen_postprocess.settings_internal.tone_curve,
            renderer.screen_postprocess.settings_internal.tone_curve_white_point,
        );

        if renderer.final_composite_pass.is_enabled(&frame_ctx) {
            // Stage 1 — engine `copy_accumulation_target` body. Composite
            // lighting_base + bloom + cg + DoF into `accum_hdr_view`.
            encoder.push_debug_group("copy_accumulation_target");
            renderer.final_composite_pass.record(&mut encoder, &frame_ctx);
            encoder.pop_debug_group();

            // Stage 2 — engine `render_lightshafts(_surface_accum_HDR,
            // _surface_accum_HDR)`. Reads + additive-blits into accum_hdr.
            // Engine gate: `(cfxs.flags & 2) != 0 && s_bUseLightShafts`.
            // Our `s_b_use_light_shafts` ships false by default (cascade
            // bug — see ScreenPostprocessSettings doc). Set
            // `PROTOMORPH_ENABLE_LIGHTSHAFTS=1` to force-enable for
            // testing the engine-faithful 5-pass chain.
            let lightshafts = renderer
                .scenario_camera_fx
                .as_ref()
                .and_then(|cfx| cfx.lightshafts.as_ref());
            let env_enabled =
                std::env::var_os("PROTOMORPH_ENABLE_LIGHTSHAFTS").is_some();
            let runtime_gate = renderer
                .screen_postprocess
                .settings_internal
                .s_b_use_light_shafts
                || env_enabled;
            if runtime_gate && lightshafts.is_some_and(|ls| ls.is_enabled()) {
                encoder.push_debug_group("render_lightshafts");
                renderer.screen_postprocess.render_lightshafts(
                    &renderer.shared,
                    &mut encoder,
                    lightshafts,
                    frame_ctx.game.camera.view,
                    frame_ctx.game.camera.projection,
                    frame_ctx.game.camera.field_of_view,
                    frame_ctx.game.camera.aspect_ratio,
                    frame_ctx.game.camera.near_clip,
                    &renderer.intermediates.accum_hdr_view,
                    &renderer.gbuffer.depth_sample_view,
                    // Engine `_surface_albedo` (full-res). The albedo
                    // pass's RT0 is done with by post-process time, so
                    // engine reuses it as the shader-97 extract scratch.
                    // We do the same with `albedo_view`.
                    &renderer.intermediates.albedo_view,
                    &renderer.intermediates.accum_hdr_view,
                );
                encoder.pop_debug_group();
            }

            // Stage 3 — engine `copy(51, _surface_accum_HDR,
            // _surface_display)`. accum_hdr → swapchain + HUD text.
            encoder.push_debug_group("accum_hdr_to_swapchain");
            renderer
                .final_composite_pass
                .record_blit_to_swapchain(&mut encoder, &frame_ctx);
            encoder.pop_debug_group();
        }

        // Auto-exposure GPU dispatch is now inlined inside
        // `postprocess_player_view` via the engine-faithful
        // `downsample_generate` (shader 35 on the freshly box-downsampled
        // aux_tiny, BEFORE the bloom pyramid blurs and tints it). The
        // slot was already pushed to the camera FIFO above; just keep
        // the local for `post_submit_exposure` below.
        drop(frame_ctx);

        renderer.shared.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Post-submit phase — start async map for the exposure slot
        // we just submitted. The callback fires on a driver thread
        // when GPU finishes the copy_texture_to_buffer; subsequent
        // frames' `try_read_exposure_sample` polls the result without
        // stalling. Replaces the prior `device.poll(wait_indefinitely)`
        // pattern that drained the entire device queue per frame.
        renderer.post_submit_exposure(exposure_slot);
        // env_probe_pass.post_submit disabled — see record-phase comment above.
        renderer.final_composite_pass.post_submit();

        // Save current VP for next frame's temporal reprojection
        renderer.prev_vp = game.camera.projection * game.camera.view;
        renderer.frame_counter = renderer.frame_counter.wrapping_add(1);

        // Return preallocated buffers
        render_list.clear();
        renderer.render_list = render_list;

    }

    /// `c_player_view::setup_camera(player_window_index, player_window_count,
    /// player_window_arrangement, user_index, observer, freeze_render_camera)`.
    pub fn setup_camera(
        &mut self,
        _player_window_index: i32,
        _player_window_count: i32,
        _player_window_arrangement: i32,
        _user_index: i32,
        _freeze_render_camera: bool,
    ) {
        todo!("c_player_view::setup_camera — phase 4");
    }

    // `c_player_view::setup_camera_fx_parameters @ 0x180689C20` is
    // implemented on `Renderer` (see `Renderer::setup_camera_fx_parameters`)
    // because the orchestrator owns the cfxs source resolution + exposure
    // FIFO + screen_postprocess snapshot refresh. PlayerView holds none
    // of those.

    /// `c_player_view::create_frame_textures(long)`.
    pub fn create_frame_textures(&mut self, _player_window_index: i32) { todo!() }

    /// `c_player_view::restore_to_display_surface()`.
    pub fn restore_to_display_surface(&mut self) { todo!() }

    /// `c_player_view::render_albedo()` — first-pass G-buffer fill.
    /// Body: dllcache `0x18068abc0`. Engine equivalent of
    /// `_entry_point_albedo` for every opaque mesh: writes albedo
    /// (RGB + spec_mask) to `_surface_albedo`, normal (encoded
    /// `n*0.5+0.5` + albedo.w) to the normal G-buffer. NO lighting,
    /// NO atmosphere, NO exposure — those run later in
    /// `render_static_lighting` which samples this G-buffer.
    pub fn render_albedo<'rp>(
        &mut self,
        encoder: &'rp mut wgpu::CommandEncoder,
        ctx: &'rp FrameContext<'rp>,
        transparency_renderer: &mut crate::halo::render::render_transparents::TransparencyRenderer,
    ) -> bool {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("player_view_render_albedo"),
            color_attachments: &[
                // Slot 0 — `_surface_albedo` (Rgba8Unorm).
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.albedo_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // Slot 1 — world-space normal G-buffer (engine
                // `_surface_post_HDR` second MRT). Read by render_ssao
                // and by static_lighting PS. Engine
                // `setup_targets_albedo @ 0x180670CF0:11` clears RT1 to
                // 0x8080FF = (128, 128, 255) → (0.502, 0.502, 1.0) =
                // encoded +Z UP normal. Uncovered pixels (sky edges,
                // hairline holes) then decode to (0, 0, 1) via *2-1
                // in SSAO + SL. Previous (0.5, 0.5, 0.5) decoded to
                // (0, 0, 0) degenerate, producing wrong AO darkening
                // and wrong dot(N, L) along uncovered regions.
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.gbuffer.normal_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 128.0 / 255.0,
                            g: 128.0 / 255.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                // atmosphere @ binding 2 — albedo pass is fog-exempt (engine
                // gates entry_point==albedo out of the override); use default.
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );

        // `c_object_renderer::render_albedo(1)` — generic object
        // albedo dispatch. record_mesh_part_draw picks `artifacts_albedo`
        // when entry_point == Albedo.
        use crate::halo::render::render_objects::{
            render_object_mesh_part_flags, ContextMeshPart, EntryPoint, ObjectRenderContext,
        };
        use crate::halo::render_methods::HaloEntryPoint;

        // `c_structure_renderer::render_decorators @ 0x1806901A0` —
        // engine-faithful order: decorators dispatch FIRST inside
        // `c_player_view::render_albedo`, BEFORE objects + structure +
        // sky. Engine call sequence verified via decompile (port 13372):
        //   setup_targets_albedo → set_alpha_blend → render_decorators
        //   → render_first_person_albedo → object_renderer.render_albedo
        //   → structure_renderer.render_albedo → submit_and_render_sky
        // Drawing decorators after objects (the previous order) caused
        // their depth to land on top of geometry that should occlude
        // them, which made them "stick through" patchy fog at later
        // pass setup time — patchy fog samples scene depth and
        // multiply-add fogs decorator pixels at incorrect distances.
        ctx.decorator_renderer.render_into_pass(&mut rpass, ctx);

        let mut object_bindings =
            crate::halo::render::render_objects::ObjectPassBindings::new();
        ctx.object_renderer.render_albedo(
            render_object_mesh_part_flags::LIT,
            |context_index: usize,
             mesh_part: &ContextMeshPart,
             _ep: EntryPoint,
             _mask: u32,
             _ctx_ref: &ObjectRenderContext| {
                ctx.object_renderer.record_mesh_part_draw(
                    &mut rpass,
                    ctx.shared,
                    ctx,
                    mesh_part,
                    context_index,
                    HaloEntryPoint::Albedo,
                    &mut object_bindings,
                );
            },
        );

        // `c_structure_renderer::render_albedo()` — BSP cluster +
        // instance draws with `_entry_point_albedo` (now wired to
        // pick `artifacts_albedo` per material).
        ctx.structure_renderer.render_albedo(&mut rpass, ctx);

        // `c_object_renderer::submit_and_render_sky(0, …)` — the engine
        // draws sky into the G-buffer here (`render_albedo @ 0x18068abc0`,
        // right after `c_structure_renderer::render_albedo`) with
        // `_z_buffer_mode_write`.
        //
        // Engine-faithful: sky writes the G-buffer + depth here.
        crate::halo::render::render_objects::submit_and_render_sky(
            0,
            0,
            &mut rpass,
            ctx,
            transparency_renderer,
        );

        // `c_object_renderer::render_albedo_decals @ 0x1806E4A60` —
        // engine call site is right AFTER sky and BEFORE
        // `c_decal_system::render_all`, also under `_z_buffer_mode_read`.
        // Walks object render contexts with mesh_part_mask = DECAL bit,
        // entry_point = _default. Engine HLSL is the rmd `default_ps`
        // body skinned per-bone — our `HaloEntryPoint::DecalObject`
        // routes through `entry_decal_object.wgsl` which takes
        // `ModelVertex` and shares the rmd PS fragment chain with
        // scene decals. `submit_visibility_from_render_list` sets the
        // DECAL bit per-part by inspecting the material's
        // `artifacts_decal_object` slot.
        let mut object_decal_bindings =
            crate::halo::render::render_objects::ObjectPassBindings::new();
        ctx.object_renderer.render_albedo_decals(
            |context_index: usize,
             mesh_part: &ContextMeshPart,
             _ep: EntryPoint,
             _mask: u32,
             _ctx_ref: &ObjectRenderContext| {
                ctx.object_renderer.record_mesh_part_draw(
                    &mut rpass,
                    ctx.shared,
                    ctx,
                    mesh_part,
                    context_index,
                    HaloEntryPoint::DecalObject,
                    &mut object_decal_bindings,
                );
            },
        );

        // `c_decal_system::render_all(_pass_post_albedo)` — engine
        // call site is INSIDE `c_player_view::render_albedo`, after
        // sky AND object decals. Engine state: `_z_buffer_mode_read`
        // (depth-test on, write off — our pipeline's
        // `depth_write_enabled=false`), `_cull_mode_off` (our
        // `cull_mode=None`),
        // high_quality_blend (alpha-blend per the rmd's blend_mode).
        // RT layout: RT0 = albedo_view (engine `_surface_accum_HDR`),
        // RT1 = normal_view (engine `_surface_post_HDR`) — bump-write
        // gated by bump_mapping option (see `pipeline_cache.rs`).
        // This is the engine-faithful path for cyberdyne's 228
        // preplaced post_albedo decals.
        //
        // Decals are DISABLED by default until D7.0a (engine-faithful
        // c_decal_system::collide cylinder-sweep) + D7.2 (wire the
        // orchestrator into scenario loader) ship. Current path emits
        // flat oriented quads at each placement, which produces
        // "flat planes extending past edges" instead of properly
        // folded surface decals. See project_decal_audit_2026_05_13.
        // Re-enable via `PROTOMORPH_ENABLE_DECALS=1` for A/B testing.
        // D7.2 SHIPPED — collide + orchestrator + writer wired through
        // the scenario loader. Decals render by default. Set
        // `PROTOMORPH_DISABLE_DECALS=1` to bypass.
        let decals_enabled = std::env::var("PROTOMORPH_DISABLE_DECALS")
            .map(|v| v == "0" || v.is_empty())
            .unwrap_or(true);
        if decals_enabled {
            if let (Some(decal_gpu), Some(decal_renderer)) =
                (ctx.decal_gpu, ctx.decal_renderer)
            {
                use blam_tags::decal_system::DecalPass;
                decal_gpu.record_render(&mut rpass, decal_renderer, DecalPass::PreLighting);
            }
        }

        // Decorators + foliage / terrain albedo entry-points still
        // pending — they currently only render in the static_lighting
        // pass. Phase B port of those entry points will populate the
        // G-buffer for those pixels.
        true
    }

    /// `c_player_view::render_static_lighting()` — second-pass lit
    /// color. Body: dllcache `0x18068aea0`. Re-walks the same opaque
    /// geometry sorted by `render_albedo`, with `_z_buffer_mode_read`
    /// (depth-test on, depth-write off), and writes the FINAL lit
    /// color to `lighting_base`. The static_lighting PS samples the
    /// G-buffer populated by render_albedo for albedo + normal (or
    /// recomputes them inline — see entry_static_*.wgsl).
    pub fn render_static_lighting<'rp>(
        &mut self,
        encoder: &'rp mut wgpu::CommandEncoder,
        ctx: &'rp FrameContext<'rp>,
        transparency_renderer: &mut crate::halo::render::render_transparents::TransparencyRenderer,
    ) -> bool {
        // Engine `setup_targets_static_lighting @ 0x180670DB0` — copy
        // `_surface_accum_HDR → _surface_screenshot_composite_depth`
        // BEFORE the SL pass starts so that decorator pixels (lit in
        // the albedo pass) flow through to the SL target. SL fragments
        // overwrite this pre-blit content where they pass depth-test;
        // pixels that no SL entry covers (e.g. decorators) keep the
        // copied lit color. Mirrors `c_screen_postprocess::copy(51,
        // _surface_accum_HDR → _surface_screenshot_composite_depth)`.
        let extent = wgpu::Extent3d {
            width: ctx.shared.config.width.max(1),
            height: ctx.shared.config.height.max(1),
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ctx.intermediates.albedo_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &ctx.intermediates.lighting_base_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );
        // Engine `setup_targets_static_lighting @ 0x180670DB0:65-72`
        // explicitly CLEARS `_surface_screenshot_composite_cubemap`
        // (= our `hdr_dark`, RT1) to black before SL pass when
        // `render_to_HDR_target` is on. We mirror via LoadOp::Clear on
        // the RT1 attachment below. The previous `albedo → hdr_dark`
        // seed copy was a protomorph-specific kludge — engine never
        // pre-seeds RT1; it relies on per-fragment SL writes to fill it.
        // For uncovered pixels (decorators, sky), engine leaves RT1
        // black, which means bloom extract reads black for those regions
        // and contributes nothing — engine-correct.

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("player_view_render_static_lighting"),
            color_attachments: &[
                // RT0 — `lighting_base` (engine `_surface_screenshot_composite_depth`,
                // the LDR target). Loaded from the accum_HDR pre-blit so
                // decorator/sky pixels (already lit in render_albedo)
                // survive where SL doesn't overwrite.
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.lighting_base_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // RT1 — `hdr_dark` (engine `_surface_screenshot_composite_cubemap`,
                // the HDR/DARK target sampled by bloom_extract). Per
                // engine `setup_targets_static_lighting` line 65-72,
                // CLEARED to black each SL frame. SL fragments fill it
                // per-pixel; uncovered regions stay black → bloom feed
                // sees no contribution from sky/decorator pixels (which
                // matches engine behavior).
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.hdr_dark_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                // Engine: _z_buffer_mode_read — depth-test on, no write.
                // wgpu approximation: load the depth from albedo pass,
                // store unchanged. The pipeline's depth_write_enabled=true
                // would still write fragment depth (== existing depth
                // for same geometry), so this works.
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group_sl,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                // atmosphere @ binding 2 — SL pass setup default; per-draw
                // sites (BSP/objects/sky) rebind with their own offset.
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );

        // Decorators are NOT dispatched here — engine renders them
        // ONLY in `render_albedo` (`c_structure_renderer::render_decorators
        // @ 0x1806901A0`). Their lit color was written to
        // `_surface_accum_HDR` and copy-blitted into `lighting_base`
        // at the start of this pass.

        use crate::halo::render::render_objects::{
            render_object_mesh_part_flags, ContextMeshPart, EntryPoint, ObjectRenderContext,
        };
        use crate::halo::render_methods::HaloEntryPoint;
        let mut object_bindings_sl =
            crate::halo::render::render_objects::ObjectPassBindings::new();
        ctx.object_renderer.render_static_lighting(
            render_object_mesh_part_flags::LIT,
            |context_index: usize,
             mesh_part: &ContextMeshPart,
             _ep: EntryPoint,
             _mask: u32,
             _ctx_ref: &ObjectRenderContext| {
                ctx.object_renderer.record_mesh_part_draw(
                    &mut rpass,
                    ctx.shared,
                    ctx,
                    mesh_part,
                    context_index,
                    HaloEntryPoint::StaticSh,
                    &mut object_bindings_sl,
                );
            },
        );

        ctx.structure_renderer.render_static_lighting(&mut rpass, ctx);

        // `c_object_renderer::submit_and_render_sky(1, …)` — sky in
        // the static_lighting pass uses the StaticSh pipeline to
        // write the FINAL lit color. Sky draws after BSP/objects but
        // gets z-rejected where they're closer.
        crate::halo::render::render_objects::submit_and_render_sky(
            1,
            0,
            &mut rpass,
            ctx,
            transparency_renderer,
        );

        // `c_decal_system::render_all(_pass_post_static_lighting)` —
        // engine call site is the LAST statement of
        // `c_player_view::render_static_lighting`. Writes to RT0
        // (`lighting_base`) with color-write 7. Cyberdyne authors no
        // post-lighting decals so this loop is normally empty; we
        // include it for correctness against other maps. post_albedo
        // decals (cyberdyne's 228 placements) are dispatched in
        // `render_albedo` per Step 2.
        //
        // Same disable gate as the PreLighting site above — defer
        // until the orchestrator is wired up (D7.0a + D7.2).
        // D7.2 SHIPPED — collide + orchestrator + writer wired through
        // the scenario loader. Decals render by default. Set
        // `PROTOMORPH_DISABLE_DECALS=1` to bypass.
        let decals_enabled = std::env::var("PROTOMORPH_DISABLE_DECALS")
            .map(|v| v == "0" || v.is_empty())
            .unwrap_or(true);
        if decals_enabled {
            if let (Some(decal_gpu), Some(decal_renderer)) =
                (ctx.decal_gpu, ctx.decal_renderer)
            {
                use blam_tags::decal_system::DecalPass;
                decal_gpu.record_render(&mut rpass, decal_renderer, DecalPass::PostLighting);
            }
        }

        true
    }


    /// `c_player_view::render_dynamic_lights(ldr, hdr, depth)`.
    pub fn render_dynamic_lights(&mut self, _ldr: Surface, _hdr: Surface, _depth: Surface) {
        todo!("phase 10")
    }

    pub fn render_lightmap_shadows(&mut self) { todo!() }

    /// `c_player_view::render_transparents()`.
    /// Body: dllcache `0x18068b3b0`.
    ///
    /// Engine sequence (with our protomorph mirror notes):
    /// ```text
    /// render_method_submit_single_extern(3, 0);   // [TODO: extern routing]
    /// render_method_submit_single_extern(45, 0);
    /// render_method_submit_single_extern(1, 1);
    /// render_method_submit_single_extern(2, 1);
    /// submit_light_draw_list_to_shader(...);      // [done at frame setup]
    /// setup_targets_static_lighting_alpha_blend(post, 1);
    /// set_active_camo_bounds(...);
    /// set_using_albedo_sampler(0);
    /// c_object_renderer::submit_and_render_sky(2, …);  // sky-only batch
    /// implicit_geometry_submit();                 // [stub]
    /// effects_render(_pass_transparent);          // [stub]
    /// c_transparency_renderer::sort();            // sort outer (regular) batch
    /// c_transparency_renderer::render(1, …);      // render outer batch
    /// ```
    ///
    /// The two `sort + render` pairs operate on different marker scopes:
    /// the sky-only batch (pushed by submit_and_render_sky(2)) and the
    /// outer regular-transparent batch (already populated during
    /// `submit_visibility`).
    pub fn render_transparents<'rp>(
        &mut self,
        encoder: &'rp mut wgpu::CommandEncoder,
        ctx: &'rp FrameContext<'rp>,
        transparency_renderer: &mut crate::halo::render::render_transparents::TransparencyRenderer,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("player_view_render_transparents"),
            color_attachments: &[
                // RT0 — `lighting_base` (LDR composite).
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.lighting_base_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // RT1 — `hdr_dark` (engine `_surface_screenshot_composite_cubemap`).
                // Transparent rmsh pipelines are 2-target (same shader
                // as opaque); rpass MUST match. On PC `DARK_MULT=1.0`
                // so RT1 == RT0 verbatim.
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.hdr_dark_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        // Sky-only sort batch FIRST. submit_and_render_sky(2) push_marker
        // → queue sky transparents → sort + render → pop_marker. This
        // keeps the outer (regular) transparent batch untouched.
        crate::halo::render::render_objects::submit_and_render_sky(
            2,
            0,
            &mut rpass,
            ctx,
            transparency_renderer,
        );

        // Outer (regular) transparent batch — sort + render the
        // already-submitted entries. Engine renders this AFTER
        // implicit_geometry_submit + effects_render(_pass_transparent),
        // both stubbed.
        let eye: glam::Vec3 = ctx.game.camera.position.into();
        let fwd: glam::Vec3 = ctx.game.camera.forward.into();
        transparency_renderer.sort(eye.to_array(), fwd.to_array());
        if std::env::var_os("PROTOMORPH_DIAG_SORT_DUMP").is_some() {
            transparency_renderer.debug_dump_sorted(eye.to_array(), "regular");
        }
        transparency_renderer.render(&mut rpass, ctx);
    }

    // =================================================================
    // Engine frame-loop pass stubs — P4 stub registry
    //
    // Each stub corresponds to a `c_player_view::*` engine function
    // called from `c_player_view::render @ 0x180688BA0`. Bodies are
    // empty no-ops; P10 (missing passes) fills them in. The frame-loop
    // structural port can reference these by name and they will not
    // panic.
    //
    // Engine call sequence: see `reference_engine_frame_loop_full.md`.
    // =================================================================

    /// `c_player_view::submit_distortions @ 0x18068b5f0`. Builds the
    /// distortion accumulator into `_surface_aux_mini2` via the
    /// `_effect_pass_distortion` effect-pass dispatcher + a
    /// half-res draw. **Stub — P10 fills in.**
    pub fn submit_distortions(&mut self) {}

    /// `c_player_view::generate_distortions_callback` — inner draw
    /// callback used by `submit_distortions`'s occlusion-submit path.
    pub fn generate_distortions(&mut self) {}

    /// `c_player_view::apply_distortions @ 0x18068b8b0`. Fullscreen
    /// composite of the distortion accumulator onto the framebuffer,
    /// with optional motion-blur path when last-frame state is close
    /// to current. **Stub — P10 fills in.**
    pub fn apply_distortions(&mut self) {}

    /// `c_player_view`'s LDR/HDR depth restore helper.
    pub fn restore_ldr_hdr_depth_buffers(&mut self) {}

    /// `c_player_view::queue_patchy_fog`. Sets up the patchy-fog draw
    /// before transparents render.
    pub fn queue_patchy_fog(&mut self) {}

    /// `c_player_view::render_first_person(render_only_transparents) @ 0x18068a5d0`.
    /// Snapshots player camera into m_first_person_view, override_projection +
    /// enable_squish, then renders FP weapon model. Called twice from
    /// the frame loop: param=0 (opaque pass) and param=1 (transparent
    /// pass). **Stub — P10 fills in.**
    pub fn render_first_person(&mut self, _render_only_transparents: bool) {}

    /// `c_player_view::render_first_person_albedo @ 0x18068a920`.
    /// FP weapon G-buffer write. Runs inside `render_albedo`. **Stub.**
    pub fn render_first_person_albedo(&mut self) {}

    /// `c_player_view::render_lens_flares @ 0x18068ab40`. Walks light
    /// tags for lens flare attachments + draws sprites with prior-
    /// frame occlusion-query gating. **Stub — P10 fills in.**
    pub fn render_lens_flares(&mut self) {}

    /// `c_player_view::render_water`. Water surface pass between SL
    /// targets bind and transparents. Today's protomorph has a
    /// working impl in `render_water.rs`. **Stub here — the live
    /// impl is invoked directly until P4 frame loop port lands.**
    pub fn render_water(&mut self) {}

    /// `c_player_view::animate_water`. Per-frame water-interaction
    /// height/slope animation step.
    pub fn animate_water(&mut self) {}

    /// `c_object_renderer::submit_attachments` / `c_lights_view::submit_attachments`.
    /// Builds the per-frame draw list of object/light attachments.
    pub fn submit_attachments(&mut self) {}

    /// `c_player_view::render_misc_transparents`. Catch-all for
    /// outside-the-sorted-list transparent ops.
    pub fn render_misc_transparents(&mut self) {}

    /// `rasterizer_occlusion_submit` / `lens_flares_submit_occlusions`.
    /// Engine submits both conditional (early) and main (late) lens
    /// flare occlusion queries. **Stub.**
    pub fn submit_occlusion_tests(&mut self, _occlusion: bool, _conditional: bool) {}

    /// `rasterizer_occlusions_retrieve`. Pulls prior-frame occlusion
    /// query results from the GPU for lens flare gating.
    pub fn retrieve_occlusion_tests(&mut self) {}

    /// `c_player_view::setup_cinematic_clip_planes`. Cinematic
    /// camera mode applies HSC-driven clip planes.
    pub fn setup_cinematic_clip_planes(&mut self) {}

    /// `c_player_view::render_patchy_fog`. Patchy-fog volume render
    /// — already shipped via `patchy_fog_pass.rs`; stub kept for
    /// engine-name parity in the P4 frame loop.
    pub fn render_patchy_fog(&mut self) {}

    /// `c_player_view::render_weather_occlusion @ 0x18068b2f0`.
    /// Generates a depth-only render into the occlusion buffer for
    /// weather geometry intersection (used by patchy fog / rain).
    /// Sets `g_weather_occlusion_available`. **Stub — P10 fills in.**
    pub fn render_weather_occlusion(&mut self) {}

    /// `chud_draw_turbulence @ 0x18071ef90`. Writes the CHUD heat-
    /// haze distortion overlay into the distortion accumulator.
    /// **Stub — P10 fills in.**
    pub fn chud_draw_turbulence(&mut self) {}

    /// `chud_generate_damage_flash_texture @ 0x18071f1e0`. Returns
    /// true if player damage flash is active; result gates the
    /// `chud_draw_screen_LDR` call. **Stub — P10 fills in (returns
    /// false).**
    pub fn chud_generate_damage_flash_texture(&mut self) -> bool { false }

    /// `screen_effect_sample @ 0x1803a4e90`. Walks scenario screen-
    /// effect settings hierarchy and samples per-frame state
    /// (exposure_boost, hue/sat/contrast, color matrix, fade).
    /// **Stub — P10 fills in.**
    pub fn screen_effect_sample(&mut self) {}

    /// `c_player_view::render_effects(pass) @ 0x180688BA0` callee.
    /// Dispatches game-effect draws for a specific pass enum
    /// (`_effect_pass_distortion`, `_effect_pass_opaque`,
    /// `_effect_pass_transparent`). **Stub — P10 fills in.**
    pub fn render_effects(&mut self, _pass: EffectPass) {}

    /// `render_setup_window`. Restores viewport + scissor before UI
    /// composite. **Stub.**
    pub fn render_setup_window(&mut self) {}

    /// `c_player_view::setup_camera_fx_parameters(exposure_boost)`.
    /// Applies screen-effect exposure boost to the camera FX state.
    /// **Stub — P10 fills in.**
    pub fn setup_camera_fx_parameters(&mut self, _exposure_boost: f32) {}
}

/// Mirror of `e_effect_pass` — selector for which game-effect entries
/// get dispatched in a given pass. Engine values:
/// 0 = distortion, 1 = opaque, 2 = transparent, 3 = post-process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EffectPass {
    Distortion = 0,
    Opaque = 1,
    Transparent = 2,
    PostProcess = 3,
}

/// Stand-in for `s_render_game_state::s_player_window`. Filled when
/// `render_game_state.rs` lands.
#[derive(Debug, Clone, Default)]
pub struct PlayerWindowGameState {
    pub _placeholder: (),
}

#[derive(Debug, Clone, Default)]
pub struct ObserverDepthOfField {
    pub flags: u32,
    pub _placeholder: (),
}

#[derive(Debug, Clone, Default)]
pub struct MotionBlurState {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub game_time: f32,
    pub view_matrix: Mat4,
    pub projection_matrix: [[f32; 4]; 4],
}
