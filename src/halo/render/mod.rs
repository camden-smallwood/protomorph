mod final_composite_pass;
pub mod atmosphere_fog_interface;
pub mod bind_group_cache;
pub mod lighting_interface;
pub mod render_method_data;
pub mod select_entry_point;
pub mod screen_postprocess;
pub mod shadow_apply_pass;
pub mod patchy_fog_pass;
pub mod shared;
pub mod spectral_constants;
pub mod structure_renderer;

// Halo-faithful renderer rebuild — mirrors Ares `source/render/`.
// `Renderer` below is the top-level. Its `render()` body mirrors
// `c_player_view::render @ 0x180688ba0` step-by-step; per-step
// helpers (animate_water, render_albedo, render_static_lighting,
// etc.) sit alongside. See RENDERER_REWRITE_PLAN.md.
pub mod camera_fx_settings;
pub mod chocalate_mountain;
pub mod color_grading;
pub mod screen_effect;
// Phase 9.2 cleanup (2026-05-26): `render::geometry_sampler` shim deleted;
// the engine-verbatim `lights_distant_lighting_at_point_new` lives in
// [`crate::halo::geometry::geometry_sampling`] using the full 504-B
// `s_geometry_sample` struct.
pub mod render_cameras;
pub mod render_game_state;
pub mod render_objects;
pub mod render_patchy_fog;
pub mod render_sky;
pub mod render_transparents;
pub mod render_water;
pub mod render_decals;
pub mod render_decorators;
pub mod render_visibility;
pub mod views;
mod scenario_lighting_bake;
mod rasg_load;

use crate::game::GameState;
use glam::Vec3;
use crate::halo::geometry::MAXIMUM_NUMBER_OF_MODEL_NODES;
use crate::halo::geometry::{ModelData, ModelMeshPart, SkyVertColorVertex};
use crate::halo::objects::ObjectIndex;
use crate::halo::render_methods::material_bindings::{fallback_view_for_param, sample_intent_for_param};
use crate::halo::render_methods::HaloEntryPoint;
use crate::halo::render_methods::materials::{InlinePixelFormat, MaterialData};
use crate::halo::render_methods::pipeline_cache::{
    HaloPipelineCache, VariantArtifacts,
};
use crate::halo::render_methods::serializer::serialize as serialize_material;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use shared::*;

/// Caller's sampling intent for a bitmap upload — combined with the
/// texture's [`TexelEncoding`] to pick the final wgpu format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleIntent {
    /// Sample per the bitmap's authored curve (the common, correct case):
    /// gamma-encoded data gets an sRGB view, linear data a linear view.
    FollowCurve,
    /// Override to linear regardless of curve — ONLY for values that are
    /// physical/packed, not display color (BRDF LUTs, normal maps, masks,
    /// noise/warp). The hard-override hatch.
    ForceLinear,
}

/// Map our layout-only [`InlinePixelFormat`] to a wgpu format, choosing
/// sRGB-vs-linear from the data's [`TexelEncoding`] and the call site's
/// [`SampleIntent`] — the single place that decision is made.
///
/// `want_srgb` is true ONLY when the caller follows the curve AND the data
/// is gamma-encoded. `ForceLinear` (BRDF LUTs / normals / masks) and
/// already-linear data both yield a linear view. The prior design baked
/// srgb-ness into the format variant and let an upload-time bool override
/// it; folding both into this one function makes the two facts impossible
/// to desync — and authored-linear data stays linear, so the shrine
/// `stone_edge_dif` ~3× darkening regression (2026-05-17) can't recur.
pub(crate) fn resolve_wgpu_format(
    fmt: InlinePixelFormat,
    encoding: crate::halo::render_methods::materials::TexelEncoding,
    intent: SampleIntent,
) -> wgpu::TextureFormat {
    use crate::halo::render_methods::materials::TexelEncoding;
    use InlinePixelFormat as F;
    use wgpu::TextureFormat as W;
    let want_srgb = matches!(intent, SampleIntent::FollowCurve)
        && matches!(encoding, TexelEncoding::Gamma);
    match fmt {
        F::Bc1RgbaUnorm => if want_srgb { W::Bc1RgbaUnormSrgb } else { W::Bc1RgbaUnorm },
        F::Bc2RgbaUnorm => if want_srgb { W::Bc2RgbaUnormSrgb } else { W::Bc2RgbaUnorm },
        F::Bc3RgbaUnorm => if want_srgb { W::Bc3RgbaUnormSrgb } else { W::Bc3RgbaUnorm },
        F::Bc4RUnorm    => W::Bc4RUnorm,
        F::Bc4RSnorm    => W::Bc4RSnorm,
        F::Bc5RgUnorm   => W::Bc5RgUnorm,
        F::Bc5RgSnorm   => W::Bc5RgSnorm,
        F::Rgba8Unorm   => if want_srgb { W::Rgba8UnormSrgb } else { W::Rgba8Unorm },
        F::Rgba8Snorm   => W::Rgba8Snorm,
        F::Bgra8Unorm   => if want_srgb { W::Bgra8UnormSrgb } else { W::Bgra8Unorm },
        F::Rg8Unorm     => W::Rg8Unorm,
        F::Rg8Snorm     => W::Rg8Snorm,
        F::R8Unorm      => W::R8Unorm,
        F::Rgba16Float  => W::Rgba16Float,
        F::Rgba16Unorm  => W::Rgba16Unorm,
        F::Rgba16Snorm  => W::Rgba16Snorm,
        F::Rgba32Float  => W::Rgba32Float,
    }
}

#[cfg(test)]
mod resolve_wgpu_format_tests {
    use super::resolve_wgpu_format;
    use super::SampleIntent::{FollowCurve, ForceLinear};
    use crate::halo::render_methods::materials::InlinePixelFormat as F;
    use crate::halo::render_methods::materials::TexelEncoding::{Gamma, Linear};
    use wgpu::TextureFormat as W;

    /// The three rules the old `(fmt, linear)` table encoded, now a pure
    /// function of (encoding, intent). Gamma+FollowCurve → sRGB; everything
    /// else → the linear (Unorm) view.
    #[test]
    fn srgb_capable_families_follow_curve_and_intent() {
        let cases = [
            (F::Bc1RgbaUnorm, W::Bc1RgbaUnormSrgb, W::Bc1RgbaUnorm),
            (F::Bc2RgbaUnorm, W::Bc2RgbaUnormSrgb, W::Bc2RgbaUnorm),
            (F::Bc3RgbaUnorm, W::Bc3RgbaUnormSrgb, W::Bc3RgbaUnorm),
            (F::Rgba8Unorm,   W::Rgba8UnormSrgb,   W::Rgba8Unorm),
            (F::Bgra8Unorm,   W::Bgra8UnormSrgb,   W::Bgra8Unorm),
        ];
        for (raw, srgb, unorm) in cases {
            assert_eq!(resolve_wgpu_format(raw, Gamma, FollowCurve), srgb, "gamma+follow → sRGB");
            assert_eq!(resolve_wgpu_format(raw, Linear, FollowCurve), unorm, "linear+follow → unorm");
            assert_eq!(resolve_wgpu_format(raw, Gamma, ForceLinear), unorm, "force-linear overrides gamma");
            assert_eq!(resolve_wgpu_format(raw, Linear, ForceLinear), unorm, "force-linear");
        }
    }

    /// Non-sRGB-capable layouts ignore encoding + intent entirely.
    #[test]
    fn non_srgb_formats_are_invariant() {
        let cases = [
            (F::Bc4RUnorm, W::Bc4RUnorm),
            (F::Bc4RSnorm, W::Bc4RSnorm),
            (F::Bc5RgUnorm, W::Bc5RgUnorm),
            (F::Bc5RgSnorm, W::Bc5RgSnorm),
            (F::Rgba8Snorm, W::Rgba8Snorm),
            (F::Rg8Unorm, W::Rg8Unorm),
            (F::R8Unorm, W::R8Unorm),
            (F::Rgba16Float, W::Rgba16Float),
            (F::Rgba32Float, W::Rgba32Float),
        ];
        for (raw, want) in cases {
            for enc in [Gamma, Linear] {
                for intent in [FollowCurve, ForceLinear] {
                    assert_eq!(resolve_wgpu_format(raw, enc, intent), want);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Max simultaneous renderable objects (scenery placements + dynamic
/// objects). Sized for deadlock-class MP maps which carry 300+ placements
/// after counting all object types. Bumped from 256 → 2048 to absorb
/// future maps without re-tuning.
pub const MAX_OBJECTS: usize = 2048;

// ---------------------------------------------------------------------------
// GPU-side model types (visible to pass modules)
// ---------------------------------------------------------------------------

pub(crate) struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub parts: Vec<ModelMeshPart>,
    /// `Some` when the source mesh carries a baked PRT Ambient transfer
    /// stream (`RenderMesh::prt_ambient_stream`). One `f32` per vertex,
    /// bound to slot 1 for `HaloEntryPoint::StaticPrtAmbient` draws.
    /// Mirrors the engine's per-mesh secondary VB used by
    /// `_entry_point_static_lighting_prt_ambient` — see
    /// `project_research_per_mesh_prt_2026_05_11.md`.
    pub prt_ambient_buffer: Option<wgpu::Buffer>,
    /// `Some` when the source mesh has the engine
    /// `_mesh_has_vertex_color_bit` flag set (sky `.render_model` meshes
    /// are the canonical user) AND the stream survives the load-time
    /// drop heuristic (see `loader::convert_mesh`). One
    /// `SkyVertColorVertex` (vec4 — RGB + 0 padding) per vertex, bound
    /// to vertex buffer slot 1 for the sky pipeline draws inside
    /// `submit_and_render_sky(1, ...)`. Engine equivalent: the
    /// `_vertex_buffer_usage_vert_color` stream consumed by
    /// `_entry_point_vertex_color_lighting` (idx=14, sky shader path).
    pub vert_color_buffer: Option<wgpu::Buffer>,
    /// `MeshFlags::UseRegionIndexForSorting` — transparent parts sort by
    /// `region_index` (constant centroid + region offset), keeping co-located
    /// transparent layers in fixed composite order. See `ModelMesh`.
    pub use_region_index_for_sorting: bool,
    /// Region this mesh belongs to (engine `region_index` sort key).
    pub region_index: u8,
}

pub(crate) struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    /// Per-scenario-cubemap-probe variants of [`Self::bind_group`] — one
    /// per probe, with the `dynamic_environment_map_*` cube slots pointed
    /// at that probe's cube. Selected at draw by the object's resolved
    /// `cubemap_probe_index` (engine `render_method_submit_dynamic_cubemap`).
    /// Empty when the material samples no dynamic-env cube or no probe
    /// atlas is loaded → draw uses `bind_group` (rasg-default cube).
    pub bind_group_probes: Vec<wgpu::BindGroup>,
    /// Pipeline + cbuffer layout for this material's static-lighting
    /// variant (HLSL `static_sh_ps` / `static_per_pixel_ps`). Drives
    /// the lit-color output in `c_player_view::render_static_lighting`.
    pub artifacts: Rc<VariantArtifacts>,
    /// Pipeline for the `_entry_point_albedo` first-pass G-buffer fill.
    /// Outputs to `_surface_albedo` + normal G-buffer; no lighting.
    /// Used by the new `c_player_view::render_albedo` body.
    pub artifacts_albedo: Option<Rc<VariantArtifacts>>,
    /// Bind group for `artifacts_albedo` (separate because the cbuffer
    /// layout for the albedo entry-point may differ from the SL one
    /// when slot ordering changes).
    pub bind_group_albedo: Option<wgpu::BindGroup>,
    /// Pipeline for the `_entry_point_static_lighting_prt_ambient`
    /// variant. Compiled only for rmsh-family subclasses (others either
    /// lack the WGSL or aren't drawn through this opaque object path).
    /// Activated at draw time when the mesh has
    /// `prt_ambient_buffer.is_some()`.
    pub artifacts_prt_ambient: Option<Rc<VariantArtifacts>>,
    /// Bind group for `artifacts_prt_ambient`.
    pub bind_group_prt_ambient: Option<wgpu::BindGroup>,
    /// Pipeline for `_entry_point_vertex_color_lighting`
    /// (`static_per_vertex_color_ps`) — the diffuse term is the per-vertex
    /// baked `vert_color` (slot 1) instead of the SH probe. Compiled only
    /// for materials on a model that carries a vert_color stream (sky
    /// `.render_model` meshes). Drawn by `submit_and_render_sky` /
    /// `SkyMeshPart` when the mesh has a `vert_color_buffer`.
    pub artifacts_vertex_color: Option<Rc<VariantArtifacts>>,
    /// Bind group for `artifacts_vertex_color`.
    pub bind_group_vertex_color: Option<wgpu::BindGroup>,
    /// Pipeline for the rmd-on-object `_entry_point_default` variant —
    /// engine `c_object_renderer::render_albedo_decals @ 0x1806E4A60`
    /// dispatches these through a skinned-VS port of `decal_fx::default_ps`
    /// (see `entry_decal_object.wgsl`). Compiled only for rmd materials
    /// when carried by an object's render_model material list (e.g.
    /// `human_military_decals_2` scenery whose mesh parts have the
    /// `_render_object_mesh_part_decal_bit` set).
    pub artifacts_decal_object: Option<Rc<VariantArtifacts>>,
    /// Bind group for `artifacts_decal_object`. Includes the per-binding
    /// `decal_constants` UBO bound to the shared
    /// `default_decal_constants_buffer` with dynamic offset 0 (engine
    /// object decals don't author per-decal sprite/fade — defaults
    /// fade=1, pixel_kill=1, tiles=1, sprite=(0,0,1,1)).
    pub bind_group_decal_object: Option<wgpu::BindGroup>,
    /// 32 B buffer holding `DecalConstantsGpu::default()` for the
    /// object-decal pipeline's dynamic-offset binding. Bound via
    /// `bind_group_decal_object` at offset 0. Lives on the material so
    /// it shares the upload lifecycle.
    pub default_decal_constants_buffer: Option<wgpu::Buffer>,
    /// Source shader tag path (e.g. `levels\…\matte_waypoint_reverse`).
    /// Used by per-shader render-state overrides (sky pass skips
    /// matte_waypoint* until depth-only state lands).
    pub shader_name: String,
    /// Engine-faithful `c_render_method::get_is_transparent`. Drives
    /// dispatch routing: opaque materials draw inline in the
    /// render_albedo pass; transparent materials get queued into
    /// `transparency_renderer` so they sort with everything else.
    pub is_transparent: bool,
    /// rmsh `blend_mode` category choice (`opaque`, `alpha_blend`,
    /// `additive`, etc.) at load time. Captured here so the
    /// `SkyMeshPart` dispatch can pick the matching variant of
    /// `SkyGpu::pipeline_*` (engine `_entry_point_vertex_color_lighting`
    /// is one shader; the `blend_mode` rmdf option toggles the
    /// device-state blend, not the shader body). For non-rmsh
    /// subclasses (rmtr/rmfl/rmw/etc.) this is whatever their rmdf
    /// authors at the same category slot — irrelevant for sky meshes
    /// which always come from rmsh / rmcs.
    pub blend_mode: String,
    /// FOURCC of the source `rm**` tag (`'rmsh'`, `'rmtr'`, `'rmw '`,
    /// etc.) — copied from the underlying `ResolvedRenderMethod` at
    /// load time. Used at draw time to honor engine
    /// `c_structure_renderer::render_cluster_mesh_part @ 0x18068F7F0:111`
    /// — terrain (`rmtr`) ALWAYS routes to `_entry_point_static_lighting_sh`
    /// regardless of lightmap atlas availability (engine
    /// `setup_default_lighting` is called as part of the swap). Without
    /// the gate, an rmtr cluster with atlas coverage would sample
    /// per-pixel SH instead of the SH-cbuffer path the engine uses.
    pub group_tag: u32,
    /// Byte-offset into `shared.atmosphere_buffer`'s per-frame table that
    /// THIS material's draws bind at `camera_bgl` binding 2 (dynamic
    /// offset). Resolved once at load from the source render_method's
    /// `flags` + `custom_fog_setting_index` via `shared::atmosphere_offset_for`
    /// — engine `render_method_submit_volatile_per_shader`'s per-shader
    /// atmosphere override (`UseCustomSetting` → setting[idx], `DontFogMe`
    /// → disabled, else → cluster default). Almost always the default (0);
    /// sky backdrop materials resolve to their custom setting (e.g.
    /// construct's `fog_waterfall`).
    pub atmosphere_offset: u32,
    /// Transparent draw-order bucket — the render_method's authored
    /// `sort layer` (`GlobalSortLayer`) mapped to the runtime
    /// `TransparentSortLayer`. Primary key of the back-to-front transparent
    /// sort (engine `transparent_layer_and_z_sort_proc`). `Normal` for
    /// opaque/stub materials.
    pub sort_layer: crate::halo::render::render_transparents::TransparentSortLayer,
    /// @group(1) bind group for the alpha-tested shadow-generate pipeline
    /// (`shadow_generate.wgsl::fs_alpha_test`): the material's
    /// `alpha_test_map` view + sampler + `alpha_test_map_xform`. `Some`
    /// only for OPAQUE materials whose render method has `alpha_test != off`
    /// (grates, chain-link, foliage props) — they cast holey shadows.
    /// Built once at load via `ShadowApplyPass::build_alpha_test_bind_group`.
    /// `None` → the part draws through the plain (opaque) shadow pipeline.
    pub shadow_alpha_test_bg: Option<wgpu::BindGroup>,
}

pub(crate) struct GpuModel {
    pub meshes: Vec<GpuMesh>,
    pub materials: Vec<GpuMaterial>,
    /// Per-animated-material state — one entry per `GpuMaterial` whose
    /// source `MaterialData` is flagged `is_animated`. Walked per
    /// frame by `update_animated_cbuffers`. Mirrors
    /// `BspGpu::animated_materials`. Empty for models with no
    /// TagFunction-time materials.
    pub animated_materials: Vec<crate::halo::render_methods::animated::AnimatedMaterial>,
    /// Model-space bounding sphere — `(center.x, center.y, center.z,
    /// radius)`. Computed at upload time from `ModelMesh::vertices`
    /// (AABB → sphere) when the source `hlmt::s_model_bounding_sphere`
    /// isn't available; this matches engine `autogen` mode. Used by
    /// shadow_apply_pass to size the per-caster ortho — engine path is
    /// `object_get_bounding_sphere` (returns world-space sphere).
    pub bounding_sphere: glam::Vec4,
    /// Model-space axis-aligned bounding box `(min, max)` over every
    /// vertex of every mesh. The engine's shadow camera (`c_lightmap_
    /// shadows_view::setup_camera @ 0x1806BD8B0`) uses
    /// `m_projection_bounds` — a 3-extent OBB oriented to the light —
    /// to size the ortho. We mirror that by projecting the 8 AABB corners
    /// into light-space at shadow-camera build time, taking the per-axis
    /// min/max for tight non-spherical bounds.
    pub bounding_aabb_min: glam::Vec3,
    pub bounding_aabb_max: glam::Vec3,
    /// Mirror of `ModelData::casts_shadow` — pre-baked at object load
    /// from the obje tag's flags + lightmap_shadow_mode + type. Bit 1 of
    /// the engine's `cluster_partition.object_visibility_cull_flags`.
    /// The shadow loop in `views/player_view.rs` skips placements whose
    /// model has `casts_shadow == false` — most static scenery on a
    /// typical map (`lightmap_shadow_mode == default` + `type == scenery`).
    pub casts_shadow: bool,
    /// Spec bind-pose: object-local node-world matrices (precomputed at
    /// load into `ModelData::node_local_matrices`). The renderer uploads
    /// these as the per-object `node_matrices` (`model_u × node ×
    /// vertex`). Static per model until the real animation system makes
    /// them per-instance.
    pub node_local_matrices: Vec<glam::Mat4>,
    /// The model's markers (name → object-local marker world matrix).
    /// Used to attach parented objects to a marker on this model (engine
    /// `object_attach_to_marker`).
    pub marker_transforms: Vec<(String, glam::Mat4)>,
}

impl GpuModel {
    /// Re-resolve every animated material's cb13 (user) cbuffer at
    /// `time_seconds` and queue the bytes into each cached
    /// `props_buffer`. No-op for models with no animated materials.
    /// Mirrors `BspGpu::update_animated_cbuffers`.
    pub fn update_animated_cbuffers(&self, queue: &wgpu::Queue, time_seconds: f32) {
        for animated in &self.animated_materials {
            animated.write_at_time(queue, time_seconds);
        }
    }
}

/// Resolved scenario simple-light — one per
/// `scenario_structure_lighting_info::generic_light_instances[i]`,
/// with the matching definition's color/intensity/cone params already
/// folded in. Per-frame, `Renderer::render` picks the 8 nearest by
/// camera distance and feeds them through
/// `LightsView::add_simple_light_to_draw_list` (which finishes the
/// shader-friendly encoding).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScenarioLight {
    pub position: Vec3,
    /// Direction the light points (= -inv_direction the shader stores).
    /// Zero vector for omni lights.
    pub direction: Vec3,
    /// HDR color (color × intensity).
    pub color: Vec3,
    /// Outer cone half-angle in radians; 0 for omni lights so the
    /// shader's cone_scale collapses (see
    /// `LightsView::initialize_simple_light`).
    pub cone_angle: f32,
    /// Cone smoothness — engine default is 1.0 for sharp cone, larger
    /// values widen falloff. We pick 1.5 by default.
    pub cone_smoothness: f32,
    /// Sphere percentage — fraction of the cone that's at full
    /// brightness with no angular falloff. 0 for spotlights, 1.0 for
    /// pure omni (so the cone term goes flat).
    pub sphere_percentage: f32,
    /// Maximum reach in world units — typically the definition's
    /// `far_attenuation_bounds.upper`. Used for both bounding-radius
    /// culling and the linear distance-falloff `(1 - d/max_dist)`.
    pub max_dist: f32,
    /// Distance-falloff "size" — additive constant in
    /// `1 / (size + d²)` that caps the max value to `1/size` at d→0.
    /// `LightsView::initialize_simple_light` ALSO divides the upload
    /// color by `size`, so smaller size → brighter peak. The H3 stli
    /// schema doesn't carry an authored size field, so we pick 1.0:
    /// keeps peak at d=0 equal to the authored color×intensity (no
    /// division blow-up) while still producing inverse-square falloff
    /// at distance.
    pub size: f32,
}

/// One placement's raw SH probe + per-type chmt key, cached at scenario
/// load so [`Renderer::refresh_engine_lighting_for_frame`] can re-apply
/// the per-frame chmt boost without re-running the BSP raycast +
/// probe-lookup chain. See OL-6 in
/// [[reference_object_lighting_full_2026_05_24]].
#[derive(Debug, Clone)]
pub struct ObjectLightingCacheEntry {
    /// Raw (un-chmt-scaled) red-channel SH3 coefficients from the
    /// dequantized probe.
    pub sh_r: [f32; 9],
    pub sh_g: [f32; 9],
    pub sh_b: [f32; 9],
    /// Cached NTSC-luminance estimate over the first 4 SH coefficients
    /// (`sh_luminance_estimate`). SH is fixed per placement, so this is
    /// computed once at bake time. Engine `setup_object_lighting_for_entry_point`
    /// recomputes per-draw — same result, we just save the cycles.
    pub sh_lum: f32,
    /// `e_object_type` discriminant — index into the chmt
    /// `per_object_type_min_luminance[]` table.
    pub object_type: crate::halo::objects::object_type::ObjectType,
    /// Byte offset into `engine_lighting_buffer` where this placement's
    /// 10-vec4 ravi cbuffer entry lives. Assigned sequentially as
    /// placements get baked (`structure_renderer.next_probe_slot`).
    pub slot_offset: u32,
    /// Baked per-placement dominant-light direction (world-space, points
    /// TO the light) from the probe at this object's position. Used by
    /// the object-shadow loop so each caster's shadow points along its
    /// OWN local dominant light (engine `object_get_cached_render_lighting`)
    /// rather than one global direction — matters where lighting varies
    /// (indoor vs sunlit doorway). `[0,0,0]` if the probe had no usable
    /// dominant; the shadow loop falls back to the global dominant then.
    pub dominant_dir: [f32; 3],
}

// ---------------------------------------------------------------------------
// Renderer orchestrator
// ---------------------------------------------------------------------------

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    shared: SharedResources,
    gbuffer: GBuffer,
    intermediates: IntermediateTargets,

    // Pass owners — typed fields per `c_player_view::render`'s
    // explicit pass calls. Replaces the old `Vec<Box<dyn RenderPass>>`
    // scheduler; render() now invokes each pass by name in the
    // canonical order.
    pub shadow_apply: shadow_apply_pass::ShadowApplyPass,
    pub patchy_fog: patchy_fog_pass::PatchyFogPass,
    final_composite_pass: final_composite_pass::FinalCompositePass,

    halo_pipelines: HaloPipelineCache,

    models: Vec<GpuModel>,

    // BSP / structure rendering — separate from per-character path.
    /// `c_structure_renderer` — owns the loaded BSPs, per-frame draw
    /// lists, identity bind groups for cluster draws, and the
    /// per-instance probe-slot allocator. Engine equivalent:
    /// `g_render_structure_globals` + the all-static
    /// `c_structure_renderer` namespace.
    structure_renderer: crate::halo::render::structure_renderer::StructureRenderer,

    // dllcache `c_render_method_data` + `c_lighting_interface` mirrors.
    // Per-pass / per-draw render state read by the 49-extern dispatch
    // (see reference_h3_extern_table.md). Populated from sky tag at
    // scenario load (lighting interface) and refreshed per pass /
    // per draw (render method data).
    render_method_data: render_method_data::CRenderMethodData,
    lighting_interface: lighting_interface::LightingInterface,

    /// Atmospheric scattering parameters from the active scenario's
    /// `atmospheric` tag (`.sky_atm_parameters`). Uploaded to the
    /// `engine_atmosphere` cbuffer per frame. Set by `load_scenario`;
    /// defaults to neutral (no fog) before any scenario loads.
    ///
    /// Treated as a fallback now — when L3 per-cluster resolve has the
    /// data it needs (`scenario_sky_atmosphere` + `active_bsp_clusters` +
    /// `active_bsp_atmosphere_palette`), `views::render_player_view` rebuilds
    /// the cbuffer per frame from the eye's containing cluster.
    scenario_atmosphere: atmosphere_fog_interface::GpuAtmosphereData,
    /// Cached `SkyAtmosphere` (the scenario's `atmospheric` →
    /// `.sky_atm_parameters`). Drives the L3 per-cluster resolve at
    /// frame time. `None` before a scenario loads.
    scenario_sky_atmosphere: Option<blam_tags::sky_atmosphere::SkyAtmosphere>,
    /// Cached active-BSP cluster bounds + atmosphere-palette indirection
    /// table for the L3 per-cluster resolve. Cluster's `atmosphere_index`
    /// (i8) → palette entry → `atmosphere_setting_index` (i16) →
    /// `scenario_sky_atmosphere.atmosphere_settings[]`. Engine
    /// `c_atmosphere_fog_interface::get_atmosphere_setting @
    /// 0x1803AFBA0`.
    scenario_active_bsp_clusters: Vec<blam_tags::structure_bsp::BspCluster>,
    scenario_active_bsp_atmosphere_palette: Vec<blam_tags::structure_bsp::BspAtmospherePaletteEntry>,
    /// Cluster portals + collision-BSP planes of the active BSP — needed by
    /// the SP `compute_cluster_weights` sphere walk (structure_clusters_in_sphere).
    scenario_active_bsp_cluster_portals: Vec<blam_tags::structure_bsp::BspClusterPortal>,
    scenario_active_bsp_planes: Vec<blam_tags::math::RealPlane3d>,
    /// `c_atmosphere_fog_interface` singleton mirror — owns the
    /// accumulated `s_weighted_atmosphere_parameters` and the
    /// last-applied-cbuffer bookkeeping. Populated each frame by
    /// `compute_cluster_weights → populate_atmosphere_parameters`
    /// (Phases 3–4 of the atmosphere port plan). Not yet consumed —
    /// existing cbuffer build paths still go through `env_probe_pass`
    /// until Phase 5 folds them onto `set_constant`.
    pub atmosphere_fog_interface:
        crate::halo::render::atmosphere_fog_interface::AtmosphereFogInterface,
    /// View exposure cached from `c_camera_fx_settings`. Used as
    /// `slot7.x` in the rebuilt atmosphere cbuffer.
    scenario_view_exposure: f32,
    /// Sky sun (`get_sun_constants_from_sky @ 0x1803adcb0`) — engine's
    /// fallback when the picked atmosphere setting's bit-1 ("Override
    /// Real Sun Values") is unset.
    scenario_sky_sun: Option<atmosphere_fog_interface::SkySun>,
    /// Active scenario's camera_fx_settings tag — drives bloom,
    /// color grading, SSAO, lightshafts via `ScreenPostprocess`.
    /// `None` until a scenario is loaded.
    pub scenario_camera_fx: Option<blam_tags::camera_fx_settings::CameraFxSettings>,
    /// Per-window stored cfxs (engine `s_render_game_state::s_player_window::
    /// m_camera_fx_settings`). Without scripted-exposure animations active
    /// it stays at the `set_defaults(use_default=1)` initial state — every
    /// parameter falls through to `scenario_camera_fx` in the chooser.
    pub script_camera_fx: blam_tags::camera_fx_settings::CameraFxSettings,
    /// `globals.default_camera_fx_settings` — the engine fallback cfxs
    /// loaded from `globals\\globals.globals`. Final level of the
    /// `c_camera_fx_values::update` 4-level chooser. `None` if the globals
    /// tag doesn't author one (rare).
    pub default_camera_fx: Option<blam_tags::camera_fx_settings::CameraFxSettings>,
    /// `c_player_view::m_render_exposure` — the linear exposure multiplier
    /// the postprocess shader reads via `g_exposure.r`. Computed by
    /// `setup_camera_fx_parameters` from `m_camera_fx_values.m_exposure.m_exposure`
    /// (log2 stops) via `pow(2, stops + scripted_stops + boost) ×
    /// tone_curve_white_point × 0.66943294`.
    pub render_exposure: f32,
    /// `c_player_view::m_illum_render_scale` — `pow(2, m_self_illum_exposure
    /// - apply_scripted(m_exposure))`. Drives `g_alt_exposure.r` (the
    /// self-illumination multiplier in opaque shaders).
    pub illum_render_scale: f32,
    /// Active scenario's `chocolate_mountain_new` tag — per-object-type
    /// minimum-luminance floor. Loaded once at scenario load. The
    /// engine applies it via
    /// `c_chocalate_moutain::apply_chocalate_mountain_lighting @
    /// 0x1806f9ee0` from `object_update_cached_render_lighting`.
    /// `None` when the scenario doesn't author one (engine returns
    /// mountain_scale = -1.0 in that case).
    pub scenario_chocolate_mountain: Option<blam_tags::chocolate_mountain::ChocolateMountain>,
    /// Tags root path of the currently-loaded scenario. Used by the
    /// shader assembler (MaterialBindings::from_resolved) to read
    /// each Bitmap parameter's bitm tag for cube-vs-2D classification.
    /// Empty before any scenario loads — calling assembly before
    /// load_scenario would (rightly) fail.
    scenario_tags_root: std::path::PathBuf,
    /// Runtime `simple_lights` source — `scenario.lights[]` placements
    /// (`light` tag instances), once that path is wired. Empty today:
    /// the prior stli `generic_light_instances` source was removed
    /// because engine doesn't route stli through runtime simple_lights
    /// — stli is an offline-bake input (atlas surface lights). See
    /// `resolve_scenario_lights` doc for the full story and the queued
    /// port.
    scenario_lights: Vec<ScenarioLight>,
    /// Engine resource registry from `globals/rasterizer_globals.rasterizer_globals`.
    /// Loaded once at scenario load so engine externs (Cook-Torrance
    /// LUTs, default fallback bitmaps for missing dynamic env probes,
    /// etc.) resolve to the authored bitmap path — same way Halo's
    /// runtime does it via `c_rasterizer_globals` lookup.
    rasterizer_globals: blam_tags::rasterizer_globals::RasterizerGlobals,

    /// The currently-loaded scenario. `None` before `load_scenario`
    /// runs. Held as `Arc` so per-frame consumers (visibility,
    /// scenario_location_from_point) can take cheap clones without
    /// fighting the borrow checker against `&mut self`. Engine
    /// equivalent: `global_scenario *` global pointer.
    pub loaded_scenario: Option<std::sync::Arc<crate::halo::scenario::LoadedScenario>>,

    /// GPU-side decal state — per-palette pipelines + bind groups +
    /// per-BSP vertex/index buffers. Built by
    /// [`crate::halo::render::render_decals::bake_decals_gpu`] inside
    /// `load_scenario`, BEFORE the `LoadedScenario` is wrapped in
    /// `Arc`. Engine equivalent: the per-decal-system runtime state
    /// the cache builder pre-resolves + the per-BSP
    /// `c_vertex_allocator` / `c_index_allocator` LRU pools.
    pub decal_gpu: crate::halo::render::render_decals::DecalGpuState,

    /// Runtime toggle for the global screen-effect (sefc) color grade — the
    /// mainmenu blue tint. `false` forces the IDENTITY grade in
    /// `render_player_view` so the scene renders untinted. Toggled with the P key
    /// (see main.rs). Default `true`.
    pub screen_effect_enabled: bool,

    /// GPU particle subsystem — the persistent RawParticleState grid +
    /// spawn/update/render compute (effects: waterfall mist, etc.).
    pub particle_gpu: crate::halo::gpu_particle::ParticleGpu,
    pub light_volume_gpu: crate::halo::gpu_particle::light_volume_gpu::LightVolumeGpu,

    /// Engine-faithful per-frame camera visibility collection
    /// (Phase F-G). Populated by [`crate::halo::render::render_visibility::render_visibility_camera_collection_compute`]
    /// at the top of each frame; consumers read its region via
    /// `FrameContext::camera_visibility_region`.
    pub camera_visibility: crate::halo::visibility::CVisibilityCollection,
    /// Engine `c_visible_items::m_items` global storage (Phase A2).
    /// Per-frame visible-items output produced by `compute()` — read
    /// by `submit_visibility_from_items` (Phase H).
    pub visible_items: crate::halo::visibility::CVisibleItems,
    /// Per-frame portal-activation state (Phase D2). All-active
    /// default; reserved for door-portal toggles when the gameplay
    /// machine layer ships.
    pub portal_activation: crate::halo::visibility::ActivePortalBitvectors,
    /// Cached camera→cluster lookup. Recomputed only when the camera
    /// position changes (engine recomputes every frame; we
    /// deduplicate to skip the BSP3D walk when the camera hasn't
    /// moved).
    pub camera_cluster_cache: crate::halo::scenario::CameraClusterCache,

    /// Loaded sky model (Phase B17). `Some` whenever the active
    /// scenario's `scenario.skies[0].sky.scenery → object → model →
    /// render_model` chain resolves; `None` for skybox-less scenarios.
    /// Stored separately from `self.models` (the placement pool) so
    /// the game-side `model_data[]` index stays in sync with the
    /// renderer's placement-driven model uploads. Per-frame the sky
    /// pass renders this with a camera-position-translated model
    /// matrix (parallax-locked sky).
    pub sky_model: Option<GpuModel>,
    /// Dedicated single-slot model_u + node_matrices buffers for the
    /// sky pass. Same bind group layouts as the placement pool, so
    /// sky pipelines (rmsh artifacts) bind seamlessly.
    pub sky_gpu: render_sky::SkyGpu,

    // Preallocated per-frame scratch buffers
    render_list: Vec<(ObjectIndex, usize)>,
    node_matrix_staging: Vec<u8>,

    // Temporal reprojection state
    prev_vp: glam::Mat4,
    frame_counter: u32,

    // -- Halo-faithful renderer-rebuild fields --
    //
    // `c_rasterizer` mirror — wraps the wgpu device/queue + state
    // caches. Phases 7c+ migrate pass bodies to issue calls via
    // this instead of touching wgpu directly.
    pub rasterizer: crate::halo::rasterizer::Rasterizer,

    /// Active world viewpoint. Carries cameras + sub-views + per-view
    /// state (exposure, observer DoF, motion blur reproj). Mirrors
    /// `c_player_view`.
    pub player_view: crate::halo::render::views::PlayerView,

    /// `c_lights_view` mirror — holds the up-to-8 simple lights for
    /// the active player view. Populated at scenario load (L2c) +
    /// per-frame culled (L2d). Built each frame from `lights_view` and
    /// uploaded to `shared.simple_lights_buffer`.
    pub lights_view: crate::halo::render::views::LightsView,

    /// `c_screen_postprocess` mirror — bloom + tonemap + color
    /// grading pipelines and scratch surfaces. Driven from
    /// `Renderer::render`'s step 19 (`postprocess_player_view`).
    pub screen_postprocess: crate::halo::render::screen_postprocess::ScreenPostprocess,

    /// `g_transparency_renderer` mirror — global transparent-element
    /// pool + sort + dispatch. Populated during `submit_visibility`
    /// and drained in `PlayerView::render_transparents`. Halo:
    /// `c_transparency_renderer @ 0x182592E57`.
    pub transparency_renderer: crate::halo::render::render_transparents::TransparencyRenderer,

    /// `c_object_renderer` mirror — per-frame object_render_contexts
    /// + context_mesh_parts + marker stack. Populated by
    /// `submit_visibility` and walked by render_albedo /
    /// render_static_lighting / render_lights / render_shadows_*.
    /// Halo: `g_render_object_globals_internal` (TLS pointer).
    pub object_renderer: crate::halo::render::render_objects::ObjectRenderer,

    /// `c_water_renderer` mirror — owns water-specific render state
    /// (visible water cluster part list, ripple buffer interaction,
    /// underwater flag) and dispatches the water-shading sub-pass
    /// between `render_first_person(0)` and `render_transparents`.
    /// Per `reference_dllcache_water_pipeline.md`. Engine: namespace
    /// class `c_water_renderer` with all-static methods + globals
    /// (`g_water_render_structure_part_indices[512]`,
    /// `g_water_render_structure_part_count`, `g_is_underwater`,
    /// `g_xm_view_proj_inv`, `g_water_tessellation_*_buffer`).
    pub water_renderer: crate::halo::render::render_water::WaterRenderer,

    /// `c_decorator_renderer` mirror — GPU-instanced foliage rendering
    /// for the scenario's painted decorator placements (~26K instances
    /// on riverworld). Loaded at scenario load via `load_scenario`,
    /// drawn per-frame in `render_decorators`. See
    /// `reference_mcc_strips_decorator_data.md` for the data path.
    pub decorator_renderer: crate::halo::render::render_decorators::DecoratorRenderer,

    /// L4 — engine_lighting buffer offsets (in bytes) for each scenery
    /// placement's baked SH probe. `Some(offset)` when the placement has a
    /// matching probe in `lightmap_bsp_data.scenery_probes`; `None` when
    /// the lookup falls through (no lightmap, out-of-range index, etc.) —
    /// the per-object draw uses the frame-default sky probe (offset 0) in
    /// that case. Mirrors engine `lights_prepare_for_object_static_new`'s
    /// type==6 branch caching the resolved sample on the object's
    /// `cached_render_lighting`.
    pub scenery_lighting_offsets: Vec<Option<u32>>,
    /// Same as `scenery_lighting_offsets` but for non-scenery dynamic
    /// placement types (crate, weapon, equipment, machine, control).
    /// Engine: when the per-object type is NOT 6 (scenery) or 7 (device
    /// machine), `lights_prepare_for_object_static_new` falls back to
    /// `c_geometry_sampler::sample_one_air_probe(position, ...)` — i.e.,
    /// nearest-airprobe-by-position. We bake the lookup at scenario load
    /// since both placement positions and airprobe positions are static.
    pub crate_lighting_offsets: Vec<Option<u32>>,
    pub vehicle_lighting_offsets: Vec<Option<u32>>,
    pub weapon_lighting_offsets: Vec<Option<u32>>,
    /// Per-scenario-cubemap-probe cube views for OBJECT dynamic-env
    /// mapping — the same atlas the BSP per-cluster routing uses
    /// (`<scenario>_<bsp>_cubemaps.bitmap`). Objects pick their nearest
    /// probe by position (engine `c_dynamic_cubemap_sample`); the
    /// resolved cube binds to the material's `dynamic_environment_map_*`
    /// slots so glass/env-mapped objects reflect the real environment
    /// instead of the rasg neutral default. Retained from BSP 0 at load
    /// (single-BSP scenarios). Empty → no atlas → rasg default cube.
    pub object_cubemap_probe_views: Vec<wgpu::TextureView>,
    /// World positions of each `object_cubemap_probe_views` entry (1:1
    /// with `scenario.cubemaps[]`), for nearest-probe selection.
    pub object_cubemap_positions: Vec<glam::Vec3>,
    pub equipment_lighting_offsets: Vec<Option<u32>>,
    pub machine_lighting_offsets: Vec<Option<u32>>,
    pub control_lighting_offsets: Vec<Option<u32>>,
    /// Per-placement cache of raw (un-chmt-scaled) SH probe data,
    /// indexed by engine_lighting slot. Populated at scenario load by
    /// `bake_object_lighting_offsets`; consumed every frame by
    /// `refresh_engine_lighting_for_frame` which applies the per-frame
    /// chmt boost (`mountain_scale = sqrt(min_lum / render_exposure)`)
    /// and writes the scaled ravi to `engine_lighting_buffer`.
    ///
    /// Engine equivalent: each `c_object`'s `render_lighting.lightprobe_sample`
    /// caches the raw SH; `c_lighting_interface::setup_object_lighting_for_entry_point
    /// @ 0x1806A9AA0` re-applies the chmt boost every draw against the
    /// current frame's `c_player_view::render_exposure`. We do it once
    /// per frame across all placements (engine's per-draw granularity
    /// doesn't help us — the result is identical for every draw of a
    /// given object within a frame). See OL-6 in
    /// [[reference_object_lighting_full_2026_05_24]].
    pub object_lighting_cache: Vec<ObjectLightingCacheEntry>,
    /// `c_camera_fx_values` runtime state — engine `s_render_game_state::
    /// player_window[i].m_camera_fx_values`. Holds the smoothed exposure,
    /// 60-frame history ring, FIFO of pending GPU readbacks, plus all the
    /// per-frame postprocess scalars (bloom_*, bling_*, self_illum_*, ...)
    /// that `c_camera_fx_values::update` (L3) animates each frame.
    ///
    /// Initialized at scenario load via `default_init_from(&cfxs)` —
    /// engine-symmetric per `s_render_game_state::initialize_for_new_map @
    /// 0x180682790`.
    pub render_game_state: crate::halo::render::render_game_state::RenderGameState,
    /// `c_camera_fx_values::m_exposure_boost` — auxiliary stops
    /// added on top of `m_exposure` and the cfxs target. Defaults to
    /// 0; v1 doesn't expose a way to drive it. Hooked into
    /// `view_exposure` for engine-faithful formula completeness.
    pub exposure_boost: f32,
    /// 60-entry ring of recent `actual_brightness` samples (log2
    /// stops). Diagnostic-only; populated each frame when a fresh
    /// FIFO sample arrives. `PROTOMORPH_DEBUG_EXPOSURE` logs
    /// per-second min/max/stddev so bloom-feed noise (Issue #3) is
    /// visible. Not used by the auto-exposure logic itself.
    actual_brightness_ring: [Option<f32>; 60],
    /// `scripted_exposure_game_globals` — engine global script-driven
    /// exposure state (FAST_ADAPT bit + history-replace bit). Default-
    /// zero matches engine behavior when no scripted exposure HSC is
    /// active. Wiring HSC support to set the flags is deferred — keep
    /// the field so the gates in `calculate_new_value` see the engine-
    /// faithful flag set.
    scripted_exposure_game_globals: crate::halo::render::camera_fx_settings::ScriptedExposure,
    /// `c_camera_fx_values::g_exposure_stops` — class-level static
    /// mutated by `g_render_set_exposure` HSC opcode and several
    /// cinematic helpers. Default 0. Feeds:
    ///   - `c_screen_postprocess::postprocess_player_view::v21`
    ///     (bloom-extract exposure compensation).
    ///   - `c_camera_fx_values::get_render_exposure` (post-tone-curve
    ///     multiplier read as `g_exposure.r` in postprocess shaders).
    ///   - `c_player_view::setup_camera_fx_parameters` self-illum
    ///     calculations (not yet wired — see Phase 7.7).
    /// Engine site: `c_camera_fx_values::g_exposure_stops` static.
    g_exposure_stops: f32,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // `DISCARD_HAL_LABELS` drops debug-label strings going to wgpu-hal —
        // they're useless without a frame capture and cost an alloc per
        // labelled object. `from_build_config()` enables DEBUG/VALIDATION
        // only in debug builds.
        let instance_flags = wgpu::InstanceFlags::from_build_config()
            | wgpu::InstanceFlags::DISCARD_HAL_LABELS;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: instance_flags,
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            // Halo forward pipeline needs more bind group slots and
            // more sampled textures per stage than wgpu's defaults:
            // - 5 bind groups (camera, model, material, node-matrices,
            //   plus env-probe pass headroom)
            // - Terrain (rmtr) declares ~17 texture samplers per
            //   variant (4× per-layer base+detail+bump+detail_bump +
            //   blend_map + rmsh-baseline union). Default cap is 16.
            //   Raise to 32 — comfortably above any single Halo shader.
            // The particle update kernel binds 9 storage buffers (grid +
            // physics + the 5 curve-eval tables + row-batch + the
            // transition/periodic LUT); wgpu's portable default caps storage
            // buffers per stage at 8. Metal (this app's target) supports 31,
            // so raise to 12 (headroom). The two `..default()` fields below are
            // already Metal-reliant in the same way.
            required_limits: wgpu::Limits {
                max_bind_groups: 5,
                max_sampled_textures_per_shader_stage: 32,
                max_storage_buffers_per_shader_stage: 12,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        // Engine's PC final_composite (`final_composite_base_hlsl.hlsl:73-76,
        // 95-132`) outputs LINEAR HDR to its LDR backbuffer; engine relies on
        // the SWAPCHAIN doing the linear→sRGB encode at present time.
        //
        // Picking a linear (non-sRGB) format here is NOT engine-faithful —
        // engine expects sRGB. It works visually on macOS only because
        // CAMetalLayer applies the sRGB encode itself when the surface format
        // is `BGRA8Unorm` (layer colorspace defaults to sRGB), so the chain
        // ends up linear-scene → Metal-encodes-on-scanout → display correctly.
        //
        // Switching to a *Srgb format on macOS double-encodes (wgpu's encode
        // stacked on Metal's), producing bright+desaturated output. Until
        // protomorph tests on Windows/Linux where Vulkan/DX12 swapchains
        // don't auto-encode for linear formats, keep the linear pick — but
        // the engine-faithful path is sRGB once the platform-quirk is solved.
        let format = caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut shared = SharedResources::new(device, queue, config);

        let gbuffer = GBuffer::new(&shared.device, shared.config.width, shared.config.height);
        let intermediates = IntermediateTargets::new(
            &shared.device,
            shared.config.width,
            shared.config.height,
            shared.config.format,
        );
        // Swap the placeholder G-buffer views in shared with the real
        // ones from intermediates + gbuffer. The static_lighting PSs
        // sample these via camera_bgl bindings 10 / 11.
        let gbuffer_albedo_view = intermediates
            .albedo_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let gbuffer_normal_view = gbuffer
            .normal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // DepthOnly scene-depth view → transparent camera bind group's
        // soft_z sampling (binding 15). View the COPY texture, not the
        // live depth attachment, so sampling doesn't alias the read-only
        // depth attachment in the transparent pass (TBDR flicker — see
        // `GBuffer::depth_sample_view`). Refreshed via
        // `GBuffer::copy_depth_for_sampling` before the transparent pass.
        let gbuffer_depth_view = gbuffer.depth_copy_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene_depth_soft_z"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        shared.set_gbuffer_views(gbuffer_albedo_view, gbuffer_normal_view, gbuffer_depth_view);

        // Producer passes first (downstream passes capture cubemap + sampler refs)
        let shadow_apply = shadow_apply_pass::ShadowApplyPass::new(&shared);
        let patchy_fog = patchy_fog_pass::PatchyFogPass::new(&shared);

        // c_screen_postprocess built early so its bloom output view
        // is available to final_composite_pass at construction.
        let mut screen_postprocess =
            crate::halo::render::screen_postprocess::ScreenPostprocess::new(&shared);
        screen_postprocess.editable_settings.postprocess = true;
        screen_postprocess.settings_internal.postprocess = true;

        let final_composite_pass = final_composite_pass::FinalCompositePass::new(
            &shared,
            &intermediates,
            &gbuffer,
            shared.config.format,
            screen_postprocess.bloom_result_view(),
            screen_postprocess.color_grading_active_view(),
            screen_postprocess.color_grading_back_view(),
            screen_postprocess.dof_blur_view(),
            screen_postprocess.bling_result_view(),
        );

        // Phase 1 frame order. Halo's `c_player_view::render` ordering
        // (per IDA): shadow_generate → albedo → static_lighting →
        // dynamic_lighting → ... → composite. Sky is rendered as an
        // OBJECT inside the geometry pass (Halo's `.sky` tag → render_model
        // path) — no separate pass needed.

        // Halo-faithful renderer-rebuild fields. Built before the
        // `Self {...}` literal so we can construct from `shared`'s
        // device/queue/config without a borrow conflict.
        let mut rasterizer = crate::halo::rasterizer::Rasterizer::new(
            shared.device.clone(),
            shared.queue.clone(),
            shared.config.width,
            shared.config.height,
        );
        rasterizer.surface_table.allocate_default(&shared.device);
        // Hand the rasterizer's Kernel5PS pool slot (0x33) over to
        // `SharedRenderState` so the camera_bgl binding 14 reads from
        // the same buffer that `set_shader_constant_bool(0x80330005, …)`
        // writes into. Pre-allocated inside `Rasterizer::new`.
        let kernel5_buffer = rasterizer
            .cbuffer_pool
            .slots[0x33]
            .gpu
            .clone()
            .expect("Rasterizer::new pre-allocates cbuffer pool slot 0x33");
        shared.set_kernel5_ps_buffer(kernel5_buffer);
        let player_view = crate::halo::render::views::PlayerView::default();
        let lights_view = crate::halo::render::views::LightsView::default();
        // Allocate the water-renderer here so we can pass `&shared`
        // before it moves into the struct literal below. Mirrors
        // `c_water_renderer::initialize @ 0x180692400` (dllcache) —
        // uploads the engine-static tessellation IB+VB and builds the
        // Phase A1d debug pipeline.
        let water_renderer =
            crate::halo::render::render_water::WaterRenderer::new(&shared, &intermediates, &gbuffer);
        let decorator_renderer =
            crate::halo::render::render_decorators::DecoratorRenderer::new(&shared);
        let sky_gpu = render_sky::SkyGpu::new(&shared);
        let particle_gpu = crate::halo::gpu_particle::ParticleGpu::new(&shared);
        let light_volume_gpu =
            crate::halo::gpu_particle::light_volume_gpu::LightVolumeGpu::new(&shared);

        Self {
            surface,
            shared,
            light_volume_gpu,
            gbuffer,
            intermediates,
            shadow_apply,
            patchy_fog,
            final_composite_pass,
            halo_pipelines: HaloPipelineCache::new(),
            models: Vec::new(),
            structure_renderer:
                crate::halo::render::structure_renderer::StructureRenderer::new(),
            render_list: Vec::with_capacity(MAX_OBJECTS),
            node_matrix_staging: vec![
                0u8;
                MAXIMUM_NUMBER_OF_MODEL_NODES
                    * std::mem::size_of::<[[f32; 4]; 4]>()
            ],
            prev_vp: glam::Mat4::IDENTITY,
            frame_counter: 0,

            render_method_data: render_method_data::CRenderMethodData::default(),
            // Initialized to the dllcache fallback (`setup_default_lighting`'s
            // built-in flat-sun + neutral-gray probe). Overwritten on
            // scenario load by D3.7's sky-tag walker, or per-instance
            // by D3.8's `select_instance_entry_point`.
            lighting_interface: lighting_interface::LightingInterface::fallback(),
            // Neutral atmosphere — `compute_scattering` returns ext=1,
            // ins=0 until a scenario is loaded.
            scenario_atmosphere: atmosphere_fog_interface::GpuAtmosphereData::neutral(),
            scenario_sky_atmosphere: None,
            scenario_active_bsp_clusters: Vec::new(),
            scenario_active_bsp_atmosphere_palette: Vec::new(),
            scenario_active_bsp_cluster_portals: Vec::new(),
            scenario_active_bsp_planes: Vec::new(),
            atmosphere_fog_interface:
                crate::halo::render::atmosphere_fog_interface::AtmosphereFogInterface::default(),
            scenario_view_exposure: 0.67,
            scenario_sky_sun: None,
            scenario_camera_fx: None,
            scenery_lighting_offsets: Vec::new(),
            crate_lighting_offsets: Vec::new(),
            vehicle_lighting_offsets: Vec::new(),
            weapon_lighting_offsets: Vec::new(),
            equipment_lighting_offsets: Vec::new(),
            machine_lighting_offsets: Vec::new(),
            control_lighting_offsets: Vec::new(),
            object_cubemap_probe_views: Vec::new(),
            object_cubemap_positions: Vec::new(),
            object_lighting_cache: Vec::new(),
            script_camera_fx: crate::halo::render::camera_fx_settings::defaulted_script_settings(),
            default_camera_fx: None,
            render_exposure: 1.0,
            illum_render_scale: 1.0,
            scenario_chocolate_mountain: None,
            scenario_tags_root: std::path::PathBuf::new(),
            scenario_lights: Vec::new(),
            rasterizer_globals: blam_tags::rasterizer_globals::RasterizerGlobals::default(),
            loaded_scenario: None,
            decal_gpu: crate::halo::render::render_decals::DecalGpuState::default(),
            screen_effect_enabled: true,
            particle_gpu,
            camera_visibility: crate::halo::visibility::CVisibilityCollection::new(),
            visible_items: crate::halo::visibility::CVisibleItems::new(),
            portal_activation:
                crate::halo::visibility::ActivePortalBitvectors::new_all_active(),
            camera_cluster_cache: crate::halo::scenario::CameraClusterCache::new(),
            sky_model: None,
            sky_gpu,

            rasterizer,
            player_view,
            lights_view,
            screen_postprocess,
            transparency_renderer:
                crate::halo::render::render_transparents::TransparencyRenderer::new(),
            object_renderer:
                crate::halo::render::render_objects::ObjectRenderer::new(),
            water_renderer,
            decorator_renderer,
            render_game_state: crate::halo::render::render_game_state::RenderGameState::initialize(),
            exposure_boost: 0.0,
            actual_brightness_ring: [None; 60],
            scripted_exposure_game_globals: Default::default(),
            g_exposure_stops: 0.0,
        }
    }

    /// Load a Halo `.crate` / `.biped` / `.weapon` / etc. (any tag
    /// inheriting `obje`) by walking the object → model → render_model
    /// chain. Materials are resolved through `halo::load_shader_material`
    /// → the blam-tags `render_method` walker.
    /// Object-local world matrix of `marker` on the sky scenery model
    /// (`node_world × marker.local`). The sky object is at the world origin,
    /// so this is also the marker's world matrix. For parent-marker
    /// attachment of sky-parented placements. `None` if there's no sky or no
    /// such marker.
    pub(crate) fn sky_marker_transform(&self, marker: &str) -> Option<glam::Mat4> {
        self.sky_model
            .as_ref()?
            .marker_transforms
            .iter()
            .find(|(n, _)| n.as_str() == marker)
            .map(|(_, m)| *m)
    }

    /// Object-local world matrix of `marker` on the model at `model_index`
    /// (`node_world × marker.local`). Compose with the object's world matrix
    /// for the marker's world pose. `None` if the model/marker is missing.
    pub(crate) fn model_marker_transform(
        &self,
        model_index: usize,
        marker: &str,
    ) -> Option<glam::Mat4> {
        self.models
            .get(model_index)?
            .marker_transforms
            .iter()
            .find(|(n, _)| n.as_str() == marker)
            .map(|(_, m)| *m)
    }

    /// Nearest object dynamic-env cubemap probe to `pos` — engine
    /// `c_dynamic_cubemap_sample::search_for_cubemap_sample` picks the
    /// nearest probe by 3D distance. `None` when no atlas is loaded (the
    /// caller then binds the rasg `DefaultDynamicCubeMap` fallback, as the
    /// engine does when `get_cluster_cubemap_index` returns -1).
    pub(crate) fn nearest_object_cubemap_probe(&self, pos: glam::Vec3) -> Option<u16> {
        nearest_by(&self.object_cubemap_positions, pos, |p| *p).map(|(i, _)| i as u16)
    }


    /// Build one render batch per particle system (loading each system's
    /// base/alpha/palette bitmaps) and register them with the GPU
    /// particle subsystem (Phase 4 per-system materials).
    pub fn register_particle_batches(
        &mut self,
        descriptors: &[crate::halo::effects::BatchDescriptor],
        tags_root: &std::path::Path,
    ) {
        use crate::halo::gpu_particle::BatchMaterial;
        // Scene lightprobe DC color (engine `effect_lightprobe_get_lightmap_tint`
        // = `light_probe_rgb[0] · 1/√(4π)`). protomorph reuses the scene
        // lightprobe (per-emitter lightprobe indices are stubbed); read it
        // once here since it's shared by every batch. SH band-0 constant
        // `1/(2√π) = 1/√(4π) ≈ 0.2821`.
        let li_dc: [f32; 3] = {
            const Y00: f32 = 0.282_094_79;
            let li = &self.lighting_interface;
            [
                Y00 * li.sh_probe_r[0],
                Y00 * li.sh_probe_g[0],
                Y00 * li.sh_probe_b[0],
            ]
        };
        let mut materials = Vec::with_capacity(descriptors.len());
        for d in descriptors {
            let ri = &d.render_info;
            // Engine `initialize_particle @0x18056ae10`: m_initial_color =
            // white, then ×lightmap_tint (bit3 — the lightprobe DC) and/or
            // ×diffuse_tint (bit4 — the probe's m_diffuse). m_diffuse isn't
            // tracked separately, so bit4 reuses the DC ambient. HDR is
            // carried directly in the float rgb, so the exponent .w stays 0.
            let initial_color: [f32; 4] = {
                let mut c = [1.0f32, 1.0, 1.0, 0.0];
                if d.tint_from_lightmap {
                    c[0] = li_dc[0];
                    c[1] = li_dc[1];
                    c[2] = li_dc[2];
                }
                if d.tint_from_diffuse {
                    c[0] *= li_dc[0];
                    c[1] *= li_dc[1];
                    c[2] *= li_dc[2];
                }
                c
            };
            let (base, base_w, base_h) = self
                .load_bitmap_view_dims(tags_root, &ri.base_map)
                .map(|(v, w, h)| (v, w, h))
                .unwrap_or_else(|| (self.particle_gpu.fallback_view(), 1, 1));
            let alpha = self
                .load_bitmap_view(tags_root, &ri.alpha_map)
                .unwrap_or_else(|| self.particle_gpu.fallback_view());
            let palette_view = ri
                .palette
                .as_deref()
                .and_then(|p| self.load_bitmap_view(tags_root, p));
            // Bake sprite-sheet frame UVs from the base bitmap's sequences,
            // selected by the particle's first_sequence_index. Empty for
            // plain (non-animated) bitmaps → render samples the full quad.
            let sequences = crate::halo::bitmaps::load_sequences(
                &blam_tags::paths::resolve_tag_path(tags_root, &ri.base_map, "bitmap"),
            );
            let frames = crate::halo::gpu_particle::bake_sprite_frames(
                &sequences,
                d.first_sequence_index,
            );
            if std::env::var("PROTOMORPH_DIAG_SPRITE").is_ok() {
                // props[8] = particle_frame: c[0]=is_constant, c[1]=value.
                let fp = &d.eval_tables.properties[8];
                eprintln!(
                    "[sprite] base={} seqs={} seq_idx={} frames={} frame_blend={} one_shot={} backwards={} | frame_prop const={} val={:.2} in_idx={}",
                    ri.base_map,
                    sequences.len(),
                    d.first_sequence_index,
                    frames.len(),
                    ri.frame_blend,
                    d.anim_one_shot,
                    d.anim_backwards,
                    fp.c[0],
                    fp.c[1],
                    fp.b[0],
                );
            }
            let ph = &d.physics;
            materials.push(BatchMaterial {
                albedo: ri.albedo,
                blend_mode: ri.blend_mode,
                black_point: ri.black_point,
                base_view: base,
                alpha_view: alpha,
                palette_view,
                physics: [
                    crate::halo::effects::particle_emitter::PARTICLE_GRAVITY * ph.gravity_mod,
                    ph.air_friction,
                    ph.rot_friction,
                    0.0,
                ],
                // Corner (the billboard quad) takes the SPRITE's aspect when the
                // bitmap is a sprite sheet — a tall+narrow bolt sprite must not
                // be stretched across a square quad (renders ~4× too wide). The
                // per-frame UV is handled separately by `bake_sprite_frames`;
                // here we take the first sprite of the selected sequence for the
                // geometry aspect (frames share aspect in practice).
                corner: usize::try_from(d.first_sequence_index)
                    .ok()
                    .and_then(|i| sequences.get(i))
                    .and_then(|seq| seq.sprites.first())
                    .map(|sp| {
                        crate::halo::gpu_particle::sprite_corner(sp, base_w, base_h, d.center_offset)
                    })
                    .unwrap_or_else(|| {
                        crate::halo::gpu_particle::default_sprite_corner(
                            base_w,
                            base_h,
                            d.center_offset,
                        )
                    }),
                rotation_offset: d.rotation_offset,
                depth_fade: d.render_info.depth_fade,
                fog: d.render_info.fog,
                distortion: d.render_info.distortion,
                billboard: d.billboard,
                motion_blur_aspect_scale: d.motion_blur_aspect_scale,
                near_range: d.near_range,
                near_cutoff: d.near_cutoff,
                edge_range: d.edge_range,
                edge_cutoff: d.edge_cutoff,
                intensity_affects_alpha: d.intensity_affects_alpha,
                flip_u: d.flip_u,
                flip_v: d.flip_v,
                frames,
                frame_blend: d.render_info.frame_blend,
                anim_one_shot: d.anim_one_shot,
                anim_backwards: d.anim_backwards,
                eval: d.eval_tables.clone(),
                initial_color,
                renders: d.renders,
            });
        }
        let size_scale = std::env::var("PROTOMORPH_PARTICLE_SIZE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        self.particle_gpu
            .register_batches(&self.shared, materials, size_scale);
    }

    /// Load + cache the `base_map` textures for the effect store's light
    /// volumes (Track R1). Call once after the effects load. Each unique
    /// bitmap path is loaded into [`light_volume::LightVolumeGpu`] so the
    /// per-frame strip render can bind it without touching the tag cache.
    pub fn register_light_volume_textures(
        &mut self,
        base_maps: &[String],
        tags_root: &std::path::Path,
    ) {
        for bm in base_maps {
            if bm.is_empty() || self.light_volume_gpu.has_texture(bm) {
                continue;
            }
            if let Some(view) = self.load_bitmap_view(tags_root, bm) {
                self.light_volume_gpu.register_texture(bm.clone(), view);
            }
        }
    }

    pub fn load_object_tag(&mut self, path: &std::path::Path) -> (usize, ModelData) {
        let model = crate::halo::loader::load_object(path)
            .unwrap_or_else(|e| panic!("failed to load Halo object {}: {e}", path.display()));
        let index = self.upload_model_data(&model);
        (index, model)
    }

    /// Load a Halo `.scenario` tag — walks the BSP / lightmap chain
    /// for the active zone set. Returns the [`LoadedScenario`]; the
    /// caller can iterate placements and feed each through
    /// [`Self::load_object_tag`] (Phase E will automate this).
    ///
    /// Mirrors Ares `scenario_load → scenario_activate_initial_zone_set`.
    ///
    /// Returns the scenario as an `Arc` so the renderer can keep a
    /// per-frame reference (Phase I0a) while the caller retains
    /// access for diagnostics + object spawning.
    /// Reset per-SCENARIO renderer state to its freshly-constructed
    /// default so a second `load_scenario` (map switch) doesn't
    /// append/leak onto the previous map's content. Called at the very
    /// top of `load_scenario`, before any population.
    ///
    /// NO-OP on a fresh renderer's first load: every field cleared here
    /// is already empty / default at that point (clearing an empty
    /// Vec, setting `next_probe_slot` 1→1), so the first/only load is
    /// byte-identical to today.
    ///
    /// What it clears and why it's safe (each is empty on first load AND
    /// repopulated afterward):
    /// - `structure_renderer.{bsps, bsp_cluster_meshes, bsp_cluster_bounds,
    ///   bsp_instance_bounds, lightmap_bsp_data, cluster_parts,
    ///   instance_parts}` + `next_probe_slot → 1` (via
    ///   `StructureRenderer::reset`) — all repopulated by
    ///   `Renderer::upload_bsp` (driven from `game.rs::load_scene` after
    ///   `load_scenario` returns) or rebuilt per-frame by
    ///   `submit_visibility`.
    /// - `self.models` — the placement model pool, repopulated by
    ///   `upload_model_data` / `load_object_tag` from `game.rs::load_scene`
    ///   after `load_scenario` returns. (Each `GpuModel` owns its
    ///   `bind_group_probes`, so this also drops the per-model cubemap
    ///   bind groups.)
    /// - `object_cubemap_probe_views` / `object_cubemap_positions` — the
    ///   object dynamic-env probe atlas; repopulated in `upload_bsp`
    ///   (which gates on `is_empty()`, so without this reset a second
    ///   load would keep the FIRST map's atlas).
    ///
    /// Deliberately NOT cleared (initialized-once / per-frame, not
    /// per-scenario content): `structure_renderer.identity_*` bindings
    /// (shared identity scratch, reinit gated on `is_none()` in
    /// `init_bsp_identity_bindings`), the device/queue/pipelines/sampler
    /// caches, and `sky_model` (unconditionally reassigned later in
    /// `load_scenario`).
    ///
    /// The process-global object/tag tables (`OBJECT_HEADER_DATA`,
    /// `OBJECT_NAME_MAP`, `TAG_CACHE`) are reset by existing helpers
    /// already invoked during this load: `LoadedScenario::load` calls
    /// `tag_init` (clears `TAG_CACHE` + sets the new root) and
    /// `populate_from_scenario` (full-`replace`s `OBJECT_HEADER_DATA` and
    /// clears `OBJECT_NAME_MAP`). No extra reset is needed here.
    fn reset_scene(&mut self) {
        self.structure_renderer.reset();
        self.models.clear();
        self.object_cubemap_probe_views.clear();
        self.object_cubemap_positions.clear();
    }

    pub fn load_scenario(
        &mut self,
        path: &std::path::Path,
    ) -> Result<std::sync::Arc<crate::halo::scenario::LoadedScenario>, crate::halo::scenario::ScenarioLoadError> {
        // Reset per-scenario state BEFORE any population so a map switch
        // (a second `load_scenario`) doesn't append/leak onto the prior
        // map. NO-OP on the first/only load (everything is already empty).
        self.reset_scene();
        // Pre-load rasg + per-BSP cubemap atlas BEFORE
        // `LoadedScenario::load` so material bind-group construction
        // (which reads `fallback_view_for_param` for cube slots) sees
        // the proper engine cube view instead of the 1×1 black-cube
        // placeholder. Without this, every BSP material bake-locks the
        // black-cube and reflections render as 0 even after we
        // populate `default_bitmap_views` post-load.
        let tags_root = blam_tags::paths::derive_tags_root(path).unwrap_or_else(
            || path.parent().map(|p| p.to_path_buf()).unwrap_or_default(),
        );
        self.preload_dynamic_cubemap_for_scenario(&tags_root, path);
        crate::halo::structures::bsp_gpu::gpu_checkpoint(&self.shared, "preload_dynamic_cubemap (rasg)");
        let loaded = crate::halo::scenario::LoadedScenario::load(path)?;
        // Cache the atmospheric scattering parameters so all halo
        // shaders can read them via the `engine_atmosphere` cbuffer
        // (binding 2 of camera_bgl). Falls back to a neutral
        // atmosphere when the scenario has no `atmospheric` tag-ref.
        // View exposure (Halo's `g_exposure.r`) — computed from
        // scenario.camera_fx_settings.exposure.exposure (stops) per
        // `c_camera_fx_values::get_render_exposure @ 0x18068e3e0`.
        // Formula: `pow(2, stops) × tone_curve_white_point × 0.66943294`.
        // tone_curve_white_point = 1.4938016 is the canonical Halo PC
        // value (final_composite.fx:tone_curve_constants.x); passing
        // 1.0 here was a 33% global darkening because the resulting
        // exposure is 0.67 instead of 1.0 at exposure_stops=0.
        const TONE_CURVE_WHITE_POINT: f32 = 1.4938016;
        // Engine-symmetric scenario-load init mirroring
        // `s_render_game_state::initialize_for_new_map @ 0x180682790` →
        // inlined `c_camera_fx_values::set @ 0x180682B40`. Snaps the cfxs
        // tag's authored targets onto the runtime values so the first frame
        // doesn't fade in from 0 — `calculate_new_value` would otherwise ramp
        // from 0 stops to the target over many frames.
        if let Some(cfx) = loaded.camera_fx.as_ref() {
            self.render_game_state.player_window[0].camera_fx_values.default_init_from(cfx);
        }
        let view_exposure = loaded.camera_fx.as_ref().map(|cfx| {
            cfx.exposure.view_exposure(TONE_CURVE_WHITE_POINT, 0.0)
        }).unwrap_or(1.0);
        // `g_alt_exposure.x = illum_scale` — engine sources from
        // `c_camera_fx_values::self_illum_scale` (camera_fx_settings.h
        // 0xD4 RealParameter). The blam-tags `CameraFxSettings` walker
        // doesn't surface the field today (older MCC schema variants
        // omit it), so default to 1.0 (the engine-side default when the
        // tag doesn't author the field). The full per-state animation
        // path lands with `c_camera_fx_values::update`.
        let illum_scale = 1.0_f32;
        eprintln!(
            "[scenario]   view_exposure = {:.3} | illum_scale = {:.3} (from camera_fx)",
            view_exposure, illum_scale,
        );
        // B1: write `g_exposure` + `g_alt_exposure` PS reg slots
        // (binding 9). HDR_target_stops = 0 → `pow(2,0) = 1.0` (our
        // Rg11b10Ufloat HDR target needs no encoding scale).
        self.shared.queue.write_buffer(
            &self.shared.engine_exposure_buffer,
            0,
            bytemuck::bytes_of(
                &crate::halo::render::shared::GpuEngineExposure::from_camera_fx(
                    view_exposure, illum_scale, 0.0,
                ),
            ),
        );

        // Stash the chocolate_mountain (chmt) per-object-type
        // luminance floor table. Engine reads this from
        // `global_scenario->chocalate_mountain_settings` inside
        // `apply_chocalate_mountain_lighting` to compute each object's
        // `mountain_scale = sqrt(min_lum / render_exposure)`.
        if let Some(chmt) = loaded.chocolate_mountain.as_ref() {
            eprintln!(
                "[scenario]   chocolate_mountain: {} per-object-type entries",
                chmt.per_object_min_luminance.len(),
            );
            self.scenario_chocolate_mountain = Some(chmt.clone());
        } else {
            self.scenario_chocolate_mountain = None;
        }

        // Snapshot the bloom + bling fields into ScreenPostprocess so
        // the postprocess pass uses this scenario's authored values
        // instead of the neutral CameraFxBloomDefaults::DEFAULT.
        if let Some(cfx) = loaded.camera_fx.as_ref() {
            self.screen_postprocess.camera_fx =
                crate::halo::render::screen_postprocess::CameraFxBloomDefaults::from_tag(cfx);
            self.scenario_camera_fx = Some(cfx.clone());
            eprintln!(
                "[scenario]   bloom: point={:.2} inherent={:.2} intensity={:.2} | bling: count={} size={:.0} angle={:.0} intensity={:.2}",
                cfx.bloom_point.value, cfx.bloom_inherent.value, cfx.bloom_intensity.value,
                cfx.bling_count.value, cfx.bling_size.value, cfx.bling_angle_deg.value, cfx.bling_intensity.value,
            );
            eprintln!(
                "[scenario]   auto-exposure: sensitivity={:.2} anti_bloom={:.2} | self_illum: pref_stops={:.2} scale={:.2}",
                cfx.auto_exposure_sensitivity.value,
                cfx.auto_exposure_anti_bloom.value,
                cfx.self_illum_preferred.value,
                cfx.self_illum_scale.value,
            );
            self.screen_postprocess
                .update_color_grading(&self.shared.queue, cfx.color_grading.as_ref(), true);
            self.final_composite_pass.set_bound_views(
                &self.shared,
                &self.intermediates,
                &self.gbuffer,
                self.screen_postprocess.bloom_result_view(),
                self.screen_postprocess.color_grading_active_view(),
                self.screen_postprocess.color_grading_back_view(),
                self.screen_postprocess.dof_blur_view(),
                self.screen_postprocess.bling_result_view(),
            );
            // Engine-faithful: route through `set_composite_params` so
            // tone curve constants are recomputed from
            // `screen_postprocess.settings_internal` instead of the
            // default-1.0 fallback baked into `set_cg_blend_factor`.
            let screen_grade = self
                .loaded_scenario
                .as_ref()
                .and_then(|s| s.screen_effect.as_ref())
                .map(|ase| {
                    let settings = crate::halo::render::screen_effect::sample_global(ase);
                    crate::halo::render::screen_effect::build_grade(&settings)
                })
                .unwrap_or(crate::halo::render::screen_effect::ScreenEffectGrade::IDENTITY);
            self.final_composite_pass.set_composite_params(
                &self.shared.queue,
                self.screen_postprocess.color_grading_blend_factor(),
                [0.0, 0.0, 0.0, 0.0],
                self.screen_postprocess.settings_internal.tone_curve,
                self.screen_postprocess.settings_internal.tone_curve_white_point,
                self.screen_postprocess.bling_scale(),
                screen_grade,
            );
            match cfx.color_grading.as_ref() {
                Some(cg) if cg.is_enabled() => {
                    eprintln!(
                        "[scenario]   color grading: ENABLED (bc={:?} hslv={:?} colorize={:?} sel={:?} cb={:?})",
                        cg.brightness_contrast.as_ref().filter(|b| b.flags.contains(blam_tags::camera_fx_settings::CameraFxEnableFlags::Enable)).map(|b| (b.brightness, b.contrast)),
                        cg.hslv.as_ref().filter(|h| h.flags.contains(blam_tags::camera_fx_settings::CameraFxEnableFlags::Enable)).map(|h| (h.hue_offset_deg, h.saturation_offset, h.vibrance, h.lightness_offset)),
                        cg.colorize.as_ref().filter(|c| c.flags.contains(blam_tags::camera_fx_settings::CameraFxEnableFlags::Enable)).map(|c| (c.blendfactor, c.target_hue_deg)),
                        cg.selective_color.as_ref().filter(|s| s.flags.contains(blam_tags::camera_fx_settings::CameraFxEnableFlags::Enable)).map(|s| s.flags.names()),
                        cg.color_balance.as_ref().filter(|c| c.flags.contains(blam_tags::camera_fx_settings::CameraFxEnableFlags::Enable)).map(|c| c.flags.names()),
                    );
                }
                Some(_) => eprintln!("[scenario]   color grading: present but disabled (identity LUT)"),
                None => eprintln!("[scenario]   color grading: tag predates struct (identity LUT)"),
            }
        }
        self.scenario_tags_root = loaded.tags_root.clone();
        // (rasg + dynamic cubemap atlas already preloaded at the top of
        // `load_scenario` so material bind groups built inside
        // `LoadedScenario::load` see the proper cube view.)
        // `get_sun_constants_from_sky` value — engine pulls this when
        // the active atmosphere setting's bit-1 ("Override Real Sun
        // Values") is unset. Without it our sun_intensity → inscatter
        // path was 50–500× too bright (raw atmosphere `color × intensity`
        // instead of sky's `lightgen[last].intensity × solid_angle ×
        // 0.2`), which washed every distant pixel toward white.
        let sky_sun = loaded.sky_lighting.as_ref().and_then(|s| {
            match (s.lightgen_sun_intensity, s.lightgen_sun_direction) {
                (Some(i), Some(d)) => Some(atmosphere_fog_interface::SkySun {
                    intensity: [i.i, i.j, i.k],
                    direction: [d.i, d.j, d.k],
                }),
                _ => None,
            }
        });
        self.scenario_atmosphere = loaded.atmosphere.as_ref()
            .map(|atm| atmosphere_fog_interface::GpuAtmosphereData::from_sky_atmosphere_with_exposure(atm, view_exposure, sky_sun))
            .unwrap_or_else(atmosphere_fog_interface::GpuAtmosphereData::neutral);
        // Cache inputs for L3 per-cluster atmosphere resolve at frame
        // time (engine-faithful MP fast-path of compute_cluster_weights).
        self.scenario_sky_atmosphere = loaded.atmosphere.clone();
        // Engine `c_atmosphere_fog_interface::initialize_for_new_map @
        // 0x1803AF4A0` runs alongside scenario load. Resets the
        // accumulated default_parameters so a stale prior-scenario blend
        // doesn't leak in, and (eventually, Phase 8) clears
        // weather_effect_indices.
        self.atmosphere_fog_interface.initialize_for_new_map();
        // A1 phase 5 — load `sky.patchy_fog_texture` noise bitmap if
        // authored. Falls through silently when unauthored / missing —
        // PatchyFogPass.noise_view stays None and the player_view
        // dispatcher falls back to the placeholder.
        if let Some(atm) = self.scenario_sky_atmosphere.as_ref() {
            self.patchy_fog.load_noise(&self.shared, &loaded.tags_root, &atm.patchy_fog_texture);
        } else {
            self.patchy_fog.load_noise(&self.shared, &loaded.tags_root, "");
        }
        // A1 phase 6 — reset per-frame fog state on scenario load so
        // the first frame seeds last_eye_position from the new camera
        // rather than reading stale state from the previous map.
        self.player_view.patchy_fog.is_first_frame = true;
        self.player_view.patchy_fog.lateral_offsets.fill(0.0);
        self.player_view.patchy_fog.vertical_offsets.fill(0.0);
        self.player_view.patchy_fog.roll = 0.0;
        if let Some(bsp) = loaded.active_bsps.first() {
            self.scenario_active_bsp_clusters = bsp.sbsp.clusters.clone();
            self.scenario_active_bsp_atmosphere_palette = bsp.sbsp.atmosphere_palette.clone();
            self.scenario_active_bsp_cluster_portals = bsp.sbsp.cluster_portals.clone();
            self.scenario_active_bsp_planes = bsp
                .sbsp
                .collision_bsp
                .as_ref()
                .map(|c| c.planes.clone())
                .unwrap_or_default();
        } else {
            self.scenario_active_bsp_clusters.clear();
            self.scenario_active_bsp_atmosphere_palette.clear();
            self.scenario_active_bsp_cluster_portals.clear();
            self.scenario_active_bsp_planes.clear();
        }
        self.scenario_view_exposure = view_exposure;
        self.scenario_sky_sun = sky_sun;
        if let Some(sun) = sky_sun {
            eprintln!(
                "[atmosphere] sky lightgen sun (sky_lights[last]): intensity {:?} dir {:?}",
                sun.intensity, sun.direction,
            );
        } else {
            eprintln!(
                "[atmosphere] sky lightgen sun: <none> — engine default white-down sun",
            );
        }
        if let Some(atm) = loaded.atmosphere.as_ref() {
            for (i, s) in atm.atmosphere_settings.iter().enumerate() {
                let snap = atmosphere_fog_interface::GpuAtmosphereData::from_atmosphere_setting(s, view_exposure, sky_sun);
                let log2e = 1.442695_f32;
                let beta_m  = [snap.slot2_beta_m_log2e_g1[0]/log2e, snap.slot2_beta_m_log2e_g1[1]/log2e, snap.slot2_beta_m_log2e_g1[2]/log2e];
                let beta_p  = [snap.slot3_beta_p_log2e_refh[0]/log2e, snap.slot3_beta_p_log2e_refh[1]/log2e, snap.slot3_beta_p_log2e_refh[2]/log2e];
                eprintln!(
                    "[atm[{}] {}] flags={:?} enabled={} override_real_sun={} ray_mult={} mie_mult={} desat={} max_thick={} dist_bias={} sea={} ray_h={} mie_h={}",
                    i, s.name, s.flags.get(), s.is_enabled(),
                    s.flags.contains(blam_tags::sky_atmosphere::AtmosphereFlags::OverrideRealSunValues),
                    s.rayleigh_multiplier, s.mie_multiplier, s.desaturation,
                    s.max_fog_thickness, s.distance_bias, s.sea_level, s.rayleigh_height_scale, s.mie_height_scale,
                );
                eprintln!(
                    "[atm[{}] {}]   β_m={:?}  β_p={:?}",
                    i, s.name, beta_m, beta_p,
                );
                eprintln!(
                    "[atm[{}] {}]   β_m_ang={:?}  β_p_ang={:?}",
                    i, s.name,
                    &snap.slot4_beta_m_angular_mieh[..3],
                    &snap.slot5_beta_p_angular_rayh[..3],
                );
            }
        }
        eprintln!(
            "[atmosphere] PRIMARY β_m·log2e: {:?}  β_p·log2e: {:?}",
            &self.scenario_atmosphere.slot2_beta_m_log2e_g1[..3],
            &self.scenario_atmosphere.slot3_beta_p_log2e_refh[..3],
        );
        eprintln!(
            "[atmosphere] PRIMARY β_m_angular: {:?}  β_p_angular: {:?}",
            &self.scenario_atmosphere.slot4_beta_m_angular_mieh[..3],
            &self.scenario_atmosphere.slot5_beta_p_angular_rayh[..3],
        );
        eprintln!(
            "[atmosphere] PRIMARY sun_int_norm: {:?}  dist_bias: {}",
            &self.scenario_atmosphere.slot1_sun_int_norm_thickness[..3],
            self.scenario_atmosphere.slot0_sun_dir_dist_bias[3],
        );
        // `c_lighting_interface::setup_default_lighting @ 0x1806AA190`.
        // Sky-tag chain → sky.render_model.lightprobe → 9 floats per
        // band. dllcache routes:
        //   calculate_and_set_ravi_constants(probe_r, g, b, 1.0)
        //   calculate_dominant_light_from_lightprobe(...) → dir, intensity
        //   set_object_dominant_light_direction_and_intensity(dir, intensity)
        // Direction is luma-weighted L1 collapse — NOT the atmospheric
        // sun direction. Intensity is SH-projected at that direction.
        self.lighting_interface = match loaded.sky_lighting.as_ref() {
            Some(sky) => {
                let li = lighting_interface::LightingInterface::from_lightprobe(
                    sky.probe.r, sky.probe.g, sky.probe.b,
                );
                eprintln!(
                    "[lighting] sky probe → dir=({:.3},{:.3},{:.3}) intensity=({:.3},{:.3},{:.3})",
                    li.dominant_light_dir.x, li.dominant_light_dir.y, li.dominant_light_dir.z,
                    li.dominant_light_intensity.x, li.dominant_light_intensity.y,
                    li.dominant_light_intensity.z,
                );
                li
            }
            None => {
                eprintln!("[lighting] no sky lighting (chain broke); falling back");
                lighting_interface::LightingInterface::fallback()
            }
        };

        // Load per-BSP lightmap atlases for terrain per-pixel lighting.
        // Riverworld has 1 active BSP; we load atlas + intensity tags
        // referenced by `LightmapBspData` and bind them at @group(0).
        // (Multi-BSP zone sets would need per-bsp swap; deferred.)
        if let Some(bsp) = loaded.active_bsps.first() {
            if let Some(lbsp) = bsp.lightmap.as_ref() {
                self.upload_lightmap_atlases(&loaded.tags_root, lbsp);
                crate::halo::structures::bsp_gpu::gpu_checkpoint(&self.shared, "upload_lightmap_atlases");
            }
        }

        // DIAGNOSTIC (PROTOMORPH_MINIMAL_LOAD=1): load ONLY scenario + BSP +
        // lightmap + rasg. Skip sky scenery, decorators, scenario lights, and
        // decals (and game.rs skips all object placements + weather). Bisection
        // tool for the huge-BSP first-frame freeze that persists even with all
        // structure DRAWING stripped — isolates whether a loaded auxiliary tag
        // (decorator instance fields, decal projection meshes, etc.) or its
        // per-frame pass is the hang, vs. the BSP/lightmap itself.
        let minimal_load = std::env::var_os("PROTOMORPH_MINIMAL_LOAD").is_some();
        if minimal_load {
            eprintln!("[load] PROTOMORPH_MINIMAL_LOAD: skipping sky/decorators/lights/decals");
        }

        // Phase B17: load the sky scenery → render_model and stash
        // its GPU meshes/materials. Renders BEFORE BSP albedo with
        // camera-position-translated model_u (parallax-locked sky).
        // Engine equivalent: `c_object_renderer::submit_and_render_sky`.
        self.sky_model = if minimal_load { None } else { self.load_sky_scenery(&loaded) };

        // Resolve wind tag + noise bitmap BEFORE decorator load_scenario
        // builds per-set bind groups (so they pick up the real noise
        // texture, not the 1×1 placeholder). Engine `get_wind_definition
        // @ 0x1809081E0` is per-camera-position (per-BSP); we don't have
        // a BSP3D walker yet so v1 picks the first active BSP's wind
        // (single-BSP scenarios are correct; multi-BSP campaign maps
        // will use the wrong wind tag for non-default BSPs until the
        // walker lands). Falls back to globals.default_wind.
        let resolved_wind = loaded
            .active_bsp_winds
            .first()
            .cloned()
            .flatten()
            .or_else(|| loaded.default_wind.clone());
        if let Some(w) = &resolved_wind {
            eprintln!(
                "[wind] resolved gust_size={:.2} bitmap={}",
                w.gust_size, w.gust_noise_bitmap_path,
            );
            // Load the noise bitmap. Engine `setup_wind_constants @
            // 0x180908290` does this lookup itself + falls back to
            // `rasterizer_globals.default_bitmaps[10]` when the wind tag
            // doesn't author one. We just resolve the tag's path and
            // load via the standard bitmap loader; failure leaves the
            // placeholder (1×1 black noise = wind motion driven by the
            // constant `bend × dir` term only).
            if !w.gust_noise_bitmap_path.is_empty() {
                let bitmap_path = blam_tags::paths::resolve_tag_path(
                    &loaded.tags_root,
                    &w.gust_noise_bitmap_path,
                    "bitmap",
                );
                if let Some(tex) = crate::halo::bitmaps::load(&bitmap_path) {
                    eprintln!(
                        "[wind] noise loaded {}×{} {:?}",
                        tex.width, tex.height, tex.format,
                    );
                    self.decorator_renderer.set_wind_noise(&self.shared, &tex);
                } else {
                    eprintln!(
                        "[wind] noise load failed {} — using 1×1 black placeholder",
                        bitmap_path.display(),
                    );
                }
            }
        } else {
            eprintln!("[wind] no .wind tag — wind animation frozen");
        }
        self.decorator_renderer.current_wind = resolved_wind;

        // Build per-set GPU buffers for every painted decorator
        // placement. Engine equivalent: `c_decorator_renderer::initialize`
        // chain. Walks `loaded.scenario.decorators[].sets[].placements[]`
        // and uploads (mesh + texture + instance buffer) per set.
        if !minimal_load {
            self.decorator_renderer.load_scenario(&self.shared, &loaded);
        }

        // L2c-real: walk each active BSP's lighting_info, resolve
        // each instance against its definition, and stash the result
        // on the renderer.
        self.scenario_lights = if minimal_load { Vec::new() } else { resolve_scenario_lights(&loaded) };
        eprintln!(
            "[scenario]   scenario_lights: {} resolved (from {} BSPs)",
            self.scenario_lights.len(),
            loaded.active_bsps.len(),
        );

        // L2 per-cluster: bake each BSP cluster's up-to-8 nearest
        // stli lights into the simple_lights_buffer at unique slots,
        // and store the slot offset per cluster. Engine-faithful: each
        // cluster's draws bind their own slot (no per-frame
        // re-selection, no popping).
        if !minimal_load {
            self.bake_cluster_simple_lights(&loaded);
        }

        // L4: pre-bake per-scenery-placement engine_lighting slots from
        // each BSP's `lightmap_bsp_data.scenery_probes`. Engine equivalent:
        // `lights_prepare_for_object_static_new @ 0x1808A2930`'s type==6
        // branch (scenery), called per object during
        // `object_update_cached_render_lighting`. Our offline bake is
        // equivalent because the probes are static authoring data — the
        // engine caches the converted sample anyway, only repeating the
        // conversion when the object is re-initialized.
        if !minimal_load {
            self.bake_scenery_lighting_offsets(&loaded);
        }

        // D2b/Step 4b — bake per-palette decal pipelines + bind groups
        // and per-BSP GPU vertex/index buffers. Must run BEFORE wrapping
        // `loaded` in `Arc` since the bake mutates `self.halo_pipelines`.
        // Engine equivalent: the cache builder pre-resolves each
        // palette entry's `c_render_method_shader_decal` chain, and
        // `c_decal_system::create` uploads the projection mesh into
        // the per-BSP allocator pools at scenario init.
        if !minimal_load {
            self.decal_gpu = crate::halo::render::render_decals::bake_decals_gpu(
                &self.shared,
                &mut self.halo_pipelines,
                &mut self.rasterizer.sampler_state_cache,
                &self.scenario_tags_root,
                &loaded,
            );
        }

        // Phase I0a — stash the scenario for per-frame visibility
        // (`scenario_location_from_point` + `compute()`).
        let arc = std::sync::Arc::new(loaded);
        self.loaded_scenario = Some(std::sync::Arc::clone(&arc));
        Ok(arc)
    }





    /// Engine-faithful per-frame chmt + ravi cbuffer write. Engine
    /// `c_lighting_interface::setup_object_lighting_for_entry_point @
    /// 0x1806A9AA0` fires per-DRAW; protomorph does it once per frame
    /// for all cached placements (same result since the cached SH is
    /// static within a frame). Must be called AFTER `render_exposure`
    /// is updated for the current frame and BEFORE any object draws
    /// read the engine_lighting cbuffer.
    ///
    /// Engine formula (verified via dllcache decompile @ 0x1806A9AA0
    /// and chmt apply @ 0x1806f9ee0):
    /// ```text
    /// mountain_scale[type] = sqrt(min_luminance[type] / render_exposure)  // chmt apply
    /// ravi_scale           = max(1, mountain_scale / sh_lum) if mountain_scale > 0 else 1
    /// ravi                 = pack_ravi_constants(sh_r * ravi_scale, sh_g * ravi_scale, sh_b * ravi_scale)
    /// ```
    /// `mountain_scale` depends only on `(object_type, render_exposure)`,
    /// so we precompute one value per type per frame; only the
    /// per-placement scale division + pack runs per entry.
    pub fn refresh_engine_lighting_for_frame(&mut self) {
        use crate::halo::lighting::{pack_ravi_constants, GpuEngineLightingPs};
        use crate::halo::objects::object_type::ObjectType;

        if self.object_lighting_cache.is_empty() {
            return;
        }

        let render_exposure = self.render_exposure.max(1.0e-6);
        // Precompute per-type chmt scales. `mountain_scale = -1` is the
        // engine sentinel for "no boost" (scale=1 fallback).
        let mut mountain_scales = [-1.0_f32; 16];
        if let Some(chmt) = self.scenario_chocolate_mountain.as_ref() {
            for (t, slot) in mountain_scales.iter_mut().enumerate() {
                *slot =
                    crate::halo::render::chocalate_mountain::apply_chocalate_mountain_lighting(
                        chmt,
                        t as i32,
                        render_exposure,
                    );
            }
        }

        // PROTOMORPH_DIAG_LIGHTING_REFRESH=<n> dumps the first n entries
        // per frame. Cheap diag for "is the per-frame scale changing?"
        let dump_limit: usize = std::env::var("PROTOMORPH_DIAG_LIGHTING_REFRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        for (idx, entry) in self.object_lighting_cache.iter().enumerate() {
            let mountain_scale = mountain_scales[entry.object_type as usize];
            let scale = if mountain_scale > 0.0 {
                (mountain_scale / entry.sh_lum.max(0.001)).max(1.0)
            } else {
                1.0
            };
            let ravi_payload = GpuEngineLightingPs {
                ravi: pack_ravi_constants(&entry.sh_r, &entry.sh_g, &entry.sh_b, scale),
            };
            if idx < dump_limit {
                let type_name: &str = match entry.object_type {
                    ObjectType::Scenery => "scenery",
                    ObjectType::Crate => "crate",
                    ObjectType::Weapon => "weapon",
                    ObjectType::Equipment => "equipment",
                    ObjectType::Machine => "machine",
                    ObjectType::Control => "control",
                    _ => "?",
                };
                eprintln!(
                    "[diag-lighting-refresh] entry[{idx}] type={type_name} \
                     sh_lum={:.4} mountain={mountain_scale:.4} scale={scale:.4} \
                     slot_offset=0x{:x}",
                    entry.sh_lum, entry.slot_offset,
                );
            }
            self.shared.queue.write_buffer(
                &self.shared.engine_lighting_buffer,
                entry.slot_offset as u64,
                bytemuck::bytes_of(&ravi_payload),
            );
        }
    }

    /// Walks the scenario's primary sky chain
    /// (`skies[0].sky.scenery → object/model → render_model`) and
    /// builds GPU buffers for its meshes/materials. Returns a
    /// dedicated `GpuModel` (not pushed into `self.models`, which is
    /// the placement pool — keeping it separate avoids drifting the
    /// game-side `model_data[]` index).
    fn load_sky_scenery(
        &mut self,
        loaded: &crate::halo::scenario::LoadedScenario,
    ) -> Option<GpuModel> {
        let sky_ref = loaded.scenario.skies.first()?;
        if sky_ref.sky.is_empty() { return None; }
        let scenery_path = blam_tags::paths::resolve_tag_path(
            &loaded.tags_root, &sky_ref.sky, "scenery",
        );
        if !scenery_path.exists() {
            eprintln!("[sky] scenery path missing: {}", scenery_path.display());
            return None;
        }
        let model = match crate::halo::loader::load_object(&scenery_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[sky] load_object failed: {e}");
                return None;
            }
        };
        eprintln!(
            "[sky] loaded render_model: {} meshes, {} materials, {} nodes",
            model.meshes.len(), model.materials.len(), model.node_local_matrices.len(),
        );

        // `node_local_matrices` already IS the per-node WORLD bind pose
        // (composed from the parent-relative `default_transform` chain at
        // load time — see `loader::convert`). Hand it straight to the sky
        // pass. The sky shader reads `Nodes[0]` (and the static_sh fallback
        // indexes by the vertex node) — without this the sky draws at
        // identity, ignoring the authored node orientation (e.g.
        // bunkerworld's 103° yaw). Engine-faithful: `SkyGpu::upload_node_matrices`.
        let world_matrices: Vec<glam::Mat4> = model.node_local_matrices.clone();
        self.sky_gpu.upload_node_matrices(&self.shared.queue, &world_matrices);

        // TEMP DIAG: dump node 0 matrix + per-mesh WORLD-space AABB so we can
        // see where the sky meshes actually land vs authored.
        if std::env::var("PROTOMORPH_DIAG_SKY").is_ok() {
            eprintln!("[sky-diag] node0 world matrix (cols): {:?}", world_matrices.get(0).map(|m| m.to_cols_array()));
            for (mi, mesh) in model.meshes.iter().enumerate() {
                let nm = world_matrices.get(0).copied().unwrap_or(glam::Mat4::IDENTITY);
                let mut mn = glam::Vec3::splat(f32::INFINITY);
                let mut mx = glam::Vec3::splat(f32::NEG_INFINITY);
                for v in &mesh.vertices {
                    let p = nm.transform_point3(glam::Vec3::from_array(v.position));
                    mn = mn.min(p); mx = mx.max(p);
                }
                let mat = mesh.parts.get(0).map(|p| p.material_index).unwrap_or(0);
                eprintln!("[sky-diag] mesh[{mi:2}] mat={mat} world_bbox=[{:.1},{:.1},{:.1}]..[{:.1},{:.1},{:.1}]",
                    mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
            }
        }

        // Borrow self mutably-then-immutably — temporarily move the
        // models pool out, upload (which appends to it), then take
        // the freshly-uploaded entry off the end + restore the pool.
        let pre_len = self.models.len();
        let _idx = self.upload_model_data(&model);
        debug_assert_eq!(self.models.len(), pre_len + 1);
        Some(self.models.pop().expect("upload_model_data appended"))
    }

}


impl Renderer {
    /// Upload a [`crate::halo::scenario::loader::LoadedBsp`] through
    /// the dedicated structure-renderer path. Returns the bsp index
    /// (used by the per-frame visibility submitter to address it).
    /// Mirrors `scenario_structure_bsp_load` semantics — populates
    /// `g_render_structure_globals.render_geometry[bsp_index]` (Ares).
    pub fn upload_bsp(
        &mut self,
        bsp: &crate::halo::scenario::loader::LoadedBsp,
        tags_root: &std::path::Path,
        scenario_cubemaps: &[blam_tags::scenario::CubemapEntry],
    ) -> usize {
        // Lazy-init shared identity bind groups for cluster draws.
        if self.structure_renderer.identity_buffers.is_none() {
            self.init_bsp_identity_bindings();
        }
        // Per-BSP cubemap routing — load the
        // `<scenario>_<bsp>_cubemaps.bitmap` atlas (one cube per
        // scenario probe) and synthesize cluster→probe assignment by
        // nearest-distance from cluster centroid to scenario probe
        // position. Mirrors the `cluster_cubemaps[]` runtime cache MCC
        // strips. Empty when the atlas isn't on disk → bind groups
        // fall back to length 1 with the rasg default cube.
        let cubemap_routing = build_bsp_cubemap_routing(
            &self.shared,
            bsp,
            scenario_cubemaps,
        );
        // Retain the probe cube atlas for OBJECT dynamic-env mapping —
        // objects resolve their nearest probe by position (engine
        // `c_dynamic_cubemap_sample`) and bind it like the BSP per-cluster
        // path. First BSP wins (single-BSP scenarios); TextureViews are
        // Arc-cheap to clone.
        if let Some(routing) = cubemap_routing.as_ref() {
            if self.object_cubemap_probe_views.is_empty() && !routing.probe_views.is_empty() {
                self.object_cubemap_probe_views = routing.probe_views.clone();
                self.object_cubemap_positions = scenario_cubemaps
                    .iter()
                    .map(|c| glam::Vec3::new(c.position.x, c.position.y, c.position.z))
                    .collect();
            }
        }
        let bsp_gpu = crate::halo::structures::BspGpu::from_loaded_bsp(
            &self.shared,
            &mut self.halo_pipelines,
            &mut self.rasterizer.sampler_state_cache,
            bsp,
            tags_root,
            &mut self.structure_renderer.next_probe_slot,
            cubemap_routing.as_ref(),
        );
        // Load EVERY rmw material on this BSP into the WaterRenderer's
        // per-material slot cache (env_cubemap + reflection/fresnel +
        // wave/foam/global_shape textures + variant pipeline). The draw
        // loop selects per water part by `material_index`. docks has two
        // (dock_water_1 / dock_water_2); rendering both is what gives its
        // harbor the right wave scale + foam (the old single-material
        // path forced every surface onto the first material).
        if !bsp_gpu.water_materials.is_empty() {
            self.water_renderer
                .load_water_materials(&self.shared, &bsp_gpu.water_materials);
        }
        let cluster_meshes = bsp.sbsp.clusters.iter().map(|c| c.mesh_index).collect();
        // Pre-bake per-cluster bounding spheres for frustum culling in
        // `submit_visibility`. Engine equivalent: `cluster_build_object_payload`
        // pre-bakes per-cluster cull flags at scenario load.
        // BspCluster::bounds_x/y/z is the AABB of the cluster's
        // GEOMETRY only. Decorators are placed by scenario authoring
        // and per-cluster simple lights have an influence radius —
        // both extend past the geometry bounds. Without engine-style
        // PVS / portal testing, we pad the sphere radius generously
        // so edge clusters with overflowing content don't pop their
        // contents. 2× is empirically conservative; tune if perf
        // regresses on dense scenes.
        const CLUSTER_CULL_RADIUS_PAD: f32 = 2.0;
        let cluster_bounds: Vec<(glam::Vec3, f32)> = bsp.sbsp.clusters.iter().map(|c| {
            let cx = (c.bounds_x.lower + c.bounds_x.upper) * 0.5;
            let cy = (c.bounds_y.lower + c.bounds_y.upper) * 0.5;
            let cz = (c.bounds_z.lower + c.bounds_z.upper) * 0.5;
            let hx = (c.bounds_x.upper - c.bounds_x.lower) * 0.5;
            let hy = (c.bounds_y.upper - c.bounds_y.lower) * 0.5;
            let hz = (c.bounds_z.upper - c.bounds_z.lower) * 0.5;
            let r = (hx*hx + hy*hy + hz*hz).sqrt() * CLUSTER_CULL_RADIUS_PAD;
            (glam::Vec3::new(cx, cy, cz), r)
        }).collect();
        // Instance bounds — sourced directly from each instance's
        // tag-time `world_bounding_sphere_*` (Halo authoring tool
        // precomputes these). Used by `submit_visibility` for instance
        // frustum cull. ~20% of riverworld + ~70% of cyberdyne instances
        // are typically out-of-view per frame on multiplayer maps.
        let instance_bounds: Vec<(glam::Vec3, f32)> = bsp
            .sbsp
            .instanced_geometry_instances
            .iter()
            .map(|inst| {
                let c = inst.world_bounding_sphere_center;
                (
                    glam::Vec3::new(c.x, c.y, c.z),
                    inst.world_bounding_sphere_radius,
                )
            })
            .collect();
        let idx = self.structure_renderer.bsps.len();
        self.structure_renderer.bsps.push(bsp_gpu);
        self.structure_renderer.bsp_cluster_meshes.push(cluster_meshes);
        self.structure_renderer.bsp_cluster_bounds.push(cluster_bounds);
        self.structure_renderer.bsp_instance_bounds.push(instance_bounds);
        idx
    }

    /// Build the identity model-uniform + 1-bone identity node-matrix
    /// bind groups that ALL cluster draws share. Cluster vertices are
    /// already in world coordinates; the existing pipeline layout
    /// requires bind groups 1 + 3 to be set, so we bind identity.
    fn init_bsp_identity_bindings(&mut self) {
        use crate::halo::geometry::ModelUniforms;
        let identity_model = ModelUniforms { model: glam::Mat4::IDENTITY.to_cols_array_2d() };
        let model_bytes = bytemuck::bytes_of(&identity_model);
        // Pad to model_stride so dynamic_offset=0 reads the right slot.
        let mut model_padded = vec![0u8; self.shared.model_stride];
        model_padded[..model_bytes.len()].copy_from_slice(model_bytes);
        let model_buffer = self.shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bsp_identity_model_uniform"),
                contents: &model_padded,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let model_bg = self.shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bsp_identity_model_bg"),
            layout: &self.shared.model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(model_bytes.len() as u64),
                }),
            }],
        });

        // Single identity 4x4 for bone 0; the rest of the bone array
        // never gets indexed (every BSP vertex has node_indices=[0;4]
        // + weights=[1,0,0,0]).
        let identity_mat = glam::Mat4::IDENTITY.to_cols_array();
        let nm_bytes = bytemuck::cast_slice(&identity_mat);
        let mut nm_padded = vec![0u8; self.shared.node_matrices_stride];
        nm_padded[..nm_bytes.len()].copy_from_slice(nm_bytes);
        let nm_buffer = self.shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bsp_identity_node_matrices"),
                contents: &nm_padded,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let nm_bg = self.shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bsp_identity_nm_bg"),
            layout: &self.shared.node_matrices_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &nm_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(self.shared.node_matrices_stride as u64),
                }),
            }],
        });

        self.structure_renderer.identity_buffers = Some((model_buffer, nm_buffer));
        self.structure_renderer.identity_model_bg = Some(model_bg);
        self.structure_renderer.identity_nm_bg = Some(nm_bg);
    }

    /// Upload a [`ModelData`] to the GPU and register it.
    pub fn upload_model_data(&mut self, model: &ModelData) -> usize {
        // Animated-material collection. The `.map` closure pushes one
        // record per material flagged `is_animated`; walked per frame
        // by `GpuModel::update_animated_cbuffers`. Mirrors the BSP
        // load path's `animated_materials` Vec.
        let mut animated_materials: Vec<
            crate::halo::render_methods::animated::AnimatedMaterial,
        > = Vec::new();
        // Compile the `_entry_point_vertex_color_lighting` variant only
        // when this model carries a per-vertex baked color stream — i.e.
        // sky `.render_model`s. Regular objects have no vc-bit meshes, so
        // gating here avoids compiling a dead pipeline for every object
        // material. Engine routes vc-bit sky meshes through this entry
        // point (`render_mesh_part_default @ 0x18069EBC0:64-82`).
        let model_has_vert_color =
            model.meshes.iter().any(|m| !m.vert_color_stream.is_empty());
        let materials: Vec<GpuMaterial> = model
            .materials
            .iter()
            .map(|mat| {
                let upload_view = |tex: &crate::halo::render_methods::materials::InlineTexture,
                                   intent: SampleIntent|
                 -> wgpu::TextureView {
                    let wgpu_format = resolve_wgpu_format(tex.format, tex.encoding, intent);
                    // BC formats need block-aligned (multiple of 4) extents
                    // at create_texture time. Halo authors bitmaps at the
                    // logical size (e.g. 474×474); the GPU allocation rounds
                    // up. Per-mip writes already handle their own block
                    // alignment. See `feedback_wgpu_bc_mip_alignment.md`.
                    let (alloc_w, alloc_h) = if tex.format.is_compressed() {
                        ((tex.width + 3) & !3, (tex.height + 3) & !3)
                    } else {
                        (tex.width, tex.height)
                    };
                    let texture = self.shared.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("halo_inline_texture"),
                        size: wgpu::Extent3d {
                            width: alloc_w,
                            height: alloc_h,
                            depth_or_array_layers: tex.layers,
                        },
                        mip_level_count: tex.mip_count,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu_format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    // tex.data layout: for each layer, the full mip
                    // pyramid back-to-back; layers are then concatenated.
                    // BC formats need block-aligned extent + bytes_per_row.
                    let mut offset = 0usize;
                    for layer in 0..tex.layers {
                        for mip in 0..tex.mip_count {
                            let mw = (tex.width >> mip).max(1);
                            let mh = (tex.height >> mip).max(1);
                            let level_bytes = tex.format.level_bytes(mw, mh);
                            let (extent_w, extent_h, rows, bytes_per_row);
                            if tex.format.is_compressed() {
                                let bw = ((mw + 3) / 4).max(1);
                                let bh = ((mh + 3) / 4).max(1);
                                extent_w = bw * 4;
                                extent_h = bh * 4;
                                rows = bh;
                                bytes_per_row = bw * tex.format.block_bytes();
                            } else {
                                extent_w = mw;
                                extent_h = mh;
                                rows = mh;
                                bytes_per_row = mw * tex.format.bytes_per_pixel();
                            }
                            self.shared.queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &texture,
                                    mip_level: mip,
                                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                &tex.data[offset..offset + level_bytes],
                                wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(bytes_per_row),
                                    rows_per_image: Some(rows),
                                },
                                wgpu::Extent3d {
                                    width: extent_w,
                                    height: extent_h,
                                    depth_or_array_layers: 1,
                                },
                            );
                            offset += level_bytes;
                        }
                    }
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(if tex.is_cube {
                            wgpu::TextureViewDimension::Cube
                        } else {
                            wgpu::TextureViewDimension::D2
                        }),
                        ..Default::default()
                    })
                };

                // rmd materials carried in an object's render_model
                // material list (e.g. `human_military_decals_2` —
                // scenery that bakes decal-shader visuals onto its
                // mesh). Engine `c_object_renderer::submit_object_mesh_parts
                // @ 0x1806E1BF0` flips the part's flags to
                // `(flags & ~LIT) | DECAL` when
                // `c_render_method::get_is_decal(material) == true`, and
                // `c_object_renderer::render_albedo_decals @ 0x1806E4A60`
                // dispatches them through entry_point=_default + the
                // DECAL mask. That path takes the rmd `default_ps`
                // body with a skinned-ModelVertex VS (our
                // `entry_decal_object.wgsl`).
                //
                // We keep a defensive rmsh stub for the SL/Albedo
                // pipeline slots so the StaticSh/Albedo entries don't
                // crash if a dispatch path ever picks rmd material —
                // those slots won't be exercised because
                // `submit_visibility_from_render_list` routes rmd parts
                // to the DECAL bit only (no LIT, no transparency
                // routing). The real rmd material drives the new
                // `artifacts_decal_object` slot below.
                let real_mat = mat;
                let is_rmd_on_object = &mat.resolved.group_tag.to_be_bytes() == b"rmd ";
                let mat_override;
                let mat: &MaterialData = if is_rmd_on_object {
                    let mut stub = MaterialData::stub_for_shader(&mat.name);
                    stub.resolved.group_tag = u32::from_be_bytes(*b"rmsh");
                    mat_override = stub;
                    &mat_override
                } else {
                    mat
                };
                let stub;
                let choices = if mat.category_choices.is_empty() {
                    stub = crate::halo::render_methods::pipeline_cache::stub_choices();
                    &stub
                } else {
                    &mat.category_choices
                };
                let artifacts = self.halo_pipelines.ensure(
                    &self.shared,
                    &mat.resolved,
                    &mat.cbuffer,
                    choices,
                    HaloEntryPoint::StaticSh,
                    &self.scenario_tags_root,
                );
                let artifacts_albedo = self.halo_pipelines.ensure(
                    &self.shared,
                    &mat.resolved,
                    &mat.cbuffer,
                    choices,
                    HaloEntryPoint::Albedo,
                    &self.scenario_tags_root,
                );
                // PRT Ambient variant — only compile for rmsh-family
                // subclasses whose WGSL assembler emits the
                // `entry_static_prt_ambient.wgsl` body. Other subclasses
                // (rmtr/rmfl/rm /etc.) don't expect the
                // `@location(11) prt_transfer` attribute and would fail
                // wgpu validation. Gating here keeps the pipeline cache
                // honest.
                let mat_group_fourcc = mat.resolved.group_tag.to_be_bytes();
                let supports_prt = matches!(&mat_group_fourcc, b"rmsh" | b"rmcs" | b"rmhg");
                let artifacts_prt_ambient = if supports_prt {
                    Some(self.halo_pipelines.ensure(
                        &self.shared,
                        &mat.resolved,
                        &mat.cbuffer,
                        choices,
                        HaloEntryPoint::StaticPrtAmbient,
                        &self.scenario_tags_root,
                    ))
                } else {
                    None
                };
                // Vertex-color lighting variant (sky meshes only). Same
                // rmsh-family gate as PRT — the WGSL assembler emits the
                // `entry_static_per_vertex_color.wgsl` body for these
                // subclasses; others lack the `@location(12) vert_color`
                // wiring.
                let artifacts_vertex_color = if supports_prt && model_has_vert_color {
                    Some(self.halo_pipelines.ensure(
                        &self.shared,
                        &mat.resolved,
                        &mat.cbuffer,
                        choices,
                        HaloEntryPoint::StaticVertexColor,
                        &self.scenario_tags_root,
                    ))
                } else {
                    None
                };

                // Build the per-variant bind group — one slot per
                // bitmap parameter the rmop declares, looked up by
                // Halo parameter name.
                let mut texture_views: Vec<(u32, wgpu::TextureView)> =
                    Vec::with_capacity(artifacts.bindings.textures.len());
                for tb in &artifacts.bindings.textures {
                    let view = match mat.find_texture_by_name(&tb.name) {
                        // Bind the authored texture only when its dimension
                        // matches the BGL slot. A force_cube / cube-classified
                        // slot fed a 2D bitmap (some rmop authors point
                        // `environment_map` at a 2D map — crashes on shrine)
                        // would otherwise put a D2 view in a Cube binding →
                        // `create_bind_group` validation panic. The mismatched
                        // cases fall through to the dimension-matched fallback
                        // below (engine binds the cube register regardless).
                        Some(tex) if tex.is_cube == tb.is_cube => {
                            upload_view(tex, sample_intent_for_param(&tb.name))
                        }
                        // Dynamic-env cube slots (`dynamic_environment_map_*`,
                        // and a fallback `environment_map`) get the rasg
                        // `DefaultDynamicCubeMap` as the BASE view (engine's
                        // fallback when the cluster has no cubemap). The
                        // `bind_group_probes` built below OVERRIDE these slots
                        // with the object's nearest cluster cubemap (engine
                        // `render_method_submit_dynamic_cubemap`), so glass /
                        // env-mapped objects reflect the real environment. The
                        // base view is used only when no probe atlas is loaded.
                        // (Also catches a cube slot whose authored bitmap was
                        // 2D — the dimension guard above fell through.)
                        _ if tb.is_cube
                            && (tb.name.starts_with("dynamic_environment_map")
                                || tb.name == "environment_map") =>
                        {
                            self.shared
                                .fallback_textures
                                .default_bitmap_view(
                                    blam_tags::rasterizer_globals::DefaultBitmap::DefaultDynamicCubeMap,
                                )
                                .clone()
                        }
                        _ => fallback_view_for_param(&self.shared, &tb.name, tb.is_cube),
                    };
                    texture_views.push((tb.slot, view));
                }

                let props_bytes = serialize_material(mat, &artifacts.layout);
                // Path B: per-instance cbuffer pool. Buffer holds
                // `MAX_INSTANCES_PER_MODEL` slots of `slot_size` bytes
                // each. Per-draw uses dynamic offset `instance_idx *
                // slot_size` to pick the slot.
                //
                // `slot_size` is aligned up to 256 — wgpu's
                // `min_uniform_buffer_offset_alignment` (the value the
                // standard requires; per-device queries can be smaller
                // but 256 is the worst case and always-valid).
                //
                // Initialize EVERY slot with the load-time bake so
                // static materials behave correctly for any
                // `instance_within_model` value (each placement sees the
                // same baked values). Animated materials overwrite the
                // per-instance slots in the per-frame update with their
                // own object context. Phase 5 wiring.
                //
                // See [[project_render_method_function_port_plan_2026_05_20]].
                let slot_size = props_bytes.len()
                    .next_multiple_of(crate::halo::render_methods::pipeline_cache::MATERIAL_SLOT_ALIGNMENT) as u64;
                let buffer_size = slot_size
                    * crate::halo::render_methods::pipeline_cache::MAX_INSTANCES_PER_MODEL as u64;
                let props_buffer = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("material_props"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                {
                    let mut all_slots = vec![0u8; buffer_size as usize];
                    for i in 0..crate::halo::render_methods::pipeline_cache::MAX_INSTANCES_PER_MODEL {
                        let off = i * slot_size as usize;
                        all_slots[off..off + props_bytes.len()].copy_from_slice(&props_bytes);
                    }
                    self.shared.queue.write_buffer(&props_buffer, 0, &all_slots);
                }
                // Record animated-material state for the per-frame
                // walker. Gated on `mat.is_animated` + a present source
                // (engine equivalent: `c_render_method` instance with
                // a live overlay list). The `mat` reference here is
                // the post-stub-swap one — in the rmd-on-object case
                // it's a fresh stub (`is_animated=false`), so this
                // skips naturally; only the SH/Albedo/PRT-Ambient
                // path's `props_buffer` is tracked. The decal-object
                // path's `real_props_buffer` is a separate concern
                // (different MaterialData, different layout) and not
                // tracked today — no shipped rmd-on-object material
                // animates.
                if mat.is_animated {
                    if let Some(source) = mat.render_method_source.as_ref() {
                        animated_materials.push(
                            crate::halo::render_methods::animated::AnimatedMaterial {
                                source: source.clone(),
                                layout: artifacts.layout.clone(),
                                primary_change_color: mat.primary_change_color,
                                secondary_change_color: mat.secondary_change_color,
                                loaded_cbuffer_bytes: mat.cbuffer.bytes.clone(),
                                props_buffers: vec![props_buffer.clone()],
                            },
                        );
                    }
                }
                let mut entries: Vec<wgpu::BindGroupEntry> =
                    Vec::with_capacity(texture_views.len() + 2);
                for (slot, view) in &texture_views {
                    entries.push(wgpu::BindGroupEntry {
                        binding: *slot,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                // Per-key samplers — one per unique `SamplerKey` in
                // the variant's dedup map. Each `BitmapBinding`'s
                // authored `BitmapAddressMode` / `BitmapFilterMode`
                // is honored via the shared dedupe binding it points at.
                // Engine equivalent: `d3d11_sampler_state_cache::get`
                // per texture stage in `c_render_method_submit_set_texture`.
                let per_key_samplers: Vec<wgpu::Sampler> = artifacts
                    .sampler_map
                    .unique_keys
                    .iter()
                    .map(|&k| {
                        self.rasterizer
                            .sampler_state_cache
                            .get(&self.shared.device, k)
                    })
                    .collect();
                for (idx, sampler) in per_key_samplers.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: artifacts.bindings.per_binding_sampler_slot(idx),
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
                entries.push(wgpu::BindGroupEntry {
                    binding: artifacts
                        .bindings
                        .per_binding_cbuffer_slot(&artifacts.sampler_map),
                    // Path B per-instance pool: bind one slot's worth of
                    // the buffer; per-draw `set_bind_group(.., &[offset])`
                    // selects the active slot.
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &props_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(slot_size),
                    }),
                });

                let bind_group = self
                    .shared
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("material_bg"),
                        layout: &artifacts.material_bgl,
                        entries: &entries,
                    });
                // Per-cluster dynamic-env cube variants — engine
                // `render_method_submit_dynamic_cubemap` binds the object's
                // CLUSTER cubemap (resolved by `c_dynamic_cubemap_sample`) to
                // the `dynamic_environment_map_*` sampler. We mirror the BSP
                // per-probe approach: one bind group per scenario cubemap probe
                // (overriding only the dynamic-env cube slots), selected at draw
                // by the object's resolved `cubemap_probe_index`. Built only for
                // materials that actually sample a dynamic-env cube AND when a
                // probe atlas is loaded; empty otherwise → draw uses `bind_group`
                // (the rasg-default base, = MCC's stripped-cubemap fallback).
                let dyn_env_slots: Vec<u32> = artifacts
                    .bindings
                    .textures
                    .iter()
                    .filter(|tb| tb.is_cube && tb.name.starts_with("dynamic_environment_map"))
                    .map(|tb| tb.slot)
                    .collect();
                let bind_group_probes: Vec<wgpu::BindGroup> =
                    if dyn_env_slots.is_empty() || self.object_cubemap_probe_views.is_empty() {
                        Vec::new()
                    } else {
                        let device = &self.shared.device;
                        let sampler_cbuffer_entries = &entries[texture_views.len()..];
                        self.object_cubemap_probe_views
                            .iter()
                            .map(|probe_view| {
                                let mut probe_entries: Vec<wgpu::BindGroupEntry> =
                                    Vec::with_capacity(entries.len());
                                for (slot, view) in &texture_views {
                                    let v = if dyn_env_slots.contains(slot) {
                                        probe_view
                                    } else {
                                        view
                                    };
                                    probe_entries.push(wgpu::BindGroupEntry {
                                        binding: *slot,
                                        resource: wgpu::BindingResource::TextureView(v),
                                    });
                                }
                                // Samplers + cbuffer are identical across probes.
                                probe_entries.extend(sampler_cbuffer_entries.iter().cloned());
                                device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("material_bg_probe"),
                                    layout: &artifacts.material_bgl,
                                    entries: &probe_entries,
                                })
                            })
                            .collect()
                    };
                // Albedo variant shares the same bindings (textures +
                // sampler + props cbuffer) but lives behind a separate
                // pipeline_layout / material_bgl, so wgpu requires its
                // own bind_group object even when the contents match.
                let bind_group_albedo =
                    self.shared
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("material_bg_albedo"),
                            layout: &artifacts_albedo.material_bgl,
                            entries: &entries,
                        });
                let bind_group_prt_ambient = artifacts_prt_ambient.as_ref().map(|arts| {
                    self.shared
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("material_bg_prt_ambient"),
                            layout: &arts.material_bgl,
                            entries: &entries,
                        })
                });
                let bind_group_vertex_color = artifacts_vertex_color.as_ref().map(|arts| {
                    self.shared
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("material_bg_vertex_color"),
                            layout: &arts.material_bgl,
                            entries: &entries,
                        })
                });

                // Classify transparency the same way BspGpu does
                // (`classify_transparency` in bsp_gpu.rs): rmd/rmhg
                // are always-transparent, rmsh/rmcs depend on
                // `blend_mode != opaque`.
                //
                // rmd-on-OBJECT materials are an exception — engine
                // `submit_object_mesh_parts @ 0x1806E1BF0` routes them
                // through `render_albedo_decals` via the DECAL bit
                // (not transparency_renderer). We mirror by reporting
                // `is_transparent=false` so
                // `submit_visibility_from_render_list` skips the
                // transparency-add branch. The artifacts_decal_object
                // slot built below holds the actual dispatch pipeline.
                let mat_group_tag = real_mat.resolved.group_tag;
                let is_transparent = {
                    let fourcc = mat_group_tag.to_be_bytes();
                    let blend = choices.get_or("blend_mode", "opaque");
                    if is_rmd_on_object {
                        false
                    } else {
                        matches!(&fourcc, b"rmd " | b"rmhg")
                            || (matches!(&fourcc, b"rmsh" | b"rmcs") && blend != "opaque")
                    }
                };

                // Build the rmd-on-object pipeline + bind group from
                // the REAL rmd material (not the rmsh stub). Compiles
                // `entry_decal_object.wgsl` with the rmd shader's
                // texture parameters; the bind group binds the per-
                // binding decal_constants slot to a shared default
                // buffer (fade=1, pixel_kill=1, tiles=1, sprite=(0,0,1,1))
                // since object-attached decals don't author sprite or
                // per-decal fade.
                let (
                    artifacts_decal_object,
                    bind_group_decal_object,
                    default_decal_constants_buffer,
                ) = if is_rmd_on_object {
                    let real_stub;
                    let real_choices = if real_mat.category_choices.is_empty() {
                        real_stub = crate::halo::render_methods::pipeline_cache::stub_choices();
                        &real_stub
                    } else {
                        &real_mat.category_choices
                    };
                    let arts = self.halo_pipelines.ensure(
                        &self.shared,
                        &real_mat.resolved,
                        &real_mat.cbuffer,
                        real_choices,
                        HaloEntryPoint::DecalObject,
                        &self.scenario_tags_root,
                    );

                    // Re-build texture views from the REAL rmd material's
                    // parameter list (the rmsh stub above has none).
                    let mut real_texture_views: Vec<(u32, wgpu::TextureView)> =
                        Vec::with_capacity(arts.bindings.textures.len());
                    for tb in &arts.bindings.textures {
                        let view = match real_mat.find_texture_by_name(&tb.name) {
                            // Dimension must match the BGL slot (see the
                            // SH/albedo path above) — a 2D bitmap in a cube
                            // slot panics `create_bind_group`.
                            Some(tex) if tex.is_cube == tb.is_cube => {
                                upload_view(tex, sample_intent_for_param(&tb.name))
                            }
                            _ => fallback_view_for_param(&self.shared, &tb.name, tb.is_cube),
                        };
                        real_texture_views.push((tb.slot, view));
                    }

                    let real_props_bytes = serialize_material(real_mat, &arts.layout);
                    // Path B: per-instance cbuffer pool (same shape as
                    // `material_props` above). Decal-object materials
                    // get N slots so per-draw `instance_within_model *
                    // slot_size` offsets stay in-bounds. Every slot
                    // initialized with load-time bake (static behavior).
                    let real_slot_size = real_props_bytes.len()
                        .next_multiple_of(crate::halo::render_methods::pipeline_cache::MATERIAL_SLOT_ALIGNMENT)
                        as u64;
                    let real_buffer_size = real_slot_size
                        * crate::halo::render_methods::pipeline_cache::MAX_INSTANCES_PER_MODEL as u64;
                    let real_props_buffer = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("material_props_decal_object"),
                        size: real_buffer_size,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    {
                        let mut all_slots = vec![0u8; real_buffer_size as usize];
                        for i in 0..crate::halo::render_methods::pipeline_cache::MAX_INSTANCES_PER_MODEL {
                            let off = i * real_slot_size as usize;
                            all_slots[off..off + real_props_bytes.len()].copy_from_slice(&real_props_bytes);
                        }
                        self.shared.queue.write_buffer(&real_props_buffer, 0, &all_slots);
                    }

                    // Shared-default DecalConstants payload. 32 B —
                    // smaller than `min_uniform_buffer_offset_alignment`
                    // but the binding's `min_binding_size` is 32 so
                    // this satisfies validation. Bound with dynamic
                    // offset 0 at draw time.
                    let default_constants =
                        crate::halo::render::render_decals::DecalConstantsGpu::default();
                    let default_constants_buffer = self.shared.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("decal_object_default_constants"),
                            contents: bytemuck::bytes_of(&default_constants),
                            usage: wgpu::BufferUsages::UNIFORM,
                        },
                    );

                    let real_per_key_samplers: Vec<wgpu::Sampler> = arts
                        .sampler_map
                        .unique_keys
                        .iter()
                        .map(|&k| {
                            self.rasterizer
                                .sampler_state_cache
                                .get(&self.shared.device, k)
                        })
                        .collect();

                    let mut entries: Vec<wgpu::BindGroupEntry> =
                        Vec::with_capacity(real_texture_views.len() + 3);
                    for (slot, view) in &real_texture_views {
                        entries.push(wgpu::BindGroupEntry {
                            binding: *slot,
                            resource: wgpu::BindingResource::TextureView(view),
                        });
                    }
                    for (idx, sampler) in real_per_key_samplers.iter().enumerate() {
                        entries.push(wgpu::BindGroupEntry {
                            binding: arts.bindings.per_binding_sampler_slot(idx),
                            resource: wgpu::BindingResource::Sampler(sampler),
                        });
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: arts
                            .bindings
                            .per_binding_cbuffer_slot(&arts.sampler_map),
                        // Path B per-instance pool: bind one slot's worth;
                        // per-draw offset selects active slot.
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &real_props_buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(real_slot_size),
                        }),
                    });
                    // Per-binding decal_constants — REQUIRED for rmd
                    // since `material_bindings` adds the slot when
                    // group_tag=="rmd ". Bound as a sub-range with
                    // dynamic_offset=true at draw time.
                    if let Some(slot) =
                        arts.bindings.per_binding_decal_constants_slot(&arts.sampler_map)
                    {
                        entries.push(wgpu::BindGroupEntry {
                            binding: slot,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &default_constants_buffer,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<
                                        crate::halo::render::render_decals::DecalConstantsGpu,
                                    >() as u64,
                                ),
                            }),
                        });
                    }
                    let bg = self.shared.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            label: Some("material_bg_decal_object"),
                            layout: &arts.material_bgl,
                            entries: &entries,
                        },
                    );
                    (Some(arts), Some(bg), Some(default_constants_buffer))
                } else {
                    (None, None, None)
                };

                // Alpha-tested shadow casters: an OPAQUE material whose
                // render method has `alpha_test = simple` must discard masked
                // texels in the shadow-generate PS (engine `shadow_generate_ps`
                // + `calc_alpha_test_on_ps`) so it casts a holey shadow rather
                // than a solid silhouette. Build the per-material @group(1)
                // bind group (alpha_test_map view + sampler + xform).
                // `multiply_map` (rmcs deadlock-net custom) is deferred — those
                // are alpha_blend/transparent and skipped from shadows anyway.
                let shadow_alpha_test_bg = if !is_transparent
                    && choices.get_or("alpha_test", "none") == "simple"
                {
                    let xform = mat
                        .cbuffer
                        .find("alpha_test_map")
                        .map(|s| s.value)
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]);
                    mat.find_texture_by_name("alpha_test_map").map(|tex| {
                        if std::env::var_os("PROTOMORPH_DIAG_SHADOW_ALPHA_TEST").is_some() {
                            eprintln!(
                                "[shadow] alpha-test caster material '{}' (xform={:?})",
                                mat.name, xform,
                            );
                        }
                        let view =
                            upload_view(tex, sample_intent_for_param("alpha_test_map"));
                        self.shadow_apply.build_alpha_test_bind_group(
                            &self.shared.device,
                            &view,
                            xform,
                        )
                    })
                } else {
                    None
                };

                GpuMaterial {
                    bind_group,
                    bind_group_probes,
                    artifacts,
                    artifacts_albedo: Some(artifacts_albedo),
                    bind_group_albedo: Some(bind_group_albedo),
                    artifacts_prt_ambient,
                    bind_group_prt_ambient,
                    artifacts_vertex_color,
                    bind_group_vertex_color,
                    artifacts_decal_object,
                    bind_group_decal_object,
                    default_decal_constants_buffer,
                    shader_name: mat.name.clone(),
                    is_transparent,
                    blend_mode: choices.get_or("blend_mode", "opaque").to_string(),
                    group_tag: mat.resolved.group_tag,
                    // Per-shader atmosphere override (engine
                    // `render_method_submit_volatile_per_shader`): resolve this
                    // material's atmosphere-table offset from its render_method
                    // flags + custom_fog_setting_index. Sky backdrop shaders set
                    // `UseCustomSetting` + an index (construct → fog_waterfall);
                    // sun rings set `DontFogMe`; everything else = default.
                    atmosphere_offset: mat
                        .render_method_source
                        .as_ref()
                        .map(|s| {
                            use blam_tags::render_method::GlobalRenderMethodFlags as F;
                            let mut bits = 0u16;
                            if s.rmsh.flags.contains(F::DontFogMe) {
                                bits |= 0x1;
                            }
                            if s.rmsh.flags.contains(F::UseCustomSetting) {
                                bits |= 0x2;
                            }
                            crate::halo::render::shared::atmosphere_offset_for(
                                bits,
                                s.rmsh.custom_fog_setting_index,
                            )
                        })
                        .unwrap_or(crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET),
                    sort_layer: mat
                        .render_method_source
                        .as_ref()
                        .map(|s| {
                            crate::halo::render::render_transparents::TransparentSortLayer::from_global(
                                s.rmsh.sort_layer.get(),
                            )
                        })
                        .unwrap_or(crate::halo::render::render_transparents::TransparentSortLayer::Normal),
                    shadow_alpha_test_bg,
                }
            })
            .collect();

        // Model-space AABB + bounding sphere. Engine reads the
        // authored `(bounding_offset, bounding_radius)` from the object
        // tag (or `model object data[0]` from the .model fallback) at
        // load time and caches it at object header +0x1C/+0x28 — that's
        // what `object_get_bounding_sphere @ 0x1802473A0` returns. We
        // mirror by preferring the authored value; the AABB is still
        // walked from vertex positions for the shadow OBB-projection
        // path. When neither authored bounds nor vertex positions are
        // usable (skinned meshes whose vertex buffer holds bone-local
        // garbage post-decompression), we fall back to a sphere
        // around the authored center.
        let (bounding_sphere, bounding_aabb_min, bounding_aabb_max) = {
            let mut have_any = false;
            let mut min = glam::Vec3::splat(f32::INFINITY);
            let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
            for mesh in &model.meshes {
                for v in &mesh.vertices {
                    let p = glam::Vec3::from_array(v.position);
                    min = min.min(p);
                    max = max.max(p);
                    have_any = true;
                }
            }
            // Prefer authored sphere from the object tag — that's the
            // engine's runtime behavior. Vertex AABB is purely for the
            // OBB shadow camera projection.
            let authored = model.authored_bounding_sphere;
            if let Some(bs) = authored {
                let center = bs.truncate();
                let radius = bs.w;
                let aabb_min = if have_any { min } else { center - glam::Vec3::splat(radius) };
                let aabb_max = if have_any { max } else { center + glam::Vec3::splat(radius) };
                (bs, aabb_min, aabb_max)
            } else if have_any {
                let center = (min + max) * 0.5;
                let mut r2 = 0.0_f32;
                for mesh in &model.meshes {
                    for v in &mesh.vertices {
                        let p = glam::Vec3::from_array(v.position);
                        r2 = r2.max((p - center).length_squared());
                    }
                }
                let radius = r2.sqrt().max(0.5);
                (
                    glam::Vec4::new(center.x, center.y, center.z, radius),
                    min,
                    max,
                )
            } else {
                (
                    glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
                    glam::Vec3::splat(-0.5),
                    glam::Vec3::splat(0.5),
                )
            }
        };

        let meshes: Vec<GpuMesh> = model
            .meshes
            .iter()
            // Skip empty meshes — wgpu rejects 0-byte buffer slices,
            // and BSP render geometry blocks sometimes carry placeholder
            // mesh slots with no vertices/indices.
            .filter(|mesh| !mesh.vertices.is_empty() && !mesh.indices.is_empty())
            .map(|mesh| {
                let verts = &mesh.vertices;
                let vertex_buffer = self.shared.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("vertex_buffer"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                );
                let index_buffer =
                    self.shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("index_buffer"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                let prt_ambient_buffer = if mesh.prt_ambient_stream.is_empty() {
                    None
                } else {
                    Some(self.shared.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("model_mesh_prt_ambient"),
                            contents: bytemuck::cast_slice(&mesh.prt_ambient_stream),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ))
                };
                let vert_color_buffer = if mesh.vert_color_stream.is_empty() {
                    None
                } else {
                    // Engine sky shader (`sky_dome_simple_hlsl.hlsl`)
                    // reads `input.color` (RealVector3d packed via the
                    // `_vertex_buffer_usage_vert_color` stream). Pad to
                    // vec4 because Metal aligns 3-component vertex
                    // attributes to 16B.
                    let padded: Vec<SkyVertColorVertex> = mesh
                        .vert_color_stream
                        .iter()
                        .map(|c| SkyVertColorVertex { color: [c[0], c[1], c[2], 0.0] })
                        .collect();
                    Some(self.shared.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("model_mesh_vert_color"),
                            contents: bytemuck::cast_slice(&padded),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ))
                };
                GpuMesh {
                    vertex_buffer,
                    index_buffer,
                    parts: mesh.parts.clone(),
                    prt_ambient_buffer,
                    vert_color_buffer,
                    use_region_index_for_sorting: mesh.use_region_index_for_sorting,
                    region_index: mesh.region_index,
                }
            })
            .collect();

        let index = self.models.len();
        // One-shot diagnostic: print per-model AABB + sphere when the
        // env var is set, so we can correlate "tiny default AABB" with
        // missing shadows.
        if std::env::var("PROTOMORPH_LOG_MODEL_AABB").map(|v| v == "1").unwrap_or(false) {
            let extent = bounding_aabb_max - bounding_aabb_min;
            eprintln!(
                "[model {}] aabb=({:.2},{:.2},{:.2})..({:.2},{:.2},{:.2}) extent=({:.2},{:.2},{:.2}) bs_radius={:.2}",
                index,
                bounding_aabb_min.x, bounding_aabb_min.y, bounding_aabb_min.z,
                bounding_aabb_max.x, bounding_aabb_max.y, bounding_aabb_max.z,
                extent.x, extent.y, extent.z,
                bounding_sphere.w,
            );
        }
        self.models.push(GpuModel {
            meshes,
            materials,
            animated_materials,
            bounding_sphere,
            bounding_aabb_min,
            bounding_aabb_max,
            casts_shadow: model.casts_shadow,
            node_local_matrices: model.node_local_matrices.clone(),
            marker_transforms: model.marker_transforms.clone(),
        });
        // Drain this model's texture/mesh staging now so the per-object
        // uploads (a scenario can place hundreds of unique object tags) don't
        // accumulate into the first-frame `queue.submit` — the GPU-watchdog
        // hang that bit the BSP material path. See `bsp_gpu::flush_staging`.
        crate::halo::structures::bsp_gpu::flush_staging(&self.shared);
        index
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.shared.config.width = width;
        self.shared.config.height = height;
        self.surface.configure(&self.shared.device, &self.shared.config);

        self.gbuffer = GBuffer::new(&self.shared.device, width, height);
        self.intermediates = IntermediateTargets::new(
            &self.shared.device,
            width,
            height,
            self.shared.config.format,
        );
        // Refresh the G-buffer views in shared *before* rebuilding the
        // camera bind groups — `rebuild_camera_bind_group` reads
        // `self.gbuffer_albedo_view` / `_normal_view`, and those still
        // point at the just-dropped textures until `set_gbuffer_views`
        // installs views into the freshly-allocated ones. Without this,
        // the SL pass samples a stale texture that frame N never
        // writes — visible as a non-moving copy of the scene blended
        // over the live render.
        let gbuffer_albedo_view = self
            .intermediates
            .albedo_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let gbuffer_normal_view = self
            .gbuffer
            .normal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Fresh DepthOnly-aspect scene-depth view for the transparent
        // camera bind group's soft_z sampling (binding 15). Views the
        // COPY texture, not the live depth attachment (TBDR flicker —
        // see `GBuffer::depth_sample_view`).
        let gbuffer_depth_view = self.gbuffer.depth_copy_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene_depth_soft_z"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.shared
            .set_gbuffer_views(gbuffer_albedo_view, gbuffer_normal_view, gbuffer_depth_view);
        self.shared.rebuild_camera_bind_group();
        // The water extern bind group references the freshly-allocated
        // scene_ldr_snapshot + scene_depth views at @group(2); rebuild
        // it now that the underlying textures changed.
        self.water_renderer.rebuild_extern_bind_group(
            &self.shared.device,
            &self.intermediates,
            &self.gbuffer,
        );

        self.final_composite_pass.resize(&self.shared, &self.gbuffer, &self.intermediates);
        // Re-allocate aux_bloom + rebuild composite bind group with new view.
        self.screen_postprocess.resize(&self.shared);
        // Distortion accumulation + scene-copy targets track the viewport.
        self.particle_gpu.resize(&self.shared.device, width, height);
        // Bind-group cache invalidation — intermediates rebuilt above
        // means cached views' addresses are dead. Drop cached entries
        // so the next frame rebuilds with the new pointers.
        self.shadow_apply.invalidate_cache();
        self.final_composite_pass.set_bound_views(
            &self.shared,
            &self.intermediates,
            &self.gbuffer,
            self.screen_postprocess.bloom_result_view(),
            self.screen_postprocess.color_grading_active_view(),
            self.screen_postprocess.color_grading_back_view(),
            self.screen_postprocess.dof_blur_view(),
            self.screen_postprocess.bling_result_view(),
        );
    }

    /// Per-frame orchestrator entry. Engine: `c_player_view::render
    /// @ 0x180688ba0`. The body lives on `PlayerView::render_player_view`;
    /// here we `mem::take` the PlayerView out to break the
    /// `&mut self.player_view` + `&mut self` borrow conflict, run, and
    /// put it back.
    pub fn render(&mut self, game: &GameState) {
        // L2: simple_lights cbuffer slots are baked once at scenario
        // load by `bake_cluster_simple_lights`. Per-cluster draws
        // bind their own slot via dynamic uniform offset. No per-frame
        // upload needed — engine-faithful: light selection is static,
        // matches dllcache `c_lights_view::add_simple_light_to_draw_list`
        // being called once per cluster part during scenario init,
        // not per frame.

        // Engine `s_scripted_exposure::update_scripted_exposure @ 0x180686A00`
        // — per-tick advance of the scripted-exposure animation +
        // counters. Engine calls this before `setup_camera_fx_parameters`
        // each tick. Inert at default-zero state; activates when HSC
        // opcodes populate the singleton (FAST_ADAPT bit lifecycle,
        // DISABLE_AUTOEXPOSURE lockout window, smoothstep blend).
        // Tick rate: 30 Hz canonical (engine `game_time_get()`).
        let game_ticks = (game.total_time * 30.0) as i32;
        self.scripted_exposure_game_globals.update_scripted_exposure(game_ticks);

        // Particle spawn → integrate. Scatter this frame's CPU-pulsed
        // spawn requests into the persistent grid (engine `spawn_all`),
        // then age + apply physics over the whole grid (`frame_advance_all`).
        // Billboard render (Phase 4) follows in the main pass.
        self.particle_gpu.dispatch_spawn(
            &self.shared,
            &game.effects.pending_spawns,
            &game.effects.pending_slots,
        );
        self.particle_gpu.dispatch_update(
            &self.shared,
            game.delta_time,
            &game.effects.batch_self_accel,
            &game.effects.batch_state,
            game.effects.row_pool.batch_table(),
        );

        // L3 master per-frame blender — port of
        // `c_player_view::setup_camera_fx_parameters @ 0x180689C20`.
        // Resolves the 4-level cfxs source chain, runs the inline
        // per-parameter chooser + L2 update_X family, computes
        // `m_render_exposure` and `m_illum_render_scale`. Subsumes the
        // prior `auto_exposure_frame_start` (exposure update is one
        // parameter inside `c_camera_fx_values::update` now).
        self.setup_camera_fx_parameters(game.camera.position);
        // Phase E: per-frame view_exposure upload from runtime
        // `m_render_exposure` (computed by the orchestrator above).
        // Threads `game.total_time` through `g_alt_exposure.w` for the
        // foliage VS wind-phase reader (see `from_camera_fx_with_time`).
        self.upload_view_exposure(game.total_time);
        // Phase 7: per-frame `ps_total_time` upload to MiscPS so the
        // PS-side time-driven shaders see a continuously advancing clock.
        self.upload_engine_misc_ps(game.total_time);
        // OL-6 — per-frame chmt boost + ravi cbuffer refresh. Engine
        // `setup_object_lighting_for_entry_point @ 0x1806A9AA0` fires
        // per-draw; we do all placements once per frame with the
        // exposure value just computed. Must run AFTER
        // `setup_camera_fx_parameters` (which sets `render_exposure`)
        // and BEFORE `render_player_view` (which submits object draws
        // that read the cbuffer).
        self.refresh_engine_lighting_for_frame();

        let mut pv = std::mem::take(&mut self.player_view);
        pv.render_player_view(self, game);
        self.player_view = pv;
    }

    /// Per-frame master cfxs orchestrator — port of
    /// `c_player_view::setup_camera_fx_parameters @ 0x180689C20`.
    ///
    /// Engine body:
    /// 1. Resolve scenario.camera_effects → `scenario_settings` (or None).
    /// 2. Resolve `c_world_view::get_starting_cluster()` →
    ///    sbsp.cluster_camera_fx_palettes[cluster.palette_idx] →
    ///    `cluster_palette_entry` (DEFERRED to L3a; None for now).
    /// 3. Resolve `global_game_globals.default_camera_fx_settings` → `default_settings`.
    /// 4. Call `c_camera_fx_values::update(values, script, palette,
    ///    scenario, default, exposure_boost)`.
    /// 5. Compute `m_lights_view.m_light_intensity_scale = pow(2,
    ///    self_illum_exposure - apply_scripted(m_exposure)) ×
    ///    g_render_light_intensity × scripted.m_light_intensity_scale`
    ///    (DEFERRED — no consumer wired yet).
    /// 6. Write `m_render_exposure = c_camera_fx_values::get_render_exposure()`.
    /// 7. Write `m_illum_render_scale = pow(2, self_illum_exposure -
    ///    apply_scripted(m_exposure))`.
    ///
    /// Replaces the prior `auto_exposure_frame_start`. Exposure update
    /// is now one parameter inside the master blender; the `update`
    /// method does the FIFO read + auto-bit gating internally.
    fn setup_camera_fx_parameters(&mut self, camera_position: glam::Vec3) {
        // Engine: read OLDEST queue entry (queue[queue_index] when
        // queue_size >= 3 — index points to next-overwrite slot,
        // which is also the oldest when the FIFO is full).
        //
        // Engine-faithful: `try_read_exposure_sample` returns None
        // when the async map callback for this slot hasn't fired yet.
        // Engine `calculate_new_value @ 0x180686F50:30` uses `v10 = 0.0`
        // when D3D11 `Map(DO_NOT_WAIT)` returns null; protomorph
        // mirrors that by passing `None` through to the math (treated
        // as 0.0 in the AUTO_BIT branch).
        // Engine `calculate_new_value @ 0x180686F50:32-43`:
        // ```c
        // if (queue_size >= 3) {
        //     v12 = m_render_target_queue[m_render_target_queue_index]; // SURFACE INDEX
        //     if (v12 != -1) {
        //         get_surface_texture(&texture, v12);
        //         v13 = lock_read(...);
        //         if (v13) v10 = -half_to_real(*v13);
        //     }
        // }
        // ```
        // The queue stores surface enum values (28..35); translate back to
        // a 0..7 slot for `aux_exposure[]` indexing via
        // `Exposure::slot_from_surface_index`.
        let fresh_sample = if self.render_game_state.player_window[0].camera_fx_values.exposure.render_target_queue_size >= 3 {
            let oldest_surface = self.render_game_state.player_window[0].camera_fx_values.exposure.render_target_queue
                [self.render_game_state.player_window[0].camera_fx_values.exposure.render_target_queue_index as usize];
            match crate::halo::render::camera_fx_settings::Exposure::slot_from_surface_index(oldest_surface) {
                Some(slot) => self.screen_postprocess.try_read_exposure_sample(&self.shared, slot),
                None => None,
            }
        } else {
            None
        };
        // Engine `calculate_new_value @ 0x180686F50:30`: initializes
        // `v10 = 0.0` and uses 0.0 when `lock_read` fails (D3D11
        // `Map(DO_NOT_WAIT)` returns null). Protomorph mirrors this by
        // passing `None` straight through — `Exposure::calculate_new_value`
        // treats `None` as `v10 = 0.0`. A stale-value fallback (kept here
        // historically) caused "wrong direction at wrong times" because
        // the math compensated by an out-of-date scene brightness sample.
        let actual_brightness = fresh_sample;

        let scenario_ref = self.scenario_camera_fx.as_ref();
        let default_ref = self.default_camera_fx.as_ref();

        // DIAG: log raw FIFO sample once per second to see what the
        // auto-exposure downsample shader returns. Engine `exposure_downsample`
        // writes log2(scene_avg_luminance) — convergence target is ~+0.035 stops
        // (auto_exposure_screen_brightness); +3-stops clamp = scene is 3 stops
        // (~8×) below target.
        //
        // Additional variance trace (PROTOMORPH_DEBUG_EXPOSURE=1): maintain
        // a 60-entry ring of recent actual_brightness samples; once per
        // second log min/max/range so we can see how much the bloom-feed
        // jitters frame-to-frame. In a STATIC camera, range > 0.5 stops
        // means patchy-fog (or other animated content) is feeding noise
        // into the auto-exposure measurement — Issue #3.
        static DIAG_TICK: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(0);
        let tick = DIAG_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(v) = fresh_sample {
            let idx = (tick as usize) % self.actual_brightness_ring.len();
            self.actual_brightness_ring[idx] = Some(v);
            // Per-frame outlier detector: if this sample is >1.0 stop
            // away from the running ring mean, log it. Helps catch
            // single-frame spikes that perturb auto-exposure even when
            // the typical reading is steady.
            if std::env::var_os("PROTOMORPH_DEBUG_EXPOSURE").is_some() {
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for s in &self.actual_brightness_ring {
                    if let Some(b) = s { sum += *b; cnt += 1; }
                }
                if cnt >= 10 {
                    let mean = sum / cnt as f32;
                    if (v - mean).abs() > 1.0 {
                        eprintln!(
                            "[exposure-outlier] frame={tick} actual_b={v:.3} ring_mean={mean:.3} delta={:.3} stops",
                            v - mean,
                        );
                    }
                }
            }
        }
        if tick % 60 == 0 && actual_brightness.is_some() {
            let v = actual_brightness.unwrap();
            eprintln!(
                "[diag] actual_brightness={:.4} stops (=> linear scene_avg={:.6}); m_exposure={:.3}, render_exposure={:.3}",
                v, v.exp2(),
                self.render_game_state.player_window[0].camera_fx_values.exposure.exposure,
                self.render_exposure,
            );
            if std::env::var_os("PROTOMORPH_DEBUG_EXPOSURE").is_some() {
                let mut min_b = f32::INFINITY;
                let mut max_b = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for s in &self.actual_brightness_ring {
                    if let Some(b) = s {
                        if *b < min_b { min_b = *b; }
                        if *b > max_b { max_b = *b; }
                        sum += *b;
                        cnt += 1;
                    }
                }
                if cnt >= 2 {
                    let mean = sum / cnt as f32;
                    let mut var = 0.0f32;
                    for s in &self.actual_brightness_ring {
                        if let Some(b) = s { var += (*b - mean).powi(2); }
                    }
                    let stddev = (var / cnt as f32).sqrt();
                    eprintln!(
                        "[exposure-variance] n={cnt} min={min_b:.3} max={max_b:.3} range={:.3} stddev={stddev:.3} stops",
                        max_b - min_b,
                    );
                }
            }
        }

        // Engine `c_camera_fx_values::update @ 0x180687CB0:47-101`
        // resolves the camera's current cluster, looks up that cluster's
        // entry in the BSP `camera fx palette` block, and (when `flags`
        // bits authorize) overrides the inherited scenario cfxs with
        // per-cluster forced_exposure / brightness target / exposure
        // min-max / inherent_bloom / bloom_intensity. We mirror the
        // lookup here so levels that author per-cluster exposure stops
        // get honored (e.g. a dim interior cluster whose `forced
        // exposure` pins m_exposure to a specific value).
        // Engine `c_world_view::get_starting_cluster` calls
        // `scenario_cluster_reference_from_point(camera.position)` which
        // walks `bsp.bsp3d` to find the leaf containing the camera,
        // then maps `bsp.leaves[leaf_index].cluster` → cluster index.
        // Protomorph mirrors this via `scenario_location_from_camera_point`
        // (Phase 7.2 fix S3). Falls back to AABB containment + nearest-
        // center when camera is outside the BSP (engine handles this
        // case via `m_stored_cluster`).
        let camera_point = blam_tags::math::RealPoint3d {
            x: camera_position.x,
            y: camera_position.y,
            z: camera_position.z,
        };
        let cluster_palette_overrides = self
            .loaded_scenario
            .as_ref()
            .and_then(|scn| {
                let location = crate::halo::scenario::location::scenario_location_from_camera_point(
                    scn, camera_point,
                );
                let bsp_index = location.cluster_reference.bsp_index;
                if bsp_index < 0 { return None; }
                let bsp = scn.active_bsps.iter()
                    .find(|b| b.scenario_bsp_index as i16 == bsp_index)?;
                let cluster_idx = location.cluster_reference.cluster_index;
                if cluster_idx < 0 { return None; }
                let cluster = bsp.sbsp.clusters.get(cluster_idx as usize)?;
                let palette_idx = cluster.camera_fx_index;
                if palette_idx < 0 { return None; }
                let entry = bsp.sbsp.camera_fx_palette.get(palette_idx as usize)?;
                Some(crate::halo::render::camera_fx_settings::ClusterCameraFxPaletteOverrides {
                    flags: entry.flags.clone(),
                    forced_exposure: entry.forced_exposure,
                    forced_auto_exposure_brightness: entry.forced_auto_exposure_brightness,
                    exposure_min: entry.exposure_min,
                    exposure_max: entry.exposure_max,
                    inherent_bloom: entry.inherent_bloom,
                    bloom_intensity: entry.bloom_intensity,
                })
            });
        // Diagnostic — log cluster palette decisions once per second
        // so we can see when overrides kick in.
        if std::env::var_os("PROTOMORPH_DEBUG_EXPOSURE").is_some() && tick % 60 == 0 {
            // Detailed diag: cluster idx + palette idx + entry count.
            // Uses the same BSP3D walker (`scenario_location_from_camera_point`)
            // as the override resolution so the diag and the path agree.
            let detail = self.loaded_scenario.as_ref().and_then(|scn| {
                let location = crate::halo::scenario::location::scenario_location_from_camera_point(
                    scn, camera_point,
                );
                let bsp_index = location.cluster_reference.bsp_index;
                let bsp = scn.active_bsps.iter()
                    .find(|b| b.scenario_bsp_index as i16 == bsp_index)?;
                let cluster_idx = location.cluster_reference.cluster_index;
                let cluster_camera_fx_idx = if cluster_idx >= 0 {
                    bsp.sbsp.clusters.get(cluster_idx as usize).map(|c| c.camera_fx_index)
                } else { None };
                Some((
                    Some(cluster_idx as usize),
                    cluster_camera_fx_idx,
                    bsp.sbsp.camera_fx_palette.len(),
                ))
            });
            if let Some(o) = cluster_palette_overrides.as_ref() {
                use blam_tags::structure_bsp::CameraFxPaletteFlags;
                let force_exp = o.flags.contains(CameraFxPaletteFlags::ForceExposure);
                let force_auto = o.flags.contains(CameraFxPaletteFlags::ForceAutoExposure);
                let minmax = o.flags.contains(CameraFxPaletteFlags::OverrideExposureBounds);
                let bloom = o.flags.contains(CameraFxPaletteFlags::OverrideInherentBloom);
                eprintln!(
                    "[exposure-cluster] flags={:?} (force_exp={force_exp} force_auto_b={force_auto} minmax={minmax} bloom={bloom}) \
                     forced_exp={:.3} forced_auto_b={:.3} min/max=[{:.3}, {:.3}] inherent={:.3} bloom={:.3}",
                    o.flags, o.forced_exposure, o.forced_auto_exposure_brightness,
                    o.exposure_min, o.exposure_max,
                    o.inherent_bloom, o.bloom_intensity,
                );
            } else if let Some((cluster_idx, cluster_cam_fx_idx, palette_n)) = detail {
                eprintln!(
                    "[exposure-cluster] no override (cluster={:?} cluster.camera_fx_index={:?} palette_entries={palette_n})",
                    cluster_idx, cluster_cam_fx_idx,
                );
            } else {
                eprintln!("[exposure-cluster] no scenario / no BSP at this position");
            }
        }
        let auto_exposure_lock = self.screen_postprocess.settings_internal.auto_exposure_lock;
        self.render_game_state.player_window[0].camera_fx_values.update(
            &self.script_camera_fx,
            None, // cluster_settings — L3a (per-cluster cfxs tag lookup deferred)
            cluster_palette_overrides.as_ref(),
            scenario_ref,
            default_ref,
            self.exposure_boost,
            actual_brightness,
            auto_exposure_lock,
            &self.scripted_exposure_game_globals,
        );

        // === Engine line 134-150 — write c_player_view fields ===
        // L6: route through `CameraFxValues::get_*` accessors so the
        // engine math lives in one place. `m_render_exposure` and
        // `m_illum_render_scale` are the per-frame exposure scalars the
        // shaders read via `g_exposure.r` / `g_alt_exposure.r`.
        let tone_curve_white_point =
            self.screen_postprocess.settings_internal.tone_curve_white_point;
        self.render_exposure = self.render_game_state.player_window[0]
            .camera_fx_values
            .get_render_exposure(
                &self.scripted_exposure_game_globals,
                self.g_exposure_stops,
                tone_curve_white_point,
            );
        // Engine `setup_camera_fx_parameters @ 0x180689C20:133-148`:
        // ```c
        // v23 = m_self_illum_exposure - render_apply_scripted_exposure(globals, m_exposure);
        // v24 = powf(2, v23);
        // m_lights_view.m_light_intensity_scale = v24
        //     * c_lights_view::g_render_light_intensity
        //     * globals.m_light_intensity_scale;
        // m_render_exposure = get_render_exposure(camera_fx_values);
        // m_illum_render_scale = powf(2, v23);
        // ```
        // v23 and v28 in the decompile are the same calculation; we
        // compute once via `get_illum_render_scale` (now scripted-aware).
        self.illum_render_scale = self
            .render_game_state
            .player_window[0]
            .camera_fx_values
            .get_illum_render_scale(&self.scripted_exposure_game_globals);
        if std::env::var_os("PROTOMORPH_DIAG_ILLUM").is_some() {
            let cfx = &self.render_game_state.player_window[0].camera_fx_values;
            eprintln!(
                "[illum] illum_render_scale={:.4} self_illum_exposure={:.4} self_illum_preferred={:.4} self_illum_scale={:.4} m_exposure={:.4} render_exposure={:.3}",
                self.illum_render_scale, cfx.self_illum_exposure, cfx.self_illum_preferred,
                cfx.self_illum_scale, cfx.exposure.exposure, self.render_exposure,
            );
        }
        // `c_lights_view::g_render_light_intensity` is a class-level static.
        // Per `scenario/loader.rs:215` + `:1398`, protomorph fixes it at 1.0
        // (the engine default — `g_render_set_light_intensity` HSC mutates).
        const G_RENDER_LIGHT_INTENSITY: f32 = 1.0;
        let light_intensity_scale = self.illum_render_scale
            * G_RENDER_LIGHT_INTENSITY
            * self.scripted_exposure_game_globals.light_intensity_scale;
        self.lights_view.set_light_intensity_scale(light_intensity_scale);

        // DIAG: if PROTOMORPH_FORCE_EXPOSURE env var is set to a stops
        // value, override m_exposure to that fixed value to bypass
        // auto-exposure. Lets us see the un-boosted scene to isolate
        // "scene is dim" vs "auto-exposure over-amplifies".
        if let Ok(s) = std::env::var("PROTOMORPH_FORCE_EXPOSURE") {
            if let Ok(stops) = s.parse::<f32>() {
                const TONE_CURVE_WHITE_POINT: f32 = 1.4938016;
                self.render_exposure =
                    stops.exp2() * TONE_CURVE_WHITE_POINT * 0.66943294;
                self.illum_render_scale = (-stops).exp2();
            }
        }

        // L3.4: refresh the postprocess snapshot from the now-blended
        // `c_camera_fx_values`. Existing screen_postprocess code reads
        // from this snapshot per-pass — until those readers move to
        // direct CameraFxValues access (L7), keep the snapshot fresh.
        self.screen_postprocess.camera_fx =
            crate::halo::render::screen_postprocess::CameraFxBloomDefaults::from_camera_fx_values(
                &self.render_game_state.player_window[0].camera_fx_values,
            );
    }

    /// Post-submit hook for the auto-exposure ring. Starts an async
    /// map of the slot we just scheduled — the callback fires on a
    /// driver thread when the GPU finishes the copy. Subsequent
    /// frames' `try_read_exposure_sample` polls the result without
    /// stalling. `None` means the readback was skipped this frame
    /// (previous map_async still pending) and there's nothing to map.
    ///
    /// The GPU dispatch + FIFO push that used to live in
    /// `dispatch_auto_exposure` are now inlined inside
    /// `ScreenPostprocess::downsample_generate` (called from
    /// `postprocess_bloom_buffer` step 1) so the exposure_downsample
    /// reads the freshly box-downsampled scene BEFORE the bloom pyramid
    /// blurs and tints aux_tiny. Engine-faithful per
    /// `c_screen_postprocess::downsample_generate @ 0x1806B5C00`.
    pub fn post_submit_exposure(&self, slot: Option<u32>) {
        if let Some(slot) = slot {
            self.screen_postprocess.begin_exposure_map(slot as usize);
        }
    }

    /// Pre-load `rasterizer_globals` BEFORE `LoadedScenario::load`
    /// runs so material bind groups built during BSP load see the
    /// engine's `default_bitmaps[]` (12 fallback textures including
    /// `DefaultDynamicCubeMap`) instead of the 1×1 placeholders.
    /// Mirrors `c_rasterizer_globals::initialize`.
    ///
    /// Per-BSP cubemap atlas + cluster routing is loaded later in
    /// `upload_bsp` (`build_bsp_cubemap_routing`) — at draw time each
    /// cluster picks its nearest scenario probe, the engine's
    /// `c_dynamic_cubemap_sample::search_for_cubemap_sample_in_cluster`
    /// equivalent.
    fn preload_dynamic_cubemap_for_scenario(
        &mut self,
        tags_root: &std::path::Path,
        _scenario_path: &std::path::Path,
    ) {
        let rasg_path = tags_root.join("globals/rasterizer_globals.rasterizer_globals");
        match blam_tags::file::TagFile::read(&rasg_path) {
            Ok(tag) => match blam_tags::rasterizer_globals::RasterizerGlobals::from_tag(&tag) {
                Ok(rasg) => {
                    eprintln!(
                        "[scenario]   rasterizer_globals: {} default_bitmaps, {} material_textures, {} explicit_shaders",
                        rasg.default_bitmaps.len(),
                        rasg.material_textures.len(),
                        rasg.explicit_shaders.len(),
                    );
                    self.rasterizer_globals = rasg;
                }
                Err(e) => eprintln!("[scenario]   rasterizer_globals parse failed: {e}"),
            },
            Err(e) => eprintln!("[scenario]   rasterizer_globals read failed: {e}"),
        }
        self.load_default_bitmaps_from_rasg(tags_root);
        self.load_material_textures_from_rasg(tags_root);
    }



    /// Re-pack `engine_exposure_buffer` from the values computed by
    /// `setup_camera_fx_parameters`. `g_exposure.r = m_render_exposure`,
    /// `g_alt_exposure.r = m_illum_render_scale`, `g_alt_exposure.w =
    /// total_time` (used by the foliage VS for wind phase). Engine
    /// writes the first two from `c_player_view::m_render_exposure` /
    /// `m_illum_render_scale`; protomorph adds `total_time` to the
    /// otherwise-unused fourth slot.
    fn upload_view_exposure(&mut self, total_time: f32) {
        self.shared.queue.write_buffer(
            &self.shared.engine_exposure_buffer,
            0,
            bytemuck::bytes_of(
                &crate::halo::render::shared::GpuEngineExposure::from_camera_fx_with_time(
                    self.render_exposure,
                    self.illum_render_scale,
                    0.0,
                    total_time,
                ),
            ),
        );
    }

    /// Per-frame `MiscPS` upload — refreshes `ps_total_time` (engine
    /// `hlsl_constant_persist_fx.hlsl:114`) so shaders that animate
    /// over time (foliage wind, scrolling textures, etc.) advance.
    /// `dynamic_environment_blend` + flags keep their default values
    /// until probe-blend / pc-flag drivers land.
    fn upload_engine_misc_ps(&mut self, total_time: f32) {
        let mut misc = crate::halo::render::shared::GpuEngineMiscPs::defaults(
            self.shared.config.width,
            self.shared.config.height,
        );
        misc.ps_total_time = [total_time, 0.0, 0.0, 0.0];
        self.shared.queue.write_buffer(
            &self.shared.engine_misc_ps_buffer,
            0,
            bytemuck::bytes_of(&misc),
        );
    }

}

/// Build the per-BSP cubemap routing — load
/// `<scenario_dir>/<scenario>_<bsp>_cubemaps.bitmap` (one cube per
/// scenario probe; 1:1 by index for single-BSP MP maps) and synthesize
/// `cluster_to_probe[cluster_idx]` by nearest-distance from cluster
/// centroid to scenario probe position. Mirrors the engine's
/// `structure_bsp_build_instance_cubemap_references` (Reach tool-time)
/// + `c_dynamic_cubemap_sample::search_for_cubemap_sample_in_cluster`
/// (MCC runtime). MCC strips `cluster_cubemaps[]` so we recompute the
/// nearest-probe assignment at scenario load.
///
/// Returns `None` when the atlas file is missing or empty (the BSP
/// then has no per-cluster routing — material bind groups bind the
/// rasg default cube as a single shared variant).
fn build_bsp_cubemap_routing(
    shared: &shared::SharedResources,
    bsp: &crate::halo::scenario::loader::LoadedBsp,
    scenario_cubemaps: &[blam_tags::scenario::CubemapEntry],
) -> Option<crate::halo::structures::bsp_gpu::BspCubemapRouting> {
    if scenario_cubemaps.is_empty() {
        return None;
    }
    // Atlas path: `<sbsp_parent>/<scenario_stem>_<bsp_stem>_cubemaps.bitmap`.
    // For Guardian (`guardian.scenario` + `guardian.scenario_structure_bsp`)
    // this resolves to `guardian_guardian_cubemaps.bitmap`. Multi-BSP
    // campaign maps name each atlas after the BSP stem.
    let scenario_dir = bsp.sbsp_path.parent()?;
    // We don't have the scenario filename directly here, but the
    // scenario stem matches the BSP's parent-directory name
    // (`levels/multi/guardian/`) for MP maps. Read it from the dir.
    let dir_stem = scenario_dir.file_name().and_then(|s| s.to_str())?;
    let bsp_stem = bsp.sbsp_path.file_stem().and_then(|s| s.to_str())?;
    let atlas_path = scenario_dir.join(format!("{dir_stem}_{bsp_stem}_cubemaps.bitmap"));
    let images = crate::halo::bitmaps::load_all_images(&atlas_path);
    if images.is_empty() {
        eprintln!(
            "[bsp]   no cubemap atlas at {} — using rasg default cube only",
            atlas_path.display(),
        );
        return None;
    }
    eprintln!(
        "[bsp]   cubemap atlas: {} probes from {}",
        images.len(),
        atlas_path.display(),
    );
    let probe_views: Vec<wgpu::TextureView> = images
        .iter()
        .map(|tex| {
            // RGBW cube: DXT5 with the HDR multiplier in alpha and RGB
            // gamma-encoded (bitmap curve = XrgbGamma2). Upload as sRGB so
            // the sampler gamma-decodes RGB while leaving alpha (the W
            // multiplier) linear — exactly the RGBW reconstruction the
            // `× a × 256` shader path expects. (Was `true`/linear, which
            // read the gamma RGB ~2-3× too bright and miscolored.)
            crate::halo::structures::bsp_gpu::upload_inline_texture(shared, tex, SampleIntent::FollowCurve)
        })
        .collect();
    crate::halo::structures::bsp_gpu::gpu_checkpoint(shared, "bsp cubemap probe atlas (27 cubes)");
    // Synthesize `cluster_to_probe[]` — engine
    // `structure_bsp_build_instance_cubemap_references` partitions
    // scenario probes into cluster_cubemaps[] by point-in-BSP test
    // and runtime picks nearest. MCC stripped the partition so we
    // pre-pick the single nearest probe per cluster (collapses the
    // runtime's nearest-of-N candidates to nearest-of-all-scenario).
    let probe_count = scenario_cubemaps.len().min(probe_views.len());
    let cluster_to_probe: Vec<u16> = bsp
        .sbsp
        .clusters
        .iter()
        .map(|c| {
            let center = glam::Vec3::new(
                (c.bounds_x.lower + c.bounds_x.upper) * 0.5,
                (c.bounds_y.lower + c.bounds_y.upper) * 0.5,
                (c.bounds_z.lower + c.bounds_z.upper) * 0.5,
            );
            nearest_by(&scenario_cubemaps[..probe_count], center, |probe| {
                glam::Vec3::new(probe.position.x, probe.position.y, probe.position.z)
            })
            .map_or(0u16, |(i, _)| i as u16)
        })
        .collect();
    Some(crate::halo::structures::bsp_gpu::BspCubemapRouting {
        probe_views,
        cluster_to_probe,
    })
}

// ---------------------------------------------------------------------------
// Shared pipeline helper (used by pass modules)
// ---------------------------------------------------------------------------

/// Return the item nearest to `pos` by squared distance, with its index.
/// `key` extracts each item's world position. Ties keep the FIRST (lowest
/// index) item — strict-less-than replacement. `None` for an empty slice.
pub(super) fn nearest_by<T>(
    items: &[T],
    pos: glam::Vec3,
    key: impl Fn(&T) -> glam::Vec3,
) -> Option<(usize, &T)> {
    let mut best: Option<(usize, &T, f32)> = None;
    for (i, item) in items.iter().enumerate() {
        let d2 = (key(item) - pos).length_squared();
        match best {
            Some((_, _, bd)) if bd <= d2 => {}
            _ => best = Some((i, item, d2)),
        }
    }
    best.map(|(i, item, _)| (i, item))
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

pub fn create_1x1_texture(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4], label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    texture
}

pub fn create_screen_texture(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

pub fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn cube_tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32, sampler_type: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

pub fn uniform_entry(binding: u32, size: u64, visibility: wgpu::ShaderStages, has_dynamic_offset: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

/// Produce the runtime `simple_lights` source for the current scenario.
///
/// **Engine-correct source is `scenario.lights[]`** — `light` tag
/// placements that pass through `add_simple_light_callback @ 0x1806c82f0`
/// (dllcache) into `c_lights_view::add_simple_light_to_draw_list`. The
/// callback reads light parameters from the resolved `light` TAG, not
/// from any stli block.
///
/// `stli.generic_light_instances` is **NOT** the runtime simple_lights
/// source — it's an offline-bake input. tool.exe's `surface_light_create
/// @ 0x140239C90` walks the BSP collision mesh, looks up each triangle's
/// material in `stli.material_info[]`, and for every material with
/// `emissive_power > 0` flood-fills connected triangles into "surface
/// lights" baked directly into the lightmap atlas. At runtime the engine
/// reads that contribution from the atlas as part of normal per-pixel SH.
/// `generic_light_instances` (defs + placements) feeds a sibling offline
/// path in the same bake.
///
/// We previously sourced `simple_lights` from `generic_light_instances`,
/// which produced the wrong shape twice over: (a) added HDR point-source
/// contributions the engine never adds at runtime, and (b) didn't
/// reproduce the atlas-baked emissive surfaces engine relies on for
/// indoor ambient cast (Guardian's blue light cast, etc.). Returning
/// empty for now leaves the runtime path engine-faithful for maps with
/// no `scenario.lights[]` placements (Guardian, most MP). The atlas-bake
/// port is queued — see [`project_stli_surface_light_bake`].
///
/// TODO(stli): once `surface_light_create` is ported, this function
/// should consume `scenario.lights[]` via the resolved `light` tag and
/// produce ScenarioLight entries for the runtime cbuffer. The
/// `runtime_lights.rs` walker already loads those tags — wire them
/// through here instead of stli.
fn resolve_scenario_lights(
    _loaded: &crate::halo::scenario::LoadedScenario,
) -> Vec<ScenarioLight> {
    Vec::new()
}

