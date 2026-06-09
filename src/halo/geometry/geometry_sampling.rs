//! Engine source: `Ares/source/geometry/geometry_sampling.{h,cpp}`.
//!
//! `c_geometry_sampler::sample` — the engine's ray-into-lighting-data lookup
//! used by airprobe testing, object refresh-lighting, decorator bake, and
//! everything else that needs "at this world position, what's the SH probe?"
//!
//! ## This phase (Phase 7a) — API surface
//!
//! Structs (verbatim, all `static_assert(sizeof)` mirrored), enums, and the
//! one small SH helper that's already well-defined. Big function bodies are
//! stubbed with engine refs for Phase 7b.
//!
//! ## To port in Phase 7b
//!
//! | Function | LoC | Engine address |
//! |---|---|---|
//! | `reconstruct_quadratic_lightprobe_from_linear_and_intensity` | 115 | `dllcache 0x18048f900` |
//! | `calculate_dominant_light_from_lightprobe` | 190 | `dllcache 0x18048dec0` |
//! | `sample_lightprobe_textures` | 294 | `dllcache 0x18048fc60` |
//! | `geometry_test_triangle` | 396 | `dllcache 0x18048d290` |
//! | `geometry_test_collision_result` | 836 | `dllcache 0x18048c620` |
//! | `geometry_test_vector` | 153 | `dllcache 0x18048c300` |
//! | `c_geometry_sampler::sample` | 635 | `dllcache 0x18048a450` |
//!
//! All decompiles live at `/Users/camden/Source/protomorph/refs/engine_geometry_sampler/*.txt`.

use std::cell::RefCell;

use blam_tags::math::{RealMatrix4x3, RealPoint2d, RealPoint3d, RealRgbColor, RealVector3d};

thread_local! {
    /// Diag-only branch counters for the bake's lightprobe dispatch
    /// (per-vertex, per-pixel, single, fallback). Used by the renderer's
    /// per-set histogram in 2026-05-26 dark-fern investigation. Caller is
    /// responsible for taking a snapshot before + after a bake batch and
    /// differencing — counters are never reset internally.
    pub static DIAG_LIGHTPROBE_BRANCH_COUNTS: RefCell<[u64; 4]> = const { RefCell::new([0; 4]) };
}
use blam_tags::render_model::RenderVertex;
use blam_tags::scenario_lightmap::{DequantizedPerVertexProbe, LightmapPolicy};
use glam::Vec3;

use super::super::bitmaps::atlas_decode::DecodedAtlas;
use super::super::math::color_math::{
    real_rgb_lightprobe_from_half, HalfLinearRgbColor, HalfRgbLightprobe,
    HalfRgbLightprobeWithDominantLight,
};
use super::super::math::matrix_math::matrix4x3_inverse_transform_point;
use super::super::physics::collision_bsp::PROJECTION3D_MAPPINGS;
use super::super::physics::collision_constants::CollisionTestFlagsPair;
use super::super::physics::collisions::CollisionResult;
use super::super::render::lighting_interface as sh;
use crate::halo::bitmaps::bitmap_group_2d_get_real_pixel;
use crate::halo::scenario::loader::{LoadedBsp, LoadedScenario};

// =============================================================================
// e_geometry_sampler_outcome — `geometry_sampling.h:14`
// =============================================================================

/// Mirrors:
///
/// ```cpp
/// enum e_geometry_sampler_outcome {
///     _geometry_sampler_outcome_success = 0,
///     _geometry_sampler_outcome_no_lightmaps,
///     _geometry_sampler_outcome_no_hit,
///     _geometry_sampler_outcome_hit_but_failure,
///     _geometry_sampler_outcome_no_lightmap_for_bsp,
///     _geometry_sampler_outcome_no_cluster_or_instance,
/// };
/// ```
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeometrySamplerOutcome {
    /// `0` — atlas-or-per-vertex lookup succeeded; `sampled_result` is filled.
    /// The bake's `if (!v98)` test in `light_placement` ONLY accumulates on
    /// this outcome.
    #[default]
    Success = 0,
    /// `1` — no lightmaps at all on the scenario (uses
    /// `m_default_lightprobe_sample`).
    NoLightmaps = 1,
    /// `2` — collision_test_vector returned hit_nothing.
    NoHit = 2,
    /// `3` — hit happened but the surface had no usable lighting data.
    HitButFailure = 3,
    /// `4` — hit happened but the structure_bsp has no scenario_lightmap_bsp_data.
    NoLightmapForBsp = 4,
    /// `5` — hit happened but the surface isn't in any cluster/instance.
    NoClusterOrInstance = 5,
}

// =============================================================================
// e_geometry_test_vector_outcome — `geometry_sampling.h:23`
// =============================================================================

/// Mirrors:
///
/// ```cpp
/// enum e_geometry_test_vector_outcome {
///     _geometry_test_vector_outcome_success = 0,
///     _geometry_test_vector_outcome_hit_object_but_failed,
///     _geometry_test_vector_outcome_hit_bsp_but_failed,
///     _geometry_test_vector_outcome_hit_instance_but_failed,
///     _geometry_test_vector_outcome_hit_nothing,
/// };
/// ```
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeometryTestVectorOutcome {
    #[default]
    Success = 0,
    HitObjectButFailed = 1,
    HitBspButFailed = 2,
    HitInstanceButFailed = 3,
    HitNothing = 4,
}

// =============================================================================
// s_geometry_sample — `geometry_sampling.h:47` (sizeof=504)
// =============================================================================
//
// One per-sample lighting payload that `c_geometry_sampler::sample` fills in.
// Engine `light_placement` (the decorator bake) reads SH from
// `m_light_probe_{r,g,b}` and color/normal from `m_diffuse`/`m_normal` after
// each sample call.

/// Engine `s_geometry_sample` (504 B). Order matches Ares header byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GeometrySample {
    /// Engine `m_sample_point` @ 0x0 — the COLLISION POINT (where the ray
    /// hit), not the ray start. Set by `geometry_test_collision_result`.
    pub sample_point: RealPoint3d,
    /// Engine `m_light_probe_r[36]` @ 0xC. SH3 stores 9 coefs; engine pads
    /// to 36 floats for the `sh_scale_add` / `sh_eval_directional_light`
    /// scratch space the bake uses on top of the raw lightprobe.
    pub light_probe_r: [f32; 36],
    /// Engine `m_light_probe_g[36]` @ 0x9C.
    pub light_probe_g: [f32; 36],
    /// Engine `m_light_probe_b[36]` @ 0x12C.
    pub light_probe_b: [f32; 36],
    /// Engine `m_diffuse` @ 0x1BC — diffuse texture color at the hit point
    /// (set by `sample_diffuse_textures`). The decorator bake reads this as
    /// the per-sample "bright color" accumulator input.
    pub diffuse: RealVector3d,
    /// Engine `m_normal` @ 0x1C8 — surface normal at the hit point.
    pub normal: RealVector3d,
    /// Engine `m_dominant_light_dir` @ 0x1D4 — extracted from L1 SH bands
    /// by `find_dominant_light_direction`.
    pub dominant_light_dir: RealVector3d,
    /// Engine `m_dominant_light_intensity` @ 0x1E0.
    pub dominant_light_intensity: RealRgbColor,
    /// Engine `m_dominant_light_contrast` @ 0x1EC — ratio used by
    /// `reconstruct_quadratic_*` to synthesise the L2 bands.
    pub dominant_light_contrast: f32,
    /// Engine `m_needs_interpolation` @ 0x1F0 — set when caller should
    /// interpolate this sample across multiple frames (e.g., moving objects).
    pub needs_interpolation: bool,
    _pad: [u8; 3],
    /// Engine `m_chocalate_mountain_scale` @ 0x1F4 (yes, the engine field
    /// has that typo). Per-object boost factor for the chmt scale path.
    pub chocolate_mountain_scale: f32,
}

const _: () = assert!(std::mem::size_of::<GeometrySample>() == 504);

impl Default for GeometrySample {
    fn default() -> Self {
        Self {
            sample_point: RealPoint3d::default(),
            light_probe_r: [0.0; 36],
            light_probe_g: [0.0; 36],
            light_probe_b: [0.0; 36],
            diffuse: RealVector3d::default(),
            normal: RealVector3d::default(),
            dominant_light_dir: RealVector3d::default(),
            dominant_light_intensity: RealRgbColor::default(),
            dominant_light_contrast: 0.0,
            needs_interpolation: false,
            _pad: [0; 3],
            chocolate_mountain_scale: 1.0,
        }
    }
}

// =============================================================================
// s_geometry_test_result — `geometry_sampling.h:63` (sizeof=344)
// =============================================================================
//
// Intermediate populated by `geometry_test_collision_result`. Carries hit
// metadata AND the resolved lighting source (per-vertex/per-pixel/single
// lightprobe variants). `c_geometry_sampler::sample` reads from here to
// build the final `GeometrySample`.

/// Engine `real_rgb_lightprobe` (108 B) — SH3 per channel. Source struct
/// from `Ares/source/math/color_math.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RealRgbLightprobe {
    pub red_terms: [f32; 9],
    pub green_terms: [f32; 9],
    pub blue_terms: [f32; 9],
}
const _: () = assert!(std::mem::size_of::<RealRgbLightprobe>() == 108);

/// Engine `real_rgb_mini_lightprobe` (48 B) — SH2 per channel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RealRgbMiniLightprobe {
    pub red_terms: [f32; 4],
    pub green_terms: [f32; 4],
    pub blue_terms: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<RealRgbMiniLightprobe>() == 48);

// `real_point2d` (8 B) — pulled from `blam_tags::math` to match the same
// type used in `RenderVertex.lightmap_texcoord`, avoiding a per-call copy at
// the geometry_test_triangle boundary. Rust default layout for a 2-field
// struct of identical types is C-compatible.
const _: () = assert!(std::mem::size_of::<RealPoint2d>() == 8);

/// Engine `s_geometry_test_result` (344 B). Filled by
/// `geometry_test_collision_result`, consumed by `c_geometry_sampler::sample`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GeometryTestResult {
    /// Engine `m_collision_point` @ 0x0.
    pub collision_point: RealPoint3d,
    /// Engine `m_surface_point` @ 0xC.
    pub surface_point: RealPoint3d,
    /// Engine `m_surface_uv` @ 0x18 (texture UV in surface space).
    pub surface_uv: RealPoint2d,
    /// Engine `m_lightmap_uv` @ 0x20 (UV into the lightprobe atlas).
    pub lightmap_uv: RealPoint2d,
    /// Engine `m_surface_normal` @ 0x28.
    pub surface_normal: RealVector3d,
    /// Engine `m_structure_bsp_index` @ 0x34. `-1` for off-BSP hits.
    pub structure_bsp_index: i32,
    /// Engine `m_cluster_index` @ 0x38. `-1` when hit is on an instance.
    pub cluster_index: i32,
    /// Engine `m_instance_index` @ 0x3C. `-1` when hit is on a cluster.
    pub instance_index: i32,
    /// Engine `m_object_index` @ 0x40. `-1` for non-object hits.
    pub object_index: i32,
    /// Engine `m_has_per_vertex_lightprobe` @ 0x44. When `true`, read
    /// [`Self::interpolated_per_vertex_lightprobe`] for SH2.
    pub has_per_vertex_lightprobe: bool,
    /// Engine `m_has_per_pixel_lightmaps` @ 0x45. When `true`,
    /// `sample_lightprobe_textures` should be called with
    /// [`Self::lightmap_uv`].
    pub has_per_pixel_lightmaps: bool,
    /// Engine `m_has_single_lightprobe` @ 0x46. When `true`, read
    /// [`Self::single_sh_lightprobe`] (SH3) directly.
    pub has_single_lightprobe: bool,
    _pad0: u8,
    /// Engine `m_single_sh_lightprobe` @ 0x48 (108 B = SH3 RGB).
    pub single_sh_lightprobe: RealRgbLightprobe,
    /// Engine `m_interpolated_per_vertex_lightprobe` @ 0xB4 (48 B = SH2 RGB).
    pub interpolated_per_vertex_lightprobe: RealRgbMiniLightprobe,
    /// Engine `m_interpolated_dominant_light_intensity` @ 0xE4.
    pub interpolated_dominant_light_intensity: RealRgbColor,
    /// Engine `m_s` @ 0xF0 — barycentric u of hit on hit triangle.
    pub s: f32,
    /// Engine `m_t` @ 0xF4 — barycentric v of hit on hit triangle.
    pub t: f32,
    /// Engine `m_triangle_index` @ 0xF8.
    pub triangle_index: i32,
    /// Engine `m_triangle_vertices[3]` @ 0xFC (36 B).
    pub triangle_vertices: [RealPoint3d; 3],
    /// Engine `m_triangle_uvs[3]` @ 0x120 (24 B).
    pub triangle_uvs: [RealPoint2d; 3],
    /// Engine `m_triangle_lightmap_uvs[3]` @ 0x138 (24 B).
    pub triangle_lightmap_uvs: [RealPoint2d; 3],
    /// Engine `m_render_method_index` @ 0x150.
    pub render_method_index: i32,
    /// Engine `m_global_material_type` @ 0x154 — `c_global_material_type`
    /// (`i16 m_index` + 2 B pad).
    pub global_material_type: i32,
}

const _: () = assert!(std::mem::size_of::<GeometryTestResult>() == 344);

impl Default for GeometryTestResult {
    fn default() -> Self {
        Self {
            collision_point: RealPoint3d::default(),
            surface_point: RealPoint3d::default(),
            surface_uv: RealPoint2d::default(),
            lightmap_uv: RealPoint2d::default(),
            surface_normal: RealVector3d::default(),
            structure_bsp_index: -1,
            cluster_index: -1,
            instance_index: -1,
            object_index: -1,
            has_per_vertex_lightprobe: false,
            has_per_pixel_lightmaps: false,
            has_single_lightprobe: false,
            _pad0: 0,
            single_sh_lightprobe: RealRgbLightprobe::default(),
            interpolated_per_vertex_lightprobe: RealRgbMiniLightprobe::default(),
            interpolated_dominant_light_intensity: RealRgbColor::default(),
            s: 0.0,
            t: 0.0,
            triangle_index: -1,
            triangle_vertices: [RealPoint3d::default(); 3],
            triangle_uvs: [RealPoint2d::default(); 3],
            triangle_lightmap_uvs: [RealPoint2d::default(); 3],
            render_method_index: -1,
            global_material_type: -1,
        }
    }
}

// =============================================================================
// s_uncompressed_triangle — `geometry_sampling.h:91` (sizeof=108)
// =============================================================================

/// Engine `s_uncompressed_triangle` (108 B). Scratch struct used by
/// `geometry_test_triangle` to interpolate UV/normal at the hit point — the
/// engine reads vertex_buffer through `s_compression_info` then decompresses
/// into this temp triangle for the ray-vs-triangle math.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UncompressedTriangle {
    pub position: [RealPoint3d; 3],     // 0x0  (36)
    pub texcoord: [RealPoint2d; 3],     // 0x24 (24)
    pub normal: [RealVector3d; 3],      // 0x3C (36)
    pub indices: [i32; 3],              // 0x60 (12)
}
const _: () = assert!(std::mem::size_of::<UncompressedTriangle>() == 108);

// =============================================================================
// SH helpers — `math/spherical_harmonics.{h,cpp}` — delegate to existing port
// =============================================================================
//
// These four functions live in Ares's `math/spherical_harmonics.h` and are
// included by BOTH `c_lighting_interface` and `c_geometry_sampler`. Protomorph
// already has verbatim ports in `render/lighting_interface.rs`; we delegate to
// avoid duplication. The right long-term home is `src/halo/math/spherical_harmonics.rs`
// — both files would `use` it after that refactor.

#[inline]
fn rvec3_to_vec3(v: RealVector3d) -> Vec3 {
    Vec3::new(v.i, v.j, v.k)
}
#[inline]
fn vec3_to_rvec3(v: Vec3) -> RealVector3d {
    RealVector3d { i: v.x, j: v.y, k: v.z }
}

/// `find_dominant_light_direction` @ dllcache `0x18048fba0`.
/// Engine signature: writes `dominant_light_dir`, returns pre-normalize magnitude.
/// Body lives in [`super::super::render::lighting_interface::find_dominant_light_direction`].
pub fn find_dominant_light_direction(
    sr: &[f32; 9],
    sg: &[f32; 9],
    sb: &[f32; 9],
    dominant_light_dir: &mut RealVector3d,
) -> f32 {
    let dir = sh::find_dominant_light_direction(sr, sg, sb);
    *dominant_light_dir = vec3_to_rvec3(dir);
    // The existing port returns a normalized direction without exposing the
    // pre-normalize magnitude. Recompute it the same way the engine does:
    // engine: `magnitude = sqrt(x² + y² + z²)` where (x,y,z) is the
    // luma-weighted L1 vector BEFORE normalization. The existing port
    // discards this, so recompute inline.
    const LR: f32 = sh::LUMA_R;
    const LG: f32 = sh::LUMA_G;
    const LB: f32 = sh::LUMA_B;
    let x = -(sr[3] * LR + sg[3] * LG + sb[3] * LB);
    let y = -(sr[1] * LR + sg[1] * LG + sb[1] * LB);
    let z = sr[2] * LR + sg[2] * LG + sb[2] * LB;
    (x * x + y * y + z * z).sqrt()
}

/// `reconstruct_quadratic_lightprobe_from_linear_and_intensity` @ dllcache
/// `0x18048f900`. Synthesises the L2 (5-coef) SH bands from L1 + dominant
/// intensity and computes a contrast residual on the green channel.
/// Body lives in [`super::super::render::lighting_interface::reconstruct_quadratic_lightprobe_from_linear_and_intensity`].
pub fn reconstruct_quadratic_lightprobe_from_linear_and_intensity(
    sh_red: &mut [f32; 9],
    sh_green: &mut [f32; 9],
    sh_blue: &mut [f32; 9],
    dominant_light_intensity: RealRgbColor,
    dominant_light_dir: &mut RealVector3d,
    contrast: &mut f32,
) {
    let dom_intensity = Vec3::new(
        dominant_light_intensity.red,
        dominant_light_intensity.green,
        dominant_light_intensity.blue,
    );
    let mut dom_dir = rvec3_to_vec3(*dominant_light_dir);
    *contrast = sh::reconstruct_quadratic_lightprobe_from_linear_and_intensity(
        sh_red, sh_green, sh_blue, dom_intensity, &mut dom_dir,
    );
    *dominant_light_dir = vec3_to_rvec3(dom_dir);
}

/// `calculate_dominant_light_from_lightprobe` @ dllcache `0x18048dec0`.
/// Body lives in [`super::super::render::lighting_interface::calculate_dominant_light_from_lightprobe`].
pub fn calculate_dominant_light_from_lightprobe(
    probe_r: &[f32; 9],
    probe_g: &[f32; 9],
    probe_b: &[f32; 9],
    dir: &mut RealVector3d,
    intensity: &mut RealRgbColor,
    contrast: &mut f32,
) {
    let (d, i, c) = sh::calculate_dominant_light_from_lightprobe(probe_r, probe_g, probe_b);
    *dir = vec3_to_rvec3(d);
    intensity.red = i.x;
    intensity.green = i.y;
    intensity.blue = i.z;
    *contrast = c;
}

// =============================================================================
// sample_lightprobe_textures — `dllcache 0x18048fc60` (verbatim Phase 7b.3)
// =============================================================================

/// `sample_lightprobe_textures` @ dllcache `0x18048fc60`.
///
/// Atlas-samples the per-pixel lightprobe textures at the hit point's
/// interpolated UV, then derives the dominant light direction / intensity /
/// contrast and reconstructs the SH3 L2 bands from the L1+intensity.
///
/// Engine signature has 10 params with `bsp_lightmap_data` and `entity_data`
/// as opaque struct pointers. Protomorph adapts: caller pre-resolves the two
/// bitmap_group references into [`DecodedAtlas`] objects (cached at
/// scenario load) and passes them directly. `pixels_per_probe` is the
/// `bitmap_data[8]` byte that the engine reads to dispatch.
///
/// Engine sentinel: when either atlas is `None`, the function returns
/// without writing any output buffers.
#[allow(clippy::too_many_arguments)]
pub fn sample_lightprobe_textures(
    _structure_bsp_index: i32,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    uv: RealPoint2d,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
    compression_vectors: &[RealVector3d],
    sh_red: &mut [f32; 36],
    sh_green: &mut [f32; 36],
    sh_blue: &mut [f32; 36],
    dominant_light_dir: &mut RealVector3d,
    dominant_light_intensity: &mut RealRgbColor,
    contrast: &mut f32,
) {
    // Engine guard (line 61): both atlases must be present.
    let (Some(lightprobe), Some(intensity)) = (lightprobe_atlas, intensity_atlas) else {
        return;
    };

    // Engine line 63-66: memset SH buffers to zero (36 floats × 4 bytes = 0x90).
    sh_red.fill(0.0);
    sh_green.fill(0.0);
    sh_blue.fill(0.0);

    let bitmap_uv = RealPoint2d { x: uv.x, y: uv.y };

    // ── Per-pixel lightprobe atlas decode ────────────────────────────────
    // bitmap_data[8] byte = pixels per probe. Engine handles 4 (single-pixel
    // SH coefs) and 8 (paired-pixel SH coefs). Other values trigger
    // `handle_slim_assert(&info_11737)`; we silently no-op the SH write.
    match lightprobe_pixels_per_probe {
        8 => {
            // Paired-pixel encoding: 4 SH coefs total (indices 0..3), each
            // coef built from 2 adjacent atlas pixels.
            //
            // Engine inner loop:
            //   scale = pixel_a.alpha * pixel_b.alpha * compression_vectors[image_index].n[0]
            //   sh_red  [k] = (pixel_a.red   + pixel_b.red   - 1.0) * 2.0 * scale
            //   sh_green[k] = (pixel_a.green + pixel_b.green - 1.0) * 2.0 * scale
            //   sh_blue [k] = (pixel_a.blue  + pixel_b.blue  - 1.0) * 2.0 * scale
            //
            // compression_vectors is indexed by image_index (NOT k) — engine
            // does `cv += 2; cv[-2]` per iteration which matches `cv[2*k]`.
            for k in 0..4 {
                let img_idx = (k * 2) as i32;
                let pixel_a = bitmap_group_2d_get_real_pixel(
                    lightprobe, img_idx, &bitmap_uv, 0, true,
                );
                let pixel_b = bitmap_group_2d_get_real_pixel(
                    lightprobe, img_idx + 1, &bitmap_uv, 0, true,
                );
                let scale = pixel_a.alpha * pixel_b.alpha * compression_vectors[(k * 2) as usize].i;
                sh_red[k] = (pixel_a.red + pixel_b.red - 1.0) * 2.0 * scale;
                sh_green[k] = (pixel_a.green + pixel_b.green - 1.0) * 2.0 * scale;
                sh_blue[k] = (pixel_a.blue + pixel_b.blue - 1.0) * 2.0 * scale;
            }
        }
        4 => {
            // Single-pixel encoding: 4 SH coefs total, each read from one
            // atlas pixel. Channel reorder: alpha→sh_red, blue→sh_green,
            // green→sh_blue. The pixel's red channel is unused.
            for k in 0..4 {
                let pixel = bitmap_group_2d_get_real_pixel(
                    lightprobe, k as i32, &bitmap_uv, 0, true,
                );
                sh_red[k] = pixel.alpha;
                sh_green[k] = pixel.blue;
                sh_blue[k] = pixel.green;
            }
        }
        _ => {
            // Engine: handle_slim_assert(&info_11737). Skip the SH writes.
        }
    }

    // ── Dominant light intensity decode ──────────────────────────────────
    // Engine reads from the dominant_intensity bitmap; uses compression_vectors[8]
    // as the universal scale for the 2-pixel path. 1-pixel path skips the
    // compression altogether and just reads the pixel directly.
    match dominant_pixels_per_probe {
        2 => {
            // Same paired-pixel scheme as the lightprobe atlas above, but
            // produces a single RGB intensity triplet (not an SH coef).
            let pixel_a = bitmap_group_2d_get_real_pixel(
                intensity, 0, &bitmap_uv, 0, true,
            );
            let pixel_b = bitmap_group_2d_get_real_pixel(
                intensity, 1, &bitmap_uv, 0, true,
            );
            let scale = pixel_a.alpha * pixel_b.alpha * compression_vectors[8].i;
            dominant_light_intensity.red =
                (pixel_a.red + pixel_b.red - 1.0) * 2.0 * scale;
            dominant_light_intensity.green =
                (pixel_a.green + pixel_b.green - 1.0) * 2.0 * scale;
            dominant_light_intensity.blue =
                (pixel_a.blue + pixel_b.blue - 1.0) * 2.0 * scale;
        }
        1 => {
            // Single-pixel encoding: alpha→red, blue→green, green→blue.
            let pixel = bitmap_group_2d_get_real_pixel(
                intensity, 0, &bitmap_uv, 0, true,
            );
            dominant_light_intensity.red = pixel.alpha;
            dominant_light_intensity.green = pixel.blue;
            dominant_light_intensity.blue = pixel.green;
        }
        _ => {
            // Engine: handle_slim_assert(&info_11738). Skip.
        }
    }

    // ── Derive dominant direction + reconstruct SH3 L2 bands ─────────────
    // Engine lines 284-291: extract dominant direction from the L1 we just
    // decoded, then synthesise the L2 (5-coef) bands from L1 + intensity.
    let sh_red_9: &mut [f32; 9] = (&mut sh_red[..9]).try_into().unwrap();
    let sh_green_9: &mut [f32; 9] = (&mut sh_green[..9]).try_into().unwrap();
    let sh_blue_9: &mut [f32; 9] = (&mut sh_blue[..9]).try_into().unwrap();
    find_dominant_light_direction(sh_red_9, sh_green_9, sh_blue_9, dominant_light_dir);
    reconstruct_quadratic_lightprobe_from_linear_and_intensity(
        sh_red_9,
        sh_green_9,
        sh_blue_9,
        *dominant_light_intensity,
        dominant_light_dir,
        contrast,
    );
}

// =============================================================================
// geometry_test_triangle — `dllcache 0x18048d290` (verbatim Phase 7b.2)
// =============================================================================

/// `c_geometry_sampler::geometry_test_triangle` @ dllcache `0x18048d290`.
///
/// Engine-private helper invoked from `geometry_test_collision_result` after
/// a candidate render triangle is fetched. Interpolates UV/normal/per-vertex-
/// SH at the hit barycentric coordinates and fills `test_result`'s
/// triangle/UV/probe fields.
///
/// Engine signature has 14 params (incl. `collision_point`,
/// `debug_sampling_transform`, `structure_bsp_index` — all unused in this
/// function's body but kept for caller-side consistency). Engine body never
/// reads `collision_point` or `debug_sampling_transform`.
///
/// `lightmap_uv_data` — flat `u16[2 * vertex_count]` of vertex lightmap UVs
/// packed as unorm16. None = mesh has no lightmap UVs.
///
/// `per_vertex_lightprobes` — flat `u32[per_vertex_count * pixels_per_vertex]`
/// where `pixels_per_vertex = per_vertex_stride / 4`. Each pixel is signed-
/// RGBE encoded `real_rgb_color`. The 5 pixels per vertex encode:
/// pixel 0 = dominant_light_intensity, pixels 1..4 = SH2 terms `[0..3]`.
///
/// `single_probes` — slice of `HalfRgbLightprobeWithDominantLight` (each
/// entry 72 B); `lightmap_probe_index` is the i32 index into this slice
/// (`-1` means no single probe). Engine resolves via TagBlock pointer chase
/// on `bsp_lightmap_data.single_probes`.
#[allow(clippy::too_many_arguments)]
pub fn geometry_test_triangle(
    _collision_point: RealPoint3d,
    triangle: &UncompressedTriangle,
    bs: f32,
    bt: f32,
    triangle_index: i32,
    lightmap_uv_data: Option<&[RealPoint2d]>,
    per_vertex_lightprobes: Option<&[DequantizedPerVertexProbe]>,
    lightmap_probe_index: i32,
    _debug_sampling_transform: Option<&RealMatrix4x3>,
    test_result: &mut GeometryTestResult,
    _structure_bsp_index: i32,
    single_probes: Option<&[HalfRgbLightprobeWithDominantLight]>,
) {
    // Engine lines 165-167: triangle_index + zero lightmap_uv.
    test_result.triangle_index = triangle_index;
    test_result.lightmap_uv = RealPoint2d::default();

    // ── lightmap UV decode + interpolate (engine lines 168-189) ──────────
    if let Some(uv_data) = lightmap_uv_data {
        // Engine: 6 unorm16 reads (3 vertices × u + v) via
        // uncompress_word_to_real_normalized. Protomorph stores RenderVertex
        // lightmap_texcoord as RealPoint2d f32 directly (the lightmap-tag
        // walker has already done the decode at scenario load), so the
        // caller passes a slice of RealPoint2d and we read it as-is.
        for corner in 0..3 {
            let i = triangle.indices[corner] as usize;
            test_result.triangle_lightmap_uvs[corner] = uv_data[i];
        }
        let v0u = test_result.triangle_lightmap_uvs[0].x;
        let v1u = test_result.triangle_lightmap_uvs[1].x;
        let v2u = test_result.triangle_lightmap_uvs[2].x;
        let v0v = test_result.triangle_lightmap_uvs[0].y;
        let v1v = test_result.triangle_lightmap_uvs[1].y;
        let v2v = test_result.triangle_lightmap_uvs[2].y;
        test_result.lightmap_uv.x = (v1u - v0u) * bs + v0u + (v2u - v0u) * bt;
        test_result.lightmap_uv.y = (v1v - v0v) * bs + v0v + (v2v - v0v) * bt;
    }

    // ── surface_point: barycentric interp of triangle.position (190-201) ─
    let p0 = triangle.position[0];
    let p1 = triangle.position[1];
    let p2 = triangle.position[2];
    test_result.surface_point.x = (p1.x - p0.x) * bs + p0.x + (p2.x - p0.x) * bt;
    test_result.surface_point.y = (p1.y - p0.y) * bs + p0.y + (p2.y - p0.y) * bt;
    test_result.surface_point.z = (p1.z - p0.z) * bs + p0.z + (p2.z - p0.z) * bt;

    // ── surface_uv: barycentric interp of triangle.texcoord (202-209) ────
    let t0 = triangle.texcoord[0];
    let t1 = triangle.texcoord[1];
    let t2 = triangle.texcoord[2];
    test_result.surface_uv.x = (t1.x - t0.x) * bs + t0.x + (t2.x - t0.x) * bt;
    test_result.surface_uv.y = (t1.y - t0.y) * bs + t0.y + (t2.y - t0.y) * bt;

    // ── surface_normal: barycentric interp of triangle.normal (210-221) ──
    let n0 = triangle.normal[0];
    let n1 = triangle.normal[1];
    let n2 = triangle.normal[2];
    test_result.surface_normal.i = (n1.i - n0.i) * bs + n0.i + (n2.i - n0.i) * bt;
    test_result.surface_normal.j = (n1.j - n0.j) * bs + n0.j + (n2.j - n0.j) * bt;
    test_result.surface_normal.k = (n1.k - n0.k) * bs + n0.k + (n2.k - n0.k) * bt;

    // ── per-vertex SH probe: barycentric interp (engine lines 222-364) ───
    //
    // Engine pulls 3 vertices × 5 RGBE pixels per vertex out of a runtime
    // per-vertex SH stream, decodes via `pixel32_rgbe_to_real_rgb_color`, and
    // interpolates. Protomorph adapts: the caller passes a slice indexed
    // by vertex (i.e., `per_vert[triangle.indices[k]]`) of pre-dequantized
    // `DequantizedPerVertexProbe`s — same logical data, no per-call RGBE
    // round-trip since `LightmapPerVertexProbe.dequantize()` already exists.
    if let Some(per_vert) = per_vertex_lightprobes {
        let v0_idx = triangle.indices[0] as usize;
        let v1_idx = triangle.indices[1] as usize;
        let v2_idx = triangle.indices[2] as usize;
        let v0 = per_vert[v0_idx];
        let v1 = per_vert[v1_idx];
        let v2 = per_vert[v2_idx];

        let lerp = |a: f32, b: f32, c: f32| -> f32 { (b - a) * bs + a + (c - a) * bt };

        test_result.interpolated_dominant_light_intensity.red = lerp(
            v0.dominant_light_intensity[0],
            v1.dominant_light_intensity[0],
            v2.dominant_light_intensity[0],
        );
        test_result.interpolated_dominant_light_intensity.green = lerp(
            v0.dominant_light_intensity[1],
            v1.dominant_light_intensity[1],
            v2.dominant_light_intensity[1],
        );
        test_result.interpolated_dominant_light_intensity.blue = lerp(
            v0.dominant_light_intensity[2],
            v1.dominant_light_intensity[2],
            v2.dominant_light_intensity[2],
        );

        for k in 0..4 {
            test_result.interpolated_per_vertex_lightprobe.red_terms[k] =
                lerp(v0.red_terms[k], v1.red_terms[k], v2.red_terms[k]);
            test_result.interpolated_per_vertex_lightprobe.green_terms[k] =
                lerp(v0.green_terms[k], v1.green_terms[k], v2.green_terms[k]);
            test_result.interpolated_per_vertex_lightprobe.blue_terms[k] =
                lerp(v0.blue_terms[k], v1.blue_terms[k], v2.blue_terms[k]);
        }
    }

    // ── single SH lightprobe: half decode (engine lines 365-386) ─────────
    if let Some(probes) = single_probes {
        if lightmap_probe_index != -1 {
            let idx = lightmap_probe_index as usize;
            if idx < probes.len() {
                // Engine: `v20 = ptr_to_first_single_probe + 72 * idx`, then
                // `real_rgb_lightprobe_from_half((const half_rgb_lightprobe *)(v20 + 16), ...)`.
                // Offset +16 lands on `.quadratic_probe`.
                real_rgb_lightprobe_from_half(
                    &probes[idx].quadratic_probe,
                    &mut test_result.single_sh_lightprobe,
                );
            }
        }
    }

    // ── triangle vertices/UVs + bs/bt copies (engine lines 387-395) ──────
    test_result.triangle_vertices = triangle.position;
    test_result.triangle_uvs = triangle.texcoord;
    test_result.s = bs;
    test_result.t = bt;
}

// =============================================================================
// geometry_sampling_point_in_triangle3d — `dllcache 0x18048f480` (Phase 7b.4a)
// =============================================================================

/// `geometry_sampling_point_in_triangle3d` @ dllcache `0x18048f480`.
///
/// Tests whether `point` lies inside triangle `(p0, p1, p2)`, computes the
/// barycentric coordinates `(s, t)` if it does, and returns the
/// "edge-distance" metric the caller uses to pick the best hit when
/// multiple candidate triangles match.
///
/// Return value:
/// - `> -eps` → point is inside (or near edge) and `(s, t)` are valid
/// - `≤ -eps` → point is outside; the magnitude is how far past the worst
///   edge the point lies. `(s, t)` are zeroed in this case.
/// - Caller picks the MAXIMUM return across candidates as the best hit.
///
/// Algorithm (engine-verbatim):
/// 1. Edge vectors: `e0 = p1 - p0`, `e1 = p2 - p0`, `v = point - p0`
/// 2. Triangle normal `n = e0 × e1`
/// 3. Project to the 2D plane perpendicular to the dominant axis of `|n|`
/// 4. Compute three signed-area metrics (`v36`, `v40`, `v41 - v40 - v36`);
///    return the MIN. If any is < `-eps`, point is outside.
/// 5. Barycentric: `s = v40 / v41`, `t = v36 / v41`.
pub fn geometry_sampling_point_in_triangle3d(
    point: RealPoint3d,
    p0: RealPoint3d,
    p1: RealPoint3d,
    p2: RealPoint3d,
    s: &mut f32,
    t: &mut f32,
) -> f32 {
    // Engine line 73-95: edge vectors.
    let v_dx = point.x - p0.x;
    let v_dy = point.y - p0.y;
    let v_dz = point.z - p0.z;
    let e0x = p1.x - p0.x;
    let e0y = p1.y - p0.y;
    let e0z = p1.z - p0.z;
    let e1x = p2.x - p0.x;
    let e1y = p2.y - p0.y;
    let e1z = p2.z - p0.z;
    let v_3d = [v_dx, v_dy, v_dz];
    let e0 = [e0x, e0y, e0z];
    let e1 = [e1x, e1y, e1z];

    // Engine line 96-99: triangle normal `n = e0 × e1` (note engine writes
    // them as `e1.* * e0.*` reorderings that are algebraically identical).
    let n0 = e1z * e0y - e1y * e0z;
    let n1 = e1x * e0z - e1z * e0x;
    let n2_val = e1y * e0x - e1x * e0y;
    let nx_abs = n0.abs();
    let ny_abs = n1.abs();
    let nz_abs = n2_val.abs();

    // Engine line 100-109: pick the dominant axis. n2 is the axis index
    // (0=X, 1=Y, 2=Z) of the largest |normal component|.
    let dominant_axis: usize = if nz_abs < ny_abs || nz_abs < nx_abs {
        if ny_abs >= nx_abs { 1 } else { 0 }
    } else {
        2
    };

    // Engine line 110-114: lookup the 2D-projection axes from the global
    // table. `v31 = 2*dominant_axis + sign_of(n[dominant_axis])`. The first
    // table dimension is `[0]` in the engine — both flat (`[v31]`) and
    // nested (`[axis][sign]`) indexing yield the same byte.
    let n_dom = [n0, n1, n2_val][dominant_axis];
    let sign_bit = if n_dom > 0.0 { 1 } else { 0 };
    let axes = PROJECTION3D_MAPPINGS[dominant_axis][sign_bit];
    let ax = axes[0] as usize;
    let ay = axes[1] as usize;

    // Engine line 115-119: 2D edge components after projection.
    let v32 = e0[ax];
    let v33 = e0[ay];
    let v34 = v_3d[ax];
    let v35 = v_3d[ay];
    // Signed area of (v_3d, e0) in projected 2D.
    let v36 = v35 * v32 - v34 * v33;
    let mut v37 = v36;
    const EDGE_EPS: f32 = -9.999_999_7e-5;
    if v36 <= EDGE_EPS {
        *s = 0.0;
        *t = 0.0;
        return v37;
    }
    let v38 = e1[ax];
    let v39 = e1[ay];
    // Signed area of (e1, v_3d) in projected 2D.
    let v40 = v39 * v34 - v38 * v35;
    if v36 > v40 {
        v37 = v40;
    }
    if v40 <= EDGE_EPS {
        *s = 0.0;
        *t = 0.0;
        return v37;
    }
    // Total triangle area (signed, in projected 2D).
    let v41 = v39 * v32 - v38 * v33;
    if v37 > v41 - (v40 + v36) {
        v37 = v41 - (v40 + v36);
    }
    // Reject near-degenerate triangles AND third-edge-outside cases.
    if (v41 >= -9.999_999_7e-5 && v41 <= 9.999_999_7e-5)
        || (v40 + v36) > (v41 + 9.999_999_7e-5)
    {
        *s = 0.0;
        *t = 0.0;
        return v37;
    }
    let inv_area = 1.0 / v41;
    *s = inv_area * v40;
    *t = inv_area * v36;
    v37
}

// =============================================================================
// get_structure_triangle — `dllcache 0x18048eee0` (Phase 7b.4b)
// =============================================================================

/// `get_structure_triangle` @ dllcache `0x18048eee0`.
///
/// Reads the 3 vertex indices at `indices[last_index-2 ..= last_index]`,
/// then copies position/texcoord/normal from each vertex into the output
/// triangle. Engine has separate PC and Xbox 360 paths — PC uses a 44-byte
/// rasterizer_vertex_world layout with packed s16x4 normal (decoded via
/// DirectX SIMD constants), Xbox 360 uses a 32-byte DHenN3-packed normal.
///
/// Protomorph adapts: takes `&[RenderVertex]` directly (the bake-time
/// representation already has full-precision f32 normal/texcoord). Engine's
/// runtime s16 quantization is purely a GPU-storage optimization; using
/// pre-decoded floats matches the engine's pre-quantization values.
pub fn get_structure_triangle(
    vertices: &[RenderVertex],
    last_index: u32,
    indices: &[u32],
    triangle: &mut UncompressedTriangle,
) {
    // Engine line 24-26: read 3 indices ending at last_index.
    let i0 = indices[(last_index as usize).wrapping_sub(2)] as usize;
    let i1 = indices[(last_index as usize).wrapping_sub(1)] as usize;
    let i2 = indices[last_index as usize] as usize;

    triangle.indices[0] = i0 as i32;
    triangle.indices[1] = i1 as i32;
    triangle.indices[2] = i2 as i32;

    for (k, &vert_idx) in [i0, i1, i2].iter().enumerate() {
        let v = &vertices[vert_idx];
        triangle.position[k] = v.position;
        triangle.texcoord[k] = v.texcoord;
        triangle.normal[k] = v.normal;
    }
}

// =============================================================================
// sample_diffuse_textures — `dllcache 0x180490220`
// =============================================================================

/// `sample_diffuse_textures` @ dllcache `0x180490220`. Resolves the
/// hit-surface's authored postprocess albedo for the bounce-light step
/// of [`sample`].
///
/// Engine flow:
///   1. Early-out if `test_result.m_render_method_index == -1`.
///   2. Resolve `c_render_method` via
///      `structure_bsp_get(bsp_idx)->materials[render_method_idx]` → rmsh
///      tag → `c_render_method_definition.m_vertex_types[0]` →
///      `s_render_method_postprocess_definition`. Protomorph keeps this
///      via the bake-time thread-local [`BakeMaterialLookup`].
///   3. Try the multi-tier blend-map path:
///        `sample_texture(_query_blend_map, surface_uv, mipmap_index=3, &blend)`
///      If it succeeds, walk channels `[blend.red, blend.green, blend.blue,
///      blend.alpha]` against `_query_albedo_base_map_{0,1,2,3}` and
///      weighted-average the resulting RGB.
///   4. Else: single-base fallback —
///        `sample_texture(_query_albedo_base_map_0, surface_uv, mipmap_index=0, &color)`
///      then if `_query_albedo_color` is bound, scale `color.rgb *
///      albedo_scale.rgb`.
///
/// `m_diffuse` left at `(0, 0, 0)` when no path resolves — same as the
/// engine, and downstream code gates the bounce-light SH addition on
/// `any(m_diffuse > 0)` to avoid contributing zero-weight noise.
fn sample_diffuse_textures_inner(
    test_result: &GeometryTestResult,
    diffuse_out: &mut RealVector3d,
) {
    use crate::halo::decorators::bake_material_lookup::with_material;
    use crate::halo::render_methods::postprocess_sampler::{
        query_real_constant, sample_texture, RuntimeQueryableProperty,
    };

    *diffuse_out = RealVector3d::default();

    thread_local! {
        static SDT_DIAG: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let diag = std::env::var_os("PROTOMORPH_DIAG_BAKE_HIST").is_some()
        && SDT_DIAG.with(|c| { let v = c.get(); c.set(v + 1); v }) < 20;

    if test_result.render_method_index == -1 {
        if diag {
            eprintln!(
                "[sdt-diag] render_method_index = -1 (bsp={}, uv=({:.3},{:.3}))",
                test_result.structure_bsp_index,
                test_result.surface_uv.x,
                test_result.surface_uv.y,
            );
        }
        return;
    }
    let bsp_idx = test_result.structure_bsp_index;
    let mat_idx = test_result.render_method_index;
    let uv = [test_result.surface_uv.x, test_result.surface_uv.y];

    with_material(bsp_idx, mat_idx, |entry_opt| {
        let Some(entry) = entry_opt else {
            if diag {
                eprintln!(
                    "[sdt-diag] no entry for (bsp={bsp_idx}, mat={mat_idx})"
                );
            }
            return;
        };
        if diag {
            eprintln!(
                "[sdt-diag] entry (bsp={bsp_idx}, mat={mat_idx}) queryable={:?} textures={}",
                entry.queryable_properties,
                entry.textures.iter().filter(|t| t.texture.is_some()).count(),
            );
        }

        // Engine line 76-143: blend-map multi-tier path.
        if let Some(blend) =
            sample_texture(entry, RuntimeQueryableProperty::BlendMap, uv, 3)
        {
            let pairs = [
                (blend.red, RuntimeQueryableProperty::AlbedoBaseMap0),
                (blend.green, RuntimeQueryableProperty::AlbedoBaseMap1),
                (blend.blue, RuntimeQueryableProperty::AlbedoBaseMap2),
                (blend.alpha, RuntimeQueryableProperty::AlbedoBaseMap3),
            ];
            let mut total_w = 0.0_f32;
            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            for (weight, prop) in pairs {
                if weight > 0.0 {
                    if let Some(color) = sample_texture(entry, prop, uv, 0) {
                        total_w += weight;
                        sum_r += weight * color.red;
                        sum_g += weight * color.green;
                        sum_b += weight * color.blue;
                    }
                }
            }
            // Engine line 138-139: `if (v15 <= 0.000099999997) v15 = ...`
            // — guards against divide-by-zero when no channel matched.
            let inv_w = 1.0 / total_w.max(9.999_999_7e-5);
            diffuse_out.i = sum_r * inv_w;
            diffuse_out.j = sum_g * inv_w;
            diffuse_out.k = sum_b * inv_w;
        } else {
            // Engine line 144-169: single-base fallback.
            let color = sample_texture(entry, RuntimeQueryableProperty::AlbedoBaseMap0, uv, 0)
                .unwrap_or(blam_tags::math::RealArgbColor::default());
            if let Some(albedo_scale) =
                query_real_constant(entry, RuntimeQueryableProperty::AlbedoColor)
            {
                diffuse_out.i = color.red * albedo_scale[0];
                diffuse_out.j = color.green * albedo_scale[1];
                diffuse_out.k = color.blue * albedo_scale[2];
            } else {
                diffuse_out.i = color.red;
                diffuse_out.j = color.green;
                diffuse_out.k = color.blue;
            }
        }
    });
}

// =============================================================================
// find_render_method_index_from_collision — `dllcache 0x18048daf0`
// =============================================================================

/// `c_geometry_sampler::find_render_method_index_from_collision` @ dllcache
/// `0x18048daf0`. Resolves a collision hit to a material slot index on the
/// hit BSP — the value that downstream consumers (`sample_diffuse_textures`)
/// use to walk `structure_bsp.materials[...]` and obtain the postprocess
/// definition for surface-shaded sampling.
///
/// Returns `-1` on:
///   - object hits (callers force `collision_object_index = -1` at this
///     entry — the object branch of the engine is unreachable from
///     `geometry_test_collision_result`'s only caller).
///   - out-of-range surface / instance / cluster / mesh indices.
///   - no mesh part contains the mapping's `triangle_index`.
///
/// Engine logic:
///   - Cluster branch (`instance_index == -1`):
///       for each (offset in 0..surface.mapping_count):
///         mapping = structure_surface_to_triangle_mappings[surface.first_mapping_index + offset]
///         cluster = clusters[mapping.section_index]
///         mesh    = meshes_metadata[cluster.mesh_index]
///         for each part in mesh.parts:
///           if part.index_start <= mapping.triangle_index
///                              < part.index_start + part.index_count:
///             return part.render_method_index   // first match wins
///   - Instance branch (`instance_index != -1`):
///       definition = instance_definitions[instance.definition_index]
///       mesh       = meshes_metadata[definition.mesh_index]
///       surface    = definition.structure_surfaces[surface_index]
///       (same iterate-mappings × scan-parts logic as cluster branch).
///
/// Engine's per-loop continue is `while ( flags == -1 )` — i.e. "keep
/// walking until we resolve a non-(-1) index." First match wins; ties are
/// not possible in practice since each `triangle_index` lives in exactly
/// one mesh part.
fn find_render_method_index_from_collision(
    sbsp: &blam_tags::structure_bsp::StructureBsp,
    collision_instanced_geometry_instance_index: i32,
    collision_surface_index: i32,
) -> i32 {
    if collision_instanced_geometry_instance_index == -1 {
        // Cluster branch (engine lines 38-104 of the dllcache decompile).
        let surfaces = &sbsp.structure_surfaces;
        if collision_surface_index < 0
            || (collision_surface_index as usize) >= surfaces.len()
        {
            return -1;
        }
        let surface = &surfaces[collision_surface_index as usize];
        let mappings = &sbsp.structure_surface_to_triangle_mappings;
        let clusters = &sbsp.clusters;
        let meshes = &sbsp.meshes_metadata;
        for offset in 0..surface.mapping_count {
            let mapping_idx = surface.first_mapping_index as usize + offset as usize;
            if mapping_idx >= mappings.len() {
                continue;
            }
            let mapping = &mappings[mapping_idx];
            let cluster_idx = mapping.section_index as usize;
            if cluster_idx >= clusters.len() {
                continue;
            }
            let mesh_idx = clusters[cluster_idx].mesh_index;
            if mesh_idx < 0 || (mesh_idx as usize) >= meshes.len() {
                continue;
            }
            let mesh = &meshes[mesh_idx as usize];
            let tri = mapping.triangle_index;
            for part in &mesh.parts {
                let end = part.index_start.saturating_add(part.index_count);
                if tri >= part.index_start && tri < end {
                    return part.render_method_index as i32;
                }
            }
        }
        -1
    } else {
        // Instance branch (engine lines 105-... of the dllcache decompile).
        let instances = &sbsp.instanced_geometry_instances;
        let inst_idx = collision_instanced_geometry_instance_index;
        if inst_idx < 0 || (inst_idx as usize) >= instances.len() {
            return -1;
        }
        let instance = &instances[inst_idx as usize];
        let def_idx = instance.definition_index;
        if def_idx < 0 || (def_idx as usize) >= sbsp.instance_definitions.len() {
            return -1;
        }
        let definition = &sbsp.instance_definitions[def_idx as usize];
        let mesh_idx = definition.mesh_index;
        if mesh_idx < 0 || (mesh_idx as usize) >= sbsp.meshes_metadata.len() {
            return -1;
        }
        let mesh = &sbsp.meshes_metadata[mesh_idx as usize];
        let surfaces = &definition.structure_surfaces;
        if collision_surface_index < 0
            || (collision_surface_index as usize) >= surfaces.len()
        {
            return -1;
        }
        let surface = &surfaces[collision_surface_index as usize];
        let mappings = &definition.structure_surface_to_triangle_mappings;
        for offset in 0..surface.mapping_count {
            let mapping_idx = surface.first_mapping_index as usize + offset as usize;
            if mapping_idx >= mappings.len() {
                continue;
            }
            let mapping = &mappings[mapping_idx];
            let tri = mapping.triangle_index;
            for part in &mesh.parts {
                let end = part.index_start.saturating_add(part.index_count);
                if tri >= part.index_start && tri < end {
                    return part.render_method_index as i32;
                }
            }
        }
        -1
    }
}

// =============================================================================
// geometry_test_collision_result — `dllcache 0x18048c620` (Phase 7b.4a/b)
// =============================================================================

/// `c_geometry_sampler::geometry_test_collision_result` @ dllcache `0x18048c620`.
///
/// Maps the closest-hit collision triple
/// `(object_index, instance_index, surface_index)` (from
/// `collision_test_vector`) into the lighting `GeometryTestResult` —
/// resolving the underlying render triangle + per-vertex/per-pixel/single
/// SH lighting source.
///
/// Three branches (engine lines 158-167 / 195-567 / 568-835):
/// 1. **Object hit** (`collision_object_index != -1`) — sets `early_out`,
///    delegates to object lighting cache. Decorator bake never takes this
///    path. **Ported in 7b.4a.**
/// 2. **Instanced geometry hit** (`collision_instanced_geometry_instance_index != -1`)
///    — walks instance mesh's structure_surfaces, calls geometry_test_triangle.
///    Has lightmap-type dispatch (per_vertex / single_probe / per_pixel).
///    **TODO Phase 7b.4c.**
/// 3. **Cluster hit** (the fallthrough) — walks structure_bsp.structure_surfaces,
///    calls geometry_test_triangle. **TODO Phase 7b.4b.**
///
/// `early_out` is an engine out-pointer mirroring "we handled this case,
/// upstream should stop searching".
///
/// **Signature adaptation:** engine takes opaque struct pointers throughout
/// (`s_scenario_lightmap_bsp_data`, `s_render_geometry`, etc.); protomorph
/// will pass pre-resolved data when 7b.4b/7b.4c land.
#[allow(clippy::too_many_arguments)]
pub fn geometry_test_collision_result(
    scenario: &LoadedScenario,
    structure_bsp_index: i32,
    collision_object_index: i32,
    collision_instanced_geometry_instance_index: i32,
    collision_surface_index: i32,
    collision_point: RealPoint3d,
    _ignore_lightmap_samples: bool,
    get_render_method_index: bool,
    early_out: &mut bool,
    test_result: &mut GeometryTestResult,
) -> bool {
    *early_out = false;

    // ── Branch 1: Object hit ────────────────────────────────────────────
    // Engine `object_has_valid_lighting` / `object_has_volume_samples` query
    // the object_render_state cache. Protomorph doesn't yet plumb object
    // collision into the bake (decorator/airprobe rays don't hit objects),
    // so we acknowledge the branch but report "no usable object lighting"
    // until 7b.4-followup wires the object lighting cache.
    if collision_object_index != -1 {
        // TODO 7b.4-followup: port object_has_valid_lighting (dllcache 0x180490cd0)
        // + object_has_volume_samples (0x180490d80). Both query
        // cached_object_render_states[object_get_cached_render_state(object_get_ultimate_parent(idx))].
        // For now: object branch is a no-op. Returns false (engine v9=0).
        return false;
    }

    // ── Common guard: no BSP → nothing to test ──────────────────────────
    if structure_bsp_index == -1 {
        return false;
    }

    // ── Resolve loaded_bsp + render_geometry ────────────────────────────
    // Engine fetches via `c_structure_renderer::get_lightmap_bsp_data` /
    // `get_render_geometry` from a global table keyed by structure_bsp_index.
    // Protomorph keeps these on `LoadedBsp.sbsp` / `LoadedBsp.meshes`, indexed
    // by `scenario_bsp_index`.
    let loaded_bsp = match scenario
        .active_bsps
        .iter()
        .find(|b| b.scenario_bsp_index == structure_bsp_index as usize)
    {
        Some(b) => b,
        None => return false,
    };
    let sbsp = &loaded_bsp.sbsp;

    // Engine line 177-192: populate `render_method_index` BEFORE branching
    // on instance vs cluster. The engine's call site forces
    // `collision_object_index = -1` here (the object branch returned
    // already at line 932); the helper itself only handles cluster +
    // instance hits. The structure_bsp_index store also happens at the
    // engine's top-level, mirrored here for parity with the per-branch
    // assignments below (which remain for now — minor redundancy).
    test_result.render_method_index = if get_render_method_index {
        find_render_method_index_from_collision(
            sbsp,
            collision_instanced_geometry_instance_index,
            collision_surface_index,
        )
    } else {
        -1
    };
    test_result.structure_bsp_index = structure_bsp_index;

    // ── Branch 2: Instanced geometry hit (engine lines 195-567) ─────────
    if collision_instanced_geometry_instance_index != -1 {
        return geometry_test_instance_hit(
            loaded_bsp,
            structure_bsp_index,
            collision_instanced_geometry_instance_index,
            collision_surface_index,
            collision_point,
            early_out,
            test_result,
        );
    }

    // ── Branch 3: Cluster hit (engine lines 568-835) ────────────────────
    *early_out = true; // engine line 569: cluster branch always sets early_out.

    let structure_surfaces = &sbsp.structure_surfaces;
    let mappings = &sbsp.structure_surface_to_triangle_mappings;
    let clusters = &sbsp.clusters;
    let meshes = &loaded_bsp.meshes;

    // Engine line 571-573: structure_bsp has no surfaces → return.
    if structure_surfaces.is_empty() {
        return false;
    }
    // Engine line 578-595: get structure_surfaces[collision_surface_index]
    // (the descriptor: first_mapping_index + mapping_count).
    let surface = match (collision_surface_index as usize).checked_sub(0) {
        Some(i) if i < structure_surfaces.len() => &structure_surfaces[i],
        _ => return false,
    };
    if surface.mapping_count == 0 {
        return false;
    }

    // Engine line 596-705: per-mapping point-in-triangle loop. Track best.
    let mut best_metric = f32::NEG_INFINITY; // engine: FLOAT_N3_4028235e38 (= -FLT_MAX)
    let mut best_mapping_offset: i32 = -1;
    let mut best_bs = 0.0_f32;
    let mut best_bt = 0.0_f32;

    let mut triangle = UncompressedTriangle::default();
    let mut found_close_inside = false;
    for offset in 0..surface.mapping_count as i32 {
        let mapping_idx = surface.first_mapping_index as i32 + offset;
        if mapping_idx < 0 || (mapping_idx as usize) >= mappings.len() {
            continue;
        }
        let mapping = &mappings[mapping_idx as usize];

        // mapping.section_index → cluster (engine: clusters[v92]). cluster.mesh_index → mesh.
        let cluster_idx = mapping.section_index as usize;
        if cluster_idx >= clusters.len() {
            continue;
        }
        let cluster_mesh_index = clusters[cluster_idx].mesh_index;
        if cluster_mesh_index < 0 || (cluster_mesh_index as usize) >= meshes.len() {
            continue;
        }
        let mesh = &meshes[cluster_mesh_index as usize];
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        // `mapping.triangle_index` is a position in the (raw) index buffer that
        // `RenderMesh::indices` preserves for triangle-list BSP meshes.
        // Engine: get_structure_triangle(vertex_data, mapping.triangle_index, indices, count, &triangle)
        let last_index = mapping.triangle_index as u32;
        if (last_index as usize) >= mesh.indices.len() {
            continue;
        }
        get_structure_triangle(&mesh.vertices, last_index, &mesh.indices, &mut triangle);

        // Engine: point_in_triangle3d(point, &tri.position[0..2], &bt, &best_bt)
        // The engine swaps the role of `bs`/`bt` between cluster (uses
        // `best_bt`/`bt`) and instance branches; we record (s, t) directly.
        let (mut s_cand, mut t_cand) = (0.0_f32, 0.0_f32);
        let metric = geometry_sampling_point_in_triangle3d(
            collision_point,
            triangle.position[0],
            triangle.position[1],
            triangle.position[2],
            &mut s_cand,
            &mut t_cand,
        );
        if metric <= best_metric {
            continue;
        }
        best_metric = metric;
        best_mapping_offset = offset;
        best_bs = s_cand;
        best_bt = t_cand;
        // Engine: if metric > -eps, break (= "we found a hit close enough to
        // inside, stop searching").
        if metric > -9.999_999_7e-5 {
            found_close_inside = true;
            break;
        }
    }
    let _ = found_close_inside; // engine just goto's out of the loop; no special handling needed

    // Engine line 710-712: no triangle matched → return 0.
    if best_mapping_offset == -1 {
        return false;
    }

    // Engine line 713-826: refetch the best mapping → cluster → mesh.
    let best_mapping_idx = surface.first_mapping_index as i32 + best_mapping_offset;
    let best_mapping = &mappings[best_mapping_idx as usize];
    let cluster_idx = best_mapping.section_index as usize;
    let cluster_mesh_index = clusters[cluster_idx].mesh_index;
    let mesh = &meshes[cluster_mesh_index as usize];

    // Engine line 749-751: set per_pixel_lightmaps flag iff the cluster
    // mesh actually carries a second vertex buffer for lightmap UVs
    // (`vertex_buffer_indices[1] != 0xFFFF`). At runtime the cache-build
    // step only allocates that buffer when the lightmap tag wrote
    // populated UVs for this mesh; the tag-build pipeline (Reach /
    // protomorph) doesn't have vertex buffers, so we detect the same
    // condition via `RenderMesh::has_lightmap_uvs` (set at LBSP
    // extraction time — true iff any raw_vertex carried a non-zero
    // lightmap texcoord).
    //
    // When false, all three has_* flags stay 0 → `c_geometry_sampler::
    // sample` falls into the warn+white default (sh[0]=1.0 for all
    // channels) and decorators on this cluster end up unlit-but-tinted
    // instead of reading garbage atlas pixels at UV (0, 0).
    let lightmap_uvs: Vec<RealPoint2d> = mesh
        .vertices
        .iter()
        .map(|v| v.lightmap_texcoord)
        .collect();
    test_result.has_per_pixel_lightmaps = mesh.has_lightmap_uvs;

    // Decode the best triangle (engine line 806-811).
    let last_index = best_mapping.triangle_index as u32;
    get_structure_triangle(&mesh.vertices, last_index, &mesh.indices, &mut triangle);

    // Engine line 803: store cluster_index in test_result.
    test_result.cluster_index = cluster_idx as i32;
    test_result.structure_bsp_index = structure_bsp_index;

    // Engine line 812-826: dispatch the populated triangle into
    // geometry_test_triangle. Clusters never have per-vertex SH probes nor
    // single-probe lookups (those are instance-only), so pass None / -1 for
    // those args. Engine: `(*best_mapping_idx - 2) / 3` = triangle's
    // first-vertex-index in the index buffer (since each triangle is 3
    // sequential indices); the engine value is what test_result.triangle_index
    // gets stored as.
    let triangle_index = ((last_index as i32) - 2) / 3;
    geometry_test_triangle(
        collision_point,
        &triangle,
        best_bs,
        best_bt,
        triangle_index,
        if lightmap_uvs.is_empty() { None } else { Some(&lightmap_uvs) },
        None, // no per-vertex SH for clusters
        -1,   // no single lightprobe for clusters
        None, // debug_sampling_transform
        test_result,
        structure_bsp_index,
        None, // single_probes (only used by instance branch)
    );

    true
}

// =============================================================================
// geometry_test_instance_hit — instance branch of dllcache `0x18048c620`
// =============================================================================
//
// Engine lines 195-567 of `geometry_test_collision_result`. Split into its
// own function to keep the dispatch readable. Resolves the instance + its
// definition, inverse-transforms the collision_point into instance-local
// space, dispatches on the LightmapPolicy attached to the instance in the
// per-BSP lightmap, then walks the definition's structure_surface →
// triangle mappings to find the best hit.
//
// **Phase 7b.4c status:** lightmap-type dispatch + matrix inverse-transform
// shipped. Triangle search loop deferred to Phase 7b.4d (gated on
// extending `BspInstanceDefinition` with `structure_surfaces[]` +
// `structure_surface_to_triangle_mappings[]`, which only the top-level
// `StructureBsp` currently exposes).

#[allow(clippy::too_many_arguments)]
fn geometry_test_instance_hit(
    loaded_bsp: &LoadedBsp,
    structure_bsp_index: i32,
    instance_index: i32,
    _collision_surface_index: i32,
    collision_point: RealPoint3d,
    early_out: &mut bool,
    test_result: &mut GeometryTestResult,
) -> bool {
    let sbsp = &loaded_bsp.sbsp;

    // Engine line 197: instance_has_lighting_samples check — skipped here
    // since protomorph's `loaded_bsp.lightmap` is None for BSPs without
    // any lightmap chain at all; that case is filtered upstream. Engine
    // also passes `ignore_lightmap_samples` which when true bypasses the
    // check; for the decorator-bake call path that flag is false.

    let inst_idx_u = match (instance_index as usize).checked_sub(0) {
        Some(i) if i < sbsp.instanced_geometry_instances.len() => i,
        _ => return false,
    };
    let instance = &sbsp.instanced_geometry_instances[inst_idx_u];

    // Engine line 225: `global_instance_geometry_definition_get(bsp_idx, def_idx)`
    let def_idx = instance.definition_index as usize;
    if def_idx >= sbsp.instance_definitions.len() {
        return false;
    }
    let definition = &sbsp.instance_definitions[def_idx];

    // Engine line 229-247: get mesh from def.mesh_index.
    let mesh_idx = definition.mesh_index;
    if mesh_idx < 0 || (mesh_idx as usize) >= loaded_bsp.meshes.len() {
        return false;
    }
    let mesh = &loaded_bsp.meshes[mesh_idx as usize];
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return false;
    }

    // Engine line 252: early_out = true for the instance branch (any
    // dispatch arrived here means we've identified the hit; the lookup
    // either fills test_result or fails gracefully — upstream shouldn't
    // re-test against other geometry).
    *early_out = true;
    test_result.instance_index = instance_index;
    test_result.structure_bsp_index = structure_bsp_index;

    // ── Lightmap-type dispatch (engine lines 261-382) ───────────────────
    //
    // Engine encodes the type as `HIWORD(matrix[1].up.k)` — a non-engine
    // overload stuffed into the high bits of the instance matrix's `up.k`.
    // Values: 0 = per-pixel (fallthrough), 1 = per-vertex SH, 2 = single
    // probe, 3 = also per-pixel.
    //
    // Protomorph derives the same dispatch from `LightmapInstanceEntry.policy()`,
    // which is the same source of truth (the lightmap tag's instance entry).
    let lightmap_data = match &loaded_bsp.lightmap {
        Some(d) => d,
        None => return false,
    };
    let lm_entry = lightmap_data.instances.get(inst_idx_u);
    match lm_entry.map(|e| e.policy()) {
        Some(LightmapPolicy::PerVertex) => {
            // Engine sets test_result.has_per_vertex_lightprobe = 1 and
            // resolves the runtime per-vertex SH buffer. Phase 7b.4d
            // will plumb the LightmapPerVertexBlock's lightprobe_data
            // into the per_vertex_lightprobes param of
            // geometry_test_triangle (requires re-encoding the
            // dequantized SH to the engine's RGBE u32 stream or adapting
            // geometry_test_triangle to accept LightmapPerVertexProbe
            // directly).
            test_result.has_per_vertex_lightprobe = true;
        }
        Some(LightmapPolicy::SingleProbe) => {
            // Engine sets test_result.has_single_lightprobe = 1 and
            // stores the probe index for later resolution. Dequantize
            // and assign in Phase 7b.4d when the triangle search wires
            // through to geometry_test_triangle.
            test_result.has_single_lightprobe = true;
        }
        Some(LightmapPolicy::PerPixel) | Some(LightmapPolicy::Fallback) | None => {
            // Engine fallthrough (lines 357-382): set per_pixel_lightmaps
            // = 1 when the def has a per-instance lightmap UV buffer.
            // Protomorph stores per-instance lightmap UVs in the lightmap
            // tag's `per_instance_lightmap_texcoords[]`, already loaded
            // into `loaded_bsp.per_instance_lightmap_uvs[]`. We mirror
            // the engine flag here even though the search loop that
            // would consume those UVs isn't wired yet.
            test_result.has_per_pixel_lightmaps = true;
        }
    }

    // ── Inverse-transform collision_point → instance-local (engine 418) ─
    let matrix = RealMatrix4x3 {
        scale: instance.scale,
        forward: instance.forward,
        left: instance.left,
        up: instance.up,
        position: instance.position,
    };
    let mut modified_point = RealPoint3d::default();
    matrix4x3_inverse_transform_point(&matrix, &collision_point, &mut modified_point);

    // ── Triangle search loop (engine lines 419-547) ─────────────────────
    let surface_idx_u = match (_collision_surface_index as usize).checked_sub(0) {
        Some(i) if i < definition.structure_surfaces.len() => i,
        _ => return false,
    };
    let surface = &definition.structure_surfaces[surface_idx_u];
    if surface.mapping_count == 0 {
        return false;
    }

    let mappings = &definition.structure_surface_to_triangle_mappings;
    let mut best_metric = f32::NEG_INFINITY;
    let mut best_mapping_offset: i32 = -1;
    let mut best_bs = 0.0_f32;
    let mut best_bt = 0.0_f32;
    let mut triangle = UncompressedTriangle::default();
    for offset in 0..surface.mapping_count as i32 {
        let mapping_idx = surface.first_mapping_index as i32 + offset;
        if mapping_idx < 0 || (mapping_idx as usize) >= mappings.len() {
            continue;
        }
        let mapping = &mappings[mapping_idx as usize];
        let last_index = mapping.triangle_index as u32;
        if (last_index as usize) >= mesh.indices.len() {
            continue;
        }
        // Engine `get_instance_triangle(vertex_data, compression_info, ...)`
        // applies s_compression_info to dequantize. Protomorph's RenderVertex
        // is already-decoded floats, so `get_structure_triangle` (which is
        // engine-distinct in name only — same indexed vertex extraction)
        // does the right thing for both instance and cluster meshes.
        get_structure_triangle(&mesh.vertices, last_index, &mesh.indices, &mut triangle);

        let (mut s_cand, mut t_cand) = (0.0_f32, 0.0_f32);
        let metric = geometry_sampling_point_in_triangle3d(
            modified_point,
            triangle.position[0],
            triangle.position[1],
            triangle.position[2],
            &mut s_cand,
            &mut t_cand,
        );
        if metric <= best_metric {
            continue;
        }
        best_metric = metric;
        best_mapping_offset = offset;
        best_bs = s_cand;
        best_bt = t_cand;
        if metric > -9.999_999_7e-5 {
            break;
        }
    }
    if best_mapping_offset == -1 {
        return false;
    }

    // Refetch best triangle.
    let best_mapping_idx = surface.first_mapping_index as i32 + best_mapping_offset;
    let best_mapping = &mappings[best_mapping_idx as usize];
    let last_index = best_mapping.triangle_index as u32;
    get_structure_triangle(&mesh.vertices, last_index, &mesh.indices, &mut triangle);

    // ── Build lightmap dispatch data for geometry_test_triangle ─────────
    // Each policy resolves a different combination of (lightmap_uv_data,
    // per_vertex_lightprobes, single_probes, lightmap_probe_index).
    let mut per_vertex_dequant: Vec<DequantizedPerVertexProbe> = Vec::new();
    let mut single_probes_buf: [HalfRgbLightprobeWithDominantLight; 1] =
        [HalfRgbLightprobeWithDominantLight::default()];
    let mut lightmap_probe_index: i32 = -1;
    let mut instance_lightmap_uvs: Vec<RealPoint2d> = Vec::new();

    let policy = lm_entry.map(|e| e.policy()).unwrap_or(LightmapPolicy::PerPixel);
    match policy {
        LightmapPolicy::PerVertex => {
            if let Some(entry) = lm_entry {
                let block_idx = entry.pervertex_block_index;
                if block_idx >= 0
                    && (block_idx as usize) < lightmap_data.bsp_per_vertex_data.len()
                {
                    let block = &lightmap_data.bsp_per_vertex_data[block_idx as usize];
                    // Engine binds the block's per-vertex SH stream as a
                    // runtime vertex buffer indexed by mesh vertex index.
                    // Protomorph dequantizes the block in advance; the
                    // resulting slice is indexed identically.
                    per_vertex_dequant = block
                        .lightprobe_data
                        .iter()
                        .map(|p| p.dequantize())
                        .collect();
                }
            }
        }
        LightmapPolicy::SingleProbe => {
            if let Some(entry) = lm_entry {
                let probe_idx = entry.probe_block_index;
                if probe_idx >= 0 && (probe_idx as usize) < lightmap_data.probes.len() {
                    let probe = &lightmap_data.probes[probe_idx as usize];
                    single_probes_buf[0] = lightmap_probe_to_half(probe);
                    lightmap_probe_index = 0;
                }
            }
        }
        LightmapPolicy::PerPixel | LightmapPolicy::Fallback => {
            // Per-instance lightmap UVs from
            // loaded_bsp.per_instance_lightmap_uvs[instance.lightmap_texcoord_block_index].
            // Engine reads them from a per-instance vertex_buffer; protomorph
            // pre-extracts to Vec<RealPoint2d> at scenario load.
            let bi = instance.lightmap_texcoord_block_index;
            if bi >= 0 {
                if let Some(entry) = loaded_bsp
                    .per_instance_lightmap_uvs
                    .iter()
                    .find(|e| e.block_index == bi as usize)
                {
                    instance_lightmap_uvs = entry.uvs.clone();
                }
            }
        }
    }

    // Engine line 812-826: dispatch into geometry_test_triangle with the
    // inverse-transformed point + triangle (both in instance-local space).
    let triangle_index = ((last_index as i32) - 2) / 3;
    geometry_test_triangle(
        modified_point,
        &triangle,
        best_bs,
        best_bt,
        triangle_index,
        if instance_lightmap_uvs.is_empty() { None } else { Some(&instance_lightmap_uvs) },
        if per_vertex_dequant.is_empty() { None } else { Some(&per_vertex_dequant) },
        lightmap_probe_index,
        Some(&matrix),
        test_result,
        structure_bsp_index,
        if lightmap_probe_index >= 0 { Some(&single_probes_buf) } else { None },
    );

    true
}

/// Convert a [`blam_tags::scenario_lightmap::LightmapProbe`] (half-float-
/// as-i16) into the engine's [`HalfRgbLightprobeWithDominantLight`] layout
/// for use as `single_probes[0]` in [`geometry_test_triangle`].
///
/// LightmapProbe's `i16` fields are already half-float bit patterns
/// (per the existing `dequantize()` which calls `f16::from_bits`); we
/// just reinterpret them as `u16` and copy.
fn lightmap_probe_to_half(
    probe: &blam_tags::scenario_lightmap::LightmapProbe,
) -> HalfRgbLightprobeWithDominantLight {
    let to_u16 = |x: i16| x as u16;
    HalfRgbLightprobeWithDominantLight {
        dominant_light_direction: [
            to_u16(probe.dominant_light_direction[0]),
            to_u16(probe.dominant_light_direction[1]),
            to_u16(probe.dominant_light_direction[2]),
        ],
        pad: 0,
        dominant_light_intensity: HalfLinearRgbColor {
            red: to_u16(probe.dominant_light_intensity[0]),
            green: to_u16(probe.dominant_light_intensity[1]),
            blue: to_u16(probe.dominant_light_intensity[2]),
            pad: 0,
        },
        quadratic_probe: HalfRgbLightprobe {
            red_terms: probe.red_terms.map(to_u16),
            green_terms: probe.green_terms.map(to_u16),
            blue_terms: probe.blue_terms.map(to_u16),
            pad: 0,
        },
    }
}

// =============================================================================
// c_geometry_sampler::geometry_test_vector — `dllcache 0x18048c300` (Phase 7b.5)
// =============================================================================

/// `c_geometry_sampler::geometry_test_vector` @ dllcache `0x18048c300`.
///
/// Thin wrapper that builds the collision-test flags from the caller's
/// boolean controls, casts the ray via
/// [`super::super::physics::collisions::collision_test_vector`], then
/// dispatches the hit triple into [`geometry_test_collision_result`].
///
/// Returns the engine's `e_geometry_test_vector_outcome` mapped to
/// [`GeometryTestVectorOutcome`]:
/// - `Success` — collision found AND test_result fully populated
/// - `HitObjectButFailed` — hit on an object but no usable lighting cache
/// - `HitInstanceButFailed` — hit on instanced geometry but lookup failed
/// - `HitBspButFailed` — hit on cluster but lookup failed
/// - `HitNothing` — no collision
///
/// **Signature adaptation:** engine takes no `scenario` (uses globals);
/// protomorph passes [`LoadedScenario`] for [`collision_test_vector`] +
/// [`geometry_test_collision_result`]. Engine also has no explicit `flags`
/// param — it builds them internally from the boolean inputs. Protomorph
/// mirrors that to keep callers from having to know the magic constants.
#[allow(clippy::too_many_arguments)]
pub fn geometry_test_vector(
    scenario: &LoadedScenario,
    ray_start: RealPoint3d,
    ray_dir: RealVector3d,
    ignore_all_objects: bool,
    ignore_object_index: i32,
    ignore_lightmap_samples: bool,
    ignore_instanced_geometry: bool,
    get_render_method_index: bool,
    test_result: &mut GeometryTestResult,
) -> GeometryTestVectorOutcome {
    // Engine line 33-40: build `s_collision_test_flags`.
    // `0x4811` (= 18449 decimal) = STRUCTURE | RENDER_ONLY_BSPS |
    //                              FRONT_FACING_SURFACES | IGNORE_INVISIBLE_SURFACES.
    // When `!ignore_instanced_geometry`, OR in bit 3 (INSTANCED_GEOMETRY):
    // `0x4819` (= 18457 decimal).
    let mut collision_flags_raw: u32 = 0x4811;
    if !ignore_instanced_geometry {
        collision_flags_raw = 0x4819;
    }
    // Object flags: 0 if ignore_all_objects, else 0x2181. Engine ORs against
    // existing `collision_flags.object_flags.m_flags` (which is 0 after
    // initialization).
    let object_flags_raw: u32 = if ignore_all_objects { 0 } else { 0x2181 };
    let flags = CollisionTestFlagsPair::build(collision_flags_raw, object_flags_raw);

    // Engine line 38-39: zero the flag fields of test_result. Engine writes
    // a single _WORD covering has_per_vertex_lightprobe + has_per_pixel_lightmaps.
    test_result.has_per_vertex_lightprobe = false;
    test_result.has_per_pixel_lightmaps = false;
    test_result.has_single_lightprobe = false;
    test_result.cluster_index = -1;
    test_result.instance_index = -1;
    test_result.object_index = -1;

    // Engine line 65-94: helper-object resolution for `ignore_object_index`.
    // The engine looks up the object's map_variant_index → game_variant →
    // helper_object_index. Protomorph doesn't yet plumb the game-variant
    // system into the sampler (decorator-bake callers always pass
    // ignore_object_index = -1), so we leave helper at -1.
    //
    // TODO follow-up: port the helper resolution once a non-decorator
    // caller appears.
    let second_ignore_object_index: i32 = -1;

    // Engine line 106-141: do the collision test + dispatch.
    let mut collision = CollisionResult::default();
    let hit = crate::halo::physics::collisions::collision_test_vector(
        flags,
        scenario,
        ray_start,
        ray_dir,
        ignore_object_index,
        second_ignore_object_index,
        &mut collision,
    );
    if !hit {
        // Engine `n4 = 4` is the initial value (HitNothing).
        return GeometryTestVectorOutcome::HitNothing;
    }

    // Hit. Dispatch into geometry_test_collision_result.
    let mut early_out = false;
    let resolved = geometry_test_collision_result(
        scenario,
        collision.structure_bsp_index,
        collision.object_index,
        collision.instanced_geometry_instance_index,
        collision.surface_index,
        collision.point,
        ignore_lightmap_samples,
        get_render_method_index,
        &mut early_out,
        test_result,
    );

    // Copy collision_point + global_material_type onto test_result (engine
    // lines 137-140 — always done when there was a hit, regardless of
    // geometry_test_collision_result's success).
    test_result.collision_point = collision.point;
    test_result.global_material_type = collision.global_material_type as i32;

    // Map outcome (engine lines 115-136).
    if resolved {
        GeometryTestVectorOutcome::Success
    } else if collision.object_index != -1 {
        GeometryTestVectorOutcome::HitObjectButFailed
    } else if collision.structure_bsp_index != -1 {
        if collision.instanced_geometry_instance_index != -1 {
            GeometryTestVectorOutcome::HitInstanceButFailed
        } else {
            GeometryTestVectorOutcome::HitBspButFailed
        }
    } else {
        GeometryTestVectorOutcome::HitNothing
    }
}

// =============================================================================
// c_geometry_sampler::sample — `dllcache 0x18048a450` (Phase 7b.6a)
// =============================================================================

/// `c_geometry_sampler::sample` @ dllcache `0x18048a450`.
///
/// Top-level geometry-sampler entry. The bake feeds rays in; we resolve a
/// lighting payload (SH3 lightprobe + dominant light + diffuse) at the hit.
///
/// Flow:
/// 1. If the scenario has no lightmap chain → return [`NoLightmaps`] with
///    the engine's default lightprobe sample.
/// 2. [`geometry_test_vector`] for the ray cast + lookup.
///    - `HitNothing` → return [`NoHit`].
///    - Other failure outcomes → return [`HitButFailure`].
/// 3. If the hit is NOT on an object:
///    - Resolve the per-BSP lightmap data. Missing → set default sky
///      lightprobe + return [`NoLightmapForBsp`].
///    - Cluster path: look up `lightmap.clusters[cluster_index]` for the
///      entity_data + copy surface_normal into sampled_result.normal.
///    - Instance path: look up `lightmap.instances[instance_index]` +
///      transform the surface_normal back to world via the instance matrix.
///    - Dispatch by `has_*` flag:
///       - per_vertex → copy SH2 [0..4] + reconstruct L2
///       - per_pixel → [`sample_lightprobe_textures`]
///       - single → copy SH3 [0..9] + calculate_dominant
///       - fallback white (engine warning path)
///    - **(Deferred Phase 7b.6b):** sample_diffuse_textures + bounce-light
///      addition via ravi_order_3 + sh_eval_directional_light + sh_add.
/// 4. **(Deferred Phase 7b.6b):** Object-hit branch — never taken by the
///    decorator bake (rays go from world positions toward the sky; they
///    never enter object collision).
///
/// **Signature adaptations from engine:**
/// - Engine has no `scenario`; protomorph uses [`LoadedScenario`] to
///   resolve per-BSP lightmap data.
/// - Engine accesses bitmaps through `c_structure_renderer::get_lightmap_bsp_data`
///   + `bitmap_group_2d_get_real_pixel`; protomorph takes pre-decoded
///   atlases as `Option<&DecodedAtlas>` params (caller pre-resolves at
///   scenario load).
#[allow(clippy::too_many_arguments)]
pub fn sample(
    scenario: &LoadedScenario,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
    ray_start: RealPoint3d,
    ray_dir: RealVector3d,
    ignore_all_objects: bool,
    ignore_object_index: i32,
    light_probe: bool,
    diffuse: bool,
    sampled_result: &mut GeometrySample,
) -> GeometrySamplerOutcome {
    // Engine line 116: needs_interpolation defaults to true.
    *sampled_result = GeometrySample::default();
    sampled_result.needs_interpolation = true;

    // Engine line 118: `global_scenario->new_lightmaps.index == -1` →
    // scenario has no lightmap chain at all. Copy `m_default_lightprobe_sample`
    // and return NoLightmaps. Protomorph proxy: any active BSP must have
    // `lightmap.is_some()` for a chain to exist. Check the FIRST one; full
    // scenario-wide check is broader but this matches the engine sentinel.
    let any_lightmap = scenario
        .active_bsps
        .iter()
        .any(|b| b.lightmap.is_some());
    if !any_lightmap {
        // Engine lines 119-148: copy m_default_lightprobe_sample (sky SH +
        // dom + diffuse=white + normal=+Z + contrast=10) before returning.
        apply_default_sky_lightprobe(scenario, sampled_result);
        sampled_result.needs_interpolation = false;
        return GeometrySamplerOutcome::NoLightmaps;
    }

    // Engine line 152-161: call geometry_test_vector with ignore_lightmap=false,
    // ignore_instanced_geometry=false, get_render_method_index=true.
    let mut test_result = GeometryTestResult::default();
    let n4 = geometry_test_vector(
        scenario, ray_start, ray_dir, ignore_all_objects, ignore_object_index,
        false, false, true, &mut test_result,
    );
    if n4 != GeometryTestVectorOutcome::Success {
        // Engine line 163: `return (n4 != 4) + 2`. 4 = HitNothing → return 2
        // (NoHit). Any other non-zero outcome → return 3 (HitButFailure).
        return if n4 == GeometryTestVectorOutcome::HitNothing {
            GeometrySamplerOutcome::NoHit
        } else {
            GeometrySamplerOutcome::HitButFailure
        };
    }

    // Engine line 165: object-hit branch fork. Decorator bake never hits
    // objects (rays go from world positions outward; they don't enter
    // object collision). Defer to Phase 7b.6b.
    if test_result.object_index != -1 {
        // TODO Phase 7b.6b: object-hit branch (engine lines 482-625) — copy
        // SH from cached_object_render_states[render_state], handle volume
        // samples via get_volume_sample_at_location.
        return GeometrySamplerOutcome::HitButFailure;
    }

    // ── Resolve loaded_bsp + lightmap_bsp_data ──────────────────────────
    let loaded_bsp = match scenario
        .active_bsps
        .iter()
        .find(|b| b.scenario_bsp_index == test_result.structure_bsp_index as usize)
    {
        Some(b) => b,
        None => return GeometrySamplerOutcome::NoLightmapForBsp,
    };
    let lightmap_bsp = match &loaded_bsp.lightmap {
        Some(d) => d,
        None => {
            // Engine line 169-189: no lightmap data → set default sky
            // lightprobe via c_lighting_interface::get_default_sky_lightprobe.
            // Protomorph: pull from scenario.sky_lighting.default_lightprobe
            // if available; otherwise leave zeros.
            apply_default_sky_lightprobe(scenario, sampled_result);
            return GeometrySamplerOutcome::NoLightmapForBsp;
        }
    };

    // Engine line 191: memset sampled_result and set needs_interpolation = 1.
    *sampled_result = GeometrySample::default();
    sampled_result.needs_interpolation = true;

    // Engine deviation (intentional): the dllcache runtime sample never
    // writes `sampled_result.m_sample_point` in the cluster/instance
    // branches — only the object branch does
    // (`sampled_result->m_sample_point = test_result.m_collision_point` @
    // dllcache 0x18048a450:502). tool.exe's bake (`sub_140C4AF00`) doesn't
    // read it from the sample either — it threads the hit point through a
    // separate byref OUT parameter on `sub_140989F30`.
    //
    // Protomorph's `light_placement` (Rust port of `sub_140C4AF00`) reads
    // `hit_sample.sample_point` for the distance-weight calculation, so
    // we mirror what tool.exe's OUT byref would have written: copy
    // `test_result.collision_point` into `sampled_result.sample_point`
    // unconditionally. Without this, `hit_sample.sample_point` stays at
    // the GeometrySample::default() `(0, 0, 0)`, distance collapses to
    // `magnitude(placement_pos)` (~146 m for far-from-origin scenes), and
    // every ray's weight ≈ 0 — the entire bake produces near-black
    // decorator colors.
    sampled_result.sample_point = test_result.collision_point;

    // Engine line 194-293: cluster vs instance dispatch.
    if test_result.instance_index == -1 {
        // Engine line 196: cluster_index == -1 → no_cluster_or_instance.
        if test_result.cluster_index == -1 {
            // Engine lines 197-225: copy m_default_lightprobe_sample (sky SH
            // + dom + diffuse + normal + contrast) before returning.
            apply_default_sky_lightprobe(scenario, sampled_result);
            sampled_result.needs_interpolation = false;
            return GeometrySamplerOutcome::NoClusterOrInstance;
        }

        // Engine line 229-246: look up cluster lightmap entity_data (not
        // currently consumed in the SH branches — only sample_diffuse_textures
        // and the per-pixel atlas lookup use it). Validate the cluster index
        // is in range.
        if (test_result.cluster_index as usize) >= lightmap_bsp.clusters.len() {
            return GeometrySamplerOutcome::NoClusterOrInstance;
        }

        // Engine lines 247-249: copy surface_normal as-is (no transform for
        // clusters since vertices are already in world space).
        sampled_result.normal = test_result.surface_normal;
    } else {
        // Instance branch (engine lines 251-293).
        if (test_result.instance_index as usize) >= lightmap_bsp.instances.len() {
            return GeometrySamplerOutcome::NoClusterOrInstance;
        }
        // Engine line 274-291: get the instance's world matrix.
        let inst_idx = test_result.instance_index as usize;
        if inst_idx >= loaded_bsp.sbsp.instanced_geometry_instances.len() {
            return GeometrySamplerOutcome::NoClusterOrInstance;
        }
        let instance = &loaded_bsp.sbsp.instanced_geometry_instances[inst_idx];
        // Engine line 292: `matrix4x3_transform_normal(matrix, surface_normal, &sampled.normal)`.
        // Transform local-space normal → world-space. For an orthonormal
        // basis with uniform scale (instance matrix), this is the same as
        // `result = M * normal` ignoring translation.
        let n = test_result.surface_normal;
        sampled_result.normal.i = instance.forward.i * n.i
            + instance.left.i * n.j
            + instance.up.i * n.k;
        sampled_result.normal.j = instance.forward.j * n.i
            + instance.left.j * n.j
            + instance.up.j * n.k;
        sampled_result.normal.k = instance.forward.k * n.i
            + instance.left.k * n.j
            + instance.up.k * n.k;
    }

    // Engine line 290-291: `if (diffuse) sample_diffuse_textures(...)`.
    // Writes `m_diffuse = albedo`-style RGB sampled from the hit
    // surface's authored postprocess textures; gates the bounce-light
    // SH addition below (the block only fires when any `m_diffuse > 0`).
    //
    // Material lookup goes through the bake-time thread-local
    // [`crate::halo::decorators::bake_material_lookup`] — see that
    // module for the scenario-load-time cache build.
    if diffuse {
        sample_diffuse_textures_inner(&test_result, &mut sampled_result.diffuse);
    }

    // ── Light-probe dispatch (engine lines 296-443) ─────────────────────
    if light_probe {
        // Diag: per-source-branch counter (decorator-bake brightness-disparity
        // investigation 2026-05-26).
        DIAG_LIGHTPROBE_BRANCH_COUNTS.with(|c| {
            let mut a = c.borrow_mut();
            if test_result.has_per_vertex_lightprobe {
                a[0] += 1;
            } else if test_result.has_per_pixel_lightmaps {
                a[1] += 1;
            } else if test_result.has_single_lightprobe {
                a[2] += 1;
            } else {
                a[3] += 1;
            }
        });
        if test_result.has_per_vertex_lightprobe {
            // Engine lines 298-342: copy SH2 [0..4] + intensity, then
            // find_dominant_light_direction + reconstruct_quadratic_*.
            let pv = &test_result.interpolated_per_vertex_lightprobe;
            for k in 0..4 {
                sampled_result.light_probe_r[k] = pv.red_terms[k];
                sampled_result.light_probe_g[k] = pv.green_terms[k];
                sampled_result.light_probe_b[k] = pv.blue_terms[k];
            }
            sampled_result.dominant_light_intensity =
                test_result.interpolated_dominant_light_intensity;
            let sh_r: &mut [f32; 9] = (&mut sampled_result.light_probe_r[..9])
                .try_into()
                .unwrap();
            let sh_g: &mut [f32; 9] = (&mut sampled_result.light_probe_g[..9])
                .try_into()
                .unwrap();
            let sh_b: &mut [f32; 9] = (&mut sampled_result.light_probe_b[..9])
                .try_into()
                .unwrap();
            find_dominant_light_direction(
                sh_r, sh_g, sh_b,
                &mut sampled_result.dominant_light_dir,
            );
            reconstruct_quadratic_lightprobe_from_linear_and_intensity(
                sh_r, sh_g, sh_b,
                sampled_result.dominant_light_intensity,
                &mut sampled_result.dominant_light_dir,
                &mut sampled_result.dominant_light_contrast,
            );
        } else if test_result.has_per_pixel_lightmaps {
            // Engine lines 343-359: sample_lightprobe_textures from the
            // per-cluster lightmap atlas at test_result.lightmap_uv.
            sample_lightprobe_textures(
                test_result.structure_bsp_index,
                lightprobe_atlas,
                intensity_atlas,
                test_result.lightmap_uv,
                lightprobe_pixels_per_probe,
                dominant_pixels_per_probe,
                &lightmap_bsp.compression_vectors,
                &mut sampled_result.light_probe_r,
                &mut sampled_result.light_probe_g,
                &mut sampled_result.light_probe_b,
                &mut sampled_result.dominant_light_dir,
                &mut sampled_result.dominant_light_intensity,
                &mut sampled_result.dominant_light_contrast,
            );
        } else if test_result.has_single_lightprobe {
            // Engine lines 360-424: copy SH3 [0..9] from
            // test_result.single_sh_lightprobe + calculate_dominant_*.
            let sp = &test_result.single_sh_lightprobe;
            for k in 0..9 {
                sampled_result.light_probe_r[k] = sp.red_terms[k];
                sampled_result.light_probe_g[k] = sp.green_terms[k];
                sampled_result.light_probe_b[k] = sp.blue_terms[k];
            }
            let sh_r: &mut [f32; 9] = (&mut sampled_result.light_probe_r[..9])
                .try_into()
                .unwrap();
            let sh_g: &mut [f32; 9] = (&mut sampled_result.light_probe_g[..9])
                .try_into()
                .unwrap();
            let sh_b: &mut [f32; 9] = (&mut sampled_result.light_probe_b[..9])
                .try_into()
                .unwrap();
            calculate_dominant_light_from_lightprobe(
                sh_r, sh_g, sh_b,
                &mut sampled_result.dominant_light_dir,
                &mut sampled_result.dominant_light_intensity,
                &mut sampled_result.dominant_light_contrast,
            );
        } else {
            // Engine lines 425-443: warning + uniform-white fallback. Engine
            // emits "Geometry sampler: sampled from something with no lighting
            // data?" and sets sh[0] = 1.0 for all channels.
            sampled_result.light_probe_r[0] = 1.0;
            sampled_result.light_probe_g[0] = 1.0;
            sampled_result.light_probe_b[0] = 1.0;
        }

        // Engine lines 440-475: bounce-light SH addition. Adds an indirect
        // contribution from the hit surface back toward the sample point,
        // modeled as a directional light pointing along `-surface_normal`
        // (the bounced light direction) with intensity = surface SH probe
        // evaluated at `+surface_normal` × surface diffuse albedo × the
        // tunable `g_bounce_light_scale` global (= 1.0 on stock H3 MCC).
        //
        // Gated on `any(m_diffuse > 0)` so a surface that returned no
        // postprocess albedo contributes nothing (`sample_diffuse_textures`
        // leaves m_diffuse zero on those paths). With the postprocess
        // sampler now wired (Phase 7b.6c), this block produces the engine-
        // intended bounce term.
        if sampled_result.diffuse.i > 0.0
            || sampled_result.diffuse.j > 0.0
            || sampled_result.diffuse.k > 0.0
        {
            // Engine: `inverse_normal = surface_normal ^ sign_bit` =
            // bitwise-XOR with the IEEE-754 sign bit ≡ scalar negation.
            let inverse_normal = RealVector3d {
                i: -test_result.surface_normal.i,
                j: -test_result.surface_normal.j,
                k: -test_result.surface_normal.k,
            };
            let surface_normal_vec3 = glam::Vec3::new(
                test_result.surface_normal.i,
                test_result.surface_normal.j,
                test_result.surface_normal.k,
            );
            let sh_r_first9: &[f32; 9] = (&sampled_result.light_probe_r[..9])
                .try_into()
                .unwrap();
            let sh_g_first9: &[f32; 9] = (&sampled_result.light_probe_g[..9])
                .try_into()
                .unwrap();
            let sh_b_first9: &[f32; 9] = (&sampled_result.light_probe_b[..9])
                .try_into()
                .unwrap();
            // Engine: `vN = max(0, ravi_order_3(surface_normal, light_probe_N))`.
            let irrad_r = crate::halo::decorators::light_placement::ravi_order_3(
                surface_normal_vec3, sh_r_first9,
            ).max(0.0);
            let irrad_g = crate::halo::decorators::light_placement::ravi_order_3(
                surface_normal_vec3, sh_g_first9,
            ).max(0.0);
            let irrad_b = crate::halo::decorators::light_placement::ravi_order_3(
                surface_normal_vec3, sh_b_first9,
            ).max(0.0);

            // Engine `g_bounce_light_only` debug flag (zero the direct SH
            // before adding the bounce contribution) — defaults false, not
            // hooked to any runtime control in protomorph. Skipping the
            // memset path matches the production runtime.
            const G_BOUNCE_LIGHT_SCALE: f32 = 1.0;
            let r_intensity = G_BOUNCE_LIGHT_SCALE * sampled_result.diffuse.i * irrad_r;
            let g_intensity = G_BOUNCE_LIGHT_SCALE * sampled_result.diffuse.j * irrad_g;
            let b_intensity = G_BOUNCE_LIGHT_SCALE * sampled_result.diffuse.k * irrad_b;
            let inverse_normal_vec3 =
                glam::Vec3::new(inverse_normal.i, inverse_normal.j, inverse_normal.k);

            let mut bounce_r = [0.0_f32; 9];
            let mut bounce_g = [0.0_f32; 9];
            let mut bounce_b = [0.0_f32; 9];
            crate::halo::render::lighting_interface::sh_eval_directional_light(
                3, inverse_normal_vec3,
                r_intensity, g_intensity, b_intensity,
                &mut bounce_r, &mut bounce_g, &mut bounce_b,
            );
            // `sh_add(out, a, b, order)` writes `out[i] = a[i] + b[i]` for
            // `i in 0..order²`. Engine call: `sh_add(probe, 3, probe, bounce)`
            // — accumulate in place. Borrow-checker workaround: collect
            // first-9 into a local, add, write back.
            let mut sum_r = [0.0_f32; 9];
            let mut sum_g = [0.0_f32; 9];
            let mut sum_b = [0.0_f32; 9];
            for k in 0..9 {
                sum_r[k] = sampled_result.light_probe_r[k] + bounce_r[k];
                sum_g[k] = sampled_result.light_probe_g[k] + bounce_g[k];
                sum_b[k] = sampled_result.light_probe_b[k] + bounce_b[k];
            }
            sampled_result.light_probe_r[..9].copy_from_slice(&sum_r);
            sampled_result.light_probe_g[..9].copy_from_slice(&sum_g);
            sampled_result.light_probe_b[..9].copy_from_slice(&sum_b);
        }
    }

    GeometrySamplerOutcome::Success
}

/// Engine `c_geometry_sampler::m_default_lightprobe_sample` initial values
/// (IDA `.data:0x18116FE10`). Populated by `setup_default_lighting` at
/// scenario load — for scenarios with sky lightprobe, those values
/// override the SH below; otherwise these stay.
///
/// Layout per `s_geometry_sample` (504 B):
/// - sample_point = (0, 0, -1)
/// - light_probe_r[36] — only first 9 = SH3 R; rest scratch
/// - light_probe_g[36]
/// - light_probe_b[36]
/// - diffuse, normal, dom_dir, dom_intensity, contrast = all zero
/// - needs_interpolation = false
const DEFAULT_LIGHTPROBE_SH_R: [f32; 9] = [
    1.89376, -1.44314, 0.19557001, 0.51599002,
    -0.30807999, -0.041730002, -0.46665001, 0.29194999, 0.10501,
];
const DEFAULT_LIGHTPROBE_SH_G: [f32; 9] = [
    2.1351199, -1.79217, 0.15439001, 0.51479,
    -0.27408999, 0.070479997, -0.62704998, 0.25751001, -0.01816,
];
const DEFAULT_LIGHTPROBE_SH_B: [f32; 9] = [
    2.2585101, -2.07371, 0.056329999, 0.44103,
    -0.19508, 0.23114, -0.76419997, 0.18839, -0.22193,
];

/// Engine fallback for `c_geometry_sampler::sample`'s three non-Success
/// branches:
/// - **NoLightmaps** (engine 118-148): copy `m_default_lightprobe_sample`
///   wholesale (SH + sample_point = (0,0,-1); diffuse/normal/dom = 0).
/// - **NoLightmapForBsp** (engine 169-189): `get_default_sky_lightprobe`
///   for SH + `diffuse = (1,1,1)` + `normal = +Z` + `contrast = 10`.
/// - **NoClusterOrInstance** (engine 197-225): same as NoLightmaps copy.
///
/// Protomorph uses `scenario.sky_lighting.probe` when available, else the
/// hardcoded constants above. `mode` selects between the two diffuse/normal
/// conventions.
///
/// **Note on the bake:** `light_placement` only accumulates on `Success`
/// outcomes — fallback outcomes get skipped. This function therefore
/// doesn't affect the decorator bake. It's correct for the airprobe bake
/// + object lighting + future callers that read fallback SH.
fn apply_default_sky_lightprobe(scenario: &LoadedScenario, sample: &mut GeometrySample) {
    // SH source: scenario sky probe if available, else hardcoded engine constants.
    let (sh_r, sh_g, sh_b) = if let Some(sky) = scenario.sky_lighting.as_ref() {
        (sky.probe.r, sky.probe.g, sky.probe.b)
    } else {
        (DEFAULT_LIGHTPROBE_SH_R, DEFAULT_LIGHTPROBE_SH_G, DEFAULT_LIGHTPROBE_SH_B)
    };
    for k in 0..9 {
        sample.light_probe_r[k] = sh_r[k];
        sample.light_probe_g[k] = sh_g[k];
        sample.light_probe_b[k] = sh_b[k];
    }
    let sh_r_mut: &mut [f32; 9] = (&mut sample.light_probe_r[..9]).try_into().unwrap();
    let sh_g_mut: &mut [f32; 9] = (&mut sample.light_probe_g[..9]).try_into().unwrap();
    let sh_b_mut: &mut [f32; 9] = (&mut sample.light_probe_b[..9]).try_into().unwrap();
    calculate_dominant_light_from_lightprobe(
        sh_r_mut, sh_g_mut, sh_b_mut,
        &mut sample.dominant_light_dir,
        &mut sample.dominant_light_intensity,
        &mut sample.dominant_light_contrast,
    );
    // Engine NoLightmapForBsp-style overrides: diffuse=(1,1,1), normal=+Z, contrast=10.
    sample.diffuse = RealVector3d { i: 1.0, j: 1.0, k: 1.0 };
    sample.normal = RealVector3d { i: 0.0, j: 0.0, k: 1.0 };
    sample.dominant_light_contrast = 10.0;
}

// =============================================================================
// lights_distant_lighting_at_point_new — `dllcache 0x1808a3220`
// =============================================================================

/// Engine `lightmap_sample_raycast_sideways @ dllcache .data:0x180f41dd0` —
/// 9 directions used by `lights_distant_lighting_at_point_new`'s SIDEWAYS
/// branch (flags & 1). Read verbatim from the binary via the IDA bridge.
///
/// Pattern: 4 horizontal-down (xy=±10, z=-5) + 1 straight-long-down
/// (z=-40) + 4 horizontal-up (xy=±10, z=+5).
pub const LIGHTMAP_SAMPLE_RAYCAST_SIDEWAYS: [RealVector3d; 9] = [
    RealVector3d { i: -10.0, j:   0.0, k: -5.0  },
    RealVector3d { i:  10.0, j:   0.0, k: -5.0  },
    RealVector3d { i:   0.0, j: -10.0, k: -5.0  },
    RealVector3d { i:   0.0, j:  10.0, k: -5.0  },
    RealVector3d { i:   0.0, j:   0.0, k: -40.0 },
    RealVector3d { i: -10.0, j:   0.0, k:  5.0  },
    RealVector3d { i:  10.0, j:   0.0, k:  5.0  },
    RealVector3d { i:   0.0, j: -10.0, k:  5.0  },
    RealVector3d { i:   0.0, j:  10.0, k:  5.0  },
];

// Engine math globals (`global_down3d` @ dllcache `0x1810D5128`, `global_up3d`
// @ `0x1810D5110`) are surfaced from [`crate::halo::math::globals`]. The IDA
// decompile mistypes these via `x_event_category_index_2520.…peer_connectivity.
// latency_estimate_msec` / `…peer_nat_type` — both are actually pointer
// dereferences into the engine's shared `.rdata` constants block.
use crate::halo::math::globals::{GLOBAL_DOWN_3D, GLOBAL_UP_3D};
use crate::halo::math::vector_math::scale_vector3d;

/// Engine `e_lights_distant_lighting_flags` bits (bitfield on the `flags`
/// param of `lights_distant_lighting_at_point_new`).
pub mod lights_distant_lighting_flags {
    /// `flags & 1` — SIDEWAYS branch (9-ray fixed table). Engine
    /// `lights_prepare_for_object_static_new` sets this when
    /// `_object_definition.flags & 2 != 0` (`OBJ_FLAG_SEARCHES_LIGHTMAPS_ON_FAILURE`).
    pub const SIDEWAYS: u8 = 1;
    /// `flags & 4` — OVERRIDE_DIRECTION branch (1-ray with caller-supplied
    /// direction; bypasses object-class dispatch).
    pub const OVERRIDE_DIRECTION: u8 = 4;
}

/// Engine `lights_distant_lighting_at_point_new @ 0x1808a3220` — multi-ray
/// dispatcher around [`sample`] for static object lighting.
///
/// **Engine signature:**
/// ```c
/// e_geometry_sampler_outcome lights_distant_lighting_at_point_new(
///     char flags,
///     int object_index,
///     int ignore_object_index,
///     bool need_valid_sample,
///     const real_point3d *position,
///     s_geometry_sample *out_sample,
///     const real_vector3d *override_direction,
///     bool is_flying)
/// ```
///
/// **Protomorph adapter:** engine reads `object_class` (n6),
/// `object_def_flags`, `object_radius`, and `up` via `object_header_data` +
/// `object_get_orientation(object_index)`. We don't have an object index for
/// statically-placed scenery at this code path, so we accept those four as
/// explicit parameters. Engine also reads `global_scenario->flags & 0x200`
/// (used by the n6==1 weapon branch); we pass it as `scenario_flag_0x200`.
///
/// `c_geometry_sampler::sample`'s state-from-globals (scenario, atlases,
/// per-probe bitmap layout) is forwarded through as parameters too.
///
/// **Engine algorithm (`lights_distant_lighting_at_point_new @ 0x1808a3220`,
/// lines 113-485):**
/// 1. Pick direction set:
///    - `flags & 1` → 9-dir SIDEWAYS table (`lightmap_sample_raycast_sideways`)
///    - `flags & 4` → 1-dir OVERRIDE (caller-supplied)
///    - else → object-class-driven:
///      - `is_flying` → 2-ray (engine lines 166-205)
///      - `v24 = (n6 == 0 && obj_def.flags & 4)` (damaged biped) → 3-ray
///        (engine lines 206-243)
///      - `v23 = (n6 == 6 || n6 == 0)` (scenery or biped) → 3-ray (engine
///        lines 244-278)
///      - else → 1-ray with optional `0.1×` scale when `n6 == 1` (weapon)
///        AND `object_radius > 2` AND `global_scenario->flags & 0x200`
/// 2. Loop over directions (engine LABEL_42, lines 338-462):
///    - per-direction start point = `position + start_offset[i]` if buffer
///      provided, else `position - 0.001 × direction[i]` (the sideways/
///      override sub-microscopic backoff).
///    - Call `c_geometry_sampler::sample(&start, &direction[i], …)`.
///    - Engine outcome dispatch (lines 376-454):
///      - n3 == 0 (Success) → copy `sampled_result` to `out_sample`, set
///        `n2 = 0` (terminates loop).
///      - n3 == 3 (HitButFailure) → set `v39 = 1`, keep iterating.
///      - other non-zero outcomes → keep iterating.
/// 3. Final outcome (engine lines 483-485):
///    - `n2 != 0 && v39` → return `HitButFailure (3)`
///    - else → return `n2` (`Success (0)` or `NoHit (2)`)
#[allow(clippy::too_many_arguments)]
pub fn lights_distant_lighting_at_point_new(
    flags: u8,
    object_class: u8,
    // `s_object_data.flags` — the runtime object DATUM's flags word, not
    // the tag's `object_definition_flags` (the IDA decompile mislabels the
    // local `obj_def`, but it reads the live datum). Bit 2 (`& 4`) =
    // damaged-biped state. Engine-internal runtime state with no tag-schema
    // names, so it stays a raw integer.
    object_def_flags: u32,
    object_radius: f32,
    up: RealVector3d,
    ignore_object_index: i32,
    _need_valid_sample: bool,
    position: RealPoint3d,
    out_sample: &mut GeometrySample,
    override_direction: Option<RealVector3d>,
    is_flying: bool,
    scenario: &LoadedScenario,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
    scenario_flag_0x200: bool,
) -> GeometrySamplerOutcome {
    use lights_distant_lighting_flags::{OVERRIDE_DIRECTION, SIDEWAYS};

    // Buffers for the DEFAULT branch's per-class direction/start-offset
    // arrays (engine: stack-local `v88..v105` and `buffer/v98..v102` —
    // 3 directions max, 3 start offsets max).
    let mut local_dirs = [RealVector3d::default(); 3];
    let mut local_offsets = [RealVector3d::default(); 3];

    // Engine: lightmap_sample_raycast_sideways pointer / override_direction /
    // local buffer pointer (`buffer_1` is null for sideways/override, set to
    // `&buffer` for default).
    let (directions, offsets, n9): (&[RealVector3d], Option<&[RealVector3d]>, usize);

    if (flags & SIDEWAYS) != 0 {
        // Engine lines 120-129: SIDEWAYS branch — 9-ray table, no per-direction
        // start offsets (start = position - 0.001*direction).
        directions = &LIGHTMAP_SAMPLE_RAYCAST_SIDEWAYS;
        offsets = None;
        n9 = 9;
    } else if (flags & OVERRIDE_DIRECTION) != 0 {
        // Engine lines 320-328: OVERRIDE branch — 1-ray with caller-supplied
        // direction. Engine asserts on null override; we silently degenerate.
        let dir = override_direction.unwrap_or_default();
        local_dirs[0] = dir;
        directions = &local_dirs[..1];
        offsets = None;
        n9 = 1;
    } else {
        // Engine lines 130-318: DEFAULT branch — object-class dispatch.
        // n6 = object class, v23/v24 = class predicates.
        let n6 = object_class;
        let v23 = n6 == 6 || n6 == 0;
        // Runtime "damaged biped" datum bit (engine `& 4`); only consulted
        // for live bipeds (object class 0).
        let v24 = n6 == 0 && (object_def_flags & 4) != 0;

        // Engine lines 152-162:
        //   __real@3ecccccd = max(object_radius, 0.4)  (default 0.4)
        //   __real@3d4ccccd = max(object_radius, 0.05) (default 0.05)
        let r_big = if object_radius > 0.05 {
            object_radius.max(0.4)
        } else {
            0.4
        };
        let r_small = object_radius.max(0.05);

        if is_flying {
            // Engine lines 166-205: 2-ray flying — both directions are
            // `(r_big * 2) × GRAVITY_DOWN`; both offsets are zero.
            n9 = 2;
            local_dirs[0] = scale_vector3d(GLOBAL_DOWN_3D, r_big * 2.0);
            local_dirs[1] = scale_vector3d(GLOBAL_UP_3D, r_big * 2.0);
            local_offsets[0] = scale_vector3d(GLOBAL_UP_3D, 0.0);
            local_offsets[1] = scale_vector3d(GLOBAL_UP_3D, 0.0);
        } else if v24 {
            // Engine lines 206-243: 3-ray damaged-biped (n6==0 + obj_def
            // bit 2). Directions: gravity*10*r_big, up*-3*r_big, up*-4*r_big.
            // Offsets: up*r_small, dir[0]*-0.01, up*r_small.
            n9 = 3;
            local_dirs[0] = scale_vector3d(GLOBAL_DOWN_3D, r_big * 10.0);
            local_dirs[1] = scale_vector3d(up, r_big * -3.0);
            local_dirs[2] = scale_vector3d(up, r_big * -4.0);
            local_offsets[0] = scale_vector3d(GLOBAL_UP_3D, r_small);
            local_offsets[1] = scale_vector3d(local_dirs[0], -0.01);
            local_offsets[2] = scale_vector3d(up, r_small);
        } else if v23 {
            // Engine lines 244-278: 3-ray scenery/biped (n6==6 or n6==0
            // without bit-2). Directions: up*-3*r_big, up*-4*r_big,
            // gravity*10*r_big. Offsets: dir[0]*-0.01, up*r_small, up*r_small.
            n9 = 3;
            local_dirs[0] = scale_vector3d(up, r_big * -3.0);
            local_dirs[1] = scale_vector3d(up, r_big * -4.0);
            local_dirs[2] = scale_vector3d(GLOBAL_DOWN_3D, r_big * 10.0);
            local_offsets[0] = scale_vector3d(local_dirs[0], -0.01);
            local_offsets[1] = scale_vector3d(up, r_small);
            local_offsets[2] = scale_vector3d(up, r_small);
        } else {
            // Engine lines 279-317: 1-ray default. Direction =
            // gravity*10*r_big. Offset = up*r_small, with optional 0.1×
            // shrink for weapons (n6==1) on scenarios with `flags & 0x200`
            // AND `r_small > 2`.
            n9 = 1;
            local_dirs[0] = scale_vector3d(GLOBAL_DOWN_3D, r_big * 10.0);
            let offset_scale =
                if n6 == 1 && r_small > 2.0 && scenario_flag_0x200 {
                    r_small * 0.1
                } else {
                    r_small
                };
            local_offsets[0] = scale_vector3d(GLOBAL_UP_3D, offset_scale);
        }
        directions = &local_dirs[..n9];
        offsets = Some(&local_offsets[..n9]);
    }

    // Engine LABEL_42: per-ray loop (engine lines 338-462).
    //
    // `n2 = 2` (NoHit) accumulator; flips to 0 on first Success. `v39 = 1`
    // when any ray returned outcome=3 (HitButFailure).
    let mut n2: u32 = 2;
    let mut v39: bool = false;
    let mut sampled = GeometrySample::default();

    for i in 0..n9 {
        if n2 != 2 {
            // Engine `if (n2 != 2) break;` (line 340).
            break;
        }
        // Engine lines 342-356: pick start point.
        let dir = directions[i];
        let start = if let Some(off) = offsets {
            // Engine `point0 = buffer[v38] + position`.
            RealPoint3d {
                x: position.x + off[i].i,
                y: position.y + off[i].j,
                z: position.z + off[i].k,
            }
        } else {
            // Engine `point0 = position - 0.001 * direction`.
            RealPoint3d {
                x: position.x - 0.001 * dir.i,
                y: position.y - 0.001 * dir.j,
                z: position.z - 0.001 * dir.k,
            }
        };

        // Engine line 368-375: c_geometry_sampler::sample call.
        let outcome = sample(
            scenario,
            lightprobe_atlas,
            intensity_atlas,
            lightprobe_pixels_per_probe,
            dominant_pixels_per_probe,
            start,
            dir,
            /*ignore_all_objects*/ false,
            ignore_object_index,
            /*light_probe*/ true,
            /*diffuse*/ true,
            &mut sampled,
        );

        match outcome {
            GeometrySamplerOutcome::Success => {
                // Engine line 411-453: copy whole sampled_result to out_sample
                // (engine uses 4×OWORD + 2×QWORD memcpy; we just struct-copy).
                *out_sample = sampled;
                n2 = 0;
            }
            GeometrySamplerOutcome::HitButFailure => {
                // Engine line 392-393: `if (n3 == 3) v39 = 1`.
                v39 = true;
            }
            _ => {
                // Engine: other non-Success outcomes just keep iterating.
            }
        }
    }

    // Engine lines 483-485: final outcome.
    if n2 != 0 && v39 {
        GeometrySamplerOutcome::HitButFailure
    } else if n2 == 0 {
        GeometrySamplerOutcome::Success
    } else {
        GeometrySamplerOutcome::NoHit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::math::RealPoint3d;

    /// Centroid (`bs = bt = 1/3`) of a triangle with positions
    /// `(0,0,0), (3,0,0), (0,3,0)` lands at `(1, 1, 0)`. Verify
    /// `geometry_test_triangle` reproduces that centroid in `surface_point`,
    /// and also that `s = bs`, `t = bt`, and `triangle_index` round-trip.
    #[test]
    fn surface_point_at_centroid() {
        let triangle = UncompressedTriangle {
            position: [
                RealPoint3d { x: 0.0, y: 0.0, z: 0.0 },
                RealPoint3d { x: 3.0, y: 0.0, z: 0.0 },
                RealPoint3d { x: 0.0, y: 3.0, z: 0.0 },
            ],
            texcoord: [RealPoint2d::default(); 3],
            normal: [RealVector3d::default(); 3],
            indices: [0, 1, 2],
        };
        let mut tr = GeometryTestResult::default();
        let bs = 1.0 / 3.0;
        let bt = 1.0 / 3.0;
        geometry_test_triangle(
            RealPoint3d::default(),
            &triangle,
            bs,
            bt,
            42,
            None,
            None,
            -1,
            None,
            &mut tr,
            0,
            None,
        );
        assert!((tr.surface_point.x - 1.0).abs() < 1e-5);
        assert!((tr.surface_point.y - 1.0).abs() < 1e-5);
        assert!(tr.surface_point.z.abs() < 1e-5);
        assert_eq!(tr.triangle_index, 42);
        assert_eq!(tr.s, bs);
        assert_eq!(tr.t, bt);
    }

    /// Vertex-0-only barycentric (`bs = bt = 0`) should reproduce vertex 0
    /// exactly for all interpolated fields.
    #[test]
    fn vertex0_barycentric_reproduces_vertex0() {
        let triangle = UncompressedTriangle {
            position: [
                RealPoint3d { x: 10.0, y: 20.0, z: 30.0 },
                RealPoint3d { x: -7.0, y: 0.0, z: 0.0 },
                RealPoint3d { x: 0.0, y: -7.0, z: 0.0 },
            ],
            texcoord: [
                RealPoint2d { x: 0.25, y: 0.75 },
                RealPoint2d { x: 0.5, y: 0.5 },
                RealPoint2d { x: 1.0, y: 1.0 },
            ],
            normal: [
                RealVector3d { i: 0.6, j: 0.8, k: 0.0 },
                RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
                RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
            ],
            indices: [0, 1, 2],
        };
        let mut tr = GeometryTestResult::default();
        geometry_test_triangle(
            RealPoint3d::default(),
            &triangle,
            0.0,
            0.0,
            7,
            None,
            None,
            -1,
            None,
            &mut tr,
            0,
            None,
        );
        assert!((tr.surface_point.x - 10.0).abs() < 1e-5);
        assert!((tr.surface_point.y - 20.0).abs() < 1e-5);
        assert!((tr.surface_point.z - 30.0).abs() < 1e-5);
        assert!((tr.surface_uv.x - 0.25).abs() < 1e-5);
        assert!((tr.surface_uv.y - 0.75).abs() < 1e-5);
        assert!((tr.surface_normal.i - 0.6).abs() < 1e-5);
        assert!((tr.surface_normal.j - 0.8).abs() < 1e-5);
    }

    /// `point_in_triangle3d`: centroid of an XY-plane right triangle
    /// `(0,0,0)-(1,0,0)-(0,1,0)` produces `(s, t) ≈ (1/3, 1/3)` and a
    /// positive return value (point inside).
    #[test]
    fn point_in_triangle3d_centroid_inside() {
        let p0 = RealPoint3d { x: 0.0, y: 0.0, z: 0.0 };
        let p1 = RealPoint3d { x: 1.0, y: 0.0, z: 0.0 };
        let p2 = RealPoint3d { x: 0.0, y: 1.0, z: 0.0 };
        let point = RealPoint3d { x: 1.0 / 3.0, y: 1.0 / 3.0, z: 0.0 };
        let (mut s, mut t) = (0.0, 0.0);
        let metric = geometry_sampling_point_in_triangle3d(point, p0, p1, p2, &mut s, &mut t);
        assert!(metric > 0.0, "metric should be > 0 inside; got {}", metric);
        assert!((s - 1.0 / 3.0).abs() < 1e-4, "s={}", s);
        assert!((t - 1.0 / 3.0).abs() < 1e-4, "t={}", t);
    }

    /// Point well outside the triangle in the same plane returns a negative
    /// metric and zeroed `(s, t)`.
    #[test]
    fn point_in_triangle3d_outside_returns_negative() {
        let p0 = RealPoint3d { x: 0.0, y: 0.0, z: 0.0 };
        let p1 = RealPoint3d { x: 1.0, y: 0.0, z: 0.0 };
        let p2 = RealPoint3d { x: 0.0, y: 1.0, z: 0.0 };
        let point = RealPoint3d { x: 2.0, y: 2.0, z: 0.0 };
        let (mut s, mut t) = (5.0, 5.0);
        let metric = geometry_sampling_point_in_triangle3d(point, p0, p1, p2, &mut s, &mut t);
        assert!(metric < 0.0, "metric should be < 0 outside; got {}", metric);
        assert_eq!(s, 0.0);
        assert_eq!(t, 0.0);
    }

    /// Z-up triangle: verify the dominant-axis projection picks Z and the
    /// 2D solve works (returns positive metric, valid barycentrics) on a
    /// point at a non-centroid barycentric.
    #[test]
    fn point_in_triangle3d_xy_plane_quarter_point() {
        let p0 = RealPoint3d { x: 0.0, y: 0.0, z: 5.0 };
        let p1 = RealPoint3d { x: 4.0, y: 0.0, z: 5.0 };
        let p2 = RealPoint3d { x: 0.0, y: 4.0, z: 5.0 };
        // Barycentric (s=0.25, t=0.5) → point = p0 + 0.25*(p1-p0) + 0.5*(p2-p0)
        //                              = (1, 2, 5)
        let point = RealPoint3d { x: 1.0, y: 2.0, z: 5.0 };
        let (mut s, mut t) = (0.0, 0.0);
        let metric = geometry_sampling_point_in_triangle3d(point, p0, p1, p2, &mut s, &mut t);
        assert!(metric > 0.0);
        assert!((s - 0.25).abs() < 1e-4, "s={}", s);
        assert!((t - 0.5).abs() < 1e-4, "t={}", t);
    }

    /// `get_structure_triangle`: 3-vertex extraction round-trips known
    /// vertex data through the indices.
    #[test]
    fn get_structure_triangle_round_trips_indices() {
        use blam_tags::render_model::{PrtVertexType, RenderVertex};
        let vertices = [
            RenderVertex {
                position: RealPoint3d { x: 10.0, y: 11.0, z: 12.0 },
                texcoord: RealPoint2d { x: 0.1, y: 0.2 },
                normal: RealVector3d { i: 0.0, j: 0.0, k: 1.0 },
                tangent: RealVector3d::default(),
                binormal: RealVector3d::default(),
                node_indices: [0; 4],
                node_weights: [0.0; 4],
                lightmap_texcoord: RealPoint2d::default(),
                vert_color: RealVector3d::default(),
            },
            RenderVertex {
                position: RealPoint3d { x: 20.0, y: 21.0, z: 22.0 },
                texcoord: RealPoint2d { x: 0.3, y: 0.4 },
                normal: RealVector3d { i: 1.0, j: 0.0, k: 0.0 },
                tangent: RealVector3d::default(),
                binormal: RealVector3d::default(),
                node_indices: [0; 4],
                node_weights: [0.0; 4],
                lightmap_texcoord: RealPoint2d::default(),
                vert_color: RealVector3d::default(),
            },
            RenderVertex {
                position: RealPoint3d { x: 30.0, y: 31.0, z: 32.0 },
                texcoord: RealPoint2d { x: 0.5, y: 0.6 },
                normal: RealVector3d { i: 0.0, j: 1.0, k: 0.0 },
                tangent: RealVector3d::default(),
                binormal: RealVector3d::default(),
                node_indices: [0; 4],
                node_weights: [0.0; 4],
                lightmap_texcoord: RealPoint2d::default(),
                vert_color: RealVector3d::default(),
            },
        ];
        // Index buffer with two extra prefix indices so last_index=4
        // points at the triangle (0, 1, 2). Engine reads
        // indices[last_index-2..=last_index] = indices[2..=4] = [0,1,2].
        let indices: [u32; 5] = [99, 99, 0, 1, 2];
        let mut tri = UncompressedTriangle::default();
        get_structure_triangle(&vertices, 4, &indices, &mut tri);
        assert_eq!(tri.indices, [0, 1, 2]);
        assert_eq!(tri.position[0].x, 10.0);
        assert_eq!(tri.position[1].x, 20.0);
        assert_eq!(tri.position[2].x, 30.0);
        assert_eq!(tri.texcoord[2].y, 0.6);
        assert_eq!(tri.normal[1].i, 1.0);
        let _ = PrtVertexType::None; // silence unused-import warning if any
    }

    /// `sample_lightprobe_textures` early-out: either atlas being `None`
    /// should leave all output buffers untouched.
    #[test]
    fn sample_lightprobe_textures_early_out_on_missing_atlas() {
        let mut sh_r = [99.0_f32; 36];
        let mut sh_g = [88.0_f32; 36];
        let mut sh_b = [77.0_f32; 36];
        let mut dir = RealVector3d { i: 1.0, j: 2.0, k: 3.0 };
        let mut intensity = RealRgbColor { red: 4.0, green: 5.0, blue: 6.0 };
        let mut contrast = 7.0_f32;

        // Both None → early return.
        sample_lightprobe_textures(
            0,
            None,
            None,
            RealPoint2d { x: 0.5, y: 0.5 },
            8,
            2,
            &[RealVector3d::default(); 9],
            &mut sh_r,
            &mut sh_g,
            &mut sh_b,
            &mut dir,
            &mut intensity,
            &mut contrast,
        );

        // Buffers preserved.
        assert_eq!(sh_r[0], 99.0);
        assert_eq!(sh_g[0], 88.0);
        assert_eq!(sh_b[0], 77.0);
        assert_eq!(dir.i, 1.0);
        assert_eq!(intensity.red, 4.0);
        assert_eq!(contrast, 7.0);
    }

    /// Build a synthetic DecodedAtlas with known per-layer pixel values and
    /// verify the engine `n==8` paired-pixel SH decode formula
    /// `(a + b - 1) * 2 * (a.alpha * b.alpha * cv.x)` reproduces byte-exact.
    /// 8 layers × 1×1 pixel × known RGBA → 4 SH coefs.
    #[test]
    fn sample_lightprobe_textures_paired_decode_matches_engine_formula() {
        // 8 layers, 1×1 each, RGBA pre-decoded.
        let mut atlas = DecodedAtlas {
            width: 1,
            height: 1,
            layers: Vec::with_capacity(8),
        };
        // Layer k → [r=0.1+k*0.05, g=0.2, b=0.3, a=0.5]
        for k in 0..8 {
            atlas
                .layers
                .push(vec![[0.1 + k as f32 * 0.05, 0.2, 0.3, 0.5]]);
        }
        // Single-layer intensity atlas (1-pixel mode → just channel reorder).
        let intensity_atlas = DecodedAtlas {
            width: 1,
            height: 1,
            layers: vec![vec![[0.4, 0.5, 0.6, 0.7]]],
        };
        let mut sh_r = [0.0_f32; 36];
        let mut sh_g = [0.0_f32; 36];
        let mut sh_b = [0.0_f32; 36];
        let mut dir = RealVector3d::default();
        let mut intensity = RealRgbColor::default();
        let mut contrast = 0.0_f32;
        let cv = [RealVector3d { i: 0.5, j: 0.0, k: 0.0 }; 9];

        sample_lightprobe_textures(
            0,
            Some(&atlas),
            Some(&intensity_atlas),
            RealPoint2d { x: 0.5, y: 0.5 },
            8,
            1,
            &cv,
            &mut sh_r,
            &mut sh_g,
            &mut sh_b,
            &mut dir,
            &mut intensity,
            &mut contrast,
        );

        // For k=0 pair (layers 0 + 1):
        //   pixel_a.red = 0.1, pixel_b.red = 0.15, a.alpha=b.alpha=0.5, cv[0].i=0.5
        //   scale = 0.5 * 0.5 * 0.5 = 0.125
        //   sh_red[0] = (0.1 + 0.15 - 1.0) * 2.0 * 0.125 = -0.1875
        assert!((sh_r[0] - (-0.1875)).abs() < 1e-6, "sh_r[0]={}", sh_r[0]);
        // sh_green[0] = (0.2 + 0.2 - 1.0) * 2.0 * 0.125 = -0.15
        assert!((sh_g[0] - (-0.15)).abs() < 1e-6, "sh_g[0]={}", sh_g[0]);
        // sh_blue[0] = (0.3 + 0.3 - 1.0) * 2.0 * 0.125 = -0.1
        assert!((sh_b[0] - (-0.1)).abs() < 1e-6, "sh_b[0]={}", sh_b[0]);

        // Intensity (1-pixel mode): alpha=0.7→red, blue=0.6→green, green=0.5→blue.
        assert!((intensity.red - 0.7).abs() < 1e-6);
        assert!((intensity.green - 0.6).abs() < 1e-6);
        assert!((intensity.blue - 0.5).abs() < 1e-6);
    }

    /// Lightmap UV pass-through + barycentric interp. Caller supplies a
    /// slice of pre-decoded `RealPoint2d`; vertex-0 barycentric reads out
    /// vertex 0's UV exactly.
    #[test]
    fn lightmap_uv_passthrough_and_interp() {
        let uv_data = [
            RealPoint2d { x: 0.0, y: 1.0 },
            RealPoint2d { x: 1.0, y: 0.0 },
            RealPoint2d { x: 0.5, y: 0.5 },
        ];
        let triangle = UncompressedTriangle {
            position: [RealPoint3d::default(); 3],
            texcoord: [RealPoint2d::default(); 3],
            normal: [RealVector3d::default(); 3],
            indices: [0, 1, 2],
        };
        let mut tr = GeometryTestResult::default();
        geometry_test_triangle(
            RealPoint3d::default(),
            &triangle,
            0.0,
            0.0,
            0,
            Some(&uv_data),
            None,
            -1,
            None,
            &mut tr,
            0,
            None,
        );
        assert!(tr.triangle_lightmap_uvs[0].x.abs() < 1e-5);
        assert!((tr.triangle_lightmap_uvs[0].y - 1.0).abs() < 1e-5);
        assert!(tr.lightmap_uv.x.abs() < 1e-5);
        assert!((tr.lightmap_uv.y - 1.0).abs() < 1e-5);
    }
}
