//! Engine source: `Ares/source/decorators/light_placement.{h,cpp}`.
//!
//! Verbatim port of the decorator-bake pipeline. Replaced the invented
//! `decorators/bake.rs` (deleted 2026-05-26) which used the now-removed
//! `lightmap_raycast.rs` XY-grid BVH. All ray casts now go through
//! [`c_geometry_sampler::sample`](crate::halo::geometry::geometry_sampling::sample).
//!
//! ## Pipeline
//!
//! Entry: [`initialize_for_new_structure_bsp`] (Reach symbol; tool.exe
//! `sub_140C4AE70`). For each BSP loaded, it calls TWO siblings in order:
//!
//! 1. [`structure_bsp_light_decorators_from_scenario`] (`sub_140C4BE80`) —
//!    walks clusters → decorator_blocks → placements, bakes each via
//!    [`light_placement`].
//! 2. [`structure_bsp_build_decorators_from_scenario`] (`sub_140C4C150`) —
//!    populates `decorator_runtime_block.model_start_index[]` so the
//!    renderer can route placements to the right model variation.
//!
//! See [`feedback_reach_name_tool_body_convention`] memory: function names
//! come from Reach symbols (preserved C++ exports); function BODIES come
//! from tool.exe (the MCC-authoritative decompile). The 2026-05-26 audit
//! found that Reach's entry point calls only `_light_` per BSP, while
//! tool.exe calls BOTH `_light_` + `_build_` — missing the second one
//! ships a decorator port with empty draw ranges.
//!
//! ## Sub-phase status
//!
//! - **Phase 8.0** (this commit): module scaffold + sample tables (verbatim
//!   from Reach IDA via the existing `bake.rs`) + `initialize_for_new_structure_bsp`
//!   entry point. The three siblings are stubbed with TODOs.
//! - **Phase 8.1**: `structure_bsp_light_decorators_from_scenario` body.
//! - **Phase 8.2**: `light_placement` body (the actual per-placement bake).
//! - **Phase 8.3**: `structure_bsp_build_decorators_from_scenario` body.

use crate::halo::bitmaps::atlas_decode::DecodedAtlas;
use crate::halo::geometry::geometry_sampling::{
    sample as c_geometry_sampler_sample, GeometrySample, GeometrySamplerOutcome,
};
use crate::halo::render::lighting_interface::sh_eval_direction;
use crate::halo::scenario::loader::LoadedScenario;
use blam_tags::decorator_set::{DecoratorSet, LightingSamplePattern};
use blam_tags::math::{RealPoint3d, RealVector3d};
use blam_tags::scenario::ScenarioDecoratorPlacement;
use glam::{Quat, Vec3};

// =============================================================================
// Sample tables — Reach `unk_8202BD58` (Ground) + `unk_8202BE70` (Hanging)
// =============================================================================
//
// Verbatim from Reach IDA database (port 13371) via the existing `bake.rs`.
// Each entry is 28 bytes: `(local_pos.xyz, local_dir.xyz, weight)` — 7 × 4 B.
// Engine `light_placement` reads one of these tables based on the decorator
// tag's `lighting_sample_pattern` byte (`unk_byte_50` in IDA): `Hanging` if
// `==1`, `Ground` otherwise.

/// One sample-pattern entry. Order + values verified against the Reach
/// disassembly 2026-05-07.
#[derive(Debug, Clone, Copy)]
pub struct SampleEntry {
    /// Local-space sample point as a unit-cube fraction (0..1). Engine
    /// transforms via `lerp(mesh_bounds_min, mesh_bounds_max, pos)`.
    pub pos: Vec3,
    /// Local-space ray direction for this sample (engine uses it for ray
    /// occlusion + ravi evaluation orientation).
    pub dir: Vec3,
    /// Per-sample weight used in the accumulated SH average.
    pub weight: f32,
}

/// Reach `unk_8202BD58` — Ground pattern (10 samples).
///
/// Engine reads when `lighting_sample_pattern != 1` (Ground or any non-
/// Hanging value). Verified 2026-05-07 via IDA bridge port 13371.
pub const GROUND_TABLE: [SampleEntry; 10] = [
    SampleEntry { pos: Vec3::new(0.500, 0.500, 0.900), dir: Vec3::new( 0.000,  0.000, -1.000), weight: 2.800 },
    SampleEntry { pos: Vec3::new(0.500, 0.500, 0.100), dir: Vec3::new( 0.000,  0.000,  1.000), weight: 2.000 },
    SampleEntry { pos: Vec3::new(0.500, 0.100, 0.500), dir: Vec3::new( 0.000,  1.000,  0.000), weight: 2.000 },
    SampleEntry { pos: Vec3::new(0.500, 0.900, 0.500), dir: Vec3::new( 0.000, -1.000,  0.000), weight: 2.000 },
    SampleEntry { pos: Vec3::new(0.100, 0.500, 0.500), dir: Vec3::new( 1.000,  0.000,  0.000), weight: 2.000 },
    SampleEntry { pos: Vec3::new(0.900, 0.500, 0.500), dir: Vec3::new(-1.000,  0.000,  0.000), weight: 2.000 },
    SampleEntry { pos: Vec3::new(0.900, 0.500, 0.900), dir: Vec3::new( 0.000,  0.000, -1.000), weight: 2.800 },
    SampleEntry { pos: Vec3::new(0.100, 0.500, 0.900), dir: Vec3::new( 0.000,  0.000, -1.000), weight: 2.800 },
    SampleEntry { pos: Vec3::new(0.500, 0.900, 0.900), dir: Vec3::new( 0.000,  0.000, -1.000), weight: 2.800 },
    SampleEntry { pos: Vec3::new(0.500, 0.100, 0.900), dir: Vec3::new( 0.000,  0.000, -1.000), weight: 2.800 },
];

/// Reach `unk_8202BE70` — Hanging pattern (10 samples).
///
/// Engine reads when `lighting_sample_pattern == 1`. Verified 2026-05-07.
pub const HANGING_TABLE: [SampleEntry; 10] = [
    SampleEntry { pos: Vec3::new(0.100, 0.500, 0.900), dir: Vec3::new( 1.000,  0.000, -0.100), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.900, 0.500, 0.900), dir: Vec3::new(-1.000,  0.000, -0.100), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.500, 0.100, 0.900), dir: Vec3::new( 0.000,  1.000, -0.100), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.500, 0.900, 0.900), dir: Vec3::new( 0.000, -1.000, -0.100), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.100, 0.100, 0.800), dir: Vec3::new( 0.700,  0.700,  0.300), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.900, 0.900, 0.800), dir: Vec3::new(-0.700, -0.700,  0.300), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.100, 0.900, 0.800), dir: Vec3::new( 0.700, -0.700,  0.300), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.900, 0.100, 0.800), dir: Vec3::new(-0.700,  0.700,  0.300), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.500, 0.500, 0.500), dir: Vec3::new( 0.000,  0.000,  1.000), weight: 1.500 },
    SampleEntry { pos: Vec3::new(0.500, 0.500, 0.000), dir: Vec3::new( 0.000,  0.000,  1.000), weight: 1.500 },
];

// =============================================================================
// initialize_for_new_structure_bsp — `tool.exe sub_140C4AE70` (Phase 8.0)
// =============================================================================

/// `s_decorator_lightmap_sampler::initialize_for_new_structure_bsp` @ tool.exe
/// `sub_140C4AE70`. Reach equivalent symbol `?initialize_for_new_structure_bsp@s_decorator_lightmap_sampler@@SAXF@Z`
/// @ Reach `0x82797880` — **CRITICAL DIVERGENCE: Reach calls only `_light_`
/// per BSP; tool.exe calls BOTH `_light_` AND `_build_`.** Reach's stripped
/// build is wrong for MCC; we follow tool.exe.
///
/// Engine signature: `(short bsp_mask)` where each bit `i` indicates that
/// BSP slot `i` (0..16) needs re-baking. For protomorph the global 16-slot
/// pool doesn't exist — callers pass a single BSP index and we iterate
/// `scenario.active_bsps` to find the match. The bitmask form can be added
/// later as a wrapper if needed.
#[allow(clippy::too_many_arguments)]
pub fn initialize_for_new_structure_bsp(
    scenario: &mut LoadedScenario,
    bsp_index: i32,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
    mesh_bounds_resolver: impl Fn(usize, usize, &DecoratorSet, &ScenarioDecoratorPlacement) -> Option<MeshBounds>,
) {
    // Validate that the BSP is active (engine's `block[3] != -1` sentinel).
    if !scenario
        .active_bsps
        .iter()
        .any(|b| b.scenario_bsp_index == bsp_index as usize)
    {
        return;
    }

    // Engine line 27-28: call _light_ then _build_ per BSP, in that order.
    // The order matters — `_build_` reads `decorator_block.placements[i].byte[7]`
    // (sort key) which `_light_` populates as part of the HDR-encoded color
    // alpha byte. Running _build_ before _light_ would sort against garbage.
    structure_bsp_light_decorators_from_scenario(
        scenario,
        bsp_index,
        lightprobe_atlas,
        intensity_atlas,
        lightprobe_pixels_per_probe,
        dominant_pixels_per_probe,
        mesh_bounds_resolver,
    );
    structure_bsp_build_decorators_from_scenario(scenario, bsp_index);
}

// =============================================================================
// structure_bsp_light_decorators_from_scenario — `tool.exe sub_140C4BE80` (Phase 8.1)
// =============================================================================

/// `structure_bsp_light_decorators_from_scenario` @ tool.exe `sub_140C4BE80`.
/// Reach equivalent `?structure_bsp_light_decorators_from_scenario@@YA_NJFPAUscenario@@@Z`
/// @ Reach `0x82797938`.
///
/// Engine walks `structure_bsp.clusters[]` → `cluster.decorator_blocks[]` →
/// `decorator_block.placements[]`. The per-cluster grouping is a cache-build
/// artifact: at cache-build time tool.exe bins placements by their containing
/// cluster + decorator_set into per-cluster `s_decorator_runtime_block`s
/// (60 B) and 16-byte `s_decorator_runtime_placement`s. Each cluster gets a
/// `processed` bit (`flags & 0x10`) to make the bake idempotent across BSPs
/// that share clusters.
///
/// **Protomorph adaptation:** we don't replicate the cache-build per-cluster
/// binning. Scenario tags hold placements flat as
/// `scenario.decorators[block].sets[set].placements: Vec<ScenarioDecoratorPlacement>`,
/// so we walk THAT shape and call [`light_placement`] per placement. The
/// per-cluster bookkeeping (idempotency bit, profiling tick) is dropped
/// since protomorph runs the bake once at scenario load.
///
/// Returns `true` per engine convention (engine line 128 unconditionally
/// returns 1). Per-block success counts are logged when any fail.
///
/// **Two-phase split-borrow pattern:** we can't simultaneously hold
/// `&mut LoadedScenario` (for the per-placement mutation) AND
/// `&LoadedScenario` (which [`light_placement`] needs to call
/// `c_geometry_sampler::sample`). Phase A walks immutably and collects
/// `(block_idx, set_idx, placement_idx, baked_color)` tuples; Phase B
/// writes them back + truncates failed entries.
///
/// `mesh_bounds_resolver` — caller-provided closure mapping
/// `(block_idx, set_idx, &DecoratorSet) → MeshBounds`. The bounds come
/// from the decorator's `.render_model` tag's subpart AABB which lives
/// outside `LoadedScenario`. Return `None` to skip placements whose mesh
/// bounds can't be resolved.
pub fn structure_bsp_light_decorators_from_scenario(
    scenario: &mut LoadedScenario,
    bsp_index: i32,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
    mesh_bounds_resolver: impl Fn(usize, usize, &DecoratorSet, &ScenarioDecoratorPlacement) -> Option<MeshBounds>,
) -> bool {
    if !scenario
        .active_bsps
        .iter()
        .any(|b| b.scenario_bsp_index == bsp_index as usize)
    {
        return false;
    }

    // Phase A — immutable walk, collect bake results.
    let mut baked: Vec<(usize, usize, usize, Option<BakedDecoratorColor>)> = Vec::new();
    {
        let scenario_ref: &LoadedScenario = scenario;
        for (block_idx, decorator_block) in scenario_ref.scenario.decorators.iter().enumerate() {
            for (set_idx, set_entry) in decorator_block.sets.iter().enumerate() {
                let set_tag = match scenario_ref
                    .decorator_sets
                    .get(block_idx)
                    .and_then(|sets| sets.get(set_idx))
                    .and_then(|opt| opt.as_ref())
                {
                    Some(t) => t,
                    None => continue,
                };
                for (p_idx, placement) in set_entry.placements.iter().enumerate() {
                    let mesh_bounds = match mesh_bounds_resolver(block_idx, set_idx, set_tag, placement) {
                        Some(b) => b,
                        None => continue,
                    };
                    let result = light_placement(
                        placement,
                        set_tag,
                        mesh_bounds,
                        bsp_index,
                        scenario_ref,
                        lightprobe_atlas,
                        intensity_atlas,
                        lightprobe_pixels_per_probe,
                        dominant_pixels_per_probe,
                    );
                    baked.push((block_idx, set_idx, p_idx, result));
                }
            }
        }
    }

    // Phase B — apply results. Engine compacts: successful placements write
    // forward, failed ones are dropped. Compact in-place per (block, set).
    // We also write each `tint_color` to the final baked RGB so the renderer
    // can read it (engine writes RGBE bytes 12-15 in place).
    use std::collections::HashMap;
    let mut grouped: HashMap<(usize, usize), Vec<(usize, Option<BakedDecoratorColor>)>> =
        HashMap::new();
    for (b, s, p, color) in baked {
        grouped.entry((b, s)).or_default().push((p, color));
    }

    // PROTOMORPH_DIAG_BAKE_HIST=1 — per-(block,set) alpha + black-output
    // histogram, printed after compaction. Used to confirm the
    // "stubbed m_diffuse + authored alpha=1 → black placement" hypothesis
    // from the 2026-05-26 chain audit before committing to a Phase 7b.6c port.
    let hist_diag = std::env::var("PROTOMORPH_DIAG_BAKE_HIST")
        .ok()
        .as_deref() == Some("1");

    for ((block_idx, set_idx), mut entries) in grouped {
        entries.sort_by_key(|(p, _)| *p);
        let placements = &mut scenario.scenario.decorators[block_idx].sets[set_idx].placements;
        let total = placements.len();

        // Pre-compaction stats (alpha distribution + black-output count) read
        // straight from the entries vector + placement.ground_tint.
        let mut alpha_zero = 0usize;
        let mut alpha_low = 0usize;
        let mut alpha_high = 0usize;
        let mut alpha_one = 0usize;
        let mut black_out = 0usize;
        let mut dropped = 0usize;
        if hist_diag {
            for (read_idx, color_opt) in entries.iter() {
                let placement = &placements[*read_idx];
                let alpha = (placement.ground_tint as i32 & 0xFF) as f32 / 255.0;
                if alpha <= 0.0 {
                    alpha_zero += 1;
                } else if alpha < 0.5 {
                    alpha_low += 1;
                } else if alpha < 0.999 {
                    alpha_high += 1;
                } else {
                    alpha_one += 1;
                }
                match color_opt {
                    None => dropped += 1,
                    Some(c) => {
                        if c.rgb.length_squared() < 1e-4 {
                            black_out += 1;
                        }
                    }
                }
            }
        }

        let mut write_idx = 0;
        for (read_idx, color_opt) in entries.iter() {
            if let Some(color) = color_opt {
                if write_idx != *read_idx {
                    placements.swap(write_idx, *read_idx);
                }
                // Engine writes RGBE bytes 12-15 to the runtime placement.
                // Protomorph stores the linear-RGB form (`color.rgb`) in
                // `placement.tint_color`; the renderer re-encodes via
                // [`hdr_encode`] at GPU-instance-buffer build time. The
                // authored `placement.ground_tint` blend factor is
                // preserved (a separate field from engine byte 15).
                placements[write_idx].tint_color = RealPoint3d {
                    x: color.rgb.x,
                    y: color.rgb.y,
                    z: color.rgb.z,
                };
                write_idx += 1;
            }
        }
        if write_idx < total {
            eprintln!(
                "decorators: could not light {} / {} decorators in block {} set {}",
                total - write_idx,
                total,
                block_idx,
                set_idx,
            );
            placements.truncate(write_idx);
        }

        if hist_diag {
            let set_path = scenario
                .scenario
                .decorators
                .get(block_idx)
                .and_then(|b| b.sets.get(set_idx))
                .map(|s| s.decorator_set.as_str())
                .unwrap_or("?");
            eprintln!(
                "[bake-hist] block={block_idx} set={set_idx} ({set_path}) \
                 n={total} alpha[=0={alpha_zero} 0<a<.5={alpha_low} .5<=a<1={alpha_high} =1={alpha_one}] \
                 black_rgb={black_out} dropped={dropped}",
            );
        }
    }

    true
}

// =============================================================================
// light_placement — STUB Phase 8.2
// =============================================================================

/// `light_placement` @ tool.exe `sub_140C4AF00`. The per-placement HDR-color
/// bake — the heart of decorator lighting.
///
/// Engine flow (paraphrased):
/// 1. Degamma placement RGB bytes 12-14 (`v9`, `v12`, `v15`) — the
///    authored tint that the bake modulates.
/// 2. If `do_bake` flag is set:
///    - Resolve decorator tag → model tag → bounds
///    - Decompress quaternion (4 × i8 → 4 floats / sqrt(2)/127)
///    - Transform local center (= 0.5,0.5,0.5 in bounds) → world center
///    - Compute world bounding sphere radius
///    - For each of 10 sample-pattern entries (Ground or Hanging):
///        - Transform local sample pos via quaternion + offset → world pos
///        - Cast ray via [`crate::halo::geometry::geometry_sampling::sample`]
///        - On hit: accumulate SH × weight, weight = 1/(dist + radius)
///    - If total_weight <= 0 → return false (drop placement)
///    - Ravi-eval SH at +Z → ground color
///    - If decorator's `lighting_sample_pattern_byte[49] == 3`:
///        compute ground-tint scale + overwrite placement byte 6
///    - Final RGB = (ravi_color × alpha) + (degamma_albedo × (1 - alpha))
/// 3. RGBE-encode the final RGB into placement bytes 12-15.
///
/// Mesh AABB used as the sample-pattern reference frame. Caller computes
/// from the render_model subpart bounds (`mode.regions[0].permutations[0]`,
/// engine reads `model + 29*sizeof_pointer + i*44` for the bounds entry).
#[derive(Debug, Clone, Copy)]
pub struct MeshBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl MeshBounds {
    pub fn extent(&self) -> Vec3 { self.max - self.min }
    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
}

/// Output of [`light_placement`]: final HDR color baked at the placement's
/// world position. Engine writes back to placement bytes 12-15 (RGBE);
/// protomorph keeps both forms so the renderer can use whichever fits.
#[derive(Debug, Clone, Copy, Default)]
pub struct BakedDecoratorColor {
    /// Final linear-RGB color after bake (brightness × tint blend).
    pub rgb: Vec3,
    /// RGBE-encoded form matching engine output bytes 12-15.
    pub hdr_encoded: [u8; 4],
    /// Ground-tint scale ∈ [0, 1] when set's tint pattern is "GroundTint"
    /// (engine `byte[49] == 3`). Maps to placement byte 6 in engine output.
    pub ground_tint: u8,
    /// Diag-only: per-channel `bright_r/g/b` = ravi_order_3 × inv_weight,
    /// the lighting contribution before the albedo blend. Lets the
    /// renderer split "low bake brightness" from "low authored albedo".
    pub diag_brightness: Vec3,
    /// Diag-only: authored `placement.tint_color` (= `albedo_lin` in
    /// the bake), so the renderer can pair it with `diag_brightness`.
    pub diag_albedo: Vec3,
    /// Diag-only: per-placement sample hit count (out of 10 ray casts).
    /// Misses imply our ray lengths or origins are off vs engine.
    pub diag_sample_hits: u8,
}

/// `light_placement` @ tool.exe `sub_140C4AF00` — per-placement HDR-color bake.
///
/// Casts 10 sample rays through [`c_geometry_sampler::sample`] over the
/// pattern selected by `set.lighting_sample_pattern`, accumulates SH ×
/// distance-weight, evaluates ravi at +Z for final brightness, blends with
/// the authored albedo by `placement.alpha`, RGBE-encodes the result.
///
/// **Returns** `Some(BakedDecoratorColor)` on success (≥ 1 ray hit);
/// `None` when no rays connected (engine returns `false` → walker drops
/// the placement).
///
/// **Signature differences from engine:**
/// - Engine mutates the 16-byte runtime placement in place (bytes 12-15).
///   Protomorph returns the baked color via [`BakedDecoratorColor`] and
///   lets the caller write it back.
/// - Engine takes the cluster's material tint params (`a4` = `v36`, 16 B)
///   and per-channel tint thresholds (`a5` = `v35`). The ground-tint
///   reduction path (engine `byte[49] == 3`) is **deferred for now** —
///   set `BakedDecoratorColor::ground_tint = 0` until material tint
///   lookup is wired through the walker.
#[allow(clippy::too_many_arguments)]
pub fn light_placement(
    placement: &ScenarioDecoratorPlacement,
    set: &DecoratorSet,
    mesh_bounds: MeshBounds,
    bsp_index: i32,
    scenario: &LoadedScenario,
    lightprobe_atlas: Option<&DecodedAtlas>,
    intensity_atlas: Option<&DecodedAtlas>,
    lightprobe_pixels_per_probe: u8,
    dominant_pixels_per_probe: u8,
) -> Option<BakedDecoratorColor> {
    // Engine `compress_placement` @ Reach `0x82ff0868` writes
    //   runtime.byte[12..14] = u8(255 × value_regamma(scenario.tint_color))
    // i.e. LINEAR → sRGB byte. Then `light_placement` reads back as
    //   v9 = value_degamma(byte / 255.0)
    // i.e. sRGB byte → LINEAR. Net round-trip is identity:
    // `scenario.tint_color` is stored LINEAR, engine `albedo_lin` equals
    // it directly (mod u8 quantization noise).
    //
    // **Previous port erroneously called `degamma(placement.tint_color)`**
    // — applying an extra sRGB→linear curve, darkening non-(1,1,1) tints
    // (e.g. 0.8 → pow(0.8,2.2) ≈ 0.61). Bright `(1,1,1)` placements
    // unaffected, dim placements darkened ~1.6×, exaggerating the
    // bright-vs-dim contrast.
    let albedo_lin = Vec3::new(
        placement.tint_color.x,
        placement.tint_color.y,
        placement.tint_color.z,
    );
    // Engine `v119 = byte[15] / 255` — alpha as a linear blend factor
    // between `avg_diffuse` (ground-bounce color) and `albedo` (painted tint).
    //
    // Engine `light_placement` reads byte[15] of the RUNTIME placement.
    // `compress_placement` (Reach `0x82FF0868:114`) writes runtime byte[15]
    // = authoring.ground_tint (pass-through). So at bake-time:
    //   runtime.byte[15] = authoring.ground_tint
    //   v119 = authoring.ground_tint / 255
    // Protomorph: `placement.ground_tint` is i8 in [-128, 127]; we map to
    // [0, 1] same way the engine maps the u8.
    let alpha = (placement.ground_tint as i32 & 0xFF) as f32 / 255.0;

    // Engine lines 151-158: pick sample table by lighting_sample_pattern.
    let table = match set.lighting_sample_pattern {
        LightingSamplePattern::Hanging => &HANGING_TABLE,
        LightingSamplePattern::Ground => &GROUND_TABLE,
    };

    // Engine lines 162-201: decompress placement transform.
    let placement_pos = Vec3::new(
        placement.position.x,
        placement.position.y,
        placement.position.z,
    );
    let quat = Quat::from_xyzw(
        placement.rotation.i,
        placement.rotation.j,
        placement.rotation.k,
        placement.rotation.w,
    )
    .normalize();
    let placement_scale = if placement.scale > 0.0 { placement.scale } else { 1.0 };

    // Engine lines 163-205: mesh extent + sample radius.
    let mesh_extent = mesh_bounds.extent();
    let mesh_min = mesh_bounds.min;
    let mesh_center_local = mesh_bounds.center();

    // Engine v113 = sqrt(x²+y²+z²) × 0.5 — NO placement_scale.
    // Scale enters separately via the ray-length multiplication below
    // (engine `v56 = v113 × sample.weight × v45` where v45 = scale).
    let radius = (mesh_extent.length() * 0.5).max(1e-3);
    // Engine v114 = max(v113 × 0.3, 1e-4) — also NO placement_scale.
    let r_threshold = (radius * 0.3).max(1e-4);

    // Engine line 196-204: bounding sphere world center for visibility precast.
    let bsphere_world = placement_pos + quat * (mesh_center_local * placement_scale);

    // Engine line 211: `dctr.flags & 2 == 0` → "don't sample through geometry"
    let visibility_gate = set.dont_sample_lighting_through_geometry();

    // Engine lines 207-219: zero SH + diffuse + normal accumulators.
    //
    // Engine accumulates `m_diffuse` (offset 444 in s_geometry_sample) and
    // `m_normal` (offset 456) — NOT m_dominant_light_intensity / m_dominant_light_dir.
    // Confirmed by tracing tool.exe `v131[36..41]` offsets + Reach's
    // sh_scale_add(v94, ...) + point_from_line3d(v92, ...) / (v91, ...) calls.
    // Audit fix 2026-05-26.
    let mut acc_sh_r = [0.0_f32; 9];
    let mut acc_sh_g = [0.0_f32; 9];
    let mut acc_sh_b = [0.0_f32; 9];
    let mut acc_diffuse = Vec3::ZERO;
    let mut acc_normal = Vec3::ZERO;
    let mut total_weight = 0.0_f32;
    let mut diag_hits = 0u8;

    // Engine lines 220-270: sample loop.
    for sample in table {
        // Engine lines 224-228: local sample position = mesh_min + sample.pos × extent.
        let local_pos = mesh_min + sample.pos * mesh_extent;
        let world_sample_pos = placement_pos + quat * (local_pos * placement_scale);
        let world_dir = quat * sample.dir;
        // Engine: v56 = v113 × sample.weight × v45 (= radius × sample.weight × placement_scale)
        let ray_dir_world = world_dir * (radius * sample.weight * placement_scale);

        // Engine lines 298-305: visibility precast (Reach decompile clarifies):
        //   if (dctr.render_flags & 2) v98 = c_geom_sampler::sample(bsphere → sample);
        //   if (v98) v98 = c_geom_sampler::sample(sample, ray_dir);
        //   if (!v98) accumulate;
        //
        // Engine USES the precast hit's data when it succeeds — falls through
        // to main only if precast returned non-Success. NOT a "skip if
        // occluded" gate.
        let mut hit_sample = GeometrySample::default();
        let mut got_hit = false;

        if visibility_gate {
            let precast_dir = world_sample_pos - bsphere_world;
            let outcome = c_geometry_sampler_sample(
                scenario,
                lightprobe_atlas,
                intensity_atlas,
                lightprobe_pixels_per_probe,
                dominant_pixels_per_probe,
                RealPoint3d { x: bsphere_world.x, y: bsphere_world.y, z: bsphere_world.z },
                RealVector3d { i: precast_dir.x, j: precast_dir.y, k: precast_dir.z },
                /*ignore_all_objects*/ true,
                /*ignore_object_index*/ -1,
                /*light_probe*/ true,
                /*diffuse*/ true,
                &mut hit_sample,
            );
            if outcome == GeometrySamplerOutcome::Success {
                got_hit = true;
            }
        }

        if !got_hit {
            let outcome = c_geometry_sampler_sample(
                scenario,
                lightprobe_atlas,
                intensity_atlas,
                lightprobe_pixels_per_probe,
                dominant_pixels_per_probe,
                RealPoint3d {
                    x: world_sample_pos.x,
                    y: world_sample_pos.y,
                    z: world_sample_pos.z,
                },
                RealVector3d {
                    i: ray_dir_world.x,
                    j: ray_dir_world.y,
                    k: ray_dir_world.z,
                },
                /*ignore_all_objects*/ true,
                /*ignore_object_index*/ -1,
                /*light_probe*/ true,
                /*diffuse*/ true,
                &mut hit_sample,
            );
            if outcome == GeometrySamplerOutcome::Success {
                got_hit = true;
            }
        }

        if !got_hit {
            continue;
        }

        // Per-sample diag.
        let per_sample_diag = std::env::var("PROTOMORPH_DIAG_BAKE_AT").ok().and_then(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 4 { return None; }
            let p: Vec<f32> = parts.iter().filter_map(|v| v.trim().parse().ok()).collect();
            if p.len() != 4 { return None; }
            let dx = placement_pos.x - p[0];
            let dy = placement_pos.y - p[1];
            let dz = placement_pos.z - p[2];
            if dx*dx + dy*dy + dz*dz <= p[3]*p[3] { Some(()) } else { None }
        }).is_some();
        if per_sample_diag {
            eprintln!(
                "[sample-diag]   ray dir=({:.2},{:.2},{:.2}) → sh_r[0..4]=({:.4},{:.4},{:.4},{:.4}) normal=({:.3},{:.3},{:.3}) sample_point=({:.2},{:.2},{:.2})",
                ray_dir_world.x, ray_dir_world.y, ray_dir_world.z,
                hit_sample.light_probe_r[0], hit_sample.light_probe_r[1], hit_sample.light_probe_r[2], hit_sample.light_probe_r[3],
                hit_sample.normal.i, hit_sample.normal.j, hit_sample.normal.k,
                hit_sample.sample_point.x, hit_sample.sample_point.y, hit_sample.sample_point.z,
            );
        }

        // Engine line 310 (Reach) / 245 (tool.exe): drop-zero gate. Reach is
        // explicit: `if (v104[0] > 0 || v104[9] > 0 || v104[18] > 0)` where
        // v104 is the flat sh array [sh_r[0..9], sh_g[0..9], sh_b[0..9]] —
        // so this is `sh_r[0] > 0 || sh_g[0] > 0 || sh_b[0] > 0`.
        if hit_sample.light_probe_r[0] <= 0.0
            && hit_sample.light_probe_g[0] <= 0.0
            && hit_sample.light_probe_b[0] <= 0.0
        {
            continue;
        }

        // Engine line 247-251: distance from placement world position to hit
        // (engine uses placement world position, NOT the per-sample ray origin).
        let hit_world = Vec3::new(
            hit_sample.sample_point.x,
            hit_sample.sample_point.y,
            hit_sample.sample_point.z,
        );
        let distance = (hit_world - placement_pos).length();
        let weight = 1.0 / (distance + r_threshold);

        // Engine line 252-264: weighted accumulate.
        for k in 0..9 {
            acc_sh_r[k] += hit_sample.light_probe_r[k] * weight;
            acc_sh_g[k] += hit_sample.light_probe_g[k] * weight;
            acc_sh_b[k] += hit_sample.light_probe_b[k] * weight;
        }
        // diffuse (offset 444 in s_geometry_sample) × weight
        acc_diffuse.x += hit_sample.diffuse.i * weight;
        acc_diffuse.y += hit_sample.diffuse.j * weight;
        acc_diffuse.z += hit_sample.diffuse.k * weight;
        // normal (offset 456) × weight
        acc_normal.x += hit_sample.normal.i * weight;
        acc_normal.y += hit_sample.normal.j * weight;
        acc_normal.z += hit_sample.normal.k * weight;

        total_weight += weight;
        diag_hits += 1;
    }

    // ── Diag: dump bake state for placements near a target world position. ──
    // Trigger via `PROTOMORPH_DIAG_BAKE_AT="x,y,z,radius"` env var. Logs the
    // per-placement post-loop state (placement_pos, total_weight, accumulated
    // SH/diffuse/normal, alpha, albedo_lin) so we can compare against the
    // engine's expected values for a specific placement.
    let diag = std::env::var("PROTOMORPH_DIAG_BAKE_AT").ok().and_then(|s| {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 4 { return None; }
        let p: Vec<f32> = parts.iter().filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() != 4 { return None; }
        let dx = placement_pos.x - p[0];
        let dy = placement_pos.y - p[1];
        let dz = placement_pos.z - p[2];
        if dx * dx + dy * dy + dz * dz <= p[3] * p[3] { Some(()) } else { None }
    }).is_some();
    if diag {
        eprintln!(
            "[bake-diag] placement pos=({:.3},{:.3},{:.3}) bsp={bsp_index} \
             albedo_lin=({:.4},{:.4},{:.4}) alpha={alpha:.4} \
             total_weight={total_weight:.4} \
             mesh_extent=({:.3},{:.3},{:.3}) radius={radius:.4} r_threshold={r_threshold:.4} \
             vis_gate={visibility_gate}",
            placement_pos.x, placement_pos.y, placement_pos.z,
            albedo_lin.x, albedo_lin.y, albedo_lin.z,
            mesh_extent.x, mesh_extent.y, mesh_extent.z,
        );
        eprintln!(
            "[bake-diag]   acc_sh_r[0..4]=({:.4},{:.4},{:.4},{:.4}) acc_sh_g[0..4]=({:.4},{:.4},{:.4},{:.4})",
            acc_sh_r[0], acc_sh_r[1], acc_sh_r[2], acc_sh_r[3],
            acc_sh_g[0], acc_sh_g[1], acc_sh_g[2], acc_sh_g[3],
        );
        eprintln!(
            "[bake-diag]   acc_diffuse=({:.4},{:.4},{:.4}) acc_normal=({:.4},{:.4},{:.4})",
            acc_diffuse.x, acc_diffuse.y, acc_diffuse.z,
            acc_normal.x, acc_normal.y, acc_normal.z,
        );
    }

    // Engine line 277-278: no hits → return false (placement dropped).
    if total_weight <= 0.0 {
        if diag { eprintln!("[bake-diag]   DROPPED (total_weight=0)"); }
        return None;
    }

    // Engine line 308 / Reach line 356: `normalize3d(&acc_normal)` — normalize
    // accumulated SURFACE NORMAL (not dom_dir).
    let normal_normalized = if acc_normal.length_squared() > 1e-12 {
        acc_normal.normalize()
    } else {
        Vec3::Z
    };

    // Engine lines 309-314 / Reach line 359: ravi at accumulated_normal over
    // accumulated SH per channel.
    let inv_weight = 1.0 / total_weight;
    let bright_r = ravi_order_3(normal_normalized, &acc_sh_r).max(0.0) * inv_weight;
    let bright_g = ravi_order_3(normal_normalized, &acc_sh_g).max(0.0) * inv_weight;
    let bright_b = ravi_order_3(normal_normalized, &acc_sh_b).max(0.0) * inv_weight;

    // Engine lines 321-324: blend brightness × (alpha × avg_diffuse + (1-alpha) × degamma_albedo).
    // Engine's m_diffuse comes from `sample_diffuse_textures` which is deferred
    // (Phase 7b.6c). Until that lands, hit_sample.diffuse is always (0,0,0)
    // so `avg_diffuse = 0` and the blend reduces to `degamma_albedo × (1-alpha)`.
    let avg_diffuse = acc_diffuse * inv_weight;
    let blended_r = (avg_diffuse.x * alpha) + (albedo_lin.x * (1.0 - alpha));
    let blended_g = (avg_diffuse.y * alpha) + (albedo_lin.y * (1.0 - alpha));
    let blended_b = (avg_diffuse.z * alpha) + (albedo_lin.z * (1.0 - alpha));

    // Engine lines 326-328: final RGB = blended × brightness.
    let final_rgb = Vec3::new(
        blended_r * bright_r,
        blended_g * bright_g,
        blended_b * bright_b,
    );

    let encoded = hdr_encode(final_rgb);

    if diag {
        eprintln!(
            "[bake-diag]   normal_normalized=({:.4},{:.4},{:.4}) bright=({:.5},{:.5},{:.5})",
            normal_normalized.x, normal_normalized.y, normal_normalized.z,
            bright_r, bright_g, bright_b,
        );
        eprintln!(
            "[bake-diag]   blended=({:.4},{:.4},{:.4}) final_rgb=({:.4},{:.4},{:.4}) rgbe=[{},{},{},{}]",
            blended_r, blended_g, blended_b,
            final_rgb.x, final_rgb.y, final_rgb.z,
            encoded[0], encoded[1], encoded[2], encoded[3],
        );
    }

    Some(BakedDecoratorColor {
        rgb: final_rgb,
        hdr_encoded: encoded,
        ground_tint: 0, // TODO: engine `v23[49] == 3` ground-tint branch.
        diag_brightness: Vec3::new(bright_r, bright_g, bright_b),
        diag_albedo: albedo_lin,
        diag_sample_hits: diag_hits,
    })
}

// =============================================================================
// SH + color helpers (Ares math/spherical_harmonics.cpp + math/color_math.cpp)
// =============================================================================

/// `value_degamma` @ Ares `math/color_math.cpp`. sRGB → linear via gamma 2.2.
/// Halo uses a piecewise sRGB curve; pow(2.2) is within 2% across [0,1].
fn degamma(c: f32) -> f32 {
    c.max(0.0).powf(2.2)
}

/// `ravi_order_3` — Lambertian-cosine-windowed SH3 evaluation. Engine
/// `sub_140984AD0`. Verified constants from engine decompile 2026-05-26:
///
/// ```text
/// engine L0 coef: 0.28209499 = Y_0           = basis[0] × 1
/// engine L1 coef: 0.32573524 = Y_1 × (2/3)   = basis[1..4] × (2/3)
/// engine L2 coefs:                           = basis[4..9] × (1/4)
/// ```
///
/// Per-band weights match Lambertian-irradiance-divided-by-π:
/// `A_l/π` = `{1, 2/3, 1/4}` for SH bands 0, 1, 2.
///
/// **Previous port omitted the band weights** (used a flat
/// `Σ basis × sh`) — that over-weighted L1 by 1.5× and L2 by 4× vs
/// engine, producing too-bright placements with high-frequency SH
/// content and roughly-correct ones with L0-dominated SH.
pub fn ravi_order_3(dir: Vec3, sh: &[f32; 9]) -> f32 {
    let basis = sh_eval_direction(3, dir);
    // L0 (band 0): weight 1
    let mut sum = basis[0] * sh[0];
    // L1 (bands 1, 2, 3): weight 2/3
    let l1_weight = 2.0_f32 / 3.0_f32;
    for k in 1..4 {
        sum += basis[k] * sh[k] * l1_weight;
    }
    // L2 (bands 4..9): weight 1/4
    let l2_weight = 0.25_f32;
    for k in 4..9 {
        sum += basis[k] * sh[k] * l2_weight;
    }
    sum
}

/// `real_rgb_color_to_pixel32_RGBE` @ Ares `math/color_math.cpp:1519`.
/// Quarter-step exponent format: `decoded = enc.rgb × exp2((enc.a − 127) / 4)`.
pub fn hdr_encode(rgb: Vec3) -> [u8; 4] {
    let peak = rgb.x.max(rgb.y).max(rgb.z).max(1e-30);
    let e = ((4.0 * peak.log2()).ceil() as i32).clamp(-127, 128);
    let scale = 2.0_f32.powf(e as f32 * 0.25);
    let inv = 1.0 / scale;
    let r = (rgb.x * inv * 255.0).round().clamp(0.0, 255.0) as u8;
    let g = (rgb.y * inv * 255.0).round().clamp(0.0, 255.0) as u8;
    let b = (rgb.z * inv * 255.0).round().clamp(0.0, 255.0) as u8;
    let a = ((e + 127).clamp(0, 255)) as u8;
    [r, g, b, a]
}

// =============================================================================
// structure_bsp_build_decorators_from_scenario — STUB Phase 8.3
// =============================================================================

/// `structure_bsp_build_decorators_from_scenario` @ tool.exe `sub_140C4C150`.
/// Reach symbol `?structure_bsp_build_decorators_from_scenario@@YA_NJFPAUscenario@@@Z`
/// @ Reach `0x82fecd90` (decompile fails — use tool.exe body).
///
/// Post-pass run AFTER `_light_` per BSP. Walks each cluster's
/// decorator_blocks: for blocks whose decorator tag has > 1 model variation,
/// sort the placements by their byte[7] (size/lod key written by `_light_`
/// as the HDR exponent), then populate `decorator_block.model_start_index[]`
/// so the renderer can split a single block into per-variation draw ranges.
///
/// **STUB Phase 8.3.** Full body at
/// `refs/engine_geometry_sampler/structure_bsp_build_decorators_from_scenario_TOOL.txt`
/// (108 LoC pseudocode).
pub fn structure_bsp_build_decorators_from_scenario(
    scenario: &mut LoadedScenario,
    bsp_index: i32,
) -> bool {
    // Engine `tag_get('sbsp', bsp_index)` null check.
    if !scenario
        .active_bsps
        .iter()
        .any(|b| b.scenario_bsp_index == bsp_index as usize)
    {
        return false;
    }

    // Engine lines 38-104: for each cluster's decorator_block, if dctr has
    // > 1 variation (`*((int*)v11 + 7) > 1` = `dctr.render_model_instance_name_valid_count > 1`),
    // sort the placement range by byte[7] (= `type_index` in protomorph's
    // scenario form) then populate `decorator_runtime_block.model_start_index`.
    //
    // **Protomorph adaptation:** the engine's per-cluster grouping is a
    // cache-build artifact we don't replicate (same as `_light_`). We sort
    // placements per (scenario_decorator_block, set_entry) instead, since
    // that's where the placements live in our data model.
    //
    // The `model_start_index[variation_count - 1]` partition table the
    // engine builds is dropped here — protomorph's renderer
    // (`render_decorators.rs`) already filters placements per `type_index`
    // at draw time, so the contiguous-range optimization isn't load-bearing.
    // If we ever need the partition for perf, the comment on engine line
    // 75-95 is the algorithm: walk sorted placements, advance v15 until
    // `placement.type_index >= variation_idx`, store the placement index
    // at `model_start_index[variation_idx]`; final pass fills tail
    // entries with placement_count.
    for (block_idx, decorator_block) in scenario.scenario.decorators.iter_mut().enumerate() {
        for (set_idx, set_entry) in decorator_block.sets.iter_mut().enumerate() {
            let variation_count = match scenario
                .decorator_sets
                .get(block_idx)
                .and_then(|sets| sets.get(set_idx))
                .and_then(|opt| opt.as_ref())
            {
                Some(t) => t.render_model_instance_name_valid_count,
                None => continue,
            };
            if variation_count <= 1 {
                continue;
            }
            // Engine `sub_140C4A9A0(placements, end, count, sort_key_offset=0)`
            // sorts placements ascending by their `type_index` byte. We use
            // stable sort to keep tie-broken ordering deterministic across
            // re-bakes (the engine's sort isn't documented as stable but
            // bake reproducibility is more important to us).
            set_entry
                .placements
                .sort_by_key(|p| p.type_index);
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_tables_have_expected_layout() {
        assert_eq!(GROUND_TABLE.len(), 10);
        assert_eq!(HANGING_TABLE.len(), 10);
        // First Ground entry: top sample, points down, weight 2.8.
        assert!((GROUND_TABLE[0].pos.z - 0.9).abs() < 1e-6);
        assert!((GROUND_TABLE[0].dir.z - (-1.0)).abs() < 1e-6);
        assert!((GROUND_TABLE[0].weight - 2.8).abs() < 1e-6);
        // Last Hanging entry: bottom-center, points up, weight 1.5.
        assert!((HANGING_TABLE[9].pos.z - 0.0).abs() < 1e-6);
        assert!((HANGING_TABLE[9].dir.z - 1.0).abs() < 1e-6);
        assert!((HANGING_TABLE[9].weight - 1.5).abs() < 1e-6);
    }
}
