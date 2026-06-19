pub mod sh;

use bytemuck::{Pod, Zeroable};

use crate::halo::render::lighting_interface::LightingInterface;
use crate::halo::render::views::lights_view::SimpleLight;

// ---------------------------------------------------------------------------
// Engine-wide forward-pipeline lighting cbuffer
// ---------------------------------------------------------------------------

/// Mirrors dllcache's "default lighting" cbuffer slots:
/// - `0x240000` = dominant_light_direction (vec4, w=1)
/// - `0x240001` = dominant_light_intensity (vec4, RGB + 0)
/// - `0x2E0000`/`0x2A0000` = ravi-packed SH3 — 10 vec4s, written by
///   `calculate_and_set_ravi_constants_internal` from the 9-float SH3
///   per channel.
///
/// Packed by [`pack_ravi_constants`] which mirrors dllcache's exact
/// shuffle (sign flips on selected components, `sqrt(3)` multiplier on
/// the L2 m=0 component). Bound at `@group(0) @binding(1)`.
///
/// Engine layout: dllcache `LightingPS` cbuffer (slot 0x30 / 0x2E)
/// — ravi SH3 only. The companion 2-vec4 dominant-light slot (0x24)
/// lives in [`GpuEngineDominantLight`] at camera_bgl binding 13.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuEngineLightingPs {
    /// 10 vec4 ravi-packed SH3 — see [`pack_ravi_constants`] for the
    /// layout. The WGSL `ravi_order_3` reads this array directly.
    pub ravi: [[f32; 4]; 10],
}

/// dllcache cbuffer slot 0x24 — analytical dominant light extracted
/// per-cluster by `c_lighting_interface` alongside the SH probe.
/// Direction is world-space unit vector pointing FROM fragment TO
/// light (engine convention). Intensity is RGB radiance.
///
/// Per-cluster like [`GpuEngineLightingPs`] — both buffers index by
/// the same `next_probe_slot` cursor; the dynamic-offset for binding
/// 13 mirrors the offset used at binding 1.
///
/// Bound at `@group(0) @binding(13)`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuEngineDominantLight {
    pub direction: [f32; 4],
    pub intensity: [f32; 4],
}

/// Faithful port of dllcache `c_lighting_interface::calculate_and_set_ravi_constants_internal`
/// @ 0x1806aa650. Packs 9-float SH3 R/G/B + scale into the 10-vec4
/// ravi cbuffer layout the static_sh / static_default pixel shaders
/// consume.
///
/// Layout (each row is one vec4 in the cbuffer):
/// ```text
///  [0] = ( sh_r[0],          sh_g[0],          sh_b[0],          0 )    // DC RGB
///  [1] = ( sh_r[3],          sh_r[1],         -sh_r[2],          0 )    // L1 R, z negated
///  [2] = ( sh_g[3],          sh_g[1],         -sh_g[2],          0 )    // L1 G
///  [3] = ( sh_b[3],          sh_b[1],         -sh_b[2],          0 )    // L1 B
///  [4] = (-sh_r[4],          sh_r[5],          sh_r[7],          0 )    // L2 R off-diag (m=±2,-1,+1)
///  [5] = (-sh_g[4],          sh_g[5],          sh_g[7],          0 )    // L2 G off-diag
///  [6] = (-sh_b[4],          sh_b[5],          sh_b[7],          0 )    // L2 B off-diag
///  [7] = (-sh_r[8], sh_r[8], -√3·sh_r[6], √3·sh_r[6] )                  // L2 R diag (m=±2,m=0)
///  [8] = (-sh_g[8], sh_g[8], -√3·sh_g[6], √3·sh_g[6] )                  // L2 G diag
///  [9] = (-sh_b[8], sh_b[8], -√3·sh_b[6], √3·sh_b[6] )                  // L2 B diag
/// ```
pub fn pack_ravi_constants(
    sh_r: &[f32; 9], sh_g: &[f32; 9], sh_b: &[f32; 9], scale: f32,
) -> [[f32; 4]; 10] {
    let s = scale;
    let sqrt3 = 3.0_f32.sqrt();
    [
        [s*sh_r[0],        s*sh_g[0],        s*sh_b[0],        0.0],
        [s*sh_r[3],        s*sh_r[1],       -s*sh_r[2],        0.0],
        [s*sh_g[3],        s*sh_g[1],       -s*sh_g[2],        0.0],
        [s*sh_b[3],        s*sh_b[1],       -s*sh_b[2],        0.0],
        [-s*sh_r[4],       s*sh_r[5],        s*sh_r[7],        0.0],
        [-s*sh_g[4],       s*sh_g[5],        s*sh_g[7],        0.0],
        [-s*sh_b[4],       s*sh_b[5],        s*sh_b[7],        0.0],
        [-s*sh_r[8],       s*sh_r[8],       -s*sqrt3*sh_r[6],  s*sqrt3*sh_r[6]],
        [-s*sh_g[8],       s*sh_g[8],       -s*sqrt3*sh_g[6],  s*sqrt3*sh_g[6]],
        [-s*sh_b[8],       s*sh_b[8],       -s*sqrt3*sh_b[6],  s*sqrt3*sh_b[6]],
    ]
}

impl GpuEngineLightingPs {
    /// Build just the ravi half from a `LightingInterface`. The
    /// dominant-light companion comes from
    /// [`GpuEngineDominantLight::from_lighting_interface`].
    pub fn from_lighting_interface(li: &LightingInterface) -> Self {
        Self {
            ravi: pack_ravi_constants(&li.sh_probe_r, &li.sh_probe_g, &li.sh_probe_b, 1.0),
        }
    }
}

impl GpuEngineDominantLight {
    /// Build just the dominant-light half from a `LightingInterface`.
    pub fn from_lighting_interface(li: &LightingInterface) -> Self {
        Self {
            direction: [
                li.dominant_light_dir.x,
                li.dominant_light_dir.y,
                li.dominant_light_dir.z,
                li.dominant_light_dir.w,
            ],
            intensity: [
                li.dominant_light_intensity.x,
                li.dominant_light_intensity.y,
                li.dominant_light_intensity.z,
                0.0,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Simple-lights cbuffer (engine: `simple_lights[N]` + `simple_light_count`)
// ---------------------------------------------------------------------------

/// Per-light layout matches `simple_lights_fx.hlsl` comment block:
///   vec4[0] = (position.xyz, size)
///   vec4[1] = (inv_direction.xyz, sphere_percentage)
///   vec4[2] = (color.xyz, smoothness)
///   vec4[3] = (distance_scale, cone_scale, distance_offset, cone_offset)
///   vec4[4] = (bounding_radius2, _, _, _)
/// 80 bytes per light (matches Ares `c_lights_view::s_simple_light`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct GpuSimpleLight {
    pub position_size: [f32; 4],
    pub inv_direction_sphere: [f32; 4],
    pub color_smooth: [f32; 4],
    pub falloff_scale_offset: [f32; 4],
    pub bounding_radius2: [f32; 4],
}

/// Maximum simple lights per cbuffer slot. Matches the engine cap of
/// 8 (`simple_lights_fx.hlsl:2`). The cbuffer holds ONE 8-light slot
/// per cluster, indexed via dynamic uniform offset at draw time —
/// per-cluster selection mirrors `c_lights_view::add_simple_light_to_draw_list`
/// being called per BSP cluster part in dllcache, where each cluster
/// gets its OWN 8-nearest-lights set, computed once and stable across
/// camera motion.
pub const SIMPLE_LIGHTS_MAX: usize = 8;

/// `simple_lights` cbuffer — 1 vec4 count header + N × `GpuSimpleLight`.
/// Bound at `@group(0) @binding(8)`. 16 + N×80 bytes.
///
/// Engine writes count to slot `0x310000` (1 vec4) and lights to slot
/// `0x310001` (40 vec4); we collapse both into one cbuffer to match
/// wgpu's single-binding-per-slot model.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuSimpleLights {
    /// `(count, _, _, _)` — only `.x` consumed; `.yzw` ignored.
    pub count: [f32; 4],
    pub lights: [GpuSimpleLight; SIMPLE_LIGHTS_MAX],
}

impl Default for GpuSimpleLights {
    fn default() -> Self {
        Self {
            count: [0.0; 4],
            lights: [GpuSimpleLight::default(); SIMPLE_LIGHTS_MAX],
        }
    }
}

impl GpuSimpleLights {
}

/// Pack a single light into the cbuffer at `slot`. Caller is
/// responsible for setting `count` after the loop.
pub fn pack_simple_light_at(
    out: &mut GpuSimpleLights,
    slot: usize,
    src: &SimpleLight,
) {
    if slot < SIMPLE_LIGHTS_MAX {
        out.lights[slot] = pack_simple_light(src);
    }
}

fn pack_simple_light(src: &SimpleLight) -> GpuSimpleLight {
    GpuSimpleLight {
        position_size: [src.position.x, src.position.y, src.position.z, src.light_source_size],
        inv_direction_sphere: [src.inv_direction.x, src.inv_direction.y, src.inv_direction.z, src.sphere],
        color_smooth: [src.color.x, src.color.y, src.color.z, src.cone_smooth],
        falloff_scale_offset: [src.distance_scale, src.cone_scale, src.distance_offset, src.cone_offset],
        bounding_radius2: [src.bounding_radius2, 0.0, 0.0, 0.0],
    }
}

