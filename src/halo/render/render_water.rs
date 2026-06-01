//! Mirror of `Ares/source/render/render_water.h` (~80 lines).
//!
//! `c_water_renderer` namespace + per-frame ripple state + the
//! tessellated water-surface render path.
//!
//! Halo's water flow:
//!   1. game tick — `game_update` + `game_interation_event_add`
//!      register splash/ripple events.
//!   2. per-frame — `ripple_check_new` / `ripple_add` populates the
//!      ripple atlas; `ripple_update` advances dynamics;
//!      `update_water_part_list` builds the visible-water draw list.
//!   3. render — `render_tessellation` rasterizes the water mesh
//!      with per-vertex Gerstner displacement; `render_shading`
//!      runs the shading pass (refraction + reflection +
//!      Schlick fresnel); `render_underwater_fog` adds depth-fog.
//!
//! v1: types + namespace shape only. The actual GPU work is heavy
//! (tessellation, ripple atlas, refraction sampling) and lands when
//! the water shader (water_fx.hlsl + water_shading_fx.hlsl) ports
//! to WGSL. Per `reference_water_fx_blueprint.md` already in memory.

use crate::halo::render::render_objects::EntryPoint;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// _vertex_type_water — D3D11 input layout mirror
// ---------------------------------------------------------------------------
//
// PC water draws use instanced patch rendering. Each "instance" is one
// source water triangle (3 control points). The input layout packs
// 156 + 12 + 72 = 240 bytes across 3 vertex streams, mirroring
// `vertex_type_water_elements` in
// `Ares/source/rasterizer/rasterizer_resource_definitions.cpp:41`.
//
// Slot mapping (engine bind order: `set_vertex_streams(start_slot=1, count=3, ...)`):
//
//   Stream 0 (D3D slot 1, per-instance, 156 bytes/inst):
//     POSITION0..7 + TEXCOORD0..1  — 3 control points worth of
//     position + texcoord + tangent + binormal + lightmap_uv,
//     interleaved per the engine's HLSL packing.
//   Stream 1 (D3D slot 2, per-vertex, 12 bytes/vert):
//     TEXCOORD2  — `bc.xyz` barycentric from the engine-static
//     `g_water_tessellation_vertex_buffer` (136 verts).
//   Stream 2 (D3D slot 3, per-instance, 72 bytes/inst):
//     NORMAL0..4  — 3 control points worth of local_info + base_texcoord
//     for foam / watercolor sampling.
//
// Vertex shader-location numbering reserves a contiguous range
// starting at 0 — the WGSL water VS will declare `@location(N)` for
// each attribute below.

/// Stream 0 (`POSITION0..7 + TEXCOORD0..1`, per-instance, 156 bytes).
/// Packs 3 control points (`pos1`, `pos2`, `pos3`) of position + texcoord
/// + tangent + binormal + lightmap_uv interleaved per the engine's
/// packing scheme. Field naming preserves the HLSL semantic-component
/// breakdown from `water_fx.hlsl:188-208` so cross-references stay
/// legible.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct WaterControlPointInstance {
    /// POSITION0 (16): pos1.xyz, tc1.x
    pub pos1_tc1x: [f32; 4],
    /// POSITION1 (16): tc1.y, tan1.xyz
    pub tc1y_tan1: [f32; 4],
    /// POSITION2 (16): bin1.xyz, lm1.x
    pub bin1_lm1x: [f32; 4],
    /// POSITION3 (16): lm1.y, pos2.xyz
    pub lm1y_pos2: [f32; 4],
    /// POSITION4 (16): tc2.xy, tan2.xy
    pub tc2_tan2xy: [f32; 4],
    /// POSITION5 (16): tan2.z, bin2.xyz
    pub tan2z_bin2: [f32; 4],
    /// POSITION6 (16): lm2.xy, pos3.xy
    pub lm2_pos3xy: [f32; 4],
    /// POSITION7 (16): pos3.z, tc3.xy, tan3.x
    pub pos3z_tc3_tan3x: [f32; 4],
    /// TEXCOORD0 (16): tan3.yz, bin3.xy
    pub tan3yz_bin3xy: [f32; 4],
    /// TEXCOORD1 (12): bin3.z, lm3.xy
    pub bin3z_lm3: [f32; 3],
}

/// Stream 2 (`NORMAL0..4`, per-instance, 72 bytes). Packs 3 control
/// points worth of `local_info` (li) + `base_texcoord` (bt) — the
/// `s_raw_water_append` data per vertex pulled from the BSP's
/// `raw_water_data.raw_water_vertices` block. Foam / watercolor /
/// global-shape sampling reads from these.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct WaterLightingInstance {
    /// NORMAL0 (16): li1.xyz, bt1.x
    pub li1_bt1x: [f32; 4],
    /// NORMAL1 (16): bt1.yz, li2.xy
    pub bt1yz_li2xy: [f32; 4],
    /// NORMAL2 (16): li2.z, bt2.xyz
    pub li2z_bt2: [f32; 4],
    /// NORMAL3 (16): li3.xyz, bt3.x
    pub li3_bt3x: [f32; 4],
    /// NORMAL4 (8): bt3.yz
    pub bt3yz: [f32; 2],
}

const _: () = {
    // Compile-time size check — wgpu pipeline build will reject any
    // mismatch but we want to catch it at compile time.
    assert!(std::mem::size_of::<WaterControlPointInstance>() == 156);
    assert!(std::mem::size_of::<WaterLightingInstance>() == 72);
};

const WATER_STREAM_0_ATTRS: [wgpu::VertexAttribute; 10] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0,   shader_location: 0 }, // POSITION0
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16,  shader_location: 1 }, // POSITION1
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32,  shader_location: 2 }, // POSITION2
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48,  shader_location: 3 }, // POSITION3
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64,  shader_location: 4 }, // POSITION4
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80,  shader_location: 5 }, // POSITION5
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 96,  shader_location: 6 }, // POSITION6
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 112, shader_location: 7 }, // POSITION7
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 128, shader_location: 8 }, // TEXCOORD0
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 144, shader_location: 9 }, // TEXCOORD1
];

const WATER_STREAM_1_ATTRS: [wgpu::VertexAttribute; 1] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 10 }, // TEXCOORD2 (barycentric)
];

const WATER_STREAM_2_ATTRS: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0,  shader_location: 11 }, // NORMAL0
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 12 }, // NORMAL1
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 13 }, // NORMAL2
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 14 }, // NORMAL3
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 64, shader_location: 15 }, // NORMAL4
];

/// Hardcoded WGSL `WaterMaterialPS` field order. Each entry is the
/// `rmt2.float_constants` source-name we expect at that struct slot.
/// `_xform`-suffixed WGSL fields strip to the bitmap source-name.
/// Slots not in this list aren't read by the shader; rmt2 entries not
/// present here get dropped during repack.
const WATER_WGSL_FIELDS: &[&str] = &[
    "slope_range_x", "slope_range_y", "wave_displacement_array",
    "time_warp", "wave_slope_array", "time_warp_aux",
    "detail_slope_scale_x", "detail_slope_scale_y", "detail_slope_scale_z",
    "detail_slope_steepness",
    "watercolor_coefficient", "reflection_coefficient",
    "sunspot_cut", "shadow_intensity_mark",
    "refraction_texcoord_shift", "water_murkiness",
    "refraction_extinct_distance", "minimal_wave_disturbance",
    "refraction_depth_dominant_ratio",
    "bankalpha_infuence_depth", "fresnel_coefficient",
    "water_diffuse",
    "foam_texture", "foam_texture_detail", "foam_height", "foam_pow",
    "displacement_range_x", "displacement_range_y", "displacement_range_z",
    "wave_height", "wave_height_aux",
    "choppiness_forward", "choppiness_backward", "choppiness_side",
    "wave_visual_damping_distance",
    "waveshape", "global_shape",
    "base_map", "detail_map", "change_color_map",
    "bump_map", "bump_detail_map", "self_illum_map",
    "albedo_color", "bump_detail_coefficient",
    "diffuse_coefficient", "specular_coefficient",
    "normal_specular_power", "glancing_specular_power",
    "fresnel_curve_steepness", "albedo_specular_tint_blend",
    "analytical_specular_contribution", "area_specular_contribution",
    "environment_map_specular_contribution", "analytical_anti_shadow_control",
    "self_illum_intensity", "normal_specular_tint", "glancing_specular_tint",
    "env_tint_color", "self_illum_color",
    "primary_change_color", "secondary_change_color",
    "watercolor_texture", "environment_map",
    "no_dynamic_lights", "global_shape_texture",
    // Phase W1 additions for non-riverworld water variants:
    //   `water_color_pure`            — `watercolor=pure` branch (zanzibar/etc)
    //   `globalshape_infuence_depth`  — `global_shape=depth` branch
    "water_color_pure", "globalshape_infuence_depth",
];

/// Repack the rmw resolved cbuffer bytes so each `WATER_WGSL_FIELDS[i]`
/// lands at byte offset `i * 16`. Different rmw rmt2s have different
/// orderings (and may omit some fields) — this lookup-by-name fix
/// shields the WGSL from rmt2 layout drift. Missing fields → 16 zero
/// bytes. Output is always `WATER_WGSL_FIELDS.len() * 16` bytes.
/// Repack rmt2-ordered cbuffer bytes into WGSL field order. Fields
/// that the input `cb` doesn't carry fall back to `fallback` (use the
/// load-time repacked bytes here to preserve static values that the
/// per-frame `resolve_pixel_user_cbuffer_at_time` doesn't re-emit —
/// specifically the "phantom" slots appended by loader.rs for params
/// the rmt2 didn't route, e.g. `fresnel_curve_steepness`).
///
/// Pass `None` for fallback at load-time (un-found = zero — fine because
/// load-time `mat.cbuffer` already had phantom slots appended by
/// `append_param_phantom_slots`, so every WGSL field is found).
fn repack_water_cbuffer_to_wgsl_order(
    cb: &blam_tags::render_method::ResolvedCbuffer,
    fallback: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = vec![0u8; WATER_WGSL_FIELDS.len() * 16];
    // Pre-fill from fallback so any field not in `cb` keeps its cached value.
    if let Some(fb) = fallback {
        let n = fb.len().min(out.len());
        out[..n].copy_from_slice(&fb[..n]);
    }
    let debug = std::env::var_os("PROTOMORPH_DEBUG_WATER_CBUFFER").is_some();
    if debug {
        eprintln!(
            "[water-cb] rmt2 has {} slots in {} bytes; WGSL expects {} fields",
            cb.slots.len(),
            cb.bytes.len(),
            WATER_WGSL_FIELDS.len(),
        );
        for s in &cb.slots {
            eprintln!(
                "[water-cb]   rmt2 slot: name={:?} byte_offset={}",
                s.source_name, s.byte_offset,
            );
        }
    }
    let mut missing = Vec::new();
    for (i, &field_name) in WATER_WGSL_FIELDS.iter().enumerate() {
        if let Some(slot) = cb.slots.iter().find(|s| s.source_name == field_name) {
            let src = slot.byte_offset as usize;
            let dst = i * 16;
            if src + 16 <= cb.bytes.len() {
                out[dst..dst + 16].copy_from_slice(&cb.bytes[src..src + 16]);
                if debug {
                    // Decode the full vec4 for sanity. Xform slots use
                    // all 4 components (scale.xy / offset.xy); scalars
                    // typically broadcast across all 4. Mismatches show
                    // up clearly when scale.y differs from scale.x.
                    let mut vals = [0f32; 4];
                    for k in 0..4 {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&cb.bytes[src + k * 4..src + k * 4 + 4]);
                        vals[k] = f32::from_le_bytes(b);
                    }
                    eprintln!(
                        "[water-cb]   WGSL[{i:02}] {field_name:35} ← rmt2[{src:4}] = ({}, {}, {}, {})",
                        vals[0], vals[1], vals[2], vals[3],
                    );
                }
            } else if debug {
                eprintln!(
                    "[water-cb]   WGSL[{i:02}] {field_name:35} OUT-OF-RANGE (src={src}, bytes={})",
                    cb.bytes.len(),
                );
            }
        } else {
            missing.push(field_name);
            if debug && fallback.is_some() {
                let dst = i * 16;
                let mut b = [0u8; 4];
                b.copy_from_slice(&out[dst..dst + 4]);
                let v = f32::from_le_bytes(b);
                eprintln!(
                    "[water-cb]   WGSL[{i:02}] {field_name:35} ← FALLBACK = {v}",
                );
            }
        }
    }
    if !missing.is_empty() && fallback.is_none() {
        eprintln!(
            "[water-cb] WARN {} WGSL fields not matched by rmt2 slot name (silently zero): {:?}",
            missing.len(),
            missing,
        );
    }
    out
}

/// `_vertex_type_water` D3D11 input layout → 3 wgpu `VertexBufferLayout`s
/// in the engine's bind-order (slot 1, 2, 3 → wgpu pipeline buffer
/// indices 0, 1, 2). Stride 156 / 12 / 72.
pub fn water_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 3] {
    [
        wgpu::VertexBufferLayout {
            array_stride: 156,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &WATER_STREAM_0_ATTRS,
        },
        wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &WATER_STREAM_1_ATTRS,
        },
        wgpu::VertexBufferLayout {
            array_stride: 72,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &WATER_STREAM_2_ATTRS,
        },
    ]
}

/// Decode mip 0 of every slice of an [`InlineTexture`] to RGBA8 and
/// write each slice as a PPM (P6) image under
/// `dump/water_bitmaps/<material>__<param>__slice<N>.ppm`. Gated by
/// the `PROTOMORPH_DUMP_WATER_BITMAPS=1` env var. Used to inspect
/// whether the on-disk bitmaps match what we see in-game (V-flip
/// suspicion check). No-op when the env var is unset.
fn dump_water_texture_ppm(
    material_name: &str,
    param_name: &str,
    tex: &crate::halo::render_methods::materials::InlineTexture,
) {
    use crate::halo::render_methods::materials::InlinePixelFormat as F;
    use std::io::Write;

    if std::env::var_os("PROTOMORPH_DUMP_WATER_BITMAPS").is_none() {
        return;
    }

    let dir = std::path::Path::new("dump/water_bitmaps");
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("[water-dump] failed to create {dir:?}: {e}");
        return;
    }

    let w = tex.width;
    let h = tex.height;
    let safe_mat: String = material_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();

    // 2D / 2D-array layout: each slice's full mip chain is contiguous
    // (matches `upload_inline_texture`'s face-major iteration).
    let mip0_bytes = tex.format.level_bytes(w, h);
    let total_slice_bytes: usize = (0..tex.mip_count)
        .map(|m| tex.format.level_bytes((w >> m).max(1), (h >> m).max(1)))
        .sum();

    for slice in 0..tex.layers {
        let slice_start = (slice as usize) * total_slice_bytes;
        let slice_end = slice_start + mip0_bytes;
        if slice_end > tex.data.len() {
            eprintln!(
                "[water-dump] {safe_mat}/{param_name} slice {slice}: data short ({} < {slice_end})",
                tex.data.len(),
            );
            continue;
        }
        let mip0 = &tex.data[slice_start..slice_end];
        // Decode this mip-0 slice into RGBA8 (memory `[R,G,B,A]`),
        // handling every InlinePixelFormat the bitmap loader can emit.
        // BC formats route through blam-tags; uncompressed formats are
        // converted inline so we don't need a round-trip through a Halo
        // BitmapFormat.
        let rgba_result: Result<Vec<u8>, String> = match tex.format {
            F::Bc1RgbaUnormSrgb | F::Bc1RgbaUnorm => {
                blam_tags::bitmap::decode::decode_to_rgba8(
                    blam_tags::bitmap::BitmapFormat::Dxt1, w, h, mip0,
                ).map_err(|e| format!("{e:?}"))
            }
            F::Bc2RgbaUnormSrgb | F::Bc2RgbaUnorm => {
                blam_tags::bitmap::decode::decode_to_rgba8(
                    blam_tags::bitmap::BitmapFormat::Dxt3, w, h, mip0,
                ).map_err(|e| format!("{e:?}"))
            }
            F::Bc3RgbaUnormSrgb | F::Bc3RgbaUnorm => {
                blam_tags::bitmap::decode::decode_to_rgba8(
                    blam_tags::bitmap::BitmapFormat::Dxt5, w, h, mip0,
                ).map_err(|e| format!("{e:?}"))
            }
            F::Bc4RUnorm | F::Bc4RSnorm => {
                blam_tags::bitmap::decode::decode_to_rgba8(
                    blam_tags::bitmap::BitmapFormat::Dxt5a, w, h, mip0,
                ).map_err(|e| format!("{e:?}"))
            }
            F::Bc5RgUnorm | F::Bc5RgSnorm => {
                blam_tags::bitmap::decode::decode_to_rgba8(
                    blam_tags::bitmap::BitmapFormat::Dxn, w, h, mip0,
                ).map_err(|e| format!("{e:?}"))
            }
            F::Rgba8UnormSrgb | F::Rgba8Unorm => Ok(mip0.to_vec()),
            F::Rgba8Snorm => Ok(mip0.iter()
                .map(|&b| ((b as i8) as i32 + 128).clamp(0, 255) as u8)
                .collect()),
            F::Bgra8Unorm | F::Bgra8UnormSrgb => {
                let mut out = Vec::with_capacity(mip0.len());
                for px in mip0.chunks_exact(4) {
                    out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                }
                Ok(out)
            }
            F::Rg8Unorm => {
                let mut out = Vec::with_capacity((mip0.len() / 2) * 4);
                for px in mip0.chunks_exact(2) {
                    out.extend_from_slice(&[px[0], px[1], 0, 255]);
                }
                Ok(out)
            }
            F::Rg8Snorm => {
                let mut out = Vec::with_capacity((mip0.len() / 2) * 4);
                for px in mip0.chunks_exact(2) {
                    let r = ((px[0] as i8) as i32 + 128).clamp(0, 255) as u8;
                    let g = ((px[1] as i8) as i32 + 128).clamp(0, 255) as u8;
                    out.extend_from_slice(&[r, g, 128, 255]);
                }
                Ok(out)
            }
            F::R8Unorm => {
                let mut out = Vec::with_capacity(mip0.len() * 4);
                for &r in mip0 {
                    out.extend_from_slice(&[r, 0, 0, 255]);
                }
                Ok(out)
            }
            F::Rgba16Float => {
                // 4 f16 per pixel — clamp to [0,1] and scale; emit a
                // companion HDR-stats line so we can spot scaling issues.
                let f16s: &[half::f16] = bytemuck::cast_slice(mip0);
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &h in f16s.iter() {
                    let v = h.to_f32();
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
                eprintln!("[water-dump] {safe_mat}/{param_name} slice {slice}: Rgba16Float min={mn} max={mx}");
                Ok(f16s.iter()
                    .map(|h| (h.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect())
            }
            F::Rgba16Unorm => {
                let u16s: &[u16] = bytemuck::cast_slice(mip0);
                Ok(u16s.iter().map(|&v| (v >> 8) as u8).collect())
            }
            F::Rgba16Snorm => {
                let i16s: &[i16] = bytemuck::cast_slice(mip0);
                Ok(i16s.iter()
                    .map(|&v| (((v as i32 + 32768) >> 8).clamp(0, 255)) as u8)
                    .collect())
            }
            F::Rgba32Float => {
                let f32s: &[f32] = bytemuck::cast_slice(mip0);
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &v in f32s.iter() {
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
                eprintln!("[water-dump] {safe_mat}/{param_name} slice {slice}: Rgba32Float min={mn} max={mx}");
                Ok(f32s.iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect())
            }
        };
        let rgba = match rgba_result {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[water-dump] decode failed for {safe_mat}/{param_name}: {e}");
                return;
            }
        };
        if rgba.len() != (w as usize) * (h as usize) * 4 {
            eprintln!(
                "[water-dump] {safe_mat}/{param_name}: unexpected decoded size {} vs {}",
                rgba.len(), (w as usize) * (h as usize) * 4,
            );
            continue;
        }

        let path = dir.join(format!("{safe_mat}__{param_name}__slice{slice}.ppm"));
        let mut file = match std::fs::File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[water-dump] failed to create {path:?}: {e}");
                continue;
            }
        };
        let header = format!("P6\n{w} {h}\n255\n");
        if let Err(e) = file.write_all(header.as_bytes()) {
            eprintln!("[water-dump] write header failed {path:?}: {e}");
            continue;
        }
        // Strip alpha — PPM is RGB only. (Alpha channel is dumped
        // separately below as a greyscale PGM so we don't lose .a-only
        // channels like global_shape's `.b` foam factor.)
        let mut rgb = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        if let Err(e) = file.write_all(&rgb) {
            eprintln!("[water-dump] write body failed {path:?}: {e}");
            continue;
        }

        // Alpha-only companion as PGM (P5).
        let alpha_path = dir.join(format!("{safe_mat}__{param_name}__slice{slice}_alpha.pgm"));
        if let Ok(mut af) = std::fs::File::create(&alpha_path) {
            let _ = af.write_all(format!("P5\n{w} {h}\n255\n").as_bytes());
            let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
            let _ = af.write_all(&alpha);
        }

        eprintln!(
            "[water-dump] wrote {path:?} ({w}×{h}, format {:?}, slice {slice}/{})",
            tex.format, tex.layers,
        );
    }
}

/// When `PROTOMORPH_DEBUG_WATER_UV=1`, replace the marker comment at
/// the top of the water `fs_main` with an unconditional `return` that
/// outputs `base_tex.xy` as RG color. Lets the lake's UV space be read
/// directly off the water surface so we can compare against the
/// `global_shape_texture` bitmap layout. No-op when the env var is
/// unset — returns the source verbatim so the include_str! borrow is
/// upgraded to `String` only when needed.
fn patch_water_uv_debug(source: &'static str) -> String {
    let mut out = source.to_string();
    if std::env::var_os("PROTOMORPH_DEBUG_WATER_UV").is_some() {
        eprintln!(
            "[water-debug] PROTOMORPH_DEBUG_WATER_UV=1 — patching fs_main to output base_tex.xy as RG",
        );
        // Trailing `//` swallows the descriptive text after the marker on
        // the same source line so the substitution leaves valid WGSL.
        out = out.replace(
            "// __WATER_UV_DEBUG_HOOK__",
            "return vec4<f32>(in.base_tex.x, in.base_tex.y, 0.0, 1.0); //",
        );
    }
    if std::env::var_os("PROTOMORPH_DEBUG_SWAP_WAVE_UV").is_some() {
        eprintln!(
            "[water-debug] PROTOMORPH_DEBUG_SWAP_WAVE_UV=1 — sampling wave bitmaps at texcoord.yx (transposed)",
        );
        out = out.replace("let sample_uv = base_uv;", "let sample_uv = base_uv.yx;");
        out = out.replace(
            "let sample_uv = tv.texcoord;",
            "let sample_uv = tv.texcoord.yx;",
        );
    }
    if std::env::var_os("PROTOMORPH_DEBUG_KILL_REFLECTION").is_some() {
        eprintln!(
            "[water-debug] PROTOMORPH_DEBUG_KILL_REFLECTION=1 — forcing color_reflection to vec3(0)",
        );
        out = out.replace(
            "    let color_reflection =\n        env_sample.rgb * env_alpha * water_material.reflection_coefficient.x;",
            "    let color_reflection = vec3<f32>(0.0); // KILLED",
        );
    }
    if std::env::var_os("PROTOMORPH_DEBUG_KILL_DISPLACEMENT").is_some() {
        eprintln!(
            "[water-debug] PROTOMORPH_DEBUG_KILL_DISPLACEMENT=1 — skipping VS wave displacement / choppiness entirely",
        );
        out = out.replace(
            "if (water_ps.k_is_water_tessellated != 0u) {",
            "if (false) {",
        );
    }
    let flip_v = std::env::var_os("PROTOMORPH_DEBUG_FLIP_TEXCOORD_V").is_some();
    let negate_v = std::env::var_os("PROTOMORPH_DEBUG_NEGATE_TEXCOORD_V").is_some();
    if flip_v || negate_v {
        let mode = if flip_v { "1 - v" } else { "-v" };
        eprintln!(
            "[water-debug] per-CP texcoord V {mode} — testing whether wave-bitmap tile-grid is from V miscoding",
        );
        let replacements = if flip_v {
            [
                ("in.tc1y_tan1.x /*HOOK_V1*/", "(1.0 - in.tc1y_tan1.x)"),
                ("in.tc2_tan2xy.y /*HOOK_V2*/", "(1.0 - in.tc2_tan2xy.y)"),
                ("in.pos3z_tc3_tan3x.z /*HOOK_V3*/", "(1.0 - in.pos3z_tc3_tan3x.z)"),
            ]
        } else {
            [
                ("in.tc1y_tan1.x /*HOOK_V1*/", "(-in.tc1y_tan1.x)"),
                ("in.tc2_tan2xy.y /*HOOK_V2*/", "(-in.tc2_tan2xy.y)"),
                ("in.pos3z_tc3_tan3x.z /*HOOK_V3*/", "(-in.pos3z_tc3_tan3x.z)"),
            ]
        };
        for (find, replace) in replacements {
            out = out.replace(find, replace);
        }
    }
    out
}

/// `c_water_renderer` — namespace class. Pattern A: all-static
/// methods. Owned by Renderer. Engine globals
/// (`g_water_render_structure_part_indices[512]`,
/// `g_water_tessellation_index_buffer`, `g_water_tessellation_vertex_buffer`,
/// `g_is_underwater`, `g_xm_view_proj_inv`, etc.) live as fields here.
#[derive(Debug)]
pub struct WaterRenderer {
    pub render_water_enabled: bool,
    pub render_water_tessellation_enabled: bool,
    pub render_water_interaction_enabled: bool,
    pub render_water_wireframe_enabled: bool,
    pub is_underwater: bool,
    pub new_interaction_event_count: i32,
    /// Mirrors engine `c_structure_renderer::lightmaps_available(bsp_index)`
    /// — set per-cluster in `c_water_renderer::render_cluster_parts`
    /// (`set_shader_constant_bool(0x520001, …)`). For protomorph's
    /// single-BSP path, set when `load_water_material` runs (rmw
    /// materials only exist when a BSP is loaded, and riverworld's
    /// BSP has a lightmap). Phase A4b will scope per-cluster.
    pub lightmaps_available: bool,
    /// `g_water_render_structure_part_indices[512]`.
    pub render_structure_part_indices: Vec<i32>,
    /// `g_water_tessellation_index_buffer @ 0x182559578`. 675 indices
    /// (225 sub-triangles per source water triangle), `Uint16`.
    /// Engine-static — generated once at `c_water_renderer::initialize`
    /// and reused for every water draw across every map. Per
    /// `reference_dllcache_water_pipeline.md`.
    pub tessellation_index_buffer: wgpu::Buffer,
    /// `g_water_tessellation_vertex_buffer @ 0x182559598`. 136 vertices
    /// (16-row triangular grid: row k has k+1 verts), each carrying
    /// barycentric (1-k/15, k/15-j/15, j/15) as `vec3<f32>`. The water
    /// VS interpolates the 3 control points (per-instance attrs) using
    /// these barycentrics to produce the rendered position.
    pub tessellation_vertex_buffer: wgpu::Buffer,
    /// Water shading pipeline — single-material minimal port of
    /// `water_shading()` (water_shading_fx.hlsl:658-1062). Built once
    /// at `WaterRenderer::new`; consumed every frame by
    /// [`Self::render_shading`]. Phase A4b/c will replace this single
    /// pipeline with a per-rmw-material cache (option-driven WGSL
    /// emission) when multi-material levels arrive.
    pub water_pipeline: wgpu::RenderPipeline,
    /// Sub-pass 2 — `_entry_point_shadow_generate` per IDA
    /// `c_water_renderer::render_shading @ 0x180693e90`:
    /// `set_color_write_enable(0, 0); set_z_buffer_mode(_z_buffer_mode_write);
    /// render_cluster_parts(_entry_point_shadow_generate, 3);`.
    /// Same VS as sub-pass 1, color writes disabled, depth write
    /// enabled — writes water surface depth into gbuffer so
    /// transparents drawn after water depth-test against it.
    pub water_depth_pipeline: wgpu::RenderPipeline,
    /// Empty `@group(2)` placeholder for the depth pipeline layout —
    /// avoids binding the real `water_extern_bind_group` (which
    /// references `scene_depth` as a sampled resource) at the same
    /// time the gbuffer depth is the writable target.
    pub depth_empty_bind_group: wgpu::BindGroup,
    /// Fullscreen pipeline ported from
    /// `water_ripple_hlsl.hlsl::underwater_vs/ps`. Engine
    /// `c_water_renderer::render_underwater_fog @ 0x180694080` runs
    /// this BEFORE `render_shading` when `g_is_underwater`. Reads
    /// scene LDR snapshot + scene depth, writes a fog-tinted copy
    /// to `lighting_base`. Uses water_engine_bgl @group(1) (for
    /// WaterSharedPS) + water_extern_bgl @group(2) (scene LDR/depth),
    /// no per-material data needed.
    pub underwater_fog_pipeline: wgpu::RenderPipeline,

    /// `WaterPS` cbuffer — `k_water_view_depth_constant` + the four
    /// `k_is_*` bools. Per
    /// `Ares/source/rasterizer/hlsl/water_registers_fx.hlsl` (DX11
    /// path) and dllcache `c_water_renderer::render_shading @
    /// 0x180693e90` (slots 0x520000 + 0x520001 + bool registers).
    pub cbuf_water_ps: wgpu::Buffer,
    /// `WaterSharedPS` cbuffer — `k_water_view_xform_inverse` (mat4
    /// for screen→world back-projection), `k_water_player_view_constant`
    /// (viewport offset/scale), `k_ps_camera_position`,
    /// `k_ps_underwater_murkiness`, `k_ps_underwater_fog_color`. Per
    /// dllcache `render_shading` (slots 0x560000 / 0x560004 /
    /// 0x560005 / 0x560006 / 0x560007).
    pub cbuf_water_shared_ps: wgpu::Buffer,
    /// `WaterSharedVS` cbuffer — `k_ripple_buffer_radius` +
    /// `k_ripple_buffer_center`. Per dllcache `render_shading` (slots
    /// 0x550000 + 0x550001).
    pub cbuf_water_shared_vs: wgpu::Buffer,
    /// BGL for the water engine cbuffer trio. Bound at `@group(1)` of
    /// every water pipeline. Slot 0 = `WaterPS` (PS), slot 1 =
    /// `WaterSharedPS` (PS), slot 2 = `WaterSharedVS` (VS).
    pub water_engine_bgl: wgpu::BindGroupLayout,
    /// Bind group instance referencing the cbuf trio buffers above.
    /// Rebuilt only if the buffers themselves change (they don't —
    /// each frame's data is `queue.write_buffer`'d in place).
    pub water_engine_bind_group: wgpu::BindGroup,

    /// Extern-bound resources for the water pass. Mirrors the
    /// dllcache `c_player_view::render_water @ 0x18068c120` extern
    /// dispatch:
    ///   - extern 45 (`_render_method_extern_scene_ldr_texture`) →
    ///     `_surface_screenshot_simple_resolve` + cbuf 0x2F (texture
    ///     size).
    ///   - extern 3 (`_render_method_extern_texture_global_target_z`)
    ///     → `_surface_depth_stencil` (read-only) + cbuf 0x3C
    ///     (depth_constants).
    /// Bound at `@group(2)` of every water pipeline. Phase A5 stands
    /// up the data path; Phase A6's full PS body starts reading scene
    /// depth for soft-edge fade and atmospheric fog distance.
    pub water_extern_bgl: wgpu::BindGroupLayout,
    pub water_extern_bind_group: wgpu::BindGroup,
    /// Clamp + bilinear sampler matching engine `set_sampler_address_mode(clamp)`
    /// + `set_sampler_filter_mode(bilinear)` for `tex_ldr_buffer` per
    /// `c_water_renderer::render_underwater_fog @ 0x180694080`. Reused
    /// by the shading pass for refraction lookups (engine binds the
    /// same sampler config via the rmt2 sampler routing for slot
    /// `tex_ldr_buffer`).
    pub scene_ldr_sampler: wgpu::Sampler,
    /// Clamp + point sampler matching engine
    /// `set_sampler_address_mode(clamp)` + `set_sampler_filter_mode(point)`
    /// for `tex_depth_buffer`. Engine forces point-filter on depth so
    /// the shader reads exact texel values without bilinear bleed
    /// across edges.
    pub scene_depth_sampler: wgpu::Sampler,
    /// `LDRTextureSize` cbuf mirror — engine slot 0x2F0000. Updated
    /// from intermediates' scene_ldr_snapshot dimensions.
    pub cbuf_ldr_texture_size: wgpu::Buffer,
    /// `DepthConstants` cbuf mirror — engine slot 0x3C0000. Updated
    /// per-frame from camera near plane (reverse-Z infinite-far →
    /// `(-near, 0, 0, 0)`).
    pub cbuf_depth_constants: wgpu::Buffer,

    /// Per-rmw-material bind group at `@group(3)`. Phase A6a — single
    /// material slot since riverworld has 1 rmw material. Contents:
    ///   0: environment_map cubemap (rmt2 PS texture)
    ///   1: env_cubemap sampler (linear, wrap)
    ///   2: rmw user cbuffer — `mat.cbuffer.bytes` uploaded verbatim
    ///      (one vec4 per slot in rmt2 declaration order, matches the
    ///      engine `submit_static_ps_parameters @ 0x180685860`
    ///      memcpy). Bound at engine PS slot 0x13 (rmt2 user cbuffer);
    ///      WGSL declares `struct WaterMaterialPS` with one vec4 per
    ///      slot in the same order.
    ///   3: wave_displacement_array (Phase A7a — VS-side rmt2
    ///      texture; 2D array, one slice per displacement keyframe).
    ///      Sampled by `transform_vertex` (`water_shading_fx.hlsl:368-485`)
    ///      to perturb position along the per-vertex tangent / binormal /
    ///      normal basis.
    ///   4: wave_displacement_array sampler (linear, repeat).
    /// Initialized with a fallback bind group (1×1 black cubemap +
    /// 1×1×1 black 2D-array + a 16-byte zeroed cbuf) at
    /// `WaterRenderer::new`; swapped in by [`Self::load_water_material`]
    /// which allocates a properly sized `cbuf_water_material_ps` buffer,
    /// uploads `mat.cbuffer.bytes`, loads the wave_displacement_array
    /// bitmap, and rebuilds the bind group. Phase A4b promotes this
    /// to a per-rmw-material cache.
    pub water_material_bgl: wgpu::BindGroupLayout,
    pub water_material_bind_group: wgpu::BindGroup,
    pub env_cubemap_sampler: wgpu::Sampler,
    pub wave_array_sampler: wgpu::Sampler,
    /// Linear+repeat sampler for watercolor / foam / global_shape
    /// 2D textures. Engine binds these via the rmt2 sampler routing;
    /// for protomorph's single-pipeline path we share one sampler.
    pub material_2d_sampler: wgpu::Sampler,
    pub cbuf_water_material_ps: wgpu::Buffer,
    /// Load-time repacked bytes (rmt2-emitted + phantom slots from
    /// `loader::append_param_phantom_slots`, all in WGSL field order).
    /// Used as the fallback in per-frame `update_constants` so phantom-
    /// slot values (e.g. `fresnel_curve_steepness`) survive the
    /// rmt2-only re-resolve.
    pub cbuf_water_material_loaded_bytes: Vec<u8>,
    /// 1×1×1 black 2D array fallback view bound to slot 3 until a
    /// real wave_displacement_array loads. Lives on `WaterRenderer`
    /// so the texture stays alive for the lifetime of the renderer.
    pub wave_array_fallback_texture: wgpu::Texture,
    pub wave_array_fallback_view: wgpu::TextureView,
    /// 1×1 white 2D fallback for missing rmw 2D textures
    /// (watercolor_texture, foam_texture, global_shape_texture). Held
    /// on `WaterRenderer` so the texture outlives any view of it.
    pub material_2d_fallback_texture: wgpu::Texture,
    pub material_2d_fallback_view: wgpu::TextureView,

    /// Source tag data for re-resolving the rmw user cbuffer per-frame
    /// (Phase A7c). Captured from the first rmw material's
    /// [`MaterialData::render_method_source`] when
    /// [`Self::load_water_material`] runs. Engine equivalent:
    /// `update_constants @ 0x180685300` re-overlays animated
    /// parameters each frame from the live `c_render_method`.
    pub water_material_source: Option<crate::halo::render_methods::materials::RenderMethodSource>,
    /// Cbuf size in bytes, captured at load. Used to bound the
    /// per-frame upload to the buffer's actual capacity.
    pub cbuf_water_material_size: u64,
}

// ---------------------------------------------------------------------------
// Water engine cbuffer trio (Pod GPU layouts)
// ---------------------------------------------------------------------------

/// `WaterPS` cbuffer Pod layout — `water_registers_fx.hlsl:CBUFFER_BEGIN(WaterPS)`.
/// Bools packed as `u32` per WGSL uniform-buffer host-shareability rules
/// (the shader does `!= 0u` checks). 32 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuWaterPS {
    pub k_water_view_depth_constant: [f32; 4],
    pub k_is_lightmap_exist: u32,
    pub k_is_water_interaction: u32,
    pub k_is_water_tessellated: u32,
    pub k_is_camera_underwater: u32,
    /// Non-engine; sourced from `PROTOMORPH_DEBUG_WATER_VIZ` env var.
    /// 0 = normal output. 1 = lightmap_intensity × 5 (diagnoses atlas-empty
    /// SH at water UVs). 2 = water_color × 5 (post-coefficient pre-blend).
    /// 3 = lm_tex.xy fract as RG (visualizes lightmap UV layout).
    pub k_debug_viz: u32,
    pub _pad_viz: [u32; 3],
}

/// `WaterSharedPS` cbuffer Pod layout. The mat4 occupies the first 64
/// bytes (4 × vec4); subsequent vec4-sized fields are padded by WGSL's
/// uniform alignment. 128 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuWaterSharedPS {
    pub k_water_view_xform_inverse: [[f32; 4]; 4],
    pub k_water_player_view_constant: [f32; 4],
    pub k_ps_camera_position: [f32; 4],
    pub k_ps_underwater_murkiness: [f32; 4],
    pub k_ps_underwater_fog_color: [f32; 4],
}

/// `WaterSharedVS` cbuffer Pod layout. Each scalar/vec2 padded out to
/// vec4 to match WGSL's uniform layout. 32 bytes total.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuWaterSharedVS {
    pub k_ripple_buffer_radius: [f32; 4],
    pub k_ripple_buffer_center: [f32; 4],
}

/// LDR texture-size cbuffer mirror — engine cbuf `0x2F0000`. Set by
/// extern 45 (`_render_method_extern_scene_ldr_texture`) when the water
/// pass binds `_surface_screenshot_simple_resolve`. Layout per
/// `submit_extern_texture` (`reference_h3_extern_table.md`):
/// `(width, height, 1/width, 1/height)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuLdrTextureSize {
    pub size: [f32; 4],
}

/// Depth-constants cbuffer mirror — engine cbuf `0x3C0000`. Set by
/// extern 3 (`_render_method_extern_texture_global_target_z`) when the
/// water pass binds `_surface_depth_stencil` for sampling. The shader
/// recovers view-space Z from the depth buffer via
/// `view_z = depth_constants.x / (depth - depth_constants.y)` per
/// `c_player_view::render_water @ 0x18068c120` and the
/// rasterizer_get_z_planes helper. For protomorph's reverse-Z
/// infinite-far perspective (`Mat4::perspective_infinite_reverse_rh`),
/// the formula reduces to `view_z = -near / depth` — i.e.
/// `depth_constants = (-near, 0, …, …)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct GpuDepthConstants {
    pub depth_constants: [f32; 4],
}


const _: () = {
    assert!(std::mem::size_of::<GpuWaterPS>() == 48);
    assert!(std::mem::size_of::<GpuWaterSharedPS>() == 128);
    assert!(std::mem::size_of::<GpuWaterSharedVS>() == 32);
    assert!(std::mem::size_of::<GpuLdrTextureSize>() == 16);
    assert!(std::mem::size_of::<GpuDepthConstants>() == 16);
};

impl WaterRenderer {
    pub fn new(
        shared: &crate::halo::render::shared::SharedResources,
        intermediates: &crate::halo::render::shared::IntermediateTargets,
        gbuffer: &crate::halo::render::shared::GBuffer,
    ) -> Self {
        let device = &shared.device;
        // Generate the engine-static tessellation IB+VB. Pattern
        // mirrors `c_water_renderer::initialize @ 0x180692400` (dllcache):
        // 15 bands × (2k+1) triangles between rows k and k+1 → 225
        // sub-triangles total; 16-row triangular vertex grid with 136
        // unique vertices.
        let (indices, vertex_floats) = generate_tessellation_data();
        debug_assert_eq!(indices.len(), 675);
        debug_assert_eq!(vertex_floats.len(), 136 * 3);

        let tessellation_index_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_water_tessellation_index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let tessellation_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g_water_tessellation_vertex_buffer"),
                contents: bytemuck::cast_slice(&vertex_floats),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // Water engine cbuffer trio — allocated zeroed; populated each
        // frame from [`Self::update_constants`]. Phase A4 binds them to
        // the real water pipeline; Phase A1d's debug stub doesn't read
        // them yet (the binding is silently ignored — wgpu permits
        // pipeline layouts that exceed shader usage).
        let cbuf_water_ps = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_WaterPS"),
            size: size_of::<GpuWaterPS>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cbuf_water_shared_ps = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_WaterSharedPS"),
            size: size_of::<GpuWaterSharedPS>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cbuf_water_shared_vs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_WaterSharedVS"),
            size: size_of::<GpuWaterSharedVS>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Water engine BGL — bound at `@group(1)` of every water
        // pipeline. Slots match the WGSL declarations in
        // `entry_water_shading.wgsl` (and Phase A4d's real water shader).
        let water_engine_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water_engine_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // VS reads `k_is_water_tessellated` (Phase A7a) to
                    // gate displacement; PS reads `k_is_lightmap_exist`,
                    // depth_constant, and the underwater bool. Both
                    // stages need access.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuWaterPS>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // VERTEX|FRAGMENT — VS uses `k_ps_camera_position`
                    // for the view-direction interpolant; PS uses the
                    // mat4 inverse + viewport constants + underwater
                    // params. Engine has VS / PS separation via two
                    // distinct named cbuffers in the DX9 path; the
                    // DX11 `WaterSharedPS` is read by both stages in
                    // protomorph (matches the unified DX11 layout in
                    // `water_registers_fx.hlsl`).
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuWaterSharedPS>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuWaterSharedVS>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let water_engine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_engine_bg"),
            layout: &water_engine_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cbuf_water_ps.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cbuf_water_shared_ps.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cbuf_water_shared_vs.as_entire_binding(),
                },
            ],
        });

        // ---- Water extern bind group (@group(2)) ----
        // Mirrors the dllcache `c_player_view::render_water` extern
        // dispatch (extern 45 + extern 3). Slot layout:
        //   0: scene_ldr texture     (extern 45 surface)
        //   1: scene_ldr sampler     (clamp + bilinear)
        //   2: scene_depth texture   (extern 3 surface, depth aspect)
        //   3: scene_depth sampler   (clamp + point)
        //   4: LDRTextureSize cbuf   (extern 45 cbuf 0x2F0000)
        //   5: DepthConstants cbuf   (extern 3 cbuf 0x3C0000)
        let scene_ldr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water_scene_ldr_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let scene_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water_scene_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let cbuf_ldr_texture_size = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_LDRTextureSize"),
            size: size_of::<GpuLdrTextureSize>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cbuf_depth_constants = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_DepthConstants"),
            size: size_of::<GpuDepthConstants>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let water_extern_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water_extern_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // Depth32Float read as `texture_depth_2d`. Engine
                    // formula linearizes via the depth_constants cbuf
                    // before use; raw sampling returns the device-space
                    // [0,1] depth.
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuLdrTextureSize>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuDepthConstants>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let water_extern_bind_group = build_water_extern_bind_group(
            device,
            &water_extern_bgl,
            &intermediates.scene_ldr_snapshot_view,
            &gbuffer.depth_sample_view,
            &scene_ldr_sampler,
            &scene_depth_sampler,
            &cbuf_ldr_texture_size,
            &cbuf_depth_constants,
        );

        // ---- Water material bind group (@group(3)) ----
        // Phase A6a: per-rmw-material slot for the env cubemap +
        // reflection / fresnel coefficients. Initialized here with a
        // fallback (1×1 black cubemap + zeroed material cbuf) so the
        // pipeline layout is valid even before any rmw material loads.
        // [`Self::load_water_material`] swaps in the real cubemap +
        // coefficients once the BSP's rmw material is resolved.
        let env_cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water_env_cubemap_sampler"),
            // Standard cube convention is ClampToEdge across all axes
            // (faces don't tile — seamless face transitions are
            // hardware-handled). Repeat here triggered visible
            // discontinuities at wave-crest pixels where reflect_dir
            // gradients are large; ClampToEdge fixes the cube boundary
            // blend to the documented spec.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Engine `sample3Dlod(wave_displacement_array, …)` uses linear
        // filtering with repeating UV addressing — the displacement
        // pattern is designed to tile seamlessly across an infinite
        // water plane.
        let wave_array_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water_wave_array_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // 1×1×1 black 2D-array fallback texture. Bound to the
        // wave_displacement_array slot until [`Self::load_water_material`]
        // swaps in the real bitmap. Without a fallback, the BGL cannot
        // be filled at construction time → pipeline build would fail.
        let wave_array_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water_wave_array_fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let wave_array_fallback_view = wave_array_fallback_texture.create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("water_wave_array_fallback_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        );
        // Linear+repeat sampler for the rmw 2D textures (watercolor,
        // foam, global_shape). Engine `sampleBiasGlobal2D` uses
        // linear filtering with global mip bias; UV addressing
        // matches the bitmap's authored mode (riverworld watercolor
        // is repeat).
        // Engine binds global_shape_texture / watercolor_texture with
        // `bitmap address mode = 1 (Clamp)` per the rmw tag. These are
        // painted-once-per-lake bitmaps (riverworld_water_multipurpose):
        // their `base_texcoord` extends past [0,1] on the larger water
        // meshes (mesh 8: x in [-0.46, 1.46]). With Repeat the painted
        // river-map TILES across the lake → visible grid artifact.
        // ClampToEdge clamps OOB UVs to the bitmap's transparent rim →
        // height_scale/choppy_scale = 0 outside the painted silhouette.
        let material_2d_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water_material_2d_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // 1×1 white fallback for unbound 2D rmw textures.
        let material_2d_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water_material_2d_fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &material_2d_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let material_2d_fallback_view = material_2d_fallback_texture.create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("water_material_2d_fallback_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            },
        );
        // Initial 16-byte zeroed cbuf — `load_water_material`
        // reallocates this to the rmw cbuffer's actual size (66 vec4 =
        // 1056 bytes for riverworld) once the material is resolved.
        // wgpu BGLs with `min_binding_size: None` accept any size
        // ≥ 16 bytes, mirroring rmsh's `material_bgl` slot 5 (cbuffer
        // size varies per VariantKey there too).
        let cbuf_water_material_ps = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_cbuf_WaterMaterialPS"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let water_material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water_material_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    // The rmw user cbuffer is read by both VS and PS —
                    // VS samples wave_displacement_array using
                    // `wave_displacement_array_xform` etc.; PS reads
                    // reflection / fresnel / etc.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // Variable per rmw material (rmt2 cbuffer size
                        // depends on parameter count). Same `None`
                        // pattern as `material_bgl` slot 5 in
                        // `shared.rs::create_material_bgl`.
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Phase A7a — wave_displacement_array. 2D array of
                // displacement keyframes; sampled in VS by
                // `transform_vertex` (`water_shading_fx.hlsl:368-485`).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    // Shared by VS (wave_displacement_array) and PS
                    // (wave_slope_array, Phase A6f). Engine pairs both
                    // textures with the same `_sampler_filter_anisotropic`
                    // + `_sampler_address_mode_wrap`.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Phase A6f — watercolor_texture (PS 2D, linear+repeat).
                // Engine: `sample2D(watercolor_texture, base_tex.xy)`
                // for the `watercolor=texture` rmdf branch (riverworld
                // uses this).
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    // VS samples global_shape_texture (binding 7) at
                    // `sample2Dlod(global_shape_texture, IN.base_tex.xy, mip)`
                    // for the `global_shape=paint` branch in
                    // `transform_vertex` (`water_shading_fx.hlsl:341-348`).
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Phase A6e — foam path. Three additional 2D textures:
                //   7: global_shape_texture (paint foam factor in .b)
                //   8: foam_texture
                //   9: foam_texture_detail (riverworld points at same
                //      bitmap as foam_texture, distinguished only by
                //      `foam_texture_detail_xform`).
                // All sampled via the shared material_2d_sampler at
                // binding 6.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    // global_shape_texture: VS samples for height_scale_global +
                    // choppy_scale_global (`water_shading_fx.hlsl:341-348`,
                    // `global_shape=paint` branch); FS samples .b channel for
                    // foam_factor_paint.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Phase A6f — wave_slope_array. Same-shape 2D array as
                // wave_displacement_array but per-pixel slope (xy)
                // keyframes; sampled by compose_slope to build the
                // per-pixel normal. Reuses wave_array_sampler at
                // binding 4 (engine pairs both with anisotropic+wrap),
                // so we promote slot 4 visibility to VERTEX_FRAGMENT.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });
        let water_material_bind_group = build_water_material_bind_group(
            device,
            &water_material_bgl,
            &shared.fallback_textures.black_cube_view,
            &env_cubemap_sampler,
            &cbuf_water_material_ps,
            &wave_array_fallback_view,
            &wave_array_sampler,
            &material_2d_fallback_view,
            &material_2d_sampler,
            &material_2d_fallback_view,
            &material_2d_fallback_view,
            &material_2d_fallback_view,
            &wave_array_fallback_view,
        );

        // Build the debug pipeline AFTER the BGLs exist — its layout
        // includes water_engine_bgl at @group(1), water_extern_bgl at
        // @group(2), and water_material_bgl at @group(3) so the shader
        // can reference all of them without a layout mismatch.
        let water_pipeline = build_water_pipeline(
            shared,
            &water_engine_bgl,
            &water_extern_bgl,
            &water_material_bgl,
            &WaterVariant::defaults_riverworld(),
        );
        let (water_depth_pipeline, depth_empty_bind_group) = build_water_depth_pipeline(
            shared,
            &water_engine_bgl,
            &water_extern_bgl,
            &water_material_bgl,
            &WaterVariant::defaults_riverworld(),
        );
        let underwater_fog_pipeline = build_underwater_fog_pipeline(
            shared,
            &water_engine_bgl,
            &water_extern_bgl,
        );

        Self {
            render_water_enabled: true,
            render_water_tessellation_enabled: false,
            render_water_interaction_enabled: false,
            render_water_wireframe_enabled: false,
            is_underwater: false,
            new_interaction_event_count: 0,
            lightmaps_available: false,
            render_structure_part_indices: Vec::with_capacity(512),
            tessellation_index_buffer,
            tessellation_vertex_buffer,
            water_pipeline,
            water_depth_pipeline,
            depth_empty_bind_group,
            underwater_fog_pipeline,
            cbuf_water_ps,
            cbuf_water_shared_ps,
            cbuf_water_shared_vs,
            water_engine_bgl,
            water_engine_bind_group,
            water_extern_bgl,
            water_extern_bind_group,
            scene_ldr_sampler,
            scene_depth_sampler,
            cbuf_ldr_texture_size,
            cbuf_depth_constants,
            water_material_bgl,
            water_material_bind_group,
            env_cubemap_sampler,
            wave_array_sampler,
            material_2d_sampler,
            cbuf_water_material_ps,
            cbuf_water_material_loaded_bytes: Vec::new(),
            wave_array_fallback_texture,
            wave_array_fallback_view,
            material_2d_fallback_texture,
            material_2d_fallback_view,
            water_material_source: None,
            cbuf_water_material_size: 16,
        }
    }

    /// Swap in a real rmw material's resources — env cubemap +
    /// rmw user cbuffer (the packed `mat.cbuffer.bytes` blob from
    /// the rmsh→rmop→rmt2 walker). Phase A6a single-material path;
    /// called once at scenario load with the first rmw material's
    /// `MaterialData`. Phase A4b replaces this with a per-rmw cache.
    ///
    /// Engine equivalents (per `reference_dllcache_cbuffer_resolution.md`):
    /// - cubemap binding ← `submit_extern_texture` for the
    ///   `environment_map` rmt2 parameter (PS slot 16 typically).
    /// - cbuffer upload ← `submit_static_ps_parameters @ 0x180685860`
    ///   which memcpy's `rmsh.postprocess.real_constants` (= our
    ///   `mat.cbuffer.bytes`) to PS slot 0x13.
    ///
    /// The shader's `WaterMaterialPS` WGSL struct must mirror the rmw
    /// rmt2 cbuffer layout — one `vec4<f32>` per slot in the same
    /// declaration order, mirroring how `submit_static_ps_parameters`
    /// memcpys with no remapping.
    pub fn load_water_material(
        &mut self,
        shared: &crate::halo::render::shared::SharedResources,
        mat: &crate::halo::render_methods::materials::MaterialData,
    ) {
        use crate::halo::render_methods::materials::MaterialTextureUsage;
        use wgpu::util::DeviceExt;

        // Mark lightmap as available — riverworld BSPs always carry a
        // lightmap atlas alongside the rmw material. Engine `is_lightmap_available`
        // is per-cluster; protomorph's single-pipeline path treats it
        // as scenario-global until A4b adds per-cluster routing.
        self.lightmaps_available = true;

        // ---- env cubemap upload ----
        // Engine: `environment_map` is a Bitmap-typed rmt2 parameter
        // resolved by the rmsh→rmop→rmt2 walker; the runtime binds it
        // to PS slot 16 via the rmt2 sampler routing.
        let env_view = match mat.find_texture(MaterialTextureUsage::Environment) {
            Some(tex) if tex.is_cube => {
                eprintln!(
                    "[water] env_cubemap loaded: {}×{} cube ({} mips)",
                    tex.width, tex.height, tex.mip_count,
                );
                // NOTE: riverworld's rmw authors `z_cubemap_temp.bitmap`
                // (64×64×6, BGRA8 sRGB) as the static environment_map.
                // Alpha distribution is engine-faithful sparse-HDR
                // encoding (~0.012 mean, sun-spot peaks near 1.0).
                // Halo's runtime overrides this static default with
                // a PER-CLUSTER dynamic environment cubemap rendered
                // each frame from the actual scene — externs
                // `TextureDynamicEnvironmentMap0/1` (ref:
                // material_bindings.rs:300). Until the per-cluster
                // dynamic cubemap pass lands, the temp placeholder
                // produces an over-bright reflection because
                // `reflection_coefficient = 600` was tuned against
                // the dynamic per-cluster encoding, not z_cubemap_temp.
                Some(crate::halo::structures::bsp_gpu::upload_inline_texture(
                    shared, tex, false,
                ))
            }
            Some(tex) => {
                eprintln!(
                    "[water] WARN environment_map not a cubemap ({}×{}, layers={}); using fallback",
                    tex.width, tex.height, tex.layers,
                );
                None
            }
            None => {
                eprintln!("[water] WARN no environment_map on rmw material; using fallback");
                None
            }
        };

        // ---- wave_displacement_array upload (Phase A7a) ----
        // Engine: VS-side rmt2 texture sampled by `transform_vertex`
        // (`water_shading_fx.hlsl:368-485`). Halo's bitm tag carries
        // type=array with depth=N slices (16 keyframes for
        // riverworld's `wave_test1_displ_water.bitmap`). Phase A7c
        // animates the `time_warp` cbuffer entry per frame so the
        // slice index ramps through these keyframes.
        let wave_view = match mat.find_texture_by_name("wave_displacement_array") {
            Some(tex) if !tex.is_cube && tex.layers > 1 => {
                eprintln!(
                    "[water] wave_displacement_array loaded: {}×{}×{} ({} mips)",
                    tex.width, tex.height, tex.layers, tex.mip_count,
                );
                dump_water_texture_ppm(&mat.name, "wave_displacement_array", tex);
                Some(crate::halo::structures::bsp_gpu::upload_inline_texture(
                    shared, tex, false,
                ))
            }
            Some(tex) => {
                eprintln!(
                    "[water] WARN wave_displacement_array unexpected shape ({}×{}, layers={}, is_cube={}); using fallback",
                    tex.width, tex.height, tex.layers, tex.is_cube,
                );
                None
            }
            None => {
                eprintln!("[water] WARN no wave_displacement_array on rmw material; using fallback");
                None
            }
        };

        // ---- watercolor_texture upload (Phase A6f) ----
        // Engine: `sampleBiasGlobal2D(watercolor_texture, base_tex.xy)`
        // for the `watercolor=texture` rmdf branch (line 788). Riverworld
        // uses `levels\multi\riverworld\bitmaps\riverworld_water_diffuse`.
        let watercolor_view = match mat.find_texture_by_name("watercolor_texture") {
            Some(tex) if !tex.is_cube && tex.layers == 1 => {
                eprintln!(
                    "[water] watercolor_texture loaded: {}×{} ({} mips)",
                    tex.width, tex.height, tex.mip_count,
                );
                dump_water_texture_ppm(&mat.name, "watercolor_texture", tex);
                Some(crate::halo::structures::bsp_gpu::upload_inline_texture(
                    shared, tex, false,
                ))
            }
            Some(tex) => {
                eprintln!(
                    "[water] WARN watercolor_texture unexpected shape ({}×{}, layers={}); using fallback",
                    tex.width, tex.height, tex.layers,
                );
                None
            }
            None => {
                eprintln!("[water] no watercolor_texture on rmw material; using white fallback");
                None
            }
        };

        // ---- foam textures (Phase A6e) ----
        // Engine line 873-921: foam computation samples three rmt2
        // textures. Riverworld foam=both branch uses ALL three:
        //   global_shape_texture: .b channel drives foam_factor_paint
        //   foam_texture        : foam color (with foam_texture_xform)
        //   foam_texture_detail : detail multiplier (with own xform).
        // Riverworld points foam_texture and foam_texture_detail at the
        // same bitmap; only the xforms differ.
        let load_2d = |name: &str| -> Option<wgpu::TextureView> {
            match mat.find_texture_by_name(name) {
                Some(tex) if !tex.is_cube && tex.layers == 1 => {
                    eprintln!(
                        "[water] {} loaded: {}×{} ({} mips)",
                        name, tex.width, tex.height, tex.mip_count,
                    );
                    dump_water_texture_ppm(&mat.name, name, tex);
                    Some(crate::halo::structures::bsp_gpu::upload_inline_texture(
                        shared, tex, false,
                    ))
                }
                Some(tex) => {
                    eprintln!(
                        "[water] WARN {} unexpected shape ({}×{}, layers={}); using fallback",
                        name, tex.width, tex.height, tex.layers,
                    );
                    None
                }
                None => {
                    eprintln!("[water] no {} on rmw material; using white fallback", name);
                    None
                }
            }
        };
        let global_shape_view = load_2d("global_shape_texture");
        let foam_view = load_2d("foam_texture");
        let foam_detail_view = load_2d("foam_texture_detail");

        // ---- wave_slope_array upload (Phase A6f) ----
        // Engine: PS-side rmt2 texture sampled by `compose_slope`
        // (`water_shading_fx.hlsl:585-607`). Same `type=array` shape
        // as wave_displacement_array — riverworld's
        // `wave_test1_slope_water.bitmap` carries the slope (xy)
        // keyframes that synchronize with the displacement array's
        // height keyframes.
        let wave_slope_view = match mat.find_texture_by_name("wave_slope_array") {
            Some(tex) if !tex.is_cube && tex.layers > 1 => {
                eprintln!(
                    "[water] wave_slope_array loaded: {}×{}×{} ({} mips)",
                    tex.width, tex.height, tex.layers, tex.mip_count,
                );
                dump_water_texture_ppm(&mat.name, "wave_slope_array", tex);
                Some(crate::halo::structures::bsp_gpu::upload_inline_texture(
                    shared, tex, false,
                ))
            }
            Some(tex) => {
                eprintln!(
                    "[water] WARN wave_slope_array unexpected shape ({}×{}, layers={}, is_cube={}); using fallback",
                    tex.width, tex.height, tex.layers, tex.is_cube,
                );
                None
            }
            None => {
                eprintln!("[water] WARN no wave_slope_array on rmw material; using fallback");
                None
            }
        };

        // ---- rmw user cbuffer (engine PS slot 0x13) ----
        // `mat.cbuffer.bytes` is the packed blob produced by the
        // rmsh→rmop→rmt2 walker — one vec4 per slot in rmt2-declared
        // source order. Different rmw shaders have DIFFERENT rmt2
        // orderings (riverworld declares `bankalpha_infuence_depth`,
        // deadlock doesn't, fresnel_coefficient drifts to a different
        // index, etc.). Our WGSL `WaterMaterialPS` struct is hardcoded
        // to riverworld's order, so for any other rmt2 we must repack
        // the bytes by NAME so each WGSL field finds its slot at the
        // expected byte offset. Slots not present in the rmt2 → zero.
        let packed_bytes = repack_water_cbuffer_to_wgsl_order(&mat.cbuffer, None);
        eprintln!(
            "[water] rmw user cbuffer: {} slots, {} repacked bytes (rmt2 had {})",
            mat.cbuffer.slots.len(),
            packed_bytes.len(),
            mat.cbuffer.bytes.len(),
        );
        self.cbuf_water_material_size = packed_bytes.len() as u64;
        self.cbuf_water_material_ps = shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("water_cbuf_WaterMaterialPS"),
                contents: &packed_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        // Cache the load-time repacked bytes as the fallback for
        // per-frame re-resolve. `resolve_pixel_user_cbuffer_at_time` only
        // emits rmt2.float_constants slots — it does NOT re-append the
        // phantom slots added by `loader::append_param_phantom_slots`
        // (rmsh/rmop params the rmt2 didn't route, e.g.
        // `fresnel_curve_steepness`). Without this cache, per-frame
        // repack zeroes the phantom slots → `fresnel_p = 0` → `pow(x, 0)
        // = 1` → fresnel saturates at 1.0 → mirror-finish lake.
        self.cbuf_water_material_loaded_bytes = packed_bytes;
        // Capture the rmw source data so `update_constants` can
        // re-resolve the cbuffer per-frame for animated parameters
        // (`time_warp` etc., Phase A7c).
        self.water_material_source = mat.render_method_source.clone();

        // ---- rebuild the bind group with the new cube view + cbuf ----
        let cube_view_ref = env_view
            .as_ref()
            .unwrap_or(&shared.fallback_textures.black_cube_view);
        let wave_view_ref = wave_view
            .as_ref()
            .unwrap_or(&self.wave_array_fallback_view);
        let watercolor_view_ref = watercolor_view
            .as_ref()
            .unwrap_or(&self.material_2d_fallback_view);
        let global_shape_view_ref = global_shape_view
            .as_ref()
            .unwrap_or(&self.material_2d_fallback_view);
        let foam_view_ref = foam_view
            .as_ref()
            .unwrap_or(&self.material_2d_fallback_view);
        let foam_detail_view_ref = foam_detail_view
            .as_ref()
            .unwrap_or(&self.material_2d_fallback_view);
        let wave_slope_view_ref = wave_slope_view
            .as_ref()
            .unwrap_or(&self.wave_array_fallback_view);
        self.water_material_bind_group = build_water_material_bind_group(
            &shared.device,
            &self.water_material_bgl,
            cube_view_ref,
            &self.env_cubemap_sampler,
            &self.cbuf_water_material_ps,
            wave_view_ref,
            &self.wave_array_sampler,
            watercolor_view_ref,
            &self.material_2d_sampler,
            global_shape_view_ref,
            foam_view_ref,
            foam_detail_view_ref,
            wave_slope_view_ref,
        );

        // Phase W2 — rebuild the shading pipeline with this material's
        // rmdf-resolved category choices so the engine HLSL category
        // branches resolve correctly (watercolor=pure/texture,
        // global_shape=none/paint/depth, bankalpha=none/depth/paint).
        // The depth-only pipeline never reads these branches; keep it
        // as built at init.
        let variant = WaterVariant::from_choices(&mat.category_choices);
        eprintln!(
            "[water] variant: watercolor={} global_shape={} bankalpha={} \
             waveshape={} foam={} reflection={} refraction={}",
            variant.watercolor, variant.global_shape, variant.bankalpha,
            variant.waveshape, variant.foam, variant.reflection, variant.refraction,
        );
        self.water_pipeline = build_water_pipeline(
            shared,
            &self.water_engine_bgl,
            &self.water_extern_bgl,
            &self.water_material_bgl,
            &variant,
        );
        // Depth-only pipeline shares vs_main with the shading pipeline,
        // so it needs the same VS-side substitutions (height_scale_global
        // depends on the global_shape choice). Bind group is identical
        // (empty placeholder); discard the new one and keep the existing.
        let (depth_pipeline, _new_bg) = build_water_depth_pipeline(
            shared,
            &self.water_engine_bgl,
            &self.water_extern_bgl,
            &self.water_material_bgl,
            &variant,
        );
        self.water_depth_pipeline = depth_pipeline;
    }

    /// Rebuild the `@group(2)` extern bind group after the underlying
    /// scene_ldr_snapshot or scene_depth views change (resize). Same
    /// pattern as `SharedResources::rebuild_camera_bind_group`.
    pub fn rebuild_extern_bind_group(
        &mut self,
        device: &wgpu::Device,
        intermediates: &crate::halo::render::shared::IntermediateTargets,
        gbuffer: &crate::halo::render::shared::GBuffer,
    ) {
        self.water_extern_bind_group = build_water_extern_bind_group(
            device,
            &self.water_extern_bgl,
            &intermediates.scene_ldr_snapshot_view,
            &gbuffer.depth_sample_view,
            &self.scene_ldr_sampler,
            &self.scene_depth_sampler,
            &self.cbuf_ldr_texture_size,
            &self.cbuf_depth_constants,
        );
    }

    /// Compute + upload the water engine cbuffer trio for the current
    /// frame. Mirrors the shader-constant uploads in
    /// `c_water_renderer::render_shading @ 0x180693e90` (dllcache).
    /// Per-cluster overrides for `k_is_lightmap_exist` happen at draw
    /// time when each cluster is bound (Phase A4 wires that).
    ///
    /// `is_water_interaction` and `is_water_tessellated` are set per-draw
    /// in the engine (`render_water_part`) — for protomorph v1 we keep
    /// them globally `false` (no ripple, no tessellation).
    pub fn update_constants(
        &self,
        queue: &wgpu::Queue,
        game: &crate::game::GameState,
        intermediates: &crate::halo::render::shared::IntermediateTargets,
    ) {
        // ---- WaterPS ----
        // `k_water_view_depth_constant` lets the shader recover view-space
        // Z from the depth_buffer. Engine formula (per
        // `rasterizer_get_z_planes` + `render_shading`):
        //   z_view = depth_constants.x / (depth - depth_constants.y)
        // For protomorph's reverse-Z infinite far perspective (depth=1
        // at near, depth=0 at infinity), `proj.z = near / -view_z` →
        // `view_z = -near / depth`, which fits the formula with
        // `depth_constants = (-near, 0, 0, 0)`.
        let near = game.camera.near_clip;
        let water_ps = GpuWaterPS {
            k_water_view_depth_constant: [-near, 0.0, 0.0, 0.0],
            k_is_lightmap_exist: u32::from(self.lightmaps_available),
            k_is_water_interaction: 0,           // Phase A9 toggles
            // Engine `c_water_renderer::render_water_part @ 0x1806943c0`
            // sets `set_shader_constant_bool(-2142109695, &is_tessellated)`
            // per-draw. On PC `c_water_renderer::render_tessellation` is
            // a no-op (no hardware tessellation), but the shader
            // constant still controls VS displacement — the engine
            // sets it to TRUE for active water rendering so the patch-
            // rendered surface gets per-vertex wave displacement. We
            // mirror that. (Was 0 in Phase A4d when displacement
            // wasn't ported; A7a turns it on.)
            k_is_water_tessellated: 1,
            // Phase A8: drives the underwater-fog branch in
            // `entry_water_shading.wgsl` final composition. Engine
            // populates from `g_is_underwater` (set by water-volume
            // containment test in the render path). For protomorph
            // we forward `self.is_underwater` directly — host wires
            // up detection via `set_camera_underwater` (defaults to
            // false; manual toggle works for testing).
            k_is_camera_underwater: u32::from(self.is_underwater),
            k_debug_viz: std::env::var("PROTOMORPH_DEBUG_WATER_VIZ")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0),
            _pad_viz: [0; 3],
        };
        queue.write_buffer(&self.cbuf_water_ps, 0, bytemuck::bytes_of(&water_ps));

        // ---- WaterSharedPS ----
        // `k_water_view_xform_inverse` = inverse(view × projection) —
        // the matrix the water shader uses for screen→world back-
        // projection in the refraction lookup
        // (`water_shading_fx.hlsl:816`).
        let view_proj = game.camera.projection * game.camera.view;
        let inv_view_proj = view_proj.inverse();
        // `k_water_player_view_constant`: HLSL `texcoord_ss = .xy +
        // texcoord_ss * .zw`. For a fullscreen single-viewport setup,
        // offset=(0,0) and scale=(1,1).
        let player_window_constants = [0.0_f32, 0.0, 1.0, 1.0];
        let camera_pos = game.camera.position;
        let water_shared_ps = GpuWaterSharedPS {
            k_water_view_xform_inverse: inv_view_proj.to_cols_array_2d(),
            k_water_player_view_constant: player_window_constants,
            k_ps_camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            // Phase A8 sources real values from the active water tag
            // (`s_water_globals` or scenario underwater metadata); for
            // now the engine defaults are murk=0.5, fog=neutral gray.
            k_ps_underwater_murkiness: [0.5, 0.0, 0.0, 0.0],
            k_ps_underwater_fog_color: [0.05, 0.18, 0.30, 1.0],
        };
        queue.write_buffer(
            &self.cbuf_water_shared_ps,
            0,
            bytemuck::bytes_of(&water_shared_ps),
        );

        // ---- WaterSharedVS ----
        // `k_ripple_buffer_radius` + `k_ripple_buffer_center` — engine
        // values from `update_water_part_list` (radius is a static
        // constant; center is `g_interaction_buffer_center` =
        // camera.forward × radius). Phase A9 wires the real ripple
        // state.
        let water_shared_vs = GpuWaterSharedVS {
            k_ripple_buffer_radius: [0.0, 0.0, 0.0, 0.0],
            k_ripple_buffer_center: [0.0, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.cbuf_water_shared_vs,
            0,
            bytemuck::bytes_of(&water_shared_vs),
        );

        // ---- Extern cbuf 0x2F (LDRTextureSize) ----
        // Engine populates this when extern 45 binds
        // `_surface_screenshot_simple_resolve` — the dispatch handler
        // writes `(w, h, 1/w, 1/h)` into cbuf 0x2F so shaders can
        // compute texel-aligned UVs. We mirror by sampling the
        // intermediates' scene_ldr_snapshot dimensions.
        let snap_size = intermediates.scene_ldr_snapshot_texture.size();
        let w = snap_size.width.max(1) as f32;
        let h = snap_size.height.max(1) as f32;
        let ldr_size = GpuLdrTextureSize {
            size: [w, h, 1.0 / w, 1.0 / h],
        };
        queue.write_buffer(
            &self.cbuf_ldr_texture_size,
            0,
            bytemuck::bytes_of(&ldr_size),
        );

        // ---- Extern cbuf 0x3C (DepthConstants) ----
        // Engine populates via `rasterizer_get_z_planes` when extern 3
        // binds `_surface_depth_stencil`. Standard formula:
        //   view_z = depth_constants.x / (depth - depth_constants.y)
        // For protomorph's reverse-Z infinite-far perspective the
        // formula collapses to `view_z = -near / depth`, which matches
        // `(-near, 0, …, …)`. .z and .w are unused in the shader's
        // current scope — Phase A6 will plumb the canonical
        // `(z_far/(z_far-z_near), z_near/(z_near-z_far),
        //   (z_far-z_near)/(z_far*z_near), -1.0/z_far)` if a non-
        // reverse-Z code path emerges.
        let depth_consts = GpuDepthConstants {
            depth_constants: [-near, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.cbuf_depth_constants,
            0,
            bytemuck::bytes_of(&depth_consts),
        );

        // ---- rmw user cbuffer overlay (Phase A7c) ----
        // Engine `update_constants @ 0x180685300` re-evaluates animated
        // parameters per frame and overlays into the rmt2 user cbuffer
        // at PS slot 0x13. We mirror by re-resolving via
        // `rebuild_cbuffer_bytes_at_time(rmsh, rmt2, rmop_params, t)`
        // — the time-aware sibling of the load-time bake — and
        // queue.write_buffer'ing the result.
        //
        // For riverworld water this animates `time_warp` (MultiSpline
        // identity → t) which drives wave_displacement_array slice
        // selection in the VS, plus `*_texture` translation channels
        // used by the wave_slope_array / foam_texture sampling.
        if let Some(src) = self.water_material_source.as_ref() {
            // Water always loads with rmt2 populated (water_material_source
            // is only set on the rmw+rmt2 path); rmt2 is None for
            // author-format-only materials which water isn't. The unwrap
            // is safe under that invariant; if a future rmw tag ships
            // without an rmt2 it'll panic loudly here instead of
            // silently freezing animation.
            let rmt2 = src.rmt2.as_ref().expect(
                "water_material_source carries an rmw material; rmt2 was loaded at scenario load",
            );
            let ctx = blam_tags::render_method::DefaultEvalContext {
                eval_time: game.total_time,
            };
            let cb = blam_tags::render_method::resolve_pixel_user_cbuffer_at_time(
                &src.rmsh, rmt2, &src.rmop_params, &ctx,
            );
            // Match the load-time path: repack rmt2-ordered bytes into
            // the WGSL struct's hardcoded field order. Use the cached
            // load-time bytes as fallback so phantom slots (rmsh/rmop
            // params not in rmt2.float_constants — e.g.
            // `fresnel_curve_steepness`) keep their authored values.
            let packed = repack_water_cbuffer_to_wgsl_order(
                &cb,
                Some(&self.cbuf_water_material_loaded_bytes),
            );
            let n = (packed.len() as u64).min(self.cbuf_water_material_size) as usize;
            queue.write_buffer(&self.cbuf_water_material_ps, 0, &packed[..n]);
        }
    }

    /// `c_water_renderer::game_update @ render_water.h:31`.
    pub fn game_update(&mut self) {}

    /// Set the camera-underwater flag, mirroring `g_is_underwater` in
    /// the engine. Drives the `k_is_camera_underwater` shader
    /// constant for the underwater fog branch
    /// (`water_shading_fx.hlsl:1042-1046`).
    ///
    /// Engine sets this via water-volume containment in the render
    /// path; protomorph defers automatic detection (would walk water
    /// mesh AABBs vs camera position). Currently call sites just
    /// leave it false; manually toggling exercises the shader path.
    pub fn set_camera_underwater(&mut self, flag: bool) {
        self.is_underwater = flag;
    }

    /// `c_water_renderer::game_interation_event_add @ h:32` (Bungie
    /// typo "interation" preserved).
    pub fn game_interaction_event_add(
        &mut self,
        _water_ripple_definition_index: i32,
        _position: Vec3,
        _object_velocity: Vec3,
        _water_velocity: Vec3,
    ) {
    }

    /// `c_water_renderer::set_performance_throttles @ h:33`.
    pub fn set_performance_throttles(&mut self) {}

    /// `c_water_renderer::ripple_check_new @ h:34`.
    pub fn ripple_check_new(&self) -> u32 {
        0
    }

    /// `c_water_renderer::ripple_add @ h:35`.
    pub fn ripple_add(&mut self, _valid_event_count: u32) {}

    /// `c_water_renderer::ripple_update @ h:36`.
    pub fn ripple_update(&mut self) {}

    /// `c_water_renderer::is_active_ripple_exist @ h:37`.
    pub fn is_active_ripple_exist(&self) -> bool {
        false
    }

    /// `c_water_renderer::update_water_part_list @ h:38`. Returns
    /// true if water surfaces are visible this frame.
    pub fn update_water_part_list(&mut self) -> bool {
        false
    }

    /// `c_water_renderer::ripple_apply @ h:39`. Pumps the ripple
    /// atlas onto the water surface normal map.
    pub fn ripple_apply(&self) {}

    /// `c_water_renderer::ripple_slope @ h:40`. Computes the slope
    /// derivative for normal reconstruction in the shader.
    pub fn ripple_slope(&self) {}

    /// `c_water_renderer::render_tessellation @ h:41`.
    pub fn render_tessellation(&self, _is_screenshot: bool) {}

    /// `c_water_renderer::render_shading @ 0x180693e90` (dllcache).
    /// Two sub-passes per `reference_dllcache_water_pipeline.md`:
    ///
    /// 1. **Shading** — `set_z_buffer_mode(read)`, color-write RGBA,
    ///    `render_cluster_parts(_entry_point_static_lighting_per_vertex, 3)`
    ///    using `_vertex_type_water` instanced patch rendering. Reads
    ///    scene LDR via extern 45, scene depth via extern 3. Outputs
    ///    refraction-baked surface color to `lighting_base`.
    /// 2. **Shadow_generate** — `set_z_buffer_mode(write)`,
    ///    color-write OFF, `render_cluster_parts(_entry_point_shadow_generate, 3)`.
    ///    Writes water depth into the main z-buffer so following
    ///    transparency draws know about water occlusion.
    ///
    /// Phase A1d body: opens a render pass on `lighting_base` (depth
    /// read-only), binds the water debug pipeline + 3-stream vertex
    /// layout (per-mesh control points / engine-static barycentric /
    /// per-mesh lighting attrs), and dispatches `draw_indexed_instanced`
    /// for every water-bearing BSP mesh. Output: barycentric tricolor
    /// visualization at correct world positions — verifies the
    /// `_vertex_type_water` packing end-to-end. The two engine
    /// sub-passes (shading proper + shadow_generate depth-only) land
    /// in Phase A4 alongside the real water shader.
    /// Run before `render_underwater_fog` + `render_shading`. Mirrors
    /// the engine's `g_is_underwater` update that happens upstream of
    /// `c_player_view::render_water`. Approximates the per-water-volume
    /// containment test via per-mesh AABB.
    pub fn update_underwater_state(
        &mut self,
        ctx: &crate::halo::render::shared::FrameContext,
    ) {
        // TEMP: underwater fog disabled until the per-water-volume
        // boundary detection is tightened. The current per-mesh AABB
        // test fires above the actual water surface (`aabb_max.z`
        // lives at the upper extent of the WHOLE water mesh, not the
        // local surface plane) — needs a per-fragment / per-region
        // test against the water mesh's actual surface, or an
        // engine-faithful port of the per-water-volume containment
        // check from `c_player_view::set_is_underwater`. Re-enable
        // by deleting the early-return below.
        let _ = ctx;
        if self.is_underwater {
            eprintln!("[water] underwater fog force-disabled (boundary work pending)");
        }
        self.is_underwater = false;

        // Phase B6: upload `EngineUnderwater` to camera_bgl @binding(7)
        // so every transparent PS can apply the engine's underwater
        // fog branch. Murkiness/fog_color come from the same scenario
        // sources WaterSharedPS uses (placeholder values today; A8
        // notes the real-source plumbing as deferred).
        let engine_uw = crate::halo::render::shared::GpuEngineUnderwater {
            fog_color: [0.05, 0.18, 0.30, 1.0],
            murkiness: [0.5, 0.0, 0.0, 0.0],
            is_underwater_flags: [
                if self.is_underwater { 1.0 } else { 0.0 },
                0.0, 0.0, 0.0,
            ],
        };
        ctx.shared.queue.write_buffer(
            &ctx.shared.engine_underwater_buffer,
            0,
            bytemuck::bytes_of(&engine_uw),
        );

        // Upload the water cbuffer trio + extern cbufs NOW so both
        // render_underwater_fog and render_shading see fresh values.
        // Engine equivalent: the shader-constant uploads inside
        // `c_water_renderer::render_shading @ 0x180693e90` plus the
        // upstream `g_xm_view_proj_inv` updates that happen before
        // `c_player_view::render_water` calls
        // `render_underwater_fog`. Both render funcs share these.
        self.update_constants(&ctx.shared.queue, ctx.game, ctx.intermediates);
    }

    pub fn render_shading(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &crate::halo::render::shared::FrameContext,
    ) {
        // Skip when water is disabled (engine: gated on
        // `render_water_enabled` in `update_water_part_list`).
        if !self.render_water_enabled {
            return;
        }

        // Count water surfaces; bail before opening an rpass when no
        // BSP carries any. Cheap path for non-water levels.
        let mut water_meshes_present = false;
        for bsp in &ctx.structure_renderer.bsps {
            if bsp.meshes.iter().any(|m| m.water.is_some()) {
                water_meshes_present = true;
                break;
            }
        }
        if !water_meshes_present {
            return;
        }

        // Cbuffers are already uploaded by `update_underwater_state`
        // (called once before `render_underwater_fog` + this pass).

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water_shading"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.lighting_base_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // RT1 — `hdr_dark` (engine `_surface_screenshot_composite
                // _cubemap`, the bloom-extract feed). Engine
                // `convert_to_render_target` writes both per fragment.
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
                // `depth_ops: None` marks the depth aspect read-only
                // for the duration of the pass, mirroring engine
                // `set_depth_stencil_surface(_surface_depth_stencil_read_only)`
                // in `c_player_view::render_water @ 0x18068c120`.
                // Read-only depth lets us BOTH depth-test (rasterizer
                // reads existing values via `depth_compare: Greater`,
                // pipeline `depth_write_enabled: false`) AND sample the
                // same texture via `@group(2) @binding(2)` for water's
                // soft-edge fade / atmospheric fog (Phase A6).
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&self.water_pipeline);
        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group_sl,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 shares the ravi cursor.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );
        // @group(1) = water engine cbuf trio (WaterPS, WaterSharedPS,
        // WaterSharedVS — uploaded above by `update_constants`).
        rpass.set_bind_group(1, &self.water_engine_bind_group, &[]);
        // @group(2) = water extern resources: scene_ldr (extern 45) +
        // scene_depth (extern 3) textures + samplers + cbuf 0x2F /
        // 0x3C. Mirrors the dllcache extern dispatch in
        // `c_player_view::render_water @ 0x18068c120`.
        rpass.set_bind_group(2, &self.water_extern_bind_group, &[]);
        // @group(3) = per-rmw-material resources: environment_map
        // cubemap + sampler + WaterMaterialPS cbuf (reflection /
        // fresnel coefficients, sunspot cut). Phase A6a — single slot.
        rpass.set_bind_group(3, &self.water_material_bind_group, &[]);
        rpass.set_index_buffer(
            self.tessellation_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        // Stream 1 — engine-static barycentric VB. Bound once; same
        // for every water draw across every BSP mesh.
        rpass.set_vertex_buffer(1, self.tessellation_vertex_buffer.slice(..));

        // Engine: `c_water_renderer::render_water_part` issues one
        // `draw_indexed_instanced` per water-flagged part — different
        // rmw materials in the same mesh route to different shader
        // pipelines via `render_method_submit`. Phase A3 mirrors that
        // dispatch granularity (per-part ranges); Phase A4 plumbs the
        // material's option-driven pipeline lookup. Until A4 lands,
        // every part draws through the same debug stub.
        for bsp in &ctx.structure_renderer.bsps {
            for mesh in &bsp.meshes {
                let Some(water) = mesh.water.as_ref() else { continue };
                if water.parts.is_empty() {
                    continue;
                }
                rpass.set_vertex_buffer(0, water.control_points.slice(..));
                rpass.set_vertex_buffer(2, water.lighting_attrs.slice(..));
                for part in &water.parts {
                    if part.instance_count == 0 {
                        continue;
                    }
                    let start = part.instance_start;
                    let end = start + part.instance_count;
                    rpass.draw_indexed(0..675, 0, start..end);
                }
            }
        }
        // Sub-pass 1 done — drop the rpass handle so we can open a new
        // one with writable depth attachment for sub-pass 2.
        drop(rpass);

        // Sub-pass 2 — engine `_entry_point_shadow_generate` per IDA
        // `c_water_renderer::render_shading @ 0x180693e90`. Same draw
        // list, color writes off, depth writes on. Lays water surface
        // depth into the gbuffer so transparents drawn after water
        // depth-test against it.
        //
        // Depth-only rpass: no color attachment, no @group(2) bind
        // (the extern resources include the scene_depth view as a
        // sampled resource — same texture we're writing depth to in
        // this pass — wgpu forbids the simultaneous use). Pipeline is
        // VS-only with a slimmer 3-group layout.
        let mut depth_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water_shadow_generate"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                // Writable depth: load existing values, store updates.
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        depth_pass.set_pipeline(&self.water_depth_pipeline);
        depth_pass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group_sl,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 shares the ravi cursor.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );
        depth_pass.set_bind_group(1, &self.water_engine_bind_group, &[]);
        // Slot 2 is an EMPTY placeholder (the real `water_extern_bind_group`
        // would re-expose scene_depth as a sampled resource that conflicts
        // with the depth-write attachment). Material stays at @group(3)
        // so the WGSL `@group(3) @binding(N)` declarations still resolve.
        depth_pass.set_bind_group(2, &self.depth_empty_bind_group, &[]);
        depth_pass.set_bind_group(3, &self.water_material_bind_group, &[]);
        depth_pass.set_index_buffer(
            self.tessellation_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        depth_pass.set_vertex_buffer(1, self.tessellation_vertex_buffer.slice(..));
        for bsp in &ctx.structure_renderer.bsps {
            for mesh in &bsp.meshes {
                let Some(water) = mesh.water.as_ref() else { continue };
                if water.parts.is_empty() {
                    continue;
                }
                depth_pass.set_vertex_buffer(0, water.control_points.slice(..));
                depth_pass.set_vertex_buffer(2, water.lighting_attrs.slice(..));
                for part in &water.parts {
                    if part.instance_count == 0 {
                        continue;
                    }
                    let start = part.instance_start;
                    let end = start + part.instance_count;
                    depth_pass.draw_indexed(0..675, 0, start..end);
                }
            }
        }
    }

    /// `c_water_renderer::render_underwater_fog @ 0x180694080`. Engine
    /// body (dllcache):
    /// ```c
    /// if ( g_is_underwater )
    /// {
    ///   set_z_buffer_mode(off); set_cull_mode(off); set_color_write_enable(0, 15);
    ///   upload k_water_view_xform_inverse / player_view / camera /
    ///     murkiness / fog_color cbuffers (slot 0x560000-0x560007);
    ///   set_explicit_shaders(58, 0x13, _none, _entry_point_shadow_apply, 0);
    ///   set_surface_as_texture(0, _surface_screenshot_simple_resolve);  // bilinear+clamp
    ///   set_surface_as_texture(1, _surface_depth_stencil);              // point+clamp
    ///   draw_vertices(triangle_strip, 0, 4);
    /// }
    /// ```
    /// Per `c_player_view::render_water @ 0x18068C120` this runs INSIDE
    /// the water-pass block, BEFORE `render_shading` — it fogs the
    /// scene LDR snapshot in-place; water surfaces draw on top of the
    /// fogged buffer in the subsequent `render_shading` call.
    pub fn render_underwater_fog(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &crate::halo::render::shared::FrameContext,
    ) {
        if !self.is_underwater {
            return;
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water_underwater_fog"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.lighting_base_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.underwater_fog_pipeline);
        // camera_bg has a dynamic offset for the engine lighting cbuf
        // slot — even though this fullscreen pass doesn't read it,
        // the layout requires the offset to be supplied. Use the
        // standard default offset (matches `water_pipeline` above).
        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group_sl,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 shares the ravi cursor.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );
        rpass.set_bind_group(1, &self.water_engine_bind_group, &[]);
        rpass.set_bind_group(2, &self.water_extern_bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    /// `c_water_renderer::render_cluster_parts @ h:46`. Internal
    /// driver (private in Halo).
    fn render_cluster_parts(&self, _entry_point: EntryPoint, _mesh_part_mask: i32) {}

    /// `water_interaction_clear_all @ h:53`.
    pub fn water_interaction_clear_all(&mut self, _game_state_restore_flags: i32) {
        self.new_interaction_event_count = 0;
    }
}

/// `s_new_interaction_event` placeholder. Halo's struct carries
/// position + velocity + ripple-definition for a single splash.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewInteractionEvent {
    pub water_ripple_definition_index: i32,
    pub position: Vec3,
    pub object_velocity: Vec3,
    pub water_velocity: Vec3,
}

/// Build the fullscreen underwater-fog pipeline. Engine equivalent:
/// the explicit-shader bind in
/// `c_water_renderer::render_underwater_fog @ 0x180694080`
/// (entry index 58 = `_entry_point_shadow_apply`, which underwater_vs/ps
/// alias to per `water_ripple_hlsl.hlsl:44-45`).
///
/// Layout matches the water_shading pipeline's first three groups:
/// camera at @group(0), water_engine_bgl at @group(1) (we read
/// WaterSharedPS from binding 1), water_extern_bgl at @group(2)
/// (scene LDR + scene depth). No vertex buffers — VS uses
/// `vertex_index` to emit 4 NDC corners as a triangle strip.
fn build_underwater_fog_pipeline(
    shared: &crate::halo::render::shared::SharedResources,
    water_engine_bgl: &wgpu::BindGroupLayout,
    water_extern_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let device = &shared.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entry_underwater_fog"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../assets/halo_shaders/entry_underwater_fog.wgsl").into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("water_underwater_fog_pipeline_layout"),
        bind_group_layouts: &[
            &shared.camera_bgl,
            water_engine_bgl,
            water_extern_bgl,
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water_underwater_fog_pipeline"),
        layout: Some(&pipeline_layout),
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
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            // Engine: `set_z_buffer_mode(off)` + `set_cull_mode(off)`.
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// rmdf category choices that drive WGSL substitution per Phase W2/W4.
/// Each field is the rmdf's chosen option name for that category, as
/// they appear in the `tag_trees/shaders/water_options/` corpus
/// (waveshape_*, watercolor_*, etc.). Built from `mat.category_choices`
/// at material load and consumed by `substitute_water_variant` to
/// expand `__CATEGORY_OPTION__` boolean tokens in the WGSL source so
/// naga DCE drops the dead branches.
///
/// Engine reference: `TEST_CATEGORY_OPTION(<category>, <option>)` macro
/// in `water_shading_fx.hlsl` (lines 343, 376, 510-511, 712, 727, 792,
/// 796, 807, 812, 880, 886, 894, 898, 902, 932, 937, 958, 1002, 1005,
/// 1009).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaterVariant {
    pub watercolor:   String, // pure / texture
    pub global_shape: String, // none / paint / depth
    pub bankalpha:    String, // none / depth / paint
    pub waveshape:    String, // default / bump (none implicit)
    pub foam:         String, // none / auto / paint / both
    pub reflection:   String, // none / static / static_ssr / dynamic
    pub refraction:   String, // none / dynamic
}

impl WaterVariant {
    /// Engine-conservative defaults (matches riverworld
    /// `_0_1_1_1_1_0_1_3`). Used at renderer init before any material
    /// is loaded.
    pub fn defaults_riverworld() -> Self {
        Self {
            watercolor:   "texture".to_string(),
            global_shape: "paint".to_string(),
            bankalpha:    "depth".to_string(),
            waveshape:    "default".to_string(),
            foam:         "both".to_string(),
            reflection:   "static".to_string(),
            refraction:   "dynamic".to_string(),
        }
    }

    pub fn from_choices(c: &crate::halo::render_methods::CategoryChoices) -> Self {
        Self {
            watercolor:   c.get_or("watercolor",   "texture").to_string(),
            global_shape: c.get_or("global_shape", "paint").to_string(),
            bankalpha:    c.get_or("bankalpha",    "depth").to_string(),
            waveshape:    c.get_or("waveshape",    "default").to_string(),
            foam:         c.get_or("foam",         "both").to_string(),
            reflection:   c.get_or("reflection",   "static").to_string(),
            refraction:   c.get_or("refraction",   "dynamic").to_string(),
        }
    }
}

/// Phase W2/W4 — substitute `__CATEGORY_OPTION__` literal-bool tokens in
/// the water WGSL source per the material's rmdf choice. Mirrors the
/// rmsh/rmtr pattern in `render_methods/mod.rs::assemble`. Naga DCE
/// drops the dead branches.
///
/// `reflection=static_ssr` maps to the `static` branch — MCC H3's
/// `water_shading_fx.hlsl` only has `none/static/dynamic` branches.
/// Reach+ adds true screen-space reflection support; for H3-equivalence
/// the static branch is the closest match (`static_ssr` was originally
/// a hint to use SSR if available, falling back to static otherwise).
fn substitute_water_variant(src: &str, v: &WaterVariant) -> String {
    let b = |on: bool| -> &'static str { if on { "true" } else { "false" } };
    src
        // watercolor
        .replace("__WATERCOLOR_PURE__",    b(v.watercolor == "pure"))
        .replace("__WATERCOLOR_TEXTURE__", b(v.watercolor == "texture"))
        // global_shape
        .replace("__GLOBAL_SHAPE_NONE__",  b(v.global_shape == "none"))
        .replace("__GLOBAL_SHAPE_PAINT__", b(v.global_shape == "paint"))
        .replace("__GLOBAL_SHAPE_DEPTH__", b(v.global_shape == "depth"))
        // bankalpha
        .replace("__BANKALPHA_NONE__",  b(v.bankalpha == "none"))
        .replace("__BANKALPHA_DEPTH__", b(v.bankalpha == "depth"))
        .replace("__BANKALPHA_PAINT__", b(v.bankalpha == "paint"))
        // waveshape
        .replace("__WAVESHAPE_DEFAULT__", b(v.waveshape == "default"))
        .replace("__WAVESHAPE_BUMP__",    b(v.waveshape == "bump"))
        .replace("__WAVESHAPE_NONE__",    b(v.waveshape == "none"))
        // foam
        .replace("__FOAM_NONE__",  b(v.foam == "none"))
        .replace("__FOAM_AUTO__",  b(v.foam == "auto"))
        .replace("__FOAM_PAINT__", b(v.foam == "paint"))
        .replace("__FOAM_BOTH__",  b(v.foam == "both"))
        // reflection — `static_ssr` falls back to `static`
        .replace("__REFLECTION_NONE__",    b(v.reflection == "none"))
        .replace("__REFLECTION_STATIC__",  b(v.reflection == "static" || v.reflection == "static_ssr"))
        .replace("__REFLECTION_DYNAMIC__", b(v.reflection == "dynamic"))
        // refraction
        .replace("__REFRACTION_NONE__",    b(v.refraction == "none"))
        .replace("__REFRACTION_DYNAMIC__", b(v.refraction == "dynamic"))
}

/// Build the water-shading pipeline — minimal `water_shading()` port
/// (`water_shading_fx.hlsl:658-1062`). Reads scene LDR snapshot for
/// refraction + Schlick fresnel against view direction.
fn build_water_pipeline(
    shared: &crate::halo::render::shared::SharedResources,
    water_engine_bgl: &wgpu::BindGroupLayout,
    water_extern_bgl: &wgpu::BindGroupLayout,
    water_material_bgl: &wgpu::BindGroupLayout,
    variant: &WaterVariant,
) -> wgpu::RenderPipeline {
    let device = &shared.device;
    let wgsl_src = patch_water_uv_debug(include_str!(
        "../../../assets/halo_shaders/entry_water_shading.wgsl"
    ));
    let wgsl_src = substitute_water_variant(&wgsl_src, variant);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entry_water_shading"),
        source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("water_water_pipeline_layout"),
        // @group(0) = camera_bgl (camera matrices + lighting +
        //   atlases — engine-shared resources).
        // @group(1) = water_engine_bgl (cbuffer trio: WaterPS,
        //   WaterSharedPS, WaterSharedVS).
        // @group(2) = water_extern_bgl (extern 45 LDR + extern 3
        //   depth + samplers + cbuf 0x2F / 0x3C).
        // @group(3) = water_material_bgl (per-rmw env cubemap +
        //   sampler + WaterMaterialPS cbuf).
        bind_group_layouts: &[
            &shared.camera_bgl,
            water_engine_bgl,
            water_extern_bgl,
            water_material_bgl,
        ],
        immediate_size: 0,
    });
    let layouts = water_vertex_buffer_layouts();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water_water_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[
                // RT0 — engine `_surface_screenshot_composite_depth` (LDR,
                // our `lighting_base`). Water draws OPAQUE
                // (`convert_to_render_target(float4(output_color, 1.0), ...)`
                // at water_shading_fx.hlsl:1057/1061). Shore softness comes
                // from the COLOR mix `output_color = lerp(color_refraction_bed,
                // output_color, bank_alpha)` — at the shoreline that lerp
                // returns the refracted scene sample which (with small
                // perturbation) matches the underlying terrain pixel.
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // RT1 — engine `_surface_screenshot_composite_cubemap` (HDR,
                // our `hdr_dark`, the bloom-extract feed). `convert_to
                // _render_target` writes both per fragment. On MCC PC
                // `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0` so RT0 == RT1.
                // Required so bright water contributes to the bloom feed
                // → auto-exposure stops over-correcting upward.
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            // Engine `c_water_renderer::render_shading @ 0x180693e90`
            // (IDA): `v2 = _cull_mode_ccw; if (!g_is_underwater) v2 =
            // _cull_mode_cw; set_cull_mode(v2);` — above water cull CW,
            // underwater cull CCW. `set_cull_mode @ 0x18066E930` maps
            // engine `_cull_mode_cw` (mode=2) → `D3DCULL_CW` = cull
            // CW-wound triangles.
            //
            // Halo content windings are mirrored at our import vs D3D9
            // viewport conventions, so wgpu `front_face: Ccw +
            // cull_mode: Some(Front)` empirically produces the same
            // visible face the engine renders above water (verified
            // when this fixed the wave-back "tooth/spike" artifacts
            // from Phase A1d's diagnostic `cull_mode: None`). (Phase A1d
            // shipped `cull_mode: None` as a diagnostic, which left
            // back faces visible — produced "tooth/spike" wave-back
            // artifacts at oblique view angles.)
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            // Engine sub-pass 1: depth-test on, depth-write OFF
            // (read-only depth; sub-pass 2 writes water depth).
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Sub-pass 2 — `_entry_point_shadow_generate` per IDA
/// `c_water_renderer::render_shading @ 0x180693e90`. Same VS as the
/// shading pipeline; FS color writes are masked off and depth-write is
/// enabled so the water surface lays its z into the gbuffer depth
/// buffer for transparency ordering. The engine sequence:
/// ```c
/// set_color_write_enable(0, 0);
/// set_depth_stencil_surface(_surface_depth_stencil, ...);  // writable
/// set_z_buffer_mode(_z_buffer_mode_write);
/// render_cluster_parts(_entry_point_shadow_generate, 3);
/// set_color_write_enable(0, 15);
/// set_depth_stencil_surface(_surface_depth_stencil_read_only, ...);
/// ```
fn build_water_depth_pipeline(
    shared: &crate::halo::render::shared::SharedResources,
    water_engine_bgl: &wgpu::BindGroupLayout,
    _water_extern_bgl: &wgpu::BindGroupLayout,
    water_material_bgl: &wgpu::BindGroupLayout,
    variant: &WaterVariant,
) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
    let device = &shared.device;
    // The depth-only pipeline reuses the same WGSL file as the shading
    // pipeline (we only run vs_main + an empty shadow_generate PS), so
    // the VS's `__GLOBAL_SHAPE_*__` / `__WATERCOLOR_*__` /
    // `__BANKALPHA_*__` tokens must also be substituted here — they
    // gate the vertex displacement amount via height_scale_global /
    // choppy_scale_global. The PS-only branches just DCE since fs_main
    // is unused in this pipeline.
    let wgsl_src = patch_water_uv_debug(include_str!(
        "../../../assets/halo_shaders/entry_water_shading.wgsl"
    ));
    let wgsl_src = substitute_water_variant(&wgsl_src, variant);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entry_water_shading_depth"),
        source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
    });
    // VS-only layout: keep @group(2) as an EMPTY placeholder so the
    // shader's `@group(3)` water_material declarations land at the
    // right index in the layout. We can't use the real
    // `water_extern_bgl` because its scene_depth view is bound as a
    // sampled resource AND is the writable depth attachment in this
    // pass — wgpu forbids that dual usage.
    let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("water_depth_empty_bgl"),
        entries: &[],
    });
    let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water_depth_empty_bg"),
        layout: &empty_bgl,
        entries: &[],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("water_depth_pipeline_layout"),
        bind_group_layouts: &[
            &shared.camera_bgl,
            water_engine_bgl,
            &empty_bgl,
            water_material_bgl,
        ],
        immediate_size: 0,
    });
    let layouts = water_vertex_buffer_layouts();
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water_depth_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &layouts,
            compilation_options: Default::default(),
        },
        // VS-only — no fragment stage. Engine sub-pass 2 does
        // `set_color_write_enable(0, 0)` then the shadow_generate
        // entry-point's PS is null (depth-only). Equivalent in wgpu:
        // omit fragment state entirely and use a depth-only rpass.
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            // Engine sub-pass 2: `set_z_buffer_mode(_z_buffer_mode_write)`.
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, empty_bg)
}

/// Build the `@group(2)` extern bind group from current view + sampler
/// + cbuffer handles. Factored out so resize can rebuild without
/// re-touching constants in `WaterRenderer`.
fn build_water_extern_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_ldr_view: &wgpu::TextureView,
    scene_depth_view: &wgpu::TextureView,
    scene_ldr_sampler: &wgpu::Sampler,
    scene_depth_sampler: &wgpu::Sampler,
    cbuf_ldr_texture_size: &wgpu::Buffer,
    cbuf_depth_constants: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water_extern_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_ldr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(scene_ldr_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(scene_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(scene_depth_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cbuf_ldr_texture_size.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: cbuf_depth_constants.as_entire_binding(),
            },
        ],
    })
}

/// Build the `@group(3)` material bind group from the env cubemap +
/// material cbuffer + sampler. Factored out so [`WaterRenderer::load_water_material`]
/// can rebuild after the BSP supplies the real cubemap.
fn build_water_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    env_cubemap_view: &wgpu::TextureView,
    env_cubemap_sampler: &wgpu::Sampler,
    cbuf_water_material_ps: &wgpu::Buffer,
    wave_array_view: &wgpu::TextureView,
    wave_array_sampler: &wgpu::Sampler,
    watercolor_view: &wgpu::TextureView,
    material_2d_sampler: &wgpu::Sampler,
    global_shape_view: &wgpu::TextureView,
    foam_view: &wgpu::TextureView,
    foam_detail_view: &wgpu::TextureView,
    wave_slope_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water_material_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(env_cubemap_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(env_cubemap_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cbuf_water_material_ps.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(wave_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(wave_array_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(watercolor_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(material_2d_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(global_shape_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(foam_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(foam_detail_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(wave_slope_view),
            },
        ],
    })
}

/// Build the engine-static water tessellation index + vertex buffers.
/// Mirrors `c_water_renderer::initialize @ 0x180692400` (dllcache):
/// 16-row triangular grid of barycentric coords + 15 bands of
/// 2k+1 sub-triangles between consecutive rows.
///
/// Returns `(indices, vertex_floats)` where:
/// - `indices`: 675 u16 indices forming 225 sub-triangles. Layout
///   per band k ∈ [0, 14]: 1 leading up-pointing triangle, then k
///   alternating down-up triangle pairs (2k+1 triangles total).
/// - `vertex_floats`: 408 f32 = 136 × vec3 of barycentric coords.
///   Vertex (row=k, col=j) where k ∈ [0, 15] and j ∈ [0, k] has
///   bc = (1 - k/15, (k-j)/15, j/15).
///
/// The water VS uses these barycentrics to interpolate between the 3
/// per-instance control points (`pos1`, `pos2`, `pos3`) packed in the
/// 156-byte slot-1 stream.
fn generate_tessellation_data() -> (Vec<u16>, Vec<f32>) {
    // ---- Index buffer: 15 bands × (2k+1) triangles ----
    let mut indices: Vec<u16> = Vec::with_capacity(675);
    let mut row_first_idx: u16 = 0; // v17 — first vertex index in current row
    for band in 0u16..15u16 {
        // v21 — first vertex index in next row (row+1).
        let next_row_first: u16 = band + row_first_idx + 1;
        // Leading up-pointing triangle of this band:
        //   (row k col 0, row k+1 col 1, row k+1 col 0)
        indices.push(row_first_idx);
        indices.push(next_row_first + 1);
        indices.push(next_row_first);
        // Inner quads — `band` of them. Each emits one down-pointing
        // then one up-pointing triangle (the alternating strip pattern
        // a triangle subdivision uses).
        for j in 0u16..band {
            // Down-pointing (top edge on row k):
            //   (row k col j, row k col j+1, row k+1 col j+1)
            indices.push(j + row_first_idx);
            indices.push(j + row_first_idx + 1);
            indices.push(j + next_row_first + 1);
            // Up-pointing (top vertex on row k):
            //   (row k col j+1, row k+1 col j+2, row k+1 col j+1)
            indices.push(j + row_first_idx + 1);
            indices.push(j + next_row_first + 2);
            indices.push(j + next_row_first + 1);
        }
        row_first_idx = next_row_first;
    }

    // ---- Vertex buffer: 16 rows, row k has k+1 vertices ----
    // Per the dllcache initialize code:
    //   bc.x = 1 - k/15
    //   bc.y = (k - j)/15
    //   bc.z = j/15
    // bc.x + bc.y + bc.z = 1 for all (k, j).
    let mut vertex_floats: Vec<f32> = Vec::with_capacity(136 * 3);
    let inv_n: f32 = 1.0 / 15.0;
    for k in 0u16..16u16 {
        let row_t = (k as f32) * inv_n;
        for j in 0u16..=k {
            let col_t = (j as f32) * inv_n;
            vertex_floats.push(1.0 - row_t);
            vertex_floats.push(row_t - col_t);
            vertex_floats.push(col_t);
        }
    }

    (indices, vertex_floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_format_sizes() {
        assert_eq!(std::mem::size_of::<WaterControlPointInstance>(), 156);
        assert_eq!(std::mem::size_of::<WaterLightingInstance>(), 72);
    }

    #[test]
    fn vertex_layouts_strides_match_engine() {
        let layouts = water_vertex_buffer_layouts();
        // Engine `set_vertex_streams` strides from
        // `c_water_renderer::render_water_part @ 0x1806943c0`.
        assert_eq!(layouts[0].array_stride, 156, "stream 0 (control points)");
        assert_eq!(layouts[1].array_stride, 12, "stream 1 (barycentric)");
        assert_eq!(layouts[2].array_stride, 72, "stream 2 (lighting attrs)");
        assert!(matches!(layouts[0].step_mode, wgpu::VertexStepMode::Instance));
        assert!(matches!(layouts[1].step_mode, wgpu::VertexStepMode::Vertex));
        assert!(matches!(layouts[2].step_mode, wgpu::VertexStepMode::Instance));
    }

    #[test]
    fn tessellation_buffer_sizes() {
        let (indices, vertices) = generate_tessellation_data();
        assert_eq!(indices.len(), 675, "expected 225 triangles × 3");
        assert_eq!(vertices.len(), 136 * 3, "expected 136 vec3 vertices");
    }

    #[test]
    fn tessellation_vertices_sum_to_one() {
        let (_indices, vertices) = generate_tessellation_data();
        for chunk in vertices.chunks_exact(3) {
            let s = chunk[0] + chunk[1] + chunk[2];
            assert!((s - 1.0).abs() < 1e-5, "barycentric sum != 1: {s}");
        }
    }

    #[test]
    fn tessellation_corners() {
        let (_indices, vertices) = generate_tessellation_data();
        // First vertex (row 0, col 0) — apex.
        assert_eq!(&vertices[0..3], &[1.0, 0.0, 0.0]);
        // Last vertex (row 15, col 15) — bottom-right corner.
        let last = vertices.len() - 3;
        assert!((vertices[last] - 0.0).abs() < 1e-5);
        assert!((vertices[last + 1] - 0.0).abs() < 1e-5);
        assert!((vertices[last + 2] - 1.0).abs() < 1e-5);
        // Bottom-left corner (row 15, col 0) — index 1+2+3+...+15 = 120.
        let bl = 120 * 3;
        assert!((vertices[bl] - 0.0).abs() < 1e-5);
        assert!((vertices[bl + 1] - 1.0).abs() < 1e-5);
        assert!((vertices[bl + 2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn tessellation_indices_in_range() {
        let (indices, vertices) = generate_tessellation_data();
        let max_vert = (vertices.len() / 3) as u16;
        for &idx in &indices {
            assert!(idx < max_vert, "index {idx} >= vertex count {max_vert}");
        }
    }

    /// W4 acceptance — `substitute_water_variant` resolves every
    /// `__CATEGORY_OPTION__` token across the full Cartesian product of
    /// rmw category options (the 16 ship variants + the unship `none`
    /// fallbacks). Any unresolved token would surface as `__FOO_BAR__`
    /// in the WGSL source and break naga at pipeline-build time.
    #[test]
    fn substitute_water_variant_resolves_all_tokens_for_every_variant() {
        let src = include_str!(
            "../../../assets/halo_shaders/entry_water_shading.wgsl"
        );

        // Option lists drawn from `tag_trees/shaders/water_options/`
        // plus the engine HLSL implicit fallbacks (`none`).
        let watercolors  = ["pure", "texture"];
        let global_shapes = ["none", "paint", "depth"];
        let bankalphas   = ["none", "depth", "paint"];
        let waveshapes   = ["default", "bump", "none"];
        let foams        = ["none", "auto", "paint", "both"];
        let reflections  = ["none", "static", "static_ssr", "dynamic"];
        let refractions  = ["none", "dynamic"];

        for &wc in &watercolors {
            for &gs in &global_shapes {
                for &ba in &bankalphas {
                    for &ws in &waveshapes {
                        for &fm in &foams {
                            for &rfl in &reflections {
                                for &rfr in &refractions {
                                    let v = WaterVariant {
                                        watercolor:   wc.to_string(),
                                        global_shape: gs.to_string(),
                                        bankalpha:    ba.to_string(),
                                        waveshape:    ws.to_string(),
                                        foam:         fm.to_string(),
                                        reflection:   rfl.to_string(),
                                        refraction:   rfr.to_string(),
                                    };
                                    let out = substitute_water_variant(src, &v);
                                    // Catch any unresolved __FOO_BAR__
                                    // tokens. A bare `__` is allowed in
                                    // identifiers, but `__UPPER_TOKEN__`
                                    // sandwiched between underscores is
                                    // our substitution pattern.
                                    for line in out.lines() {
                                        // Skip lines inside comments
                                        // (substitution-token references
                                        // in commentary).
                                        let trimmed = line.trim_start();
                                        if trimmed.starts_with("//") { continue; }
                                        for tok in line.split_whitespace() {
                                            if tok.starts_with("__")
                                                && tok.ends_with("__")
                                                && tok.len() > 4
                                                && tok.contains(char::is_uppercase)
                                            {
                                                panic!(
                                                    "Unresolved token `{tok}` for variant \
                                                     wc={wc} gs={gs} ba={ba} ws={ws} fm={fm} \
                                                     rfl={rfl} rfr={rfr}\nIn line: {line}",
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
