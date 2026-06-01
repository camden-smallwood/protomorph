// Engine `decal_fx.hlsl::default_vs` + `default_ps` (line 87 + 459).
//
// Vertex layout — `rasterizer_vertex_world` (44 B) from
// `c_decal::write_mesh_fragment`:
//   pos f32x3 + texcoord f32x2 + normal/tangent/binormal Snorm16x4
// (the i16 lanes decode to normalized [-1, 1] floats at attribute fetch).
//
// Bindings:
//   @group(0) @binding(0)  camera (shared with rest of the renderer)
//   @group(2) @binding(N)  per-rmop bitmaps (base_map / alpha_map /
//                          bump_map) + sampler `s_material` — emitted
//                          by `material_bindings::emit_wgsl`
//
// Engine's per-decal cbuffer constants (fade / sprite / pixel_kill /
// u_tiles / v_tiles) come from `set_shader_constant(0x140000, ...)` +
// `set_shader_constant(0x130000, ...)` in `c_decal::render`. The
// `decal_constants` UBO declaration is emitted by
// `material_bindings::emit_wgsl` when the material is rmd. We read
// it directly below; per-decal animation feeds these via the
// dynamic-offset ring buffer rebuilt each frame by the orchestrator.
//
// Per-option helper fragments are concatenated BEFORE this file by
// `render_methods::assemble`. Required helpers:
//   sample_diffuse(texcoord_tile, texcoord, palette_v) -> vec4<f32>
//   sample_bump(...)                                    -> vec3<f32>
//   tint_and_modulate(diffuse)                         -> vec4<f32>
//   fade_out(color)                                    -> vec4<f32>
//   const IS_FLAT_VERTEX               : bool
//   const BLEND_MODE_SELF_ILLUM        : bool
//   const BLEND_MODE_USES_SRC_ALPHA    : bool

struct DecalVertex {
    @location(0) position: vec3<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) binormal: vec4<f32>,
}

// `s_decal_interpolators` (decal_fx.hlsl:72-85). For IS_FLAT_VERTEX
// (bump_mapping=leave) the tangent/binormal/normal interpolators are
// stripped — WGSL doesn't support conditional struct fields, so we
// always declare them but the standard sample_bump variant only reads
// them when IS_FLAT_VERTEX = false.
struct DecalInterpolators {
    @builtin(position) m_position: vec4<f32>,
    @location(0) m_texcoord: vec4<f32>,
    @location(1) m_pos_w: f32,
    @location(2) m_tangent: vec3<f32>,
    @location(3) m_binormal: vec3<f32>,
    @location(4) m_normal: vec3<f32>,
}

// Engine `default_vs` body (decal_fx.hlsl:87-118).
// `always_local_to_view` is inlined: decal vertices already live in
// world space (write_mesh_fragment writes world positions when the
// fragment isn't instance-geom), so the transform is just
// camera.projection * camera.view * world_pos.
@vertex
fn vs_main(in: DecalVertex) -> DecalInterpolators {
    var out: DecalInterpolators;
    let world_pos = vec4<f32>(in.position, 1.0);
    out.m_position = camera.projection * camera.view * world_pos;

    // HLSL line 95-102: store both un-sprited and sprite-mapped UVs.
    let pixel_kill = decal_constants.pixel_kill_enabled != 0u;
    let sprite = decal_constants.sprite_corner;
    if (pixel_kill) {
        out.m_texcoord = vec4<f32>(
            in.texcoord,
            in.texcoord * sprite.zw + sprite.xy,
        );
    } else {
        out.m_texcoord = vec4<f32>(
            0.5, 0.5,
            in.texcoord * sprite.zw + sprite.xy,
        );
    }
    // HLSL line 104: m_texcoord.zw *= float2(u_tiles, v_tiles).
    out.m_texcoord = vec4<f32>(
        out.m_texcoord.x,
        out.m_texcoord.y,
        out.m_texcoord.z * decal_constants.u_tiles,
        out.m_texcoord.w * decal_constants.v_tiles,
    );
    out.m_pos_w = out.m_position.w;

    // IS_FLAT_VERTEX: engine guards these with `#if !IS_FLAT_VERTEX`.
    // We always populate; the PS only reads them when the bump
    // helper is the `standard` variant (IS_FLAT_VERTEX = false).
    out.m_tangent = in.tangent.xyz;
    out.m_binormal = in.binormal.xyz;
    out.m_normal = in.normal.xyz;
    return out;
}

// Dual-target output. Engine `default_ps` returns
// `convert_to_decal_target(color, bump, pos_w)` which for
// `RENDER_TARGET_ALBEDO_AND_NORMAL` expands to
// `convert_to_albedo_target_no_srgb` (albedo_pass_fx.hlsl:42):
//   result.albedo_specmask = albedo;      // RT0
//   result.normal.xyz      = bump * 0.5 + 0.5;  // RT1.xyz
//   result.normal.w        = albedo.w;    // RT1.w
// Engine `c_decal::render` does NOT mask RT1 for non-bump_modulate
// decals (only `set_color_write_enable(1, 0)` when def_flags & 2),
// so RT1 inherits the default-enabled mask and gets the bump-packed
// normal alpha-blended in alongside the albedo. Without this, SL
// reads the rock's unchanged normal at decal pixels and the decal's
// own surface detail (the bump_map authored alongside its base_map)
// never affects shading — exactly the "decal blends color but no
// bump shows" symptom on shrine/snowbound 2026-05-17.
struct DecalOut {
    @location(0) color: vec4<f32>,
    @location(1) dark: vec4<f32>,
}

// Debug visualization switch — substituted by `render_methods::assemble`
// from `PROTOMORPH_DECAL_VIZ` env var. Modes:
//   0 = normal output (engine default_ps)
//   1 = bitmap alpha as grayscale (.aaa, opaque)
//   2 = bitmap RGB only (.rgb, alpha=1.0)
//   3 = solid magenta — geometry visibility check
const DECAL_VIZ_MODE: u32 = __DECAL_VIZ_MODE__;

// Pre/post-lighting selector — substituted by `render_methods::assemble`
// from the rmd's `render_pass` choice. Pre-lighting writes bump to RT1
// (G-buffer normal); post-lighting writes duplicated color (PC
// `DARK_COLOR_MULTIPLIER = 1.0` so RT1 mirrors RT0 for bloom).
const RENDER_PASS_PRE_LIGHTING: bool = __DECAL_RENDER_PASS_PRE_LIGHTING__;

// Engine `default_ps` body (decal_fx.hlsl:459-475).
@fragment
fn fs_main(in: DecalInterpolators) -> DecalOut {
    // Engine `decal_fx.hlsl:465` — `clip(float4(uv.xy, 1-uv.xy))`.
    // Discards pixels outside the [0,1] sprite UV range so sprite-atlas
    // decals don't bleed into neighboring tiles at edges. HLSL forces
    // `pixel_kill_enabled = true` (line 461).
    let uv = in.m_texcoord.xy;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        discard;
    }
    let diffuse = sample_diffuse(in.m_texcoord.xy, in.m_texcoord.zw, 0.0);
    let tinted = tint_and_modulate(diffuse);
    let faded = fade_out(tinted);
    // Engine `decal_fx.hlsl:472` — sample_bump called regardless of
    // RT1 mask; `leave` variant returns (0,0,0) so the packed value
    // is the flat-vertex sentinel (0.5, 0.5, 0.5). Pipeline RT1
    // write_mask is empty for `leave` to preserve the rock normal.
    let bump = sample_bump(
        in.m_texcoord.xy,
        in.m_texcoord.zw,
        in.m_tangent,
        in.m_binormal,
        in.m_normal,
    );
    _ = in.m_pos_w;

    var out: DecalOut;
    if (DECAL_VIZ_MODE == 1u) {
        out.color = vec4<f32>(diffuse.a, diffuse.a, diffuse.a, 1.0);
        out.dark = out.color;
    } else if (DECAL_VIZ_MODE == 2u) {
        out.color = vec4<f32>(diffuse.rgb, 1.0);
        out.dark = out.color;
    } else if (DECAL_VIZ_MODE == 3u) {
        out.color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        out.dark = out.color;
    } else {
        out.color = faded;
        if (RENDER_PASS_PRE_LIGHTING) {
            // `convert_to_albedo_target_no_srgb`. RT1.xyz = packed
            // bump, RT1.w = albedo.w (decal alpha). Pipeline gates the
            // actual store via write_mask (empty for IS_FLAT_VERTEX so
            // the rock normal survives at scorch / leave variants).
            out.dark = vec4<f32>(bump * 0.5 + 0.5, faded.w);
        } else {
            // `_pass_post_static_lighting` — RT1 is hdr_dark accumulator,
            // duplicates RT0 because PC `DARK_COLOR_MULTIPLIER = 1.0`.
            out.dark = faded;
        }
    }
    return out;
}
