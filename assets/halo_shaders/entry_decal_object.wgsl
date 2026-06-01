// Engine `decal_fx.hlsl::default_ps` body, dispatched through
// `c_object_renderer::render_albedo_decals @ 0x1806E4A60` for object
// mesh parts whose material's `c_render_method::get_is_decal` returns
// true (i.e. `group_tag == 'rmd '`). The engine compiles the rmd
// shader for the OBJECT vertex type (skinned ModelVertex) AND for the
// rasterizer_vertex_world type (scene decals); we mirror the same PS
// body but with a VS that takes `ModelVertex` and skins per-bone.
//
// Visibility submit (`c_object_renderer::submit_object_mesh_parts @
// 0x1806E1BF0`) detects rmd materials via
// `c_render_method::get_is_decal @ 0x1806C0D60` and sets the
// `_render_object_mesh_part_decal_bit` (= 1 << 5 = 0x20) on the
// part's render_mask_flags. The wrapper `render_albedo_decals` then
// dispatches via `render_object_contexts(_entry_point_default, 0x20)`,
// AFTER sky and BEFORE `c_decal_system::render_all`, under
// `_z_buffer_mode_read` (depth test only, no write — see
// `feedback_decal_no_depth_write`).
//
// RT layout matches `DecalAlbedo` — RT0 = `_surface_accum_HDR`,
// RT1 = `_surface_post_HDR` (bump packed via
// `convert_to_albedo_target_no_srgb`).
//
// `decal_constants` cbuffer is bound per-material with default values
// (`fade=1`, `sprite_corner=(0,0,1,1)`, `pixel_kill_enabled=1`,
// `u_tiles=v_tiles=1`) — object-attached decals don't author sprite
// atlases or distance-fade, but the rmd PS reads these constants
// regardless and we keep the same cbuffer shape to share the
// existing helper fragments (sample_diffuse, fade_out, etc.) without
// per-entry forking.

// Skinned ModelVertex input — same layout as `entry_albedo.wgsl`.
struct ObjectDecalVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
}

// `s_decal_interpolators` (decal_fx.hlsl:72-85) — identical shape to
// the scene-decal VS output so the PS body is unchanged.
struct DecalInterpolators {
    @builtin(position) m_position: vec4<f32>,
    @location(0) m_texcoord: vec4<f32>,
    @location(1) m_pos_w: f32,
    @location(2) m_tangent: vec3<f32>,
    @location(3) m_binormal: vec3<f32>,
    @location(4) m_normal: vec3<f32>,
}

// Octahedral decode for the packed object-vertex normal. Same impl as
// `entry_albedo.wgsl`.
fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

@vertex
fn vs_main(in: ObjectDecalVertex) -> DecalInterpolators {
    // Skinning — mirrors entry_albedo.wgsl::vs_main exactly so the
    // decal sits on the same world position the opaque albedo pass
    // produced. Falls back to identity when total weight ~= 0
    // (rigid mesh part).
    var skin = mat4x4<f32>(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    skin += node_matrices[in.node_indices[0]] * in.node_weights[0];
    skin += node_matrices[in.node_indices[1]] * in.node_weights[1];
    skin += node_matrices[in.node_indices[2]] * in.node_weights[2];
    skin += node_matrices[in.node_indices[3]] * in.node_weights[3];
    let weight_sum = in.node_weights[0] + in.node_weights[1]
        + in.node_weights[2] + in.node_weights[3];
    if (weight_sum < 0.0001) {
        skin = mat4x4(vec4(1.,0.,0.,0.), vec4(0.,1.,0.,0.),
                      vec4(0.,0.,1.,0.), vec4(0.,0.,0.,1.));
    }
    let skinned_model = model_u.model * skin;

    let world_pos = skinned_model * vec4<f32>(in.position, 1.0);
    let nrm_mat = mat3x3<f32>(
        skinned_model[0].xyz,
        skinned_model[1].xyz,
        skinned_model[2].xyz,
    );
    let obj_normal = oct_decode(in.normal_oct);
    let obj_tangent = normalize(in.tangent_sign.xyz);
    let obj_binormal = cross(obj_normal, obj_tangent) * sign(in.tangent_sign.w);

    var out: DecalInterpolators;
    out.m_position = camera.projection * camera.view * world_pos;

    // Mirror `default_vs`'s sprite_corner + tiles mapping for shared
    // PS compatibility. For object decals the constants default to
    // sprite=(0,0,1,1), pixel_kill=1, tiles=1 → no-op mapping (`.zw =
    // .xy`).
    let pixel_kill = decal_constants.pixel_kill_enabled != 0u;
    let sprite = decal_constants.sprite_corner;
    if (pixel_kill) {
        out.m_texcoord = vec4<f32>(
            in.tex_coords,
            in.tex_coords * sprite.zw + sprite.xy,
        );
    } else {
        out.m_texcoord = vec4<f32>(
            0.5, 0.5,
            in.tex_coords * sprite.zw + sprite.xy,
        );
    }
    out.m_texcoord = vec4<f32>(
        out.m_texcoord.x,
        out.m_texcoord.y,
        out.m_texcoord.z * decal_constants.u_tiles,
        out.m_texcoord.w * decal_constants.v_tiles,
    );
    out.m_pos_w = out.m_position.w;

    out.m_tangent  = normalize(nrm_mat * obj_tangent);
    out.m_binormal = normalize(nrm_mat * obj_binormal);
    out.m_normal   = normalize(nrm_mat * obj_normal);
    return out;
}

// Dual-target output — same as `entry_decal.wgsl`. RT1 carries the
// bump-packed normal for bumped variants (per pipeline mask) so SL at
// these pixels can use the decal's own surface detail.
struct DecalOut {
    @location(0) color: vec4<f32>,
    @location(1) dark: vec4<f32>,
}

// Debug viz switch — same env-var-substituted const as entry_decal.
const DECAL_VIZ_MODE: u32 = __DECAL_VIZ_MODE__;

// Pre/post-lighting selector — object decals are ALWAYS pre_lighting
// (engine dispatches via `render_albedo_decals` which runs inside
// `c_player_view::render_albedo`). The substitution still flows from
// `render_methods::assemble` for consistency with entry_decal.wgsl.
const RENDER_PASS_PRE_LIGHTING: bool = __DECAL_RENDER_PASS_PRE_LIGHTING__;

@fragment
fn fs_main(in: DecalInterpolators) -> DecalOut {
    let uv = in.m_texcoord.xy;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        discard;
    }
    let diffuse = sample_diffuse(in.m_texcoord.xy, in.m_texcoord.zw, 0.0);
    let tinted = tint_and_modulate(diffuse);
    let faded = fade_out(tinted);
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
            out.dark = vec4<f32>(bump * 0.5 + 0.5, faded.w);
        } else {
            out.dark = faded;
        }
    }
    return out;
}
