//! Per-VariantKey user-parameter cbuffer layout, driven by Halo's
//! rmt2 routing model.
//!
//! Halo's rmt2 carries a `s_render_method_routing_info` table whose
//! entries map a destination register (e.g. 0x7100 = `cb13[0]`) to a
//! source-name index (`rmt2.float_constants[i]`). At runtime the
//! engine evaluates each rmsh parameter's animated_parameters and
//! channel-routes the values into the destination register's vec4.
//! [`blam_tags::render_method::resolve_pixel_user_cbuffer`] does the
//! evaluation; this module turns the resulting [`ResolvedCbuffer`] into
//! a WGSL struct + offset table.
//!
//! Every cbuffer slot is a `vec4<f32>` regardless of whether it carries
//! a scalar, an xform, or a color — that's the actual GPU layout the
//! pixl shader expects (DXBC asm reads `cb13[N].x`/`.xy`/`.xyzw`).
//! Member names mirror the rmt2's `float_constants` table; xform slots
//! get a `_xform` suffix so HLSL→WGSL ports keep their familiar names
//! (`material.base_map_xform`, etc.).

use blam_tags::render_method::ResolvedCbuffer;

/// One vec4 slot in the generated WGSL struct.
#[derive(Debug, Clone)]
pub struct CbufferMember {
    pub name: String,
    pub byte_offset: u32,
}

/// Computed layout for one variant's user-parameter cbuffer. Members
/// appear in destination-register order (matches the rmt2 routing).
#[derive(Debug, Clone)]
pub struct CbufferLayout {
    pub members: Vec<CbufferMember>,
    pub total_size: u32,
}

/// Append a slot to a `ResolvedCbuffer` if it doesn't already exist.
/// Used to pad cbuffers with terrain-baseline fields the WGSL entry
/// references unconditionally (so naga compilation succeeds even when
/// the rmt2 declared fewer than 4 layers).
fn ensure_slot(cb: &mut blam_tags::render_method::ResolvedCbuffer,
                name: &str, is_xform: bool, value: [f32; 4]) {
    if cb.slots.iter().any(|s| s.source_name == name) { return; }
    let byte_offset = cb.total_bytes;
    cb.slots.push(blam_tags::render_method::CbufferSlot {
        source_name: name.to_string(),
        is_xform,
        byte_offset,
        value,
    });
    cb.total_bytes += 16;
    let mut buf = vec![0u8; 16];
    for (c, v) in value.iter().enumerate() {
        buf[c * 4..c * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    cb.bytes.extend_from_slice(&buf);
}

/// Force a slot to be a texture xform (`is_xform=true`), regardless
/// of what the rmt2 declared. Some terrain rmt2s declare layer slots
/// (e.g. `detail_bump_m_0`) without the xform flag — our WGSL fragment
/// always references `<name>_xform`, so we override to keep the field
/// name predictable. If the rmt2 had a non-xform value here, it gets
/// replaced with identity (1, 1, 0, 0) — coefficient is lost but the
/// shader compiles. (Faithful per-rmt2 fragment generation deferred.)
fn force_xform_slot(cb: &mut blam_tags::render_method::ResolvedCbuffer, name: &str) {
    const IDENTITY: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
    if let Some(idx) = cb.slots.iter().position(|s| s.source_name == name) {
        if !cb.slots[idx].is_xform {
            cb.slots[idx].is_xform = true;
            cb.slots[idx].value = IDENTITY;
            let off = cb.slots[idx].byte_offset as usize;
            for (c, v) in IDENTITY.iter().enumerate() {
                cb.bytes[off + c * 4..off + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        return;
    }
    ensure_slot(cb, name, true, IDENTITY);
}

/// Pad a terrain (rmtr) cbuffer with the per-layer slots
/// `entry_terrain_static_sh.wgsl` references unconditionally.
/// Riverworld terrains declare 3 or 4 layers depending on the shader;
/// the WGSL is a fixed 4-layer body, so missing slots get filled with
/// neutral defaults (identity xform / 1.0 diffuse / 0.0 specular).
pub fn pad_terrain_cbuffer(cb: &mut blam_tags::render_method::ResolvedCbuffer) {
    const IDENTITY_XFORM: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
    const DEFAULT_DIFFUSE: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    const DEFAULT_SPEC_POWER: [f32; 4] = [16.0, 0.0, 0.0, 0.0];
    const DEFAULT_SPEC_TINT: [f32; 4] = [0.5, 0.5, 0.5, 0.0];
    // global_albedo_tint default is 1.0 — the 4.59479 magic factor
    // lives inline in terrain_fx.hlsl (`mult.x = tint * 4.59479`),
    // not in the cbuffer slot. WGSL applies it at composition time.
    const DEFAULT_GLOBAL_TINT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

    // blend_map xform + global_albedo_tint. Force-xform: rmt2 may
    // declare these without the xform flag for some shaders.
    force_xform_slot(cb, "blend_map");
    ensure_slot(cb, "global_albedo_tint", false, DEFAULT_GLOBAL_TINT);

    // Per-layer xforms (4 layers × 4 textures each). Force them all
    // to xform=true so the WGSL field name is consistently
    // `<name>_xform` regardless of how the rmt2 typed the slot.
    for i in 0..4 {
        force_xform_slot(cb, &format!("base_map_m_{i}"));
        force_xform_slot(cb, &format!("detail_map_m_{i}"));
        force_xform_slot(cb, &format!("bump_map_m_{i}"));
        force_xform_slot(cb, &format!("detail_bump_m_{i}"));
    }
    // Per-layer diffuse + specular params — full set per HLSL
    // `DECLARE_MATERIAL` macro (terrain_new_fx.hlsl:135-158). Used by
    // `blend_surface_parameters` to accumulate the
    // `specular_parameters` struct.
    const DEFAULT_FRESNEL_STEEPNESS: [f32; 4] = [5.0, 0.0, 0.0, 0.0];
    const DEFAULT_CONTRIBUTION: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    const DEFAULT_SELF_ILLUM_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    for i in 0..4 {
        ensure_slot(cb, &format!("diffuse_coefficient_m_{i}"), false, DEFAULT_DIFFUSE);
        ensure_slot(cb, &format!("specular_coefficient_m_{i}"), false, [0.0; 4]);
        ensure_slot(cb, &format!("specular_power_m_{i}"), false, DEFAULT_SPEC_POWER);
        ensure_slot(cb, &format!("specular_tint_m_{i}"), false, DEFAULT_SPEC_TINT);
        ensure_slot(cb, &format!("fresnel_curve_steepness_m_{i}"), false, DEFAULT_FRESNEL_STEEPNESS);
        ensure_slot(cb, &format!("analytical_specular_contribution_m_{i}"), false, DEFAULT_CONTRIBUTION);
        ensure_slot(cb, &format!("area_specular_contribution_m_{i}"), false, DEFAULT_CONTRIBUTION);
        ensure_slot(cb, &format!("environment_specular_contribution_m_{i}"), false, DEFAULT_CONTRIBUTION);
        ensure_slot(cb, &format!("albedo_specular_tint_blend_m_{i}"), false, [0.0; 4]);
        // Self-illum slots per layer (terrain_new_fx.hlsl:153-158).
        // Engine `blend_surface_parameters` only consults layers 0-2
        // (line 753-765 has no material_3_type branch). We still
        // declare all 4 to match the cbuffer layout the rmt2 routing
        // may emit. Default intensity = 0 → disabled when unauthored.
        force_xform_slot(cb, &format!("self_illum_map_m_{i}"));
        force_xform_slot(cb, &format!("self_illum_detail_map_m_{i}"));
        ensure_slot(cb, &format!("self_illum_color_m_{i}"), false, DEFAULT_SELF_ILLUM_COLOR);
        ensure_slot(cb, &format!("self_illum_intensity_m_{i}"), false, [0.0; 4]);
    }
    // `morph_dynamic` blend-type globals (terrain_new_fx.hlsl:128-132).
    // Only used when the rmtr authors `blend_type = dynamic`; default
    // zeros are safe (`transition_threshold=0` means no morph).
    ensure_slot(cb, "dynamic_material", false, [0.0; 4]);
    ensure_slot(cb, "transition_threshold", false, [0.0; 4]);
    ensure_slot(cb, "transition_sharpness", false, [1.0, 0.0, 0.0, 0.0]);

    // Envmap globals (environment_mapping_fx.hlsl:15-21). All envmap
    // variants reference these; pad with neutral defaults so the
    // calc_environment_map call doesn't read uninitialized memory when
    // the rmtr authors environment_mapping=none. env_tint_color
    // identity = (1,1,1) so any value the cube reflects passes through.
    ensure_slot(cb, "env_tint_color", false, [1.0, 1.0, 1.0, 0.0]);
    ensure_slot(cb, "env_roughness_scale", false, [1.0, 0.0, 0.0, 0.0]);
    ensure_slot(cb, "env_bias", false, [0.0; 4]);
    ensure_slot(cb, "env_topcoat_color", false, [1.0, 1.0, 1.0, 0.0]);
    ensure_slot(cb, "env_topcoat_bias", false, [0.0; 4]);
}

/// Pad an rmfl (`shader_foliage`) cbuffer with the slots the foliage
/// entry VSs reference. The foliage rmop chain doesn't always declare
/// `animation_amplitude_horizontal` — when it doesn't, the slot is
/// missing from the cbuffer and the WGSL `material.animation_amplitude_horizontal`
/// read fails layout validation. Engine HLSL declares the param at
/// `foliage_fx.hlsl:38` and reads it inside `animation_offset()`.
pub fn pad_foliage_cbuffer(
    cb: &mut blam_tags::render_method::ResolvedCbuffer,
) {
    // Default wind amplitude — small per-vertex displacement in
    // world units. Tag-authored value (when present) wins via the
    // ensure_slot "first writer wins" semantics.
    ensure_slot(cb, "animation_amplitude_horizontal", false, [0.05, 0.0, 0.0, 0.0]);
}

/// Pad an rmd (`shader_decal`) cbuffer with the xform slots the decal
/// option WGSL fragments reference. The rmd's rmt2 doesn't always
/// declare `bump_map_xform` even when the bump_mapping option resolves
/// to `standard` / `standard_mask`, so we force-add an identity slot
/// when missing. Engine `calc_bumpmap_default_ps` applies
/// `transform_texcoord(uv, bump_map_xform)` (`bump_mapping_fx.hlsl:63`)
/// — without the slot the WGSL fragment can't compile.
pub fn pad_rmd_material_cbuffer(
    cb: &mut blam_tags::render_method::ResolvedCbuffer,
    choices: &crate::halo::render_methods::CategoryChoices,
) {
    // bump_mapping = standard | standard_mask both sample `bump_map`
    // through `transform_texcoord(uv, bump_map_xform)`. `leave` skips
    // the sample entirely so no xform needed.
    let bump = choices.get_or("bump_mapping", "leave");
    if matches!(bump, "standard" | "standard_mask") {
        force_xform_slot(cb, "bump_map");
    }
    // albedo = vector_alpha samples `base_map` through
    // `transform_texcoord(uv, base_map_xform)`. Other albedo variants
    // (diffuse_only, diffuse_plus_alpha, palettized, etc.) sample raw.
    let albedo = choices.get_or("albedo", "diffuse_only");
    if matches!(albedo, "vector_alpha") {
        force_xform_slot(cb, "base_map");
    }
}

/// Pad an rmsh-style cbuffer with the scalar params the dispatched
/// option WGSL fragments reference. Looks up category choices by name
/// from the rmdf-derived [`CategoryChoices`] map — same way Halo's
/// offline compiler resolves which `PARAM` initializers to include.
pub fn pad_rmsh_material_cbuffer(
    cb: &mut blam_tags::render_method::ResolvedCbuffer,
    choices: &crate::halo::render_methods::CategoryChoices,
) {
    const ONE_SCALAR: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    const ZERO: [f32; 4] = [0.0; 4];
    const DEFAULT_FRESNEL: [f32; 4] = [5.0, 0.0, 0.0, 0.0];
    const DEFAULT_TINT: [f32; 4] = [0.5, 0.5, 0.5, 0.0];
    const DEFAULT_SPEC_POWER: [f32; 4] = [16.0, 0.0, 0.0, 0.0];

    // Shared scalars — every material variant references these via
    // the umbrella's CALC_MATERIAL dispatch + the spec composition.
    ensure_slot(cb, "diffuse_coefficient", false, ONE_SCALAR);
    ensure_slot(cb, "specular_coefficient", false, ZERO);
    ensure_slot(cb, "analytical_specular_contribution", false, ONE_SCALAR);
    ensure_slot(cb, "area_specular_contribution", false, ONE_SCALAR);
    ensure_slot(cb, "environment_map_specular_contribution", false, ZERO);

    // albedo
    ensure_slot(cb, "albedo_color", false, [1.0, 1.0, 1.0, 1.0]);
    let albedo = choices.get_or("albedo", "default");
    if albedo != "constant_color" {
        force_xform_slot(cb, "base_map");
        force_xform_slot(cb, "detail_map");
    }
    if matches!(albedo, "two_detail" | "two_detail_black_point" | "detail_blend") {
        force_xform_slot(cb, "detail_map2");
    }
    if albedo == "two_detail_overlay" {
        force_xform_slot(cb, "detail_map2");
        force_xform_slot(cb, "detail_map_overlay");
    }
    if albedo == "three_detail_blend" {
        force_xform_slot(cb, "detail_map2");
        force_xform_slot(cb, "detail_map3");
    }
    if albedo == "color_mask" {
        force_xform_slot(cb, "color_mask_map");
        // Three tint slots, one per cm channel. albedo_color is already
        // declared above (shared across all albedo variants).
        ensure_slot(cb, "albedo_color2", false, [1.0, 1.0, 1.0, 1.0]);
        ensure_slot(cb, "albedo_color3", false, [1.0, 1.0, 1.0, 1.0]);
        // Default `neutral_gray` matches the engine's authored gray
        // (mid-gray reference for the artist-tuned tint divide); 0.5
        // gives a no-op identity for albedo_color = (0.5, 0.5, 0.5, 1).
        ensure_slot(cb, "neutral_gray", false, [0.5, 0.5, 0.5, 1.0]);
    }

    // bump_mapping (`bump_mapping_fx.hlsl`)
    let bump = choices.get_or("bump_mapping", "off");
    if matches!(bump, "default" | "standard" | "detail" | "detail_unorm" | "detail_masked" | "detail_plus_detail_masked") {
        force_xform_slot(cb, "bump_map");
    }
    if matches!(bump, "detail" | "detail_unorm" | "detail_masked" | "detail_plus_detail_masked") {
        force_xform_slot(cb, "bump_detail_map");
        ensure_slot(cb, "bump_detail_coefficient", false, [1.0, 0.0, 0.0, 0.0]);
    }
    if matches!(bump, "detail_masked" | "detail_plus_detail_masked") {
        force_xform_slot(cb, "bump_detail_mask_map");
    }
    if bump == "detail_masked" {
        // PARAM(bool, invert_mask) — declared only for `detail_masked`.
        ensure_slot(cb, "invert_mask", false, [0.0, 0.0, 0.0, 0.0]);
    }
    if bump == "detail_plus_detail_masked" {
        force_xform_slot(cb, "bump_detail_masked_map");
        ensure_slot(cb, "bump_detail_masked_coefficient", false, [1.0, 0.0, 0.0, 0.0]);
    }

    // alpha_test — enumerate every option `pick_alpha_test` knows
    // about so adding a new arm there without a cbuffer pad shows up
    // here as a hard panic, not a silent missing-xform-slot bug.
    match choices.get_or("alpha_test", "none") {
        "none" | "off" => {} // off/none: no map sampled, no xform needed.
        "simple" => {
            force_xform_slot(cb, "alpha_test_map");
        }
        "multiply_map" => {
            // `calc_alpha_test_multiply_map_ps` (alpha_test_custom_fx.hlsl)
            // samples both `alpha_test_map` and `multiply_map`, then
            // clips on the product of the two alpha channels.
            force_xform_slot(cb, "alpha_test_map");
            force_xform_slot(cb, "multiply_map");
        }
        n => panic!(
            "[protomorph] unsupported alpha_test option '{n}' — add a \
             cbuffer pad arm here AND a `pick_alpha_test` arm in \
             render_methods/mod.rs (per feedback_wgsl_must_mirror_hlsl.md)"
        ),
    }

    // specular_mask
    if matches!(choices.get_or("specular_mask", "no_specular_mask"),
        "specular_mask_from_texture" | "specular_mask_from_color_texture")
    {
        force_xform_slot(cb, "specular_mask_texture");
    }

    // material_model — per-variant params, sourced from each rmop's
    // PARAM declarations in the corresponding *_fx.hlsl file.
    match choices.get_or("material_model", "diffuse_only") {
        // `cook_torrance_from_albedo` is the same WGSL alias as
        // `cook_torrance` (PBR params read from the diffuse texture
        // instead of dedicated maps — see pick_material_model). Baseline
        // cbuffer slots match so the WGSL's `material.specular_tint` /
        // `material.roughness` accessors resolve when the rmop chain
        // doesn't declare them.
        "cook_torrance" | "cook_torrance_from_albedo" | "cook_torrance_rim_fresnel" => {
            ensure_slot(cb, "fresnel_color", false, [0.5, 0.5, 0.5, 0.0]);
            ensure_slot(cb, "roughness", false, [0.4, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "albedo_blend", false, ZERO);
            ensure_slot(cb, "specular_tint", false, [1.0, 1.0, 1.0, 0.0]);
            ensure_slot(cb, "analytical_anti_shadow_control", false, ZERO);
        }
        // `two_lobe_phong_tint_map` is the same option layout — the
        // tint-map textures (`normal_specular_tint_map`,
        // `glancing_specular_tint_map`) are sampler bindings, not
        // cbuffer slots. v1 aliases here use the flat tint cbuffer
        // values; texture binding lands when the sampler plumbing does.
        "two_lobe_phong" | "two_lobe_phong_tint_map" => {
            ensure_slot(cb, "normal_specular_power", false, DEFAULT_SPEC_POWER);
            ensure_slot(cb, "normal_specular_tint", false, DEFAULT_TINT);
            ensure_slot(cb, "glancing_specular_power", false, DEFAULT_SPEC_POWER);
            ensure_slot(cb, "glancing_specular_tint", false, DEFAULT_TINT);
            ensure_slot(cb, "fresnel_curve_steepness", false, DEFAULT_FRESNEL);
            ensure_slot(cb, "albedo_specular_tint_blend", false, ZERO);
            ensure_slot(cb, "analytical_anti_shadow_control", false, ZERO);
        }
        "glass" => {
            ensure_slot(cb, "fresnel_coefficient", false, [0.1, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "fresnel_curve_steepness", false, [5.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "fresnel_curve_bias", false, ZERO);
            ensure_slot(cb, "roughness", false, ZERO);
        }
        "single_lobe_phong" => {
            ensure_slot(cb, "roughness", false, ONE_SCALAR);
            ensure_slot(cb, "specular_tint", false, [1.0, 1.0, 1.0, 0.0]);
        }
        // Options whose runtime cbuffer slots come entirely from the
        // rmop chain — no baseline pad needed here. `cook_torrance_pbr_maps`
        // + `organism` reference `material.*` fields that the rmt2's
        // own `PARAM(real, ...)` declarations route; if those rmops are
        // ever stripped or fail to apply, the WGSL won't compile and
        // we'll find out at pipeline-build time (not silently render
        // black). `diffuse_only` / `foliage` / `none` genuinely need no
        // material-model-specific slots.
        "diffuse_only"
        | "foliage"
        | "none"
        | "cook_torrance_pbr_maps"
        | "organism" => {}
        n => panic!(
            "[protomorph] unsupported material_model option '{n}' — add a \
             cbuffer pad arm here AND a `pick_material_model` arm in \
             render_methods/mod.rs (per feedback_wgsl_must_mirror_hlsl.md)"
        ),
    }

    // env_mapping — explicit enumeration of every option in the rmsh
    // and shader_custom rmdfs (`pick_env_mapping`).
    match choices.get_or("environment_mapping", "none") {
        "per_pixel" => {
            ensure_slot(cb, "env_tint_color", false, [1.0, 1.0, 1.0, 0.0]);
        }
        "dynamic" => {
            ensure_slot(cb, "env_tint_color", false, [1.0, 1.0, 1.0, 0.0]);
            ensure_slot(cb, "env_roughness_scale", false, ONE_SCALAR);
            // `dynamic_environment_blend` is engine-managed now —
            // consumed from `engine_misc_ps.dynamic_environment_blend`
            // (camera_bgl @ binding 12). Was a per-material baseline
            // slot before the MiscPS cbuffer was wired through.
        }
        // Options whose env_mapping path reads slots provided by the
        // rmop chain (or no cbuffer slots at all). `none` is the literal
        // no-env-mapping option. `from_flat_texture` /
        // `from_flat_texture_as_cubemap` / `custom_map` are routed
        // through their own rmop param tables.
        "none"
        | "from_flat_texture"
        | "from_flat_texture_as_cubemap"
        | "custom_map" => {}
        n => panic!(
            "[protomorph] unsupported environment_mapping option '{n}' — \
             add a cbuffer pad arm here AND a `pick_env_mapping` arm in \
             render_methods/mod.rs (per feedback_wgsl_must_mirror_hlsl.md)"
        ),
    }

    // parallax
    if matches!(choices.get_or("parallax", "off"), "simple" | "interpolated") {
        force_xform_slot(cb, "height_map");
        ensure_slot(cb, "height_scale", false, ZERO);
    }

    // self_illumination
    match choices.get_or("self_illumination", "none") {
        // `simple` and `simple_with_alpha_mask` share the cbuffer
        // layout — both sample `self_illum_map`, multiply by
        // `self_illum_color`, and weight by `self_illum_intensity`. The
        // alpha-mask variant additionally uses `.a` of the texture
        // sample as a multiplicative gate (WGSL fragment differs;
        // cbuffer is identical).
        "simple" | "simple_with_alpha_mask" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
        }
        // MCC schema sometimes labels the same option as `3_channel_self_illum`.
        "3_channel" | "3_channel_self_illum" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
            ensure_slot(cb, "channel_a", false, [1.0, 0.0, 0.0, 1.0]);
            ensure_slot(cb, "channel_b", false, [0.0, 1.0, 0.0, 1.0]);
            ensure_slot(cb, "channel_c", false, [0.0, 0.0, 1.0, 1.0]);
        }
        "from_diffuse" => {
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
        }
        "illum_detail" => {
            force_xform_slot(cb, "self_illum_map");
            force_xform_slot(cb, "self_illum_detail_map");
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
        }
        "self_illum_times_diffuse" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
            ensure_slot(cb, "primary_change_color", false, [1.0, 1.0, 1.0, 0.0]);
            ensure_slot(cb, "primary_change_color_blend", false, ZERO);
        }
        // ⚠️  Reconstructed. No HLSL ships in MCC corpus or Reach MCC for
        // `calc_self_illumination_simple_four_change_color_ps`. The
        // option is selected by 343i-ported MP maps (e.g. s3d_powerhouse)
        // but the engine source was never carried forward; the WGSL
        // matches an extended `three_channel_ps` shape (per-channel
        // weighting of 4 change colors). Revisit when newer-game tags
        // become available.
        "simple_four_change_color" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
            ensure_slot(cb, "primary_change_color",    false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "secondary_change_color",  false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "tertiary_change_color",   false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "quaternary_change_color", false, [1.0, 1.0, 1.0, 1.0]);
        }
        // Halogram rmdf self-illum options routing to
        // `calc_self_illumination_multilayer_ps` (parallax depth-layered;
        // `self_illumination_halogram_fx.hlsl:14-47`). Three distinct
        // rmdf option names share the same function + cbuffer slots.
        "multilayer_additive"
        | "ml_add_four_change_color"
        | "ml_add_five_change_color" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
            ensure_slot(cb, "layer_depth", false, ZERO);
            ensure_slot(cb, "layer_contrast", false, [1.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "texcoord_aspect_ratio", false, [1.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "depth_darken", false, [0.9, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "layers_of_4", false, [1.0, 0.0, 0.0, 0.0]);
        }
        // Halogram-specific: 4-tap diagonal blur + 2-channel weighted
        // blend between self_illum_color and self_illum_heat_color
        // (`self_illumination_halogram_fx.hlsl:130-159`). Used by
        // weapon scope HUD overlays.
        "scope_blur" => {
            force_xform_slot(cb, "self_illum_map");
            ensure_slot(cb, "self_illum_color", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
            ensure_slot(cb, "self_illum_heat_color", false, [1.0, 1.0, 1.0, 0.0]);
        }
        // Reconstructed `calc_self_illumination_meter_ps` — HLSL is
        // stripped from MCC corpus; rmop params drive the slot list.
        // (illum_meter.render_method_option: meter_map, meter_color_off,
        // meter_color_on, meter_value, plus shared self_illum_intensity.)
        "meter" => {
            force_xform_slot(cb, "meter_map");
            ensure_slot(cb, "meter_color_off", false, [1.0, 0.0, 0.0, 1.0]);
            ensure_slot(cb, "meter_color_on",  false, [0.0, 1.0, 0.0, 1.0]);
            ensure_slot(cb, "meter_value",     false, [0.5, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
        }
        // Engine `calc_self_illumination_plasma_ps`
        // (self_illumination_fx.hlsl:52-72). Three textures + 9 cbuffer
        // floats (color_*.rgba × 3, thinness_* × 3) + self_illum_intensity.
        "plasma" => {
            force_xform_slot(cb, "alpha_mask_map");
            force_xform_slot(cb, "noise_map_a");
            force_xform_slot(cb, "noise_map_b");
            ensure_slot(cb, "color_medium", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "color_sharp", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "color_wide", false, [1.0, 1.0, 1.0, 1.0]);
            ensure_slot(cb, "thinness_medium", false, [1.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "thinness_sharp", false, [1.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "thinness_wide", false, [1.0, 0.0, 0.0, 0.0]);
            ensure_slot(cb, "self_illum_intensity", false, ZERO);
        }
        // No-extra-slots options. `none` / `off` are the literal no-self-
        // illum arms. The mismatched names (`3_channel` schema vs
        // `3_channel_self_illum` MCC) and the multilayer variants are
        // handled above; this catches anything else explicitly.
        "none" | "off" => {}
        n => panic!(
            "[protomorph] unsupported self_illumination option '{n}' — \
             add a cbuffer pad arm here AND a `pick_self_illum` arm in \
             render_methods/mod.rs (per feedback_wgsl_must_mirror_hlsl.md)"
        ),
    }
}

impl CbufferLayout {
    /// Build the layout from a `ResolvedCbuffer`. Names follow the
    /// rmt2's source-name table; xform-typed slots gain a `_xform`
    /// suffix so HLSL/WGSL fragments can reference them as
    /// `material.<bitmap>_xform`.
    pub fn from_resolved(cb: &ResolvedCbuffer) -> Self {
        let members = cb
            .slots
            .iter()
            .map(|s| CbufferMember {
                name: if s.is_xform {
                    format!("{}_xform", s.source_name)
                } else {
                    s.source_name.clone()
                },
                byte_offset: s.byte_offset,
            })
            .collect();
        // wgpu requires uniform buffers ≥ 16 bytes — cbuffer's
        // total_bytes already pads to that, but bottom-clamp for
        // empty-slot edge cases (entry points with no user params).
        let total_size = cb.total_bytes.max(16);
        Self { members, total_size }
    }

    pub fn find(&self, name: &str) -> Option<&CbufferMember> {
        self.members.iter().find(|m| m.name == name)
    }
}

/// Emit the WGSL struct definition for this layout, named `struct_name`.
/// Includes the `@group/@binding` declaration for the cbuffer at
/// `(group, binding)`. WGSL requires structs to have at least one
/// field, so empty layouts get a `_unused: vec4<f32>` placeholder.
pub fn emit_wgsl(layout: &CbufferLayout, struct_name: &str, group: u32, binding: u32) -> String {
    let mut out = String::new();
    out.push_str(&format!("struct {} {{\n", struct_name));
    if layout.members.is_empty() {
        out.push_str("    _unused: vec4<f32>,\n");
    } else {
        // vec4<f32> is 16-byte aligned and sized — sequential members
        // naturally land at offsets 0, 16, 32, … which is exactly the
        // rmt2 destination-register order. No explicit @offset required.
        for m in &layout.members {
            out.push_str(&format!("    {}: vec4<f32>,\n", m.name));
        }
    }
    out.push_str("}\n");
    out.push_str(&format!(
        "@group({}) @binding({}) var<uniform> material: {};\n",
        group, binding, struct_name,
    ));
    out
}
