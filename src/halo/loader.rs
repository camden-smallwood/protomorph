use std::collections::HashMap;
use std::path::{Path, PathBuf};

use blam_tags::animation::{Animation, NodeTransform, Skeleton};
use blam_tags::math::{RealPoint3d, RealQuaternion, RealVector3d};
use blam_tags::paths::{derive_tags_root, resolve_tag_path, tag_ref_path};
use blam_tags::render_method::{
    compile_real_constant_at_time, resolve_pixel_user_cbuffer, BitmapBinding, CbufferSlot,
    ParameterSource, RenderMethod, RenderMethodDefinition, RenderMethodOption,
    RenderMethodTemplate, ResolvedCbuffer, ResolvedRenderMethod, ResolvedValue,
};
use blam_tags::{RenderMesh, RenderModel, RenderModelError, RenderNode, RenderVertex, TagFile};
use blam_tags::TagReadError;
use glam::{Mat4, Quat, Vec3};
use half::f16;

use crate::halo::animations::{
    AnimationChannel, AnimationChannelType, AnimationData, PositionKey, RotationKey, ScalingKey,
};
use crate::halo::bitmaps;
use crate::halo::render_methods::materials::{MaterialData, MaterialTexture, MaterialTextureUsage};
use crate::halo::geometry::{
    decompose_transform, oct_encode_snorm8, pack_tangent_with_sign, pack_weights_unorm8,
    ModelData, ModelMarker, ModelMesh, ModelMeshPart, ModelNode, ModelVertex,
};

/// Hardcoded playback rate for Halo animations. The jmad tag carries
/// no frame_rate field — the engine assumes 30 fps universally. We
/// emit `ticks_per_second = 30.0` and one key per frame at integer
/// times so [`crate::halo::animations::AnimationManager`]'s tick-based clock
/// advances at one frame per 1/30 s.
const HALO_FPS: f32 = 30.0;

/// Errors loading a Halo object as a [`ModelData`]. Carries the
/// failing path or field so the renderer can log meaningful messages.
#[derive(Debug)]
pub enum HaloLoadError {
    /// `TagFile::read` failed at any level of the chain
    /// (.crate / .model / .render_model / .model_animation_graph).
    /// Carries which file and the underlying error.
    ReadTag { path: PathBuf, source: TagReadError },
    /// `RenderModel::from_tag` rejected the render_model.
    Extract(RenderModelError),
    /// The input path doesn't sit under a `tags/` directory, so we
    /// can't resolve cross-tag references against a root.
    MissingTagsRoot(PathBuf),
    /// The expected reference field (`model` on .crate, `render
    /// model` on .model) was empty.
    MissingRef { tag: PathBuf, field: &'static str },
}

impl std::fmt::Display for HaloLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadTag { path, source } => write!(f, "failed to read tag {}: {}", path.display(), source),
            Self::Extract(e) => write!(f, "render_model extract failed: {e}"),
            Self::MissingTagsRoot(p) => write!(f, "no `tags/` ancestor for {}", p.display()),
            Self::MissingRef { tag, field } => write!(f, "{} missing required tag-ref field `{}`", tag.display(), field),
        }
    }
}

impl std::error::Error for HaloLoadError {}

/// Load an object tag (`.crate`, `.scenery`, `.biped`, etc. — any
/// tag that inherits `object` and carries a `model` ref) into
/// protomorph's [`ModelData`]. Walks the chain → renders the first
/// permutation of every region → stubs a default material per shader
/// → decodes any animations from the referenced
/// `.model_animation_graph`.
pub fn load_object(crate_path: &Path) -> Result<ModelData, HaloLoadError> {
    let tags_root = derive_tags_root(crate_path)
        .ok_or_else(|| HaloLoadError::MissingTagsRoot(crate_path.to_path_buf()))?;

    let crate_tag = TagFile::read(crate_path)
        .map_err(|e| HaloLoadError::ReadTag { path: crate_path.to_path_buf(), source: e })?;

    // Single canonical reader of the engine `object_struct_definition`
    // common substruct — handles the inheritance-chain prefix probe
    // internally (`weapon/item/object`, `unit/object`, `device/object`,
    // `item/object`, `object`, ``). Replaces the prior in-file
    // MODEL_REF_PATHS / OBJECT_STRUCT_PATHS prober pair.
    let obj_def = blam_tags::object::ObjectDefinition::from_tag(&crate_tag)
        .unwrap_or_default();

    if obj_def.model.is_empty() {
        return Err(HaloLoadError::MissingRef {
            tag: crate_path.to_path_buf(),
            field: "model",
        });
    }
    let model_path = resolve_tag_path(&tags_root, &obj_def.model, "model");

    let model_tag = TagFile::read(&model_path)
        .map_err(|e| HaloLoadError::ReadTag { path: model_path.clone(), source: e })?;
    let rm_rel = tag_ref_path(&model_tag.root(), "render model")
        .ok_or_else(|| HaloLoadError::MissingRef { tag: model_path.clone(), field: "render model" })?;
    let rm_path = resolve_tag_path(&tags_root, &rm_rel, "render_model");

    let rm_tag = TagFile::read(&rm_path)
        .map_err(|e| HaloLoadError::ReadTag { path: rm_path.clone(), source: e })?;
    let rm = RenderModel::from_tag(&rm_tag).map_err(HaloLoadError::Extract)?;

    let materials = load_materials(&rm, &tags_root);
    let mut model = convert(&rm, materials);
    model.authored_bounding_sphere = if obj_def.has_authored_bounding_sphere() {
        let o = obj_def.bounding_offset;
        Some(glam::Vec4::new(o.x, o.y, o.z, obj_def.bounding_radius))
    } else {
        // .model's `model object data[0]` auto-bake — the cache
        // builder fills this even when the object tag's radius is 0.
        read_model_object_data_sphere(&model_tag)
    };
    model.casts_shadow = obj_def.casts_shadow_runtime();

    // Optionally chase the .model's `animation` tag-ref. Empty refs
    // (no animation graph) and inherited-only graphs (animations[]
    // empty, parent set) both yield zero animations — same as an
    // FBX with no anim tracks.
    if let Some(jmad_rel) = tag_ref_path(&model_tag.root(), "animation") {
        let jmad_path = resolve_tag_path(&tags_root, &jmad_rel, "model_animation_graph");
        if jmad_path.exists() {
            let jmad_tag = TagFile::read(&jmad_path)
                .map_err(|e| HaloLoadError::ReadTag { path: jmad_path.clone(), source: e })?;
            model.animations = convert_animations(&jmad_tag, &model.nodes);
        }
    }

    Ok(model)
}

/// `.model` (hlmt) tag `model object data[0]` auto-bake sphere — the
/// cache builder fills this with a vertex-walked sphere even when the
/// object tag's `bounding_radius` is 0 ("use autogen" convention).
/// Used as the second-stage fallback in [`load_object`] when
/// `ObjectDefinition::has_authored_bounding_sphere` is false.
///
/// Engine equivalent: at object creation,
/// `c_object::initialize` copies the object-tag values into the
/// runtime object header at +0x1C/+0x28; when those are 0 the
/// `s_model_object_data[0]` values flow through instead.
/// `object_get_bounding_sphere @ 0x1802473A0` returns them transformed
/// by the object matrix.
fn read_model_object_data_sphere(model_tag: &TagFile) -> Option<glam::Vec4> {
    let elem = model_tag
        .root()
        .field("model object data")
        .and_then(|f| f.as_block())?
        .element(0)?;
    let radius = elem.read_real("radius").unwrap_or(0.0);
    if radius <= 0.0 {
        return None;
    }
    let off = elem.read_point3d("offset");
    Some(glam::Vec4::new(off.x, off.y, off.z, radius))
}

fn convert(rm: &RenderModel, materials: Vec<MaterialData>) -> ModelData {
    // v1 mesh selection: first permutation of each region. For damage
    // states / color variants, this picks the "intact" / "default"
    // entry. Most static scenery has 1 region × 1 perm so it's a no-op.
    let selected = select_meshes(rm);

    let mut model = ModelData {
        materials,
        meshes: Vec::with_capacity(selected.len()),
        nodes: Vec::with_capacity(rm.nodes.len()),
        markers: Vec::with_capacity(rm.markers.len()),
        animations: Vec::new(),
        root_inverse_transform: Mat4::IDENTITY,
        authored_bounding_sphere: None,
        // Default to true; load_object overrides via read_casts_shadow_flag
        // after reading the object tag. Non-object loaders (FBX path) leave
        // this true since they don't carry the obje gate fields.
        casts_shadow: true,
    };

    convert_nodes(&rm.nodes, &mut model);

    for &mi in &selected {
        model.meshes.push(convert_mesh(&rm.meshes[mi]));
    }

    // Diag: bone-index range vs node count. If max_idx >= nodes.len(),
    // vertex skinning samples uninitialized GPU bone matrices and
    // produces the exploded-mesh look.
    let node_count = model.nodes.len();
    let mut max_bone_idx: u8 = 0;
    let mut total_verts = 0usize;
    for mesh in &model.meshes {
        for v in &mesh.vertices {
            total_verts += 1;
            for &i in &v.node_indices {
                if i > max_bone_idx { max_bone_idx = i; }
            }
        }
    }
    eprintln!(
        "[halo] {} verts; node_count={node_count}; max bone_index={max_bone_idx}{}",
        total_verts,
        if max_bone_idx as usize >= node_count { " ⚠ OUT OF RANGE" } else { "" },
    );

    for marker in &rm.markers {
        model.markers.push(ModelMarker {
            name: marker.name.clone(),
            node_index: marker.node_index as i32,
            position: real_point_to_vec3(marker.translation),
            rotation: Vec3::ZERO,
        });
    }

    model
}

/// Build a [`MaterialData`] per render_model material slot. Each slot
/// names a `.shader` (or variant) tag — we open that, walk its
/// `render_method/parameters[]` block, follow each `bitmap`-typed
/// parameter's tag-ref to the bitm tag, and load the texture inline.
///
/// Per `feedback_wgsl_must_mirror_hlsl.md`: silent fallbacks hide what
/// needs porting. We panic on any failure rather than substituting a
/// stub material. Empty `shader_path` is the one engine-faithful
/// no-op case (the slot is genuinely unauthored at tag time).
fn load_materials(rm: &RenderModel, tags_root: &Path) -> Vec<MaterialData> {
    rm.materials.iter().map(|m| {
        if m.shader_path.is_empty() {
            return MaterialData::stub_for_shader(&m.shader_name);
        }
        // Use the actual group-tag-derived extension (rmsh→shader,
        // rmtr→shader_terrain, etc.).
        let shader_abs = resolve_tag_path(tags_root, &m.shader_path, m.shader_extension());
        load_shader_material(&shader_abs, tags_root, &m.shader_name)
            .unwrap_or_else(|| panic!(
                "[protomorph] failed to load shader material '{}' at {}: \
                 render-method walker returned None. Per \
                 `feedback_wgsl_must_mirror_hlsl.md` we don't substitute \
                 stubs — fix the loader or the tag.",
                m.shader_name, shader_abs.display(),
            ))
    }).collect()
}

/// Resolve an `rmsh` (or any `rm**`) tag end-to-end via the blam-tags
/// `render_method` walker — loads the rmsh, follows its rmdf, loads
/// every per-category rmop, and produces a flat parameter map with
/// rmsh overrides on top of rmop defaults. Also resolves the rmsh's
/// `options` vector against the rmdf into a name-keyed
/// [`CategoryChoices`] map (the same shape Halo's offline shader
/// compiler used to pick HLSL macro expansions). Returns `None` when
/// any link in the chain (rmsh / rmdf / rmop) fails to read.
fn resolve_render_method(
    shader_path: &Path,
    tags_root: &Path,
) -> Option<(
    ResolvedRenderMethod,
    ResolvedCbuffer,
    crate::halo::render_methods::CategoryChoices,
    Option<crate::halo::render_methods::materials::RenderMethodSource>,
)> {
    let rmsh_tag = TagFile::read(shader_path).ok()?;
    let rmsh = RenderMethod::from_tag(&rmsh_tag).ok()?;
    resolve_render_method_from_rm(&rmsh, tags_root)
}

/// Read a `.shader*` tag from disk and run the engine-faithful bake
/// pass (`RenderMethod::bake`) so the returned `RenderMethod`'s
/// `postprocess_definition` carries the runtime view tool.exe would
/// have written into the cache. Returns `None` when any link
/// (rmsh / rmdf / rmt2) can't be loaded — the postprocess sampler then
/// treats the material as having no contribution.
///
/// Lighter-weight than [`resolve_render_method`]: skips ResolvedRenderMethod
/// + cbuffer + RenderMethodSource construction. Used by callers that
/// only need the baked postprocess data (textures, real_constants,
/// runtime_queryable_properties) — e.g., the decorator-bake
/// postprocess sampler in `bake_material_lookup.rs`.
pub fn load_and_bake_render_method(
    shader_path: &Path,
    tags_root: &Path,
) -> Option<RenderMethod> {
    let rmsh_tag = TagFile::read(shader_path).ok()?;
    let mut rmsh = RenderMethod::from_tag(&rmsh_tag).ok()?;
    let rmdf_path = resolve_tag_path(tags_root, &rmsh.definition_path, "render_method_definition");
    let rmdf_tag = TagFile::read(&rmdf_path).ok()?;
    let rmdf = RenderMethodDefinition::from_tag(&rmdf_tag).ok()?;
    let load_rmop = |opt_path: &str| -> Option<RenderMethodOption> {
        let abs = resolve_tag_path(tags_root, opt_path, "render_method_option");
        let tag = TagFile::read(&abs).ok()?;
        RenderMethodOption::from_tag(&tag).ok()
    };
    let rmt2 = rmsh
        .postprocess_definition
        .as_ref()
        .map(|pp| pp.template_path.as_str())
        .filter(|s| !s.is_empty())
        .and_then(|tp| {
            let path = locate_rmt2(tags_root, tp)?;
            let tag = TagFile::read(&path).ok()?;
            RenderMethodTemplate::from_tag(&tag).ok()
        })?;
    let _ = rmsh.bake(&rmdf, &rmt2, load_rmop, 0.0);
    Some(rmsh)
}

/// Resolve an in-memory `RenderMethod` — same end-to-end walker as
/// [`resolve_render_method`], but skips the on-disk rmsh read. Used by
/// the decal pipeline: `decal_system.definitions[i].shader` is parsed
/// inline by the `decs` tag walker, so we already have a `RenderMethod`
/// in hand. The rmdf / rmop / rmt2 chain is still loaded from disk.
pub(crate) fn resolve_render_method_from_rm(
    rmsh: &RenderMethod,
    tags_root: &Path,
) -> Option<(
    ResolvedRenderMethod,
    ResolvedCbuffer,
    crate::halo::render_methods::CategoryChoices,
    Option<crate::halo::render_methods::materials::RenderMethodSource>,
)> {
    let mut rmsh = rmsh.clone();
    let rmdf_path = resolve_tag_path(tags_root, &rmsh.definition_path, "render_method_definition");
    let rmdf_tag = TagFile::read(&rmdf_path).ok()?;
    let rmdf = RenderMethodDefinition::from_tag(&rmdf_tag).ok()?;

    let category_choices = crate::halo::render_methods::CategoryChoices::resolve(&rmsh, &rmdf);

    // Load every rmop in the chain ONCE (used by ResolvedRenderMethod,
    // build_rmop_param_list, and RenderMethod::bake — all share this Fn).
    let load_rmop = |opt_path: &str| -> Option<RenderMethodOption> {
        let abs = resolve_tag_path(tags_root, opt_path, "render_method_option");
        let tag = TagFile::read(&abs).ok()?;
        RenderMethodOption::from_tag(&tag).ok()
    };

    // Pre-load the rmt2 (template). Author-format rmsh tags carry only
    // the template_path in `postprocess_definition`; the rest of the
    // postprocess block is empty until tool.exe bakes it. We populate
    // it ourselves via `RenderMethod::bake` below so the rest of the
    // pipeline sees an engine-faithfully filled `postprocess_definition`.
    let rmt2_loaded: Option<RenderMethodTemplate> = rmsh
        .postprocess_definition
        .as_ref()
        .map(|pp| pp.template_path.as_str())
        .filter(|s| !s.is_empty())
        .and_then(|tp| {
            let path = locate_rmt2(tags_root, tp)?;
            let tag = TagFile::read(&path).ok()?;
            RenderMethodTemplate::from_tag(&tag).ok()
        });

    // Engine-faithful pre-bake (best-effort). Mirrors tool.exe's
    // `update_postprocess @ 0x140C52530`. Skipped when rmt2 didn't
    // load — the legacy walker/cbuffer path below still works in that
    // case. Idempotent: a no-op if rmsh already ships pre-baked.
    if let Some(rmt2) = rmt2_loaded.as_ref() {
        let _ = rmsh.bake(&rmdf, rmt2, load_rmop, 0.0);
    }

    let resolved = ResolvedRenderMethod::resolve(&rmsh, &rmdf, load_rmop);

    let rmop_params = blam_tags::render_method::build_rmop_param_list(&rmsh, &rmdf, load_rmop);

    // Cache-build replay (canonical): synthesize the per-rmsh
    // `constants[]` array via the merge from
    // `c_render_method::compile_single_real_constant @ 0x826E42E8`
    // (Reach tag-debug). Stage 1 sets rmop default; Stage 2 overrides
    // conditionally based on rmsh.parameters[] declarations.
    //
    // Also captures `(rmsh_clone, rmt2, rmop_params_clone)` into a
    // `RenderMethodSource` for rmw materials so the renderer can
    // re-resolve the cbuffer per frame for animated parameters
    // (Phase A7c). Engine equivalent: the runtime keeps
    // `c_render_method` instances alive for `update_constants` to
    // re-overlay each frame.
    let mut rmt2_for_source: Option<RenderMethodTemplate> = None;
    let cbuffer_from_rmt2 = rmt2_loaded.as_ref().map(|rmt2| {
        let resolved_cb = resolve_pixel_user_cbuffer(&rmsh, rmt2, &rmop_params);
        rmt2_for_source = Some(rmt2.clone());
        resolved_cb
    });
    // Author-format rmsh tags ship with an empty `postprocess_definition`
    // — Tag Tool synthesizes the cache-built view at edit time, but the
    // tag on disk only carries the parameters block. In that case the
    // rmt2 template-path is unknown, so we can't run
    // `resolve_pixel_user_cbuffer` (which is keyed off `rmt2.float_constants[]`).
    // Fall back to walking `rmop_params` directly: each rmop parameter
    // becomes one cbuffer slot, and `compile_real_constant_at_time`
    // handles the canonical Stage 1 (rmop default) + Stage 2 (rmsh
    // overrides via animated_parameters) merge — same math as the
    // engine's cache builder. This recovers per-parameter texture xform
    // scales (e.g., `bump_map.scale_y = 3.5` for vahalla_waterfall) that
    // the (1,1,0,0) default would otherwise drop.
    let mut cbuffer = cbuffer_from_rmt2.unwrap_or_else(|| {
        let mut slots = Vec::with_capacity(rmop_params.len());
        let mut bytes = Vec::with_capacity(rmop_params.len() * 16);
        for (i, op) in rmop_params.iter().enumerate() {
            let rmsh_param = rmsh
                .parameters
                .iter()
                .find(|p| p.parameter_name == op.parameter_name);
            // Load-time bake: time = 0, no object context → engine-faithful
            // "object initial state" defaults (DefaultEvalContext returns 0
            // for any named input, matching engine LABEL_82 fallback).
            let ctx = blam_tags::render_method::DefaultEvalContext { eval_time: 0.0 };
            let (value, is_xform) = compile_real_constant_at_time(op, rmsh_param, &ctx);
            let byte_offset = (i as u32) * 16;
            for v in value.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            slots.push(CbufferSlot {
                source_name: op.parameter_name.clone(),
                is_xform,
                byte_offset,
                value,
            });
        }
        let total_bytes = (slots.len() as u32 * 16).max(16);
        if bytes.len() < total_bytes as usize {
            bytes.resize(total_bytes as usize, 0);
        }
        ResolvedCbuffer { slots, total_bytes, bytes }
    });
    append_extern_slots(&mut cbuffer);
    // Append slots for any walker-resolved parameter the rmt2 didn't
    // route (covers cases where the rmop declares a parameter that
    // doesn't appear in rmt2.float_constants — typically engine
    // externs). Existing slots from the rmt2 routing have correct
    // canonical-merge values and are NOT overwritten.
    append_param_phantom_slots(&mut cbuffer, &resolved);

    // Diag: PROTOMORPH_DIAG_RM=1 prints every shader's diffuse_coefficient
    // (and related multipliers) so we can grep for the value the engine
    // sees at draw time. Verifies whether the function-eval port
    // produces the constant=1.0 we expect, vs. silently returning 0.
    if std::env::var("PROTOMORPH_DIAG_RM").is_ok() {
        let bitmap_hint = rmsh
            .parameters
            .iter()
            .find(|p| !p.bitmap_path.is_empty())
            .map(|p| p.bitmap_path.as_str())
            .unwrap_or("?");
        eprintln!(
            "[diag-rm] rmdf={} bitmap_hint={} slots={}",
            rmsh.definition_path, bitmap_hint, cbuffer.slots.len(),
        );
        for slot in &cbuffer.slots {
            eprintln!("  {:40} value={:?}", slot.source_name, slot.value);
        }
    }

    // Apply material-model-specific cbuffer padding HERE (at load time)
    // so the values flow into MaterialData.cbuffer.bytes, not just the
    // per-variant layout used by assemble() to emit the WGSL struct.
    // Previously the pad ran inside assemble() on a clone, so any slots
    // it added (e.g., cook_torrance's fresnel_color/roughness/albedo_blend)
    // existed in the layout but read as ZERO at runtime — making
    // cook_torrance materials render fully black.
    let group = rmsh.group_tag.to_be_bytes();
    if &group == b"rmtr" {
        crate::halo::render_methods::cbuffer::pad_terrain_cbuffer(&mut cbuffer);
    } else if &group == b"rmsh" || &group == b"rmcs" {
        crate::halo::render_methods::cbuffer::pad_rmsh_material_cbuffer(&mut cbuffer, &category_choices);
    } else if &group == b"rmfl" {
        crate::halo::render_methods::cbuffer::pad_foliage_cbuffer(&mut cbuffer);
    } else if &group == b"rmd " {
        crate::halo::render_methods::cbuffer::pad_rmd_material_cbuffer(&mut cbuffer, &category_choices);
    }

    // P6.5 — capture source data for ANY material we may need to
    // re-resolve per-frame. `rmt2` is None for author-format tags
    // with empty `postprocess_definition` (e.g. vahalla_waterfall);
    // those still get a source so the rmop_params fallback can
    // re-evaluate at time T. `is_cbuffer_animated` (run after this)
    // decides whether the material actually animates.
    //
    // Previously this was gated on `group == b"rmw " &&
    // rmt2_for_source.is_some()` — meant rmsh waterfalls / scrolling
    // banners / pulsing lights all froze at t=0.
    let source = Some(crate::halo::render_methods::materials::RenderMethodSource {
        rmsh: rmsh.clone(),
        rmt2: rmt2_for_source.clone(),
        rmop_params: rmop_params.clone(),
    });

    Some((resolved, cbuffer, category_choices, source))
}

/// Defensive: append a cbuffer slot for any rmsh/rmop parameter the
/// rmt2 didn't route. WGSL fragments reference common names
/// (`bump_map_xform`, `albedo_color`, `normal_specular_tint`, …) and
/// fail to compile when the slot is missing — this guarantees they
/// exist with sensible defaults.
fn append_param_phantom_slots(
    cbuffer: &mut ResolvedCbuffer,
    resolved: &ResolvedRenderMethod,
) {
    use blam_tags::render_method::{
        CbufferSlot, ParameterSource, RenderMethodParameterType as Type, ResolvedValue,
    };

    let mut byte_offset = cbuffer.total_bytes;

    let needed_name = |p: &blam_tags::render_method::ResolvedParameter| -> String {
        if matches!(p.parameter_type, Type::Bitmap) {
            format!("{}_xform", p.name)
        } else {
            p.name.clone()
        }
    };

    for p in &resolved.parameters {
        let target = needed_name(p);
        // Slot already exists from rmt2 routing → already filled by the
        // canonical merge in `resolve_pixel_user_cbuffer`. Skip.
        if cbuffer.slots.iter().any(|s| {
            let s_name = if s.is_xform {
                format!("{}_xform", s.source_name)
            } else {
                s.source_name.clone()
            };
            s_name == target || s.source_name == p.name
        }) {
            continue;
        }

        // Slot is NOT in rmt2 routing — append a phantom slot so WGSL
        // fragments that reference this name unconditionally compile.
        // Use the walker's already-resolved value (which merges rmop
        // default + rmsh override per the same rules as the canonical
        // merge for the rmt2 path).
        let (is_xform, value) = match (&p.parameter_type, &p.source) {
            (Type::Bitmap, _) => (true, [1.0, 1.0, 0.0, 0.0]),
            (_, ParameterSource::Inline(ResolvedValue::Real(v))) => {
                (false, [*v, 0.0, 0.0, 0.0])
            }
            (_, ParameterSource::Inline(ResolvedValue::Int(v))) => {
                (false, [*v as f32, 0.0, 0.0, 0.0])
            }
            (_, ParameterSource::Inline(ResolvedValue::Bool(v))) => {
                (false, [if *v { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0])
            }
            (_, ParameterSource::Inline(ResolvedValue::Color(c))) => (false, *c),
            _ => (false, [0.0; 4]),
        };

        // The CbufferLayout::from_resolved emits slot names suffixed
        // by `_xform` only when is_xform=true — so we store the bare
        // name for both bitmap and non-bitmap and let the layout add
        // the suffix.
        cbuffer.slots.push(CbufferSlot {
            source_name: p.name.clone(),
            is_xform,
            byte_offset,
            value,
        });
        for c in value.iter() {
            cbuffer.bytes.extend_from_slice(&c.to_le_bytes());
        }
        byte_offset += 16;
    }

    cbuffer.total_bytes = byte_offset.max(16);
}

/// Names + default kinds the protomorph WGSL fragments reference.
/// Every variant of the assembler can request any subset of these;
/// we ensure each one has a cbuffer slot with a sensible default so
/// WGSL parsing never fails because a slot was missing.
///
/// `(name_in_wgsl, default_value, is_xform)`. WGSL refers to bitmap
/// xforms as `<name>_xform`; the layout emit appends `_xform` only
/// when `is_xform=true`, so we store the bare name (without suffix)
/// and let layout add it.
pub const WGSL_REQUIRED_SLOTS: &[(&str, [f32; 4], bool)] = &[
    // Bitmap xforms (identity).
    ("base_map", [1.0, 1.0, 0.0, 0.0], true),
    ("detail_map", [1.0, 1.0, 0.0, 0.0], true),
    ("change_color_map", [1.0, 1.0, 0.0, 0.0], true),
    ("bump_map", [1.0, 1.0, 0.0, 0.0], true),
    ("bump_detail_map", [1.0, 1.0, 0.0, 0.0], true),
    ("self_illum_map", [1.0, 1.0, 0.0, 0.0], true),
    // Reals + colors with reasonable visually-neutral defaults.
    ("albedo_color", [1.0, 1.0, 1.0, 1.0], false),
    ("bump_detail_coefficient", [0.0, 0.0, 0.0, 0.0], false),
    ("diffuse_coefficient", [1.0, 0.0, 0.0, 0.0], false),
    ("specular_coefficient", [0.0, 0.0, 0.0, 0.0], false),
    ("normal_specular_power", [10.0, 0.0, 0.0, 0.0], false),
    ("glancing_specular_power", [10.0, 0.0, 0.0, 0.0], false),
    ("fresnel_curve_steepness", [5.0, 0.0, 0.0, 0.0], false),
    ("albedo_specular_tint_blend", [0.0, 0.0, 0.0, 0.0], false),
    ("analytical_specular_contribution", [1.0, 0.0, 0.0, 0.0], false),
    ("area_specular_contribution", [0.0, 0.0, 0.0, 0.0], false),
    ("environment_map_specular_contribution", [0.0, 0.0, 0.0, 0.0], false),
    ("analytical_anti_shadow_control", [0.0, 0.0, 0.0, 0.0], false),
    ("self_illum_intensity", [0.0, 0.0, 0.0, 0.0], false),
    ("normal_specular_tint", [1.0, 1.0, 1.0, 1.0], false),
    ("glancing_specular_tint", [1.0, 1.0, 1.0, 1.0], false),
    ("env_tint_color", [1.0, 1.0, 1.0, 1.0], false),
    ("self_illum_color", [0.0, 0.0, 0.0, 0.0], false),
    // Engine externs (typed-MaterialData fields override these at
    // serialize time — see render_methods/serializer.rs).
    ("primary_change_color", [0.0, 0.0, 0.0, 0.0], false),
    ("secondary_change_color", [0.0, 0.0, 0.0, 0.0], false),
];

/// Engine externs that `serialize_material` overlays from typed
/// MaterialData fields. Subset of WGSL_REQUIRED_SLOTS that takes
/// per-material values (everyone else uses defaults).
pub const EXTERN_SLOTS: &[&str] = &["primary_change_color", "secondary_change_color"];

/// Ensure every `WGSL_REQUIRED_SLOTS` name has a cbuffer slot — append
/// missing ones with their default values + xform flag. Defensive
/// padding so the WGSL fragments compile regardless of which rmt2
/// variant supplied the routing.
pub fn append_extern_slots(cbuffer: &mut ResolvedCbuffer) {
    use blam_tags::render_method::CbufferSlot;
    let mut byte_offset = cbuffer.total_bytes;
    for (name, default, is_xform) in WGSL_REQUIRED_SLOTS {
        // First: if a slot for `name` already exists but disagrees on
        // is_xform, FORCE the existing slot's xform flag to match what
        // WGSL expects. The layout uses is_xform to decide whether to
        // suffix with `_xform`; a mismatch here is the difference
        // between `material.bump_map` (wrong) and `material.bump_map_xform`
        // (right).
        if let Some(existing) = cbuffer
            .slots
            .iter_mut()
            .find(|s| s.source_name == *name)
        {
            if existing.is_xform != *is_xform {
                existing.is_xform = *is_xform;
            }
            continue;
        }
        cbuffer.slots.push(CbufferSlot {
            source_name: name.to_string(),
            is_xform: *is_xform,
            byte_offset,
            value: *default,
        });
        for c in default.iter() {
            cbuffer.bytes.extend_from_slice(&c.to_le_bytes());
        }
        byte_offset += 16;
    }
    cbuffer.total_bytes = byte_offset.max(16);
}

/// Find the on-disk rmt2 file for a tag path stored in the rmsh. The
/// stored path may be truncated (older MCC schemas dropped trailing _0
/// options), so we fall back to a prefix glob within the directory.
fn locate_rmt2(tags_root: &Path, template_path: &str) -> Option<PathBuf> {
    let exact = resolve_tag_path(tags_root, template_path, "render_method_template");
    if exact.exists() {
        return Some(exact);
    }
    let dir = exact.parent()?;
    let stem_prefix = exact.file_stem()?.to_string_lossy().trim_end_matches("_0").to_string();
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("render_method_template") {
            continue;
        }
        let stem = path.file_stem()?.to_string_lossy().to_string();
        if stem.starts_with(&stem_prefix)
            && stem.trim_start_matches(&stem_prefix).chars().all(|c| c == '_' || c == '0')
        {
            return Some(path);
        }
    }
    None
}

/// Build a [`MaterialData`] from a Halo `rm**` tag.
///
/// Walks the resolved parameter list (rmsh overrides ⊕ rmop defaults
/// ⊕ engine externs) and dispatches each known parameter to the right
/// `MaterialData` field. Unknown / not-yet-mapped parameters are
/// silently dropped — the WGSL only consumes a subset today and the
/// walker emits everything the shader system declares.
pub(crate) fn load_shader_material_public(
    shader_path: &Path,
    tags_root: &Path,
    mat_name: &str,
) -> Option<MaterialData> {
    load_shader_material(shader_path, tags_root, mat_name)
}

fn load_shader_material(shader_path: &Path, tags_root: &Path, mat_name: &str) -> Option<MaterialData> {
    let (resolved, cbuffer, category_choices, source) =
        resolve_render_method(shader_path, tags_root)?;
    Some(build_material_from_resolved(
        resolved, cbuffer, category_choices, source, tags_root, mat_name,
    ))
}

/// Build a [`MaterialData`] from an in-memory `RenderMethod`. Same body
/// as [`load_shader_material`], but the rmsh source comes from the
/// inline `decal_system.definitions[i].shader` struct rather than a
/// separate `.shader` tag on disk.
pub(crate) fn load_shader_material_from_rm(
    rmsh: &RenderMethod,
    tags_root: &Path,
    mat_name: &str,
) -> Option<MaterialData> {
    let (resolved, cbuffer, category_choices, source) =
        resolve_render_method_from_rm(rmsh, tags_root)?;
    Some(build_material_from_resolved(
        resolved, cbuffer, category_choices, source, tags_root, mat_name,
    ))
}

/// Shared `MaterialData` builder — drains the resolver output (resolved
/// rmop chain + cbuffer + choices + source) into the typed fields the
/// renderer consumes. Same logic whether the rmsh came from a file or
/// from an in-memory tag struct.
fn build_material_from_resolved(
    resolved: ResolvedRenderMethod,
    cbuffer: ResolvedCbuffer,
    category_choices: crate::halo::render_methods::CategoryChoices,
    source: Option<crate::halo::render_methods::materials::RenderMethodSource>,
    tags_root: &Path,
    mat_name: &str,
) -> MaterialData {
    let mut mat = MaterialData::stub_for_shader(mat_name);
    mat.category_choices = category_choices;
    // Check animation BEFORE moving `source` into `mat`. Materials
    // without a cached `RenderMethodSource` (most non-water rm**
    // subclasses today) stay flagged static — we have no way to
    // re-evaluate them at time T anyway, so the flag is meaningless.
    mat.is_animated = source
        .as_ref()
        .map(|s| {
            blam_tags::render_method::is_cbuffer_animated(
                &s.rmsh,
                s.rmt2.as_ref(),
                &s.rmop_params,
            )
        })
        .unwrap_or(false);
    mat.render_method_source = source;

    // Hold onto the resolved-render-method walker output — the
    // cbuffer layout + binding generators consume it directly to
    // emit per-variant WGSL. (We still also dispatch by name into
    // the typed MaterialData fields below so the renderer's serializer
    // can find values.)
    mat.resolved = resolved.clone();
    mat.cbuffer = cbuffer;

    // Terrain rmt2s declare 3 or 4 blend-map layers. Our terrain WGSL
    // is a fixed 4-layer body, so pad the cbuffer with neutral
    // defaults for any missing layer slot — keeps upload-bytes in
    // sync with the WGSL struct layout.
    if mat.resolved.group_tag == u32::from_be_bytes(*b"rmtr") {
        crate::halo::render_methods::cbuffer::pad_terrain_cbuffer(&mut mat.cbuffer);
    }

    for p in &resolved.parameters {
        match (&p.name[..], &p.source) {
            // ---- Textures ----
            ("base_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                push_bitmap(b, MaterialTextureUsage::Diffuse, tags_root, &mut mat.textures, &p.name);
            }
            ("bump_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                // BC5/DXN bump maps pass through to wgpu's `Bc5RgSnorm`;
                // shader reconstructs Z = sqrt(1 - X² - Y²).
                push_bitmap(b, MaterialTextureUsage::Normal, tags_root, &mut mat.textures, &p.name);
            }
            ("self_illum_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                push_bitmap(b, MaterialTextureUsage::Emissive, tags_root, &mut mat.textures, &p.name);
            }
            ("change_color_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                // 4-channel region mask. The Halo WGSL samples this at
                // the slot the FBX path uses for opacity — different
                // shaders, no collision.
                push_bitmap(b, MaterialTextureUsage::ChangeColorMap, tags_root, &mut mat.textures, &p.name);
            }
            ("detail_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                // High-frequency surface variation, multiplied into
                // base_map. Bound at material_bgl slot 7 (Halo-only).
                push_bitmap(b, MaterialTextureUsage::DetailMap, tags_root, &mut mat.textures, &p.name);
            }
            ("bump_detail_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                // High-frequency tangent-space normal map summed onto
                // bump_map.xy in calc_bumpmap_detail_ps. Same BC5/DXN
                // decode path as bump_map.
                push_bitmap(b, MaterialTextureUsage::BumpDetail, tags_root, &mut mat.textures, &p.name);
            }
            ("bump_detail_coefficient", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.bump_detail_coefficient = v.max(0.0);
            }

            // Environment-mapping reflection cube + its tint. The
            // shader's per_pixel variant samples env_cubemap with the
            // view-reflect direction and multiplies by env_tint_color
            // × env_map_specular_contribution × spec_mask × specular_coef.
            ("environment_map", ParameterSource::Inline(ResolvedValue::Bitmap(b))) => {
                push_bitmap(b, MaterialTextureUsage::Environment, tags_root, &mut mat.textures, &p.name);
            }
            ("env_tint_color", ParameterSource::Inline(ResolvedValue::Color([r, g, b, _]))) => {
                mat.env_tint_color = Vec3::new(*r, *g, *b);
            }

            ("diffuse_coefficient", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.diffuse_coefficient = *v;
            }
            ("specular_coefficient", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.specular_coefficient = *v;
            }
            ("self_illum_intensity", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.self_illum_intensity = *v;
            }
            ("normal_specular_power", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.specular_power = v.max(1.0);
            }
            ("glancing_specular_power", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.glancing_specular_power = v.max(1.0);
            }
            ("fresnel_curve_steepness", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.fresnel_curve_steepness = v.max(0.1);
            }

            // Two-lobe Phong tints + contribution weights. `normal_specular_tint`
            // and `glancing_specular_tint` carry warm/cool reflectance colors
            // (grunt: orange normal, blue glancing). Contribution scalars
            // re-weight analytic vs area vs env-map lobes (grunt: 0.8/0.11/1.0).
            // `albedo_specular_tint_blend` mixes albedo into the spec tint.
            ("normal_specular_tint", ParameterSource::Inline(ResolvedValue::Color([r, g, b, _]))) => {
                mat.normal_specular_tint = Vec3::new(*r, *g, *b);
            }
            ("glancing_specular_tint", ParameterSource::Inline(ResolvedValue::Color([r, g, b, _]))) => {
                mat.glancing_specular_tint = Vec3::new(*r, *g, *b);
            }
            ("albedo_specular_tint_blend", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.albedo_specular_tint_blend = v.clamp(0.0, 1.0);
            }
            ("analytical_specular_contribution", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.analytical_specular_contribution = v.max(0.0);
            }
            ("area_specular_contribution", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.area_specular_contribution = v.max(0.0);
            }
            ("environment_map_specular_contribution", ParameterSource::Inline(ResolvedValue::Real(v))) => {
                mat.environment_map_specular_contribution = v.max(0.0);
            }

            // Change colors: usually engine externs (sourced from biped
            // variant data); we keep the stub_for_shader defaults until
            // protomorph wires that pipeline. When the rmsh inlines a
            // literal color (rare), we use it.
            ("primary_change_color", ParameterSource::Inline(ResolvedValue::Color([r, g, b, _]))) => {
                mat.primary_change_color = Vec3::new(*r, *g, *b);
            }
            ("secondary_change_color", ParameterSource::Inline(ResolvedValue::Color([r, g, b, _]))) => {
                mat.secondary_change_color = Vec3::new(*r, *g, *b);
            }

            _ => {} // many resolved params are walker-output but not yet wired into WGSL
        }
    }

    // Generic catch-all: load every remaining Bitmap-typed parameter
    // by name. The match arms above handle rmsh-specific names that
    // also need typed-field side-effects (`diffuse_coefficient`,
    // `normal_specular_tint`, etc.); this loop covers the per-layer
    // terrain textures (`base_map_m_0..3`, `bump_map_m_0..3`,
    // `blend_map`, etc.) and any rmop-declared bitmap that wasn't
    // pattern-matched. Per-variant binding looks up by name so all
    // declared textures need to be present.
    for p in &resolved.parameters {
        let ParameterSource::Inline(ResolvedValue::Bitmap(b)) = &p.source else { continue };
        if mat.textures.iter().any(|t| t.name == p.name) {
            continue;
        }
        let usage = usage_hint_for_param_name(&p.name);
        push_bitmap(b, usage, tags_root, &mut mat.textures, &p.name);
    }

    // Halo's `specular_mask = from_diffuse` flow doesn't need a separate
    // texture — the shader reads spec mask from `albedo.a` (sampled from
    // base_map). When walker surfaces a different `specular_mask` rmdf
    // option (e.g. `specular_mask_from_texture`), we'll branch here.
    mat
}

/// Pick a `MaterialTextureUsage` based on the Halo parameter name —
/// drives texture format decode (Bc5 SNORM for normals, sRGB for
/// diffuse, etc). Used by the generic Bitmap-param catch-all loader.
/// Pattern-matches the conventional Halo rmop parameter naming.
fn usage_hint_for_param_name(name: &str) -> MaterialTextureUsage {
    let n = name.to_ascii_lowercase();
    if n.starts_with("bump_") || n.contains("_bump_") || n.ends_with("_bump") {
        MaterialTextureUsage::Normal
    } else if n.starts_with("self_illum") {
        MaterialTextureUsage::Emissive
    } else if n == "environment_map" || n.contains("_cubemap") {
        MaterialTextureUsage::Environment
    } else if n == "change_color_map" {
        MaterialTextureUsage::ChangeColorMap
    } else if n == "blend_map" || n.starts_with("detail_") {
        MaterialTextureUsage::DetailMap
    } else {
        // base_map, base_map_m_0..3, anything else — diffuse decode.
        MaterialTextureUsage::Diffuse
    }
}

fn push_bitmap(
    b: &BitmapBinding,
    usage: MaterialTextureUsage,
    tags_root: &Path,
    out: &mut Vec<MaterialTexture>,
    param_name: &str,
) {
    if b.bitmap_path.is_empty() { return; }
    let bitm_abs = resolve_tag_path(tags_root, &b.bitmap_path, "bitmap");
    match bitmaps::load(&bitm_abs) {
        Some(inline) => out.push(MaterialTexture {
            name: param_name.to_string(),
            usage,
            texture: inline,
        }),
        None => eprintln!(
            "[halo] failed to load bitmap {} for parameter '{}'",
            bitm_abs.display(), param_name,
        ),
    }
}

fn select_meshes(rm: &RenderModel) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for region in &rm.regions {
        let Some(perm) = region.permutations.first() else { continue };
        if perm.mesh_index < 0 || perm.mesh_count <= 0 { continue; }
        for off in 0..perm.mesh_count as i32 {
            let mi = perm.mesh_index as i32 + off;
            if mi < 0 || (mi as usize) >= rm.meshes.len() { continue; }
            let mi = mi as usize;
            if !out.contains(&mi) { out.push(mi); }
        }
    }
    if out.is_empty() {
        // Some tags omit regions/permutations entirely (unusual but
        // legal) — fall back to all meshes so we still see something.
        out = (0..rm.meshes.len()).collect();
    }
    out
}

fn convert_nodes(nodes: &[RenderNode], model: &mut ModelData) {
    // Halo stores nodes parent-before-child with parent-relative TRS.
    // World-space bind-pose is the chain of locals through parents;
    // its inverse is the GPU `offset_matrix` (vertex-to-bone-local).
    let locals: Vec<Mat4> = nodes.iter()
        .map(|n| Mat4::from_rotation_translation(quat_from_real(n.default_rotation), real_point_to_vec3(n.default_translation)))
        .collect();
    let mut worlds: Vec<Mat4> = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        let world = if n.parent_index < 0 || (n.parent_index as usize) >= worlds.len() {
            locals[i]
        } else {
            worlds[n.parent_index as usize] * locals[i]
        };
        worlds.push(world);
    }

    for (i, n) in nodes.iter().enumerate() {
        let local = locals[i];
        let model_node = ModelNode {
            name: n.name.clone(),
            parent_index: -1,
            first_child_index: -1,
            next_sibling_index: -1,
            offset_matrix: worlds[i].inverse(),
            default_transform: local,
            default_decomposed: decompose_transform(&local),
        };
        model.add_child_node(n.parent_index as i32, model_node);
    }
}

fn convert_mesh(mesh: &RenderMesh) -> ModelMesh {
    let vertices: Vec<ModelVertex> = mesh.vertices.iter().map(convert_vertex).collect();
    let indices: Vec<u32> = mesh.indices.clone();
    let parts: Vec<ModelMeshPart> = mesh.parts.iter().map(|p| ModelMeshPart {
        material_index: p.material_index as usize,
        index_start: p.index_start,
        index_count: p.index_count,
        part_type: p.part_type,
    }).collect();
    // Per-vertex baked color — populated when the engine sets
    // `_mesh_has_vertex_color_bit` on the source mesh (sky `.render_model`
    // meshes are the canonical user).
    //
    // BUG WORKAROUND: blam-tags currently reads vert_color from the
    // `raw_vertex` block's "vertex color" field. For Halo 3 MCC sky
    // dome meshes the real per-vertex baked sky gradient lives in a
    // SEPARATE vertex buffer at `vertex_buffer_indices[2]` (engine
    // `_vertex_buffer_usage_vert_color`) which blam-tags doesn't yet
    // surface. Two failure modes the workaround handles:
    //
    //   1. Whole-stream zero (e.g. riverworld mesh 0): raw_vertex
    //      field is absent → blam-tags returns ZERO for every
    //      vertex. Heuristic: `max < 1/255`.
    //
    //   2. Partial data (e.g. guardian mesh 0 with min ~0.001 max
    //      ~0.92): some vertices have real values, others are zero
    //      placeholders. The black-vert triangles render solid black
    //      via the sky shader (`vert_color × g_exposure = 0`).
    //      Heuristic: `min < 0.01` flags the partial-data case.
    //
    // When dropped the dispatcher routes the mesh through Pass B
    // (engine's `_entry_point_static_lighting_sh` fallback per
    // `render_mesh_part_default @ 0x18069EBC0:64-82`).
    let vert_color_stream: Vec<[f32; 3]> = if mesh.has_vertex_color {
        let stream: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|v| [v.vert_color.i, v.vert_color.j, v.vert_color.k])
            .collect();
        let mut max_seen = 0f32;
        let mut min_seen = f32::INFINITY;
        for c in &stream {
            for k in 0..3 {
                if c[k] > max_seen { max_seen = c[k]; }
                if c[k] < min_seen { min_seen = c[k]; }
            }
        }
        let drop_zero_stream = max_seen < 1.0 / 255.0;
        let drop_partial_stream = min_seen < 0.01;
        let drop = drop_zero_stream || drop_partial_stream;
        if std::env::var_os("PROTOMORPH_TRACE_SKY_VC").is_some() {
            let n = stream.len() as f32;
            if n > 0.0 {
                let mut sum = [0f32; 3];
                for c in &stream {
                    for k in 0..3 { sum[k] += c[k]; }
                }
                let tag = if drop_zero_stream {
                    "[DROPPED — zero stream]"
                } else if drop_partial_stream {
                    "[DROPPED — partial data: some vertices near-zero]"
                } else { "" };
                eprintln!(
                    "[sky/vc-stats] {} verts: avg=({:.4},{:.4},{:.4}) min={:.4} max={:.4} {}",
                    stream.len(),
                    sum[0] / n, sum[1] / n, sum[2] / n,
                    min_seen, max_seen, tag,
                );
            }
        }
        if drop { Vec::new() } else { stream }
    } else {
        Vec::new()
    };
    ModelMesh {
        vertices,
        indices,
        parts,
        prt_ambient_stream: mesh.prt_ambient_stream.clone(),
        vert_color_stream,
    }
}

fn convert_vertex(v: &RenderVertex) -> ModelVertex {
    let normal = Vec3::new(v.normal.i, v.normal.j, v.normal.k);
    let texcoord_u = f16::from_f32(v.texcoord.x);
    // Halo stores V with 0 at top (DX convention). wgpu's
    // `textureSample` also treats v=0 as top, so we pass V through
    // unchanged. The FBX path flips because Blender stores v=0 at
    // bottom — that flip is FBX-specific and would produce
    // per-UV-island mirroring on Halo geometry if applied here.
    let texcoord_v = f16::from_f32(v.texcoord.y);

    // raw_vertex_block carries the artist-authored tangent + binormal
    // alongside the normal. Tags without tangent-space data leave
    // them zero — fall back to an arbitrary perpendicular so the
    // packing math doesn't degenerate (still wrong for normal mapping,
    // but unreachable on rendered Halo geometry).
    let raw_tangent = Vec3::new(v.tangent.i, v.tangent.j, v.tangent.k);
    let raw_binormal = Vec3::new(v.binormal.i, v.binormal.j, v.binormal.k);
    let (tangent_dir, binormal_dir) = if raw_tangent.length_squared() > 0.0001 {
        (raw_tangent.normalize(), raw_binormal.normalize_or_zero())
    } else {
        let stub = arbitrary_tangent(normal);
        (stub, normal.cross(stub))
    };

    ModelVertex {
        position: [v.position.x, v.position.y, v.position.z],
        texcoord: [texcoord_u.to_bits(), texcoord_v.to_bits()],
        normal: oct_encode_snorm8(normal),
        _pad: [0, 0],
        // The shader reconstructs binormal as `cross(N, T) × sign(w)`,
        // so we encode the handedness sign by comparing the stored
        // binormal against `N × T` — `pack_tangent_with_sign` does
        // exactly that internally.
        tangent: pack_tangent_with_sign(tangent_dir, normal, binormal_dir),
        node_indices: v.node_indices,
        node_weights: pack_weights_unorm8(v.node_weights),
        // Lightmap UV is zero on objects (only BSP cluster meshes carry
        // it). The BSP loader has its own `convert_bsp_vertex` that
        // pulls real values from the lightmap tag's `imported geometry`.
        lightmap_texcoord: [
            f16::from_f32(v.lightmap_texcoord.x).to_bits(),
            f16::from_f32(v.lightmap_texcoord.y).to_bits(),
        ],
    }
}

/// Fallback when raw_vertex carries no tangent (legacy / non-rendered
/// meshes). Picks a vector perpendicular to `normal` by orthogonalizing
/// against the world-axis least parallel to it. Stable but not
/// UV-aligned — wrong for normal mapping but never used on real geometry.
fn arbitrary_tangent(normal: Vec3) -> Vec3 {
    let basis = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    (basis - normal * basis.dot(normal)).normalize_or_zero()
}

/// Decode every animation in a `.model_animation_graph` and convert
/// to protomorph's per-bone keyed [`AnimationData`]. Bones that don't
/// match a [`ModelNode`] by name are silently dropped (jmad bones are
/// typically a superset — collision-only or attachment bones the
/// render_model doesn't carry). Returns an empty vec for jmads that
/// inherit all their animations from a parent (out of scope for v1)
/// or whose decode fails entirely.
fn convert_animations(jmad_tag: &TagFile, render_model_nodes: &[ModelNode]) -> Vec<AnimationData> {
    let animation = match Animation::new(jmad_tag) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[halo] Animation::new failed: {e}");
            return Vec::new();
        }
    };
    if animation.is_empty() {
        if let Some(parent) = animation.parent() {
            eprintln!("[halo] jmad inherits from {parent} — inherited animations not loaded (v1)");
        }
        return Vec::new();
    }

    let skeleton = Skeleton::from_tag(jmad_tag);
    if skeleton.is_empty() {
        return Vec::new();
    }

    // Map jmad bone name → protomorph node index (None when the
    // render_model doesn't have a bone of that name).
    let pm_index_by_name: HashMap<&str, usize> = render_model_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();
    let bone_to_pm: Vec<Option<usize>> = skeleton.nodes.iter()
        .map(|n| pm_index_by_name.get(n.name.as_str()).copied())
        .collect();

    // Defaults for `clip.pose` — for any jmad bone not flagged static
    // or animated by the codec, the composer falls back to this. The
    // canonical default is the render_model's per-bone bind pose
    // (matches TagTool's AnimationDefaultNodeHelper).
    let defaults: Vec<NodeTransform> = skeleton.nodes.iter()
        .map(|n| match pm_index_by_name.get(n.name.as_str()) {
            Some(&pm_idx) => {
                let (s, r, t) = render_model_nodes[pm_idx].default_decomposed;
                NodeTransform {
                    translation: RealPoint3d { x: t.x, y: t.y, z: t.z },
                    rotation: RealQuaternion { i: r.x, j: r.y, k: r.z, w: r.w },
                    scale: s.x.max(0.001), // assume uniform — Halo bones don't shear
                }
            }
            None => NodeTransform::IDENTITY,
        })
        .collect();

    let pm_node_count = render_model_nodes.len();
    let mut out = Vec::new();
    let mut decode_failures = 0usize;

    for group in animation.iter() {
        let clip = match group.decode() {
            Ok(c) => c,
            Err(_) => { decode_failures += 1; continue; }
        };
        let pose = clip.pose(&skeleton, Some(&defaults));
        let frame_count = pose.frames.len();
        if frame_count == 0 { continue; }

        let name = group.name.clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("anim_{}", group.index));

        let mut channels: Vec<AnimationChannel> = Vec::with_capacity(skeleton.len());
        for (bone_idx, &maybe_pm) in bone_to_pm.iter().enumerate() {
            let Some(pm_idx) = maybe_pm else { continue };
            let mut position_keys = Vec::with_capacity(frame_count);
            let mut rotation_keys = Vec::with_capacity(frame_count);
            let mut scaling_keys = Vec::with_capacity(frame_count);
            let mut times = Vec::with_capacity(frame_count);
            for f in 0..frame_count {
                let nt = pose.frames[f][bone_idx];
                let t = f as f32;
                position_keys.push(PositionKey {
                    time: t,
                    position: Vec3::new(nt.translation.x, nt.translation.y, nt.translation.z),
                });
                rotation_keys.push(RotationKey {
                    time: t,
                    rotation: Quat::from_xyzw(nt.rotation.i, nt.rotation.j, nt.rotation.k, nt.rotation.w),
                });
                scaling_keys.push(ScalingKey {
                    time: t,
                    scaling: Vec3::splat(nt.scale.max(0.001)),
                });
                times.push(t);
            }
            channels.push(AnimationChannel {
                channel_type: AnimationChannelType::Node,
                target_index: pm_idx as i32,
                position_keys,
                rotation_keys,
                scaling_keys,
                mesh_keys: Vec::new(),
                morph_keys: Vec::new(),
                position_times: times.clone(),
                rotation_times: times.clone(),
                scaling_times: times,
            });
        }

        let mut node_channel_map: Vec<Vec<usize>> = vec![Vec::new(); pm_node_count];
        for (ci, ch) in channels.iter().enumerate() {
            if ch.target_index >= 0 && (ch.target_index as usize) < pm_node_count {
                node_channel_map[ch.target_index as usize].push(ci);
            }
        }

        out.push(AnimationData {
            name,
            // One key per frame at integer times → duration = frame_count
            // ticks. ticks_per_second = HALO_FPS so the AnimationManager
            // tick clock advances 30 ticks per second of real time.
            duration: frame_count as f32,
            ticks_per_second: HALO_FPS,
            channels,
            node_channel_map,
        });
    }

    if decode_failures > 0 {
        eprintln!("[halo] {decode_failures} jmad animations failed to decode (likely SharedStatic codec or corrupt blob)");
    }

    out
}

fn quat_from_real(q: RealQuaternion) -> Quat {
    Quat::from_xyzw(q.i, q.j, q.k, q.w)
}

fn real_point_to_vec3(p: RealPoint3d) -> Vec3 {
    Vec3::new(p.x, p.y, p.z)
}

#[allow(dead_code)]
fn real_vector_to_vec3(v: RealVector3d) -> Vec3 {
    Vec3::new(v.i, v.j, v.k)
}
