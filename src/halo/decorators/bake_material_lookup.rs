//! Per-bake `(scenario_bsp_index, render_method_index) → postprocess
//! sampler data` lookup. Built at decorator bake start, dropped at end.
//!
//! Engine equivalent: at `c_geometry_sampler::sample` →
//! `sample_diffuse_textures` time, the engine walks the global structure_bsp
//! registry: `structure_bsp_get(bsp_idx)->materials[render_method_idx]` →
//! `c_render_method.m_definition` → tag handle → `c_render_method_definition`
//! → `m_vertex_types[0]` → `s_render_method_postprocess_definition`. All of
//! this resolves through the global tag instance table.
//!
//! Protomorph adapts: we PREBUILD the (bsp_idx, mat_idx) → sampler data map
//! at the start of decorator bake by walking each active BSP's
//! `sbsp.materials[]` and loading the referenced rmsh tag + its postprocess
//! bitmaps. The result lives in a thread-local for the duration of the
//! bake; cleared on bake exit. Single-threaded by construction (the bake
//! runs on the main thread).
//!
//! The cache stores ONLY the slim postprocess subset needed by the
//! `sample_texture` / `query_real_constant` chain: textures (with full mip
//! chain + format), real_constants, and the i16[8] `runtime_queryable_properties`
//! lookup table. Total memory ~50-200 MB during bake (matches engine's
//! tag heap footprint for these materials), freed at bake end.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use blam_tags::paths::resolve_tag_path;

use crate::halo::loader::load_and_bake_render_method;
use crate::halo::render_methods::materials::InlineTexture;
use crate::halo::scenario::loader::LoadedScenario;

/// One postprocess texture's CPU sampler payload — bitmap pixel data +
/// the bake-side metadata needed to apply `query_texture_transform`'s
/// xform lookup.
#[derive(Debug)]
pub struct MaterialSamplerTexture {
    /// Decoded bitmap (mip pyramid intact). `None` when the postprocess
    /// references an extern texture (engine-side `s_shader_extern_info`
    /// supplies these per-frame; sample_diffuse_textures never reaches
    /// extern slots for albedo because all `_query_*` properties map to
    /// authored bitmaps).
    pub texture: Option<InlineTexture>,
    /// `s_render_method_postprocess_texture.texture_transform_constant_index`
    /// — i8 in [-1, 7]. `-1` means "no xform" (identity); otherwise it
    /// indexes into the parent entry's `real_constants[]`.
    pub xform_constant_index: i8,
}

/// One material's postprocess sampler payload. Engine source:
/// `s_render_method_postprocess_definition` (140 B). We retain only the
/// fields read by `sample_texture` / `query_real_constant`.
#[derive(Debug)]
pub struct MaterialSamplerEntry {
    /// Engine `m_runtime_queryable_properties[8]`. Each `i16` is either
    /// `-1` (no slot) or an index into [`Self::textures`] (for property
    /// slots 0..3, 5, 6) or [`Self::real_constants`] (for slot 4 =
    /// `_query_albedo_color`).
    pub queryable_properties: [i16; 8],
    /// Engine `m_textures[]`. Index by the resolved
    /// `queryable_properties[property]` value.
    pub textures: Vec<MaterialSamplerTexture>,
    /// Engine `m_real_constants[]`. The xform lookup
    /// (`query_texture_transform`) indexes here via the texture entry's
    /// `xform_constant_index`; the albedo-color constant lookup
    /// (`query_real_constant`) indexes here via `queryable_properties[4]`.
    pub real_constants: Vec<[f32; 4]>,
}

/// Owned per-bake (bsp_idx, mat_idx) → sampler-entry map.
#[derive(Debug, Default)]
pub struct BakeMaterialLookup {
    data: HashMap<(i32, i32), MaterialSamplerEntry>,
}

impl BakeMaterialLookup {
    /// Walk every active BSP's `sbsp.materials[]` and load each
    /// material's rmsh + postprocess bitmaps. Materials referenced by
    /// `sbsp.materials[i].render_method` but missing from disk are
    /// skipped silently (engine falls back to `shaders\invalid.shader`;
    /// sampling against a stub returns nothing, which is the correct
    /// "no diffuse contribution" behavior for the bounce path).
    pub fn build(scenario: &LoadedScenario, tags_root: &Path) -> Self {
        let mut data = HashMap::new();
        for loaded_bsp in &scenario.active_bsps {
            let bsp_idx = loaded_bsp.scenario_bsp_index as i32;
            for (mat_idx, mat) in loaded_bsp.sbsp.materials.iter().enumerate() {
                if mat.render_method.is_empty() {
                    continue;
                }
                let rmsh_path =
                    resolve_tag_path(tags_root, &mat.render_method, mat.render_method_extension());
                // Author-format rmsh tags ship with an unbaked postprocess
                // block — tool.exe normally synthesizes textures +
                // real_constants + runtime_queryable_properties at cache
                // build. We run the same `RenderMethod::bake` pass here so
                // the sampler sees a fully-populated postprocess (same
                // engine-faithful bake `loader.rs::resolve_render_method`
                // runs for the GPU material walk).
                let Some(rmsh) = load_and_bake_render_method(&rmsh_path, tags_root) else {
                    continue;
                };
                let Some(pp) = rmsh.postprocess_definition.as_ref() else {
                    continue;
                };
                let mut textures: Vec<MaterialSamplerTexture> = Vec::with_capacity(pp.textures.len());
                for pp_tex in &pp.textures {
                    let inline = if pp_tex.bitmap_path.is_empty() {
                        None
                    } else {
                        let bitm_path = resolve_tag_path(tags_root, &pp_tex.bitmap_path, "bitmap");
                        crate::halo::bitmaps::load_image(
                            &bitm_path,
                            pp_tex.bitmap_index.max(0) as usize,
                        )
                    };
                    textures.push(MaterialSamplerTexture {
                        texture: inline,
                        xform_constant_index: pp_tex.texture_transform_constant_index,
                    });
                }
                data.insert(
                    (bsp_idx, mat_idx as i32),
                    MaterialSamplerEntry {
                        queryable_properties: pp.runtime_queryable_properties,
                        textures,
                        real_constants: pp.real_constants.clone(),
                    },
                );
            }
        }
        if std::env::var_os("PROTOMORPH_DIAG_BAKE_HIST").is_some() {
            eprintln!(
                "[bake-material-lookup] built {} entries across {} active BSPs",
                data.len(),
                scenario.active_bsps.len(),
            );
            for ((bsp, mat), entry) in data.iter().take(3) {
                let with_tex = entry.textures.iter().filter(|t| t.texture.is_some()).count();
                eprintln!(
                    "  (bsp={bsp}, mat={mat}): queryable_properties={:?} textures={}/{} real_constants={}",
                    entry.queryable_properties,
                    with_tex,
                    entry.textures.len(),
                    entry.real_constants.len(),
                );
            }
        }
        Self { data }
    }

    pub fn get(&self, scenario_bsp_index: i32, render_method_index: i32) -> Option<&MaterialSamplerEntry> {
        self.data.get(&(scenario_bsp_index, render_method_index))
    }
}

thread_local! {
    /// Thread-local cache active during `c_geometry_sampler::sample`'s
    /// bake-driven loops. Set by [`with_bake_lookup`]; readers see
    /// `Some(&lookup)` only during the scoped closure.
    static BAKE_LOOKUP: RefCell<Option<BakeMaterialLookup>> = const { RefCell::new(None) };
}

/// Install `lookup` as the thread-local bake lookup for the duration of
/// `f`. Restores the previous value (typically `None`) on return.
pub fn with_bake_lookup<F, R>(lookup: BakeMaterialLookup, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prior = BAKE_LOOKUP.with(|c| c.borrow_mut().replace(lookup));
    let result = f();
    BAKE_LOOKUP.with(|c| {
        *c.borrow_mut() = prior;
    });
    result
}

/// RAII guard variant of [`with_bake_lookup`]. Installs `lookup` on
/// construction and restores the prior thread-local on drop. Use when
/// the bake scope spans an entire function body — wrapping it in a
/// closure would force re-indenting hundreds of lines.
pub struct BakeLookupGuard {
    prior: Option<BakeMaterialLookup>,
}

impl BakeLookupGuard {
    pub fn install(lookup: BakeMaterialLookup) -> Self {
        let prior = BAKE_LOOKUP.with(|c| c.borrow_mut().replace(lookup));
        Self { prior }
    }
}

impl Drop for BakeLookupGuard {
    fn drop(&mut self) {
        BAKE_LOOKUP.with(|c| {
            *c.borrow_mut() = self.prior.take();
        });
    }
}

/// Borrow the material's sampler entry for the duration of `f`. `f`
/// receives `None` outside a [`with_bake_lookup`] scope, or when the
/// lookup doesn't carry an entry for the given key (uncached materials,
/// load failures, etc.) — `sample_diffuse_textures` interprets `None` as
/// "no postprocess data; return all-zero diffuse" (matches the engine's
/// path when `m_render_method_index == -1` short-circuits at the top).
///
/// Callback-based to keep the `RefCell` borrow scoped to `f` — escaping
/// the borrow across closure boundaries is rejected by the borrow
/// checker, so the only way to use the entry is through this scope.
pub fn with_material<F, R>(
    scenario_bsp_index: i32,
    render_method_index: i32,
    f: F,
) -> R
where
    F: FnOnce(Option<&MaterialSamplerEntry>) -> R,
{
    BAKE_LOOKUP.with(|c| {
        let borrow = c.borrow();
        let entry = borrow
            .as_ref()
            .and_then(|l| l.get(scenario_bsp_index, render_method_index));
        f(entry)
    })
}
