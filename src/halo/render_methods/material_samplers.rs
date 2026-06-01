//! Per-binding sampler deduplication for engine-faithful sampler routing.
//!
//! ## Problem
//! Engine binds **one sampler per bitmap parameter** via the rmt2's
//! per-binding `BitmapAddressMode` + `BitmapFilterMode`. Naive per-binding
//! emission overflows Metal's `max_samplers_per_shader_stage = 16` cap
//! because [`MaterialBindings`] declares 17-21 *baseline* textures so
//! WGSL fragments referencing common names (`base_map`, `bump_map`, etc.)
//! always have valid declarations.
//!
//! ## Solution
//! Group `TextureBindingSlot`s by their resolved `SamplerKey`. Per
//! material the unique-key count is typically 2-4 (most bindings share
//! Linear/Repeat; only a handful author Clamp). The BGL emits one sampler
//! entry per unique key; the WGSL assembler substitutes per-texture
//! `<tex>_sampler` references with the shared `s_dedupe_K` name based
//! on the per-material binding→key-index map.
//!
//! ## Variant cache key
//! Two materials with identical sampler dedup signatures can share the
//! same WGSL output (variant cache hit). Materials with different
//! signatures get distinct WGSL — the `signature` field is the stable
//! hash that feeds into the variant cache key.

use std::collections::{BTreeMap, HashMap};

use crate::halo::rasterizer::dx11::SamplerKey;
use crate::halo::render_methods::material_bindings::{MaterialBindings, TextureBindingSlot};
use crate::halo::render_methods::materials::MaterialData;

/// Per-material map from texture binding name → unique sampler key
/// index. The renderer:
/// - Emits one sampler binding per `unique_keys[i]` in the BGL.
/// - Creates one `wgpu::Sampler` per `unique_keys[i]` via
///   `SamplerStateCache::get`.
/// - Substitutes `<tex>_sampler` in WGSL fragments with `s_dedupe_K`
///   where `K = binding_to_key_index[<tex>]`.
#[derive(Debug, Clone)]
pub struct MaterialSamplerMap {
    /// Unique `SamplerKey`s observed across this material's bindings,
    /// in stable order (sorted by debug-derived equality). Each index
    /// maps to one WGSL `var s_dedupe_K: sampler;` declaration.
    pub unique_keys: Vec<SamplerKey>,
    /// For each `TextureBindingSlot.name` → index into `unique_keys`.
    /// Baseline textures with no `BitmapBinding` map to whichever entry
    /// corresponds to `SamplerKey::FILTERING_FALLBACK` (which is guaranteed
    /// to be in `unique_keys` if any baseline texture exists).
    pub binding_to_key_index: HashMap<String, usize>,
    /// Stable 64-bit signature of `unique_keys` — used as part of the
    /// variant cache key so materials with different sampler dedup
    /// signatures compile distinct WGSL while identical-signature
    /// materials share.
    pub signature: u64,
}

impl MaterialSamplerMap {
    /// Build the dedup map for a material given its bindings.
    /// `mat.resolved.parameters` carries each bitmap binding's authored
    /// `BitmapAddressMode` / `BitmapFilterMode`; baseline textures
    /// without a binding fall back to `SamplerKey::FILTERING_FALLBACK`.
    pub fn from_material(mat: &MaterialData, bindings: &MaterialBindings) -> Self {
        Self::from_resolved(&mat.resolved, bindings)
    }

    /// Same as [`from_material`] but takes the resolved render method
    /// directly. Used by the WGSL assembler which doesn't carry a full
    /// `MaterialData`.
    pub fn from_resolved(
        rm: &blam_tags::render_method::ResolvedRenderMethod,
        bindings: &MaterialBindings,
    ) -> Self {
        // Per-binding key resolution — for each TextureBindingSlot,
        // either pull its key from the rmt2 (`for_binding`) or fall
        // back to `FILTERING_FALLBACK` (baseline textures, externs).
        let mut per_binding: Vec<(String, SamplerKey)> =
            Vec::with_capacity(bindings.textures.len());
        for tb in &bindings.textures {
            let key = SamplerKey::for_binding_in_resolved(rm, &tb.name)
                .unwrap_or(SamplerKey::FILTERING_FALLBACK);
            per_binding.push((tb.name.clone(), key));
        }

        // Dedup keys with stable ordering. BTreeMap keyed on the debug
        // representation of `SamplerKey` gives a deterministic order
        // independent of binding iteration order — important so the
        // dedup signature is stable across runs (and across materials
        // with the same set of keys but different binding orderings).
        let mut key_index: BTreeMap<String, (SamplerKey, usize)> = BTreeMap::new();
        for (_, key) in &per_binding {
            let key_repr = format!("{:?}", key);
            let next_idx = key_index.len();
            key_index.entry(key_repr).or_insert((*key, next_idx));
        }
        // Unique keys in stable order — index = the value `usize` we
        // assigned above. Re-index sequentially for compactness.
        let mut unique_keys: Vec<(SamplerKey, usize)> =
            key_index.into_values().collect();
        // Sort by the original insertion index to keep the assignment.
        unique_keys.sort_by_key(|(_, idx)| *idx);
        // Re-index after sort (the original indices may have gaps if
        // future code drops entries; this guarantees [0..N) contiguous).
        let unique_keys: Vec<SamplerKey> = unique_keys
            .into_iter()
            .enumerate()
            .map(|(_, (k, _))| k)
            .collect();

        // Per-binding name → unique-key index lookup. Re-resolve the
        // key per binding (cheap) and find its position in `unique_keys`.
        let mut binding_to_key_index: HashMap<String, usize> = HashMap::with_capacity(
            per_binding.len(),
        );
        for (name, key) in per_binding {
            let idx = unique_keys
                .iter()
                .position(|k| format!("{k:?}") == format!("{key:?}"))
                .expect("key was just inserted above");
            binding_to_key_index.insert(name, idx);
        }

        // Stable signature for variant cache. Hash the unique keys in
        // their stable order so two materials with identical sampler
        // dedup signatures hash to the same value (cache hit).
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for k in &unique_keys {
            // SamplerKey implements Hash directly.
            k.hash(&mut hasher);
        }
        let signature = hasher.finish();

        Self {
            unique_keys,
            binding_to_key_index,
            signature,
        }
    }

    /// WGSL identifier for the deduped sampler at `unique_keys[idx]`.
    /// Used both by `emit_wgsl` (to declare bindings) and by the
    /// fragment substitution pass (to replace `<tex>_sampler`).
    pub fn dedupe_name(idx: usize) -> String {
        format!("s_dedupe_{idx}")
    }

    /// Look up the dedupe sampler name for a binding by texture name.
    /// Returns `None` if the binding isn't in this material — caller
    /// can fall back to the legacy `s_material` slot.
    pub fn lookup_sampler_name(&self, binding_name: &str) -> Option<String> {
        self.binding_to_key_index
            .get(binding_name)
            .map(|&idx| Self::dedupe_name(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_name_is_deterministic() {
        assert_eq!(MaterialSamplerMap::dedupe_name(0), "s_dedupe_0");
        assert_eq!(MaterialSamplerMap::dedupe_name(7), "s_dedupe_7");
    }

    #[test]
    fn empty_material_produces_empty_map() {
        // Sanity: a material with no bindings should produce empty
        // unique_keys + empty binding map (degenerate case — shouldn't
        // happen in practice since baseline textures always exist, but
        // covers the algorithm's edge).
        let bindings = MaterialBindings {
            textures: Vec::new(),
            sampler_slot: 0,
            cbuffer_slot: 1,
            decal_constants_slot: None,
        };
        let mat = MaterialData::stub_for_shader("test_empty");
        let map = MaterialSamplerMap::from_material(&mat, &bindings);
        assert!(map.unique_keys.is_empty());
        assert!(map.binding_to_key_index.is_empty());
    }

    #[test]
    fn baseline_only_material_dedupes_to_one_key() {
        // All bindings without rmt2 BitmapBindings → all fall back to
        // FILTERING_FALLBACK → one unique key, all bindings point at
        // index 0.
        let bindings = MaterialBindings {
            textures: vec![
                TextureBindingSlot {
                    name: "base_map".into(),
                    slot: 0,
                    is_cube: false,
                    source_extern: None,
                },
                TextureBindingSlot {
                    name: "bump_map".into(),
                    slot: 1,
                    is_cube: false,
                    source_extern: None,
                },
                TextureBindingSlot {
                    name: "detail_map".into(),
                    slot: 2,
                    is_cube: false,
                    source_extern: None,
                },
            ],
            sampler_slot: 3,
            cbuffer_slot: 4,
            decal_constants_slot: None,
        };
        let mat = MaterialData::stub_for_shader("test_baseline");
        let map = MaterialSamplerMap::from_material(&mat, &bindings);
        assert_eq!(map.unique_keys.len(), 1, "all baseline → one unique key");
        assert_eq!(map.binding_to_key_index["base_map"], 0);
        assert_eq!(map.binding_to_key_index["bump_map"], 0);
        assert_eq!(map.binding_to_key_index["detail_map"], 0);
    }

    #[test]
    fn signature_is_stable_across_calls() {
        let bindings = MaterialBindings {
            textures: vec![
                TextureBindingSlot {
                    name: "base_map".into(),
                    slot: 0,
                    is_cube: false,
                    source_extern: None,
                },
            ],
            sampler_slot: 1,
            cbuffer_slot: 2,
            decal_constants_slot: None,
        };
        let mat = MaterialData::stub_for_shader("test_stable");
        let a = MaterialSamplerMap::from_material(&mat, &bindings);
        let b = MaterialSamplerMap::from_material(&mat, &bindings);
        assert_eq!(a.signature, b.signature);
    }
}
