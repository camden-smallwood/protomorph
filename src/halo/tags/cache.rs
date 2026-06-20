//! Global tag cache. Engine equivalent: `global_tag_instances[]` —
//! the resolved tag table indexed by `tag_reference` handle. Protomorph
//! keys by `(fourcc, relative_path)` instead of by handle since we
//! don't have a runtime tag-handle pool yet, but the semantics match:
//! one parse per unique tag, shared via `Arc`.
//!
//! ## Path convention
//!
//! Engine tag-reference paths use Windows separators (`\`) and omit
//! the on-disk extension — the extension is derived from the
//! FOURCC via [`blam_tags::paths::group_tag_to_extension`]. Examples:
//!
//! ```text
//!   tag_get(*b"hlmt", "objects\\weapons\\melee\\gravity_hammer\\gravity_hammer")
//!   tag_get(*b"scen", "objects\\levels\\multi\\shrine\\marinebeacon\\marinebeacon")
//! ```
//!
//! Internally the cache normalizes lookups to lowercase since the
//! engine treats tag paths case-insensitively on disk.
//!
//! ## Loading strategy
//!
//! Lazy: the first request for a `(fourcc, path)` reads + parses;
//! subsequent requests return the cached `Arc<LoadedTag>` in O(1).
//! Load failures (file missing / parse error) are cached as `None`
//! so we don't retry every frame — matches engine's
//! "tag failed to resolve" diagnostic behavior. Each failure logs
//! once on insertion.

use blam_tags::biped::BipedDefinition;
use blam_tags::crate_definition::CrateDefinition;
use blam_tags::creature::CreatureDefinition;
use blam_tags::device_control::ControlDefinition;
use blam_tags::device_machine::MachineDefinition;
use blam_tags::device_terminal::TerminalDefinition;
use blam_tags::effect_scenery::EffectSceneryDefinition;
use blam_tags::equipment::EquipmentDefinition;
use blam_tags::file::TagFile;
use blam_tags::giant::GiantDefinition;
use blam_tags::object::ObjectDefinition;
use blam_tags::paths::{group_tag_to_extension, resolve_tag_path};
use blam_tags::projectile::ProjectileDefinition;
use blam_tags::scenery::SceneryDefinition;
use blam_tags::sound_scenery::SoundSceneryDefinition;
use blam_tags::vehicle::VehicleDefinition;
use blam_tags::weapon::WeaponDefinition;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

/// 4-byte big-endian tag group identifier — matches what
/// `tag_reference` fields store on disk (e.g. `*b"hlmt"`, `*b"scen"`,
/// `*b"weap"`).
pub type FourCC = [u8; 4];

/// Parsed runtime view of a tag. One variant per consumer-facing
/// tag shape. Add variants as new code paths come online.
///
/// Each variant holds its payload in an inner `Arc<T>` so consumers
/// can `Arc::clone` the typed body without copying. The outer
/// `Arc<LoadedTag>` (returned by [`tag_get`]) is what the cache
/// stores; the inner `Arc<T>` is for callers that want a typed
/// handle they can pass around.
#[derive(Debug)]
pub enum LoadedTag {
    /// Any of the `obje` subgroups that don't have their own typed
    /// blam-tags reader yet (`scen`, `bipd`, `crat`/`bloc`, `vehi`,
    /// `eqip`, `mach`, `ctrl`, `term`, `proj`, `ssce`, `crea`, `gint`,
    /// `efsc`). All share the same parsed shape because the `object`
    /// substruct is engine-common across them.
    ObjectDefinition(Arc<ObjectDefinition>),
    /// `weap` — full weapon definition (item + weapon-specific
    /// authored fields). Inner `item.object` chain gives any consumer
    /// that wants the obje-layer view a cheap `Arc::clone`.
    Weapon(Arc<WeaponDefinition>),
    /// `eqip` — full equipment definition (item + equipment-specific
    /// authored fields + per-type sub-blocks). Same inner-Arc story
    /// as `Weapon` for the object-layer view.
    Equipment(Arc<EquipmentDefinition>),
    /// `mach` — full machine definition (device parent + machine-
    /// specific fields). Inner `device.object` chain reachable.
    Machine(Arc<MachineDefinition>),
    /// `ctrl` — full control definition (device parent + control-
    /// specific fields).
    Control(Arc<ControlDefinition>),
    /// `term` — full terminal definition (device parent + terminal-
    /// specific authored fields + per-difficulty content blocks).
    Terminal(Arc<TerminalDefinition>),
    /// `proj` — full projectile definition (object parent + projectile-
    /// specific ballistics fields).
    Projectile(Arc<ProjectileDefinition>),
    /// `bipd` — biped definition (unit parent + biped-specific fields
    /// + physics substruct).
    Biped(Arc<BipedDefinition>),
    /// `vehi` — vehicle definition (unit parent + vehicle-specific
    /// fields).
    Vehicle(Arc<VehicleDefinition>),
    /// `gint` — giant definition (unit parent + giant-specific fields).
    Giant(Arc<GiantDefinition>),
    /// `crea` — creature definition (object parent + creature fields).
    Creature(Arc<CreatureDefinition>),
    /// `scen` — scenery definition (object parent + pathfinding/
    /// lightmapping policies).
    Scenery(Arc<SceneryDefinition>),
    /// `bloc` — crate definition (object parent + self-destruct timer).
    Crate(Arc<CrateDefinition>),
    /// `ssce` — sound scenery definition (object parent only).
    SoundScenery(Arc<SoundSceneryDefinition>),
    /// `efsc` — effect scenery definition (object parent only).
    EffectScenery(Arc<EffectSceneryDefinition>),
    /// `hlmt` — model tag. Carries the variants[] block name list
    /// needed by `object_compute_function_value` case `variant`.
    Model(Arc<LoadedModel>),
}

/// Lightweight `.model` (hlmt) tag view. Only fields actively read by
/// the runtime today are surfaced; grow as consumers need more.
#[derive(Debug, Clone, Default)]
pub struct LoadedModel {
    /// Variant `name` (string_id) per entry of the `variants[]` block.
    /// Used by `ObjectDatum::from_placement` to resolve a placement's
    /// `permutation_data.variant_name` → numeric `variant_index`, and
    /// by `object_compute_function_value` case `variant` for the
    /// count (engine sid 501: `(index+1) / count`).
    pub variant_names: Vec<String>,

    /// `model object data[0].radius` — the cache-builder's vertex-walked
    /// auto-bake bounding sphere radius. The object datum's
    /// `bounding_sphere_radius` is sourced from the object tag's
    /// `bounding_radius` when authored, else this (engine: object-tag
    /// 0 → `s_model_object_data[0]` flows through). `lights_distant_
    /// lighting_at_point_new @ 0x1808A3220` reads that radius to size the
    /// per-object lighting raycast (`ray = 10·max(radius, 0.4)`), so a
    /// missing model radius makes tall objects (zanzibar main_crane) cast
    /// a too-short ray, miss all geometry, and render black. `0.0` when the
    /// hlmt has no `model object data`.
    pub bounding_sphere_radius: f32,
}

impl LoadedTag {
    /// Cheap typed clone of the inner `Arc<ObjectDefinition>` for any
    /// variant that wraps one — `ObjectDefinition` directly, or
    /// `Weapon` / `Equipment` via the `item.object` chain. Future
    /// derived variants (`Biped`, etc.) plug into this same pattern.
    pub fn as_object_definition(&self) -> Option<Arc<ObjectDefinition>> {
        match self {
            LoadedTag::ObjectDefinition(a) => Some(a.clone()),
            LoadedTag::Weapon(w) => Some(w.item.object.clone()),
            LoadedTag::Equipment(e) => Some(e.item.object.clone()),
            LoadedTag::Machine(m) => Some(m.device.object.clone()),
            LoadedTag::Control(c) => Some(c.device.object.clone()),
            LoadedTag::Terminal(t) => Some(t.device.object.clone()),
            LoadedTag::Projectile(p) => Some(p.object.clone()),
            LoadedTag::Biped(b) => Some(b.unit.object.clone()),
            LoadedTag::Vehicle(v) => Some(v.unit.object.clone()),
            LoadedTag::Giant(g) => Some(g.unit.object.clone()),
            LoadedTag::Creature(c) => Some(c.object.clone()),
            LoadedTag::Scenery(s) => Some(s.object.clone()),
            LoadedTag::Crate(c) => Some(c.object.clone()),
            LoadedTag::SoundScenery(s) => Some(s.object.clone()),
            LoadedTag::EffectScenery(e) => Some(e.object.clone()),
            _ => None,
        }
    }
    pub fn as_biped_definition(&self) -> Option<Arc<BipedDefinition>> {
        match self {
            LoadedTag::Biped(a) => Some(a.clone()),
            _ => None,
        }
    }
    pub fn as_projectile_definition(&self) -> Option<Arc<ProjectileDefinition>> {
        match self {
            LoadedTag::Projectile(a) => Some(a.clone()),
            _ => None,
        }
    }
    /// Cheap typed clone of the inner `Arc<WeaponDefinition>` when
    /// this is a `weap` tag.
    pub fn as_weapon_definition(&self) -> Option<Arc<WeaponDefinition>> {
        match self {
            LoadedTag::Weapon(a) => Some(a.clone()),
            _ => None,
        }
    }
    /// Cheap typed clone of the inner `Arc<EquipmentDefinition>`
    /// when this is an `eqip` tag.
    pub fn as_equipment_definition(&self) -> Option<Arc<EquipmentDefinition>> {
        match self {
            LoadedTag::Equipment(a) => Some(a.clone()),
            _ => None,
        }
    }
    pub fn as_machine_definition(&self) -> Option<Arc<MachineDefinition>> {
        match self {
            LoadedTag::Machine(a) => Some(a.clone()),
            _ => None,
        }
    }
    pub fn as_control_definition(&self) -> Option<Arc<ControlDefinition>> {
        match self {
            LoadedTag::Control(a) => Some(a.clone()),
            _ => None,
        }
    }
    pub fn as_terminal_definition(&self) -> Option<Arc<TerminalDefinition>> {
        match self {
            LoadedTag::Terminal(a) => Some(a.clone()),
            _ => None,
        }
    }
    /// Cheap typed clone of the inner `Arc<LoadedModel>` when this is
    /// an `hlmt` tag.
    pub fn as_model(&self) -> Option<Arc<LoadedModel>> {
        match self {
            LoadedTag::Model(a) => Some(a.clone()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache state
// ---------------------------------------------------------------------------

type CacheKey = (FourCC, String);

struct CacheState {
    /// Absolute filesystem root containing the `tags/` tree. Set by
    /// [`tag_init`] at scenario load.
    tags_root: Option<PathBuf>,
    /// `Some(Arc<LoadedTag>)` for successful loads; `None` for cached
    /// failures (negative cache).
    entries: HashMap<CacheKey, Option<Arc<LoadedTag>>>,
}

// `HashMap::new` isn't const-callable, so we defer the cache's
// allocation to first use via `OnceLock`. Same single-tenant
// semantics as a `static RwLock`, just with lazy init.
static TAG_CACHE: OnceLock<RwLock<CacheState>> = OnceLock::new();

fn cache() -> &'static RwLock<CacheState> {
    TAG_CACHE.get_or_init(|| {
        RwLock::new(CacheState {
            tags_root: None,
            entries: HashMap::new(),
        })
    })
}

/// Initialize the cache with `tags_root`. Call once per scenario
/// load; resets any previously-cached entries.
pub fn tag_init(tags_root: PathBuf) {
    let mut g = cache().write().unwrap();
    g.tags_root = Some(tags_root);
    g.entries.clear();
}

/// Engine `tag_get(group, name)` equivalent. Returns the loaded tag
/// for `(group, rel_path)` or `None` if the tag failed to load.
///
/// Lazy: first call for a key reads from disk + parses; subsequent
/// calls return the cached `Arc`. Failures are cached as `None` and
/// logged once per `(group, path)` pair.
pub fn tag_get(group: FourCC, rel_path: &str) -> Option<Arc<LoadedTag>> {
    // Normalize the path key — engine matches case-insensitively.
    let key: CacheKey = (group, rel_path.to_ascii_lowercase());

    let cache = cache();

    // Fast path: read lock, cache hit.
    {
        let g = cache.read().unwrap();
        if let Some(slot) = g.entries.get(&key) {
            return slot.clone();
        }
    }

    // Cache miss — promote to write lock, double-check, then load.
    let mut g = cache.write().unwrap();
    if let Some(slot) = g.entries.get(&key) {
        return slot.clone();
    }
    let loaded = load_uncached(g.tags_root.as_ref(), group, rel_path);
    g.entries.insert(key, loaded.clone());
    loaded
}

fn load_uncached(
    tags_root: Option<&PathBuf>,
    group: FourCC,
    rel_path: &str,
) -> Option<Arc<LoadedTag>> {
    let Some(tags_root) = tags_root else {
        eprintln!(
            "[tag_cache] tag_get({}, {:?}) before tag_init — tags_root unset",
            fourcc_str(group),
            rel_path,
        );
        return None;
    };
    if rel_path.is_empty() {
        return None;
    }
    let ext = match group_tag_to_extension(u32::from_be_bytes(group)) {
        Some(e) => e,
        None => {
            eprintln!(
                "[tag_cache] unknown FOURCC {:?} for {:?} — extend \
                 paths::group_tag_to_extension",
                fourcc_str(group),
                rel_path,
            );
            return None;
        }
    };
    let abs = resolve_tag_path(tags_root, rel_path, ext);
    let tag = match TagFile::read(&abs) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[tag_cache] TagFile::read({}) failed: {} — caching as miss",
                abs.display(),
                e,
            );
            return None;
        }
    };
    parse(group, &tag).map(Arc::new)
}

fn parse(group: FourCC, tag: &TagFile) -> Option<LoadedTag> {
    // Per-subgroup typed parsers are matched first so a derived tag
    // (e.g. `weap`) gets its richer view. Fall through to the
    // generic `ObjectDefinition` walk for any subgroup without its
    // own typed reader yet.
    match &group {
        b"weap" => {
            return WeaponDefinition::from_tag(tag)
                .ok()
                .map(|w| LoadedTag::Weapon(Arc::new(w)));
        }
        b"eqip" => {
            return EquipmentDefinition::from_tag(tag)
                .ok()
                .map(|e| LoadedTag::Equipment(Arc::new(e)));
        }
        b"mach" => {
            return MachineDefinition::from_tag(tag)
                .ok()
                .map(|m| LoadedTag::Machine(Arc::new(m)));
        }
        b"ctrl" => {
            return ControlDefinition::from_tag(tag)
                .ok()
                .map(|c| LoadedTag::Control(Arc::new(c)));
        }
        b"term" => {
            return TerminalDefinition::from_tag(tag)
                .ok()
                .map(|t| LoadedTag::Terminal(Arc::new(t)));
        }
        b"proj" => {
            return ProjectileDefinition::from_tag(tag)
                .ok()
                .map(|p| LoadedTag::Projectile(Arc::new(p)));
        }
        b"bipd" => {
            return BipedDefinition::from_tag(tag)
                .ok()
                .map(|b| LoadedTag::Biped(Arc::new(b)));
        }
        b"vehi" => {
            return VehicleDefinition::from_tag(tag)
                .ok()
                .map(|v| LoadedTag::Vehicle(Arc::new(v)));
        }
        b"gint" => {
            return GiantDefinition::from_tag(tag)
                .ok()
                .map(|g| LoadedTag::Giant(Arc::new(g)));
        }
        b"crea" => {
            return CreatureDefinition::from_tag(tag)
                .ok()
                .map(|c| LoadedTag::Creature(Arc::new(c)));
        }
        b"scen" => {
            return SceneryDefinition::from_tag(tag)
                .ok()
                .map(|s| LoadedTag::Scenery(Arc::new(s)));
        }
        b"bloc" => {
            return CrateDefinition::from_tag(tag)
                .ok()
                .map(|c| LoadedTag::Crate(Arc::new(c)));
        }
        b"ssce" => {
            return SoundSceneryDefinition::from_tag(tag)
                .ok()
                .map(|s| LoadedTag::SoundScenery(Arc::new(s)));
        }
        b"efsc" => {
            return EffectSceneryDefinition::from_tag(tag)
                .ok()
                .map(|e| LoadedTag::EffectScenery(Arc::new(e)));
        }
        b"hlmt" => return Some(LoadedTag::Model(Arc::new(parse_model(tag)))),
        _ => {}
    }
    // Object subgroups without a typed reader (scen, bipd, crat/bloc,
    // vehi, eqip, mach, ctrl, term, proj, ssce, crea, gint, efsc) all
    // parse into the engine-common `ObjectDefinition` shape.
    if blam_tags::object::OBJECT_SUBGROUPS.contains(&group) {
        return ObjectDefinition::from_tag(tag)
            .ok()
            .map(|o| LoadedTag::ObjectDefinition(Arc::new(o)));
    }
    eprintln!(
        "[tag_cache] no parser registered for FOURCC {:?} — \
         add a LoadedTag variant in src/halo/tags/cache.rs",
        fourcc_str(group),
    );
    None
}

/// Walk a `.model` (hlmt) tag's `variants[]` block and collect the
/// `name` string_id of each entry. Other model fields stay deferred
/// until a consumer needs them.
///
/// Schema reference: `definitions/halo3_mcc/model.json` →
/// `model_variant_block` (48 bytes per entry, first field `name`).
fn parse_model(tag: &TagFile) -> LoadedModel {
    let variant_names = tag
        .root()
        .field("variants")
        .and_then(|f| f.as_block())
        .map(|block| {
            block
                .iter()
                .map(|entry| entry.read_string_id("name").unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    // `model object data[0].radius` — auto-bake bounding sphere. Source for
    // the object datum's bounding_sphere_radius when the object tag authored 0.
    let bounding_sphere_radius = tag
        .root()
        .field("model object data")
        .and_then(|f| f.as_block())
        .and_then(|b| b.element(0))
        .and_then(|e| e.read_real("radius"))
        .filter(|r| *r > 0.0)
        .unwrap_or(0.0);
    LoadedModel { variant_names, bounding_sphere_radius }
}

fn fourcc_str(g: FourCC) -> String {
    std::str::from_utf8(&g)
        .map(|s| s.to_string())
        .unwrap_or_else(|_| format!("{:02x?}", g))
}
