//! `e_object_type` — engine object-type discriminant. Identifies what
//! `<type>_compute_function_value` (and other type-specific code paths)
//! to dispatch to.
//!
//! Source: dllcache `e_object_type` enum (14 variants + sentinels).
//! Verified via `get_type_info?name=e_object_type` 2026-05-20.

/// 1:1 port of `e_object_type` from dllcache (`u32` discriminant).
/// Discriminant order matches the engine enum so the integer cast
/// from a tag-side `type` byte is direct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ObjectType {
    Biped         = 0,
    Vehicle       = 1,
    Weapon        = 2,
    Equipment     = 3,
    Terminal      = 4,
    Projectile    = 5,
    Scenery       = 6,
    Machine       = 7,
    Control       = 8,
    SoundScenery  = 9,
    Crate         = 10,
    Creature      = 11,
    Giant         = 12,
    EffectScenery = 13,
}

/// `k_object_types_count` = 14.
pub const OBJECT_TYPES_COUNT: usize = 14;

impl ObjectType {
    /// On-disk tag extension for this object type. Matches the engine's
    /// per-type tag-group name from `blam_tags::paths::engine_name_for_group_tag`.
    /// Used to resolve a placement's tag-relative path
    /// (e.g. `objects\weapons\melee\gravity_hammer\gravity_hammer`) to
    /// an absolute file path for `TagFile::read`.
    pub fn tag_extension(self) -> &'static str {
        match self {
            Self::Biped         => "biped",
            Self::Vehicle       => "vehicle",
            Self::Weapon        => "weapon",
            Self::Equipment     => "equipment",
            Self::Terminal      => "device_terminal",
            Self::Projectile    => "projectile",
            Self::Scenery       => "scenery",
            Self::Machine       => "device_machine",
            Self::Control       => "device_control",
            Self::SoundScenery  => "sound_scenery",
            Self::Crate         => "crate",
            Self::Creature      => "creature",
            Self::Giant         => "giant",
            Self::EffectScenery => "effect_scenery",
        }
    }

    /// 4-byte FOURCC tag group identifier — what `tag_reference`
    /// fields store on disk. Used as the cache key for
    /// [`crate::halo::tags::tag_get`].
    ///
    /// Note: the `.crate` extension uses FOURCC `bloc` in H3 MCC
    /// (verified via `crate.json:3`), not `crat` — this method
    /// returns the MCC value.
    pub fn tag_group_fourcc(self) -> [u8; 4] {
        match self {
            Self::Biped         => *b"bipd",
            Self::Vehicle       => *b"vehi",
            Self::Weapon        => *b"weap",
            Self::Equipment     => *b"eqip",
            Self::Terminal      => *b"term",
            Self::Projectile    => *b"proj",
            Self::Scenery       => *b"scen",
            Self::Machine       => *b"mach",
            Self::Control       => *b"ctrl",
            Self::SoundScenery  => *b"ssce",
            Self::Crate         => *b"bloc",
            Self::Creature      => *b"crea",
            Self::Giant         => *b"gint",
            Self::EffectScenery => *b"efsc",
        }
    }
}
