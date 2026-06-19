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

impl ObjectType {
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

    /// Inverse of [`Self::tag_group_fourcc`] — map a tag-reference group
    /// FOURCC back to its `ObjectType`. Used when spawning a tag named by
    /// a generic `tag_reference` (e.g. a model-variant child object whose
    /// group is `vehi`/`weap`). `None` for non-object groups.
    pub fn from_group_fourcc(group: [u8; 4]) -> Option<Self> {
        Some(match &group {
            b"bipd" => Self::Biped,
            b"vehi" => Self::Vehicle,
            b"weap" => Self::Weapon,
            b"eqip" => Self::Equipment,
            b"term" => Self::Terminal,
            b"proj" => Self::Projectile,
            b"scen" => Self::Scenery,
            b"mach" => Self::Machine,
            b"ctrl" => Self::Control,
            b"ssce" => Self::SoundScenery,
            b"bloc" => Self::Crate,
            b"crea" => Self::Creature,
            b"gint" => Self::Giant,
            b"efsc" => Self::EffectScenery,
            _ => return None,
        })
    }
}
