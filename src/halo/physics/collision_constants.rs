//! Engine source: `Ares/source/physics/collision_constants.h`.
//!
//! Bit-flag enums + flag-pair struct that gate every `collision_test_*` call
//! in the engine. Pure data + tiny inline helpers; no behavior. Verbatim port
//! of the Ares header — every enumerant value and helper signature mirrors
//! the C++ exactly.

use bitflags::bitflags;

// =============================================================================
// e_collision_test_flag
// =============================================================================
//
// Mirrors:
//
// ```cpp
// enum e_collision_test_flag
// {
//     _collision_test_structure_bit = 0,
//     _collision_test_media_bit,
//     _collision_test_soft_ceilings_bit,
//     ...
//     k_collision_test_flags_count,
// };
// ```
//
// Engine wraps this in `c_flags_no_init<e_collision_test_flag, unsigned long,
// k_collision_test_flags_count>` (= [`CollisionTestFlags`] below).

bitflags! {
    /// One bit per `e_collision_test_flag` value. The bit number is the
    /// enum value, exactly matching the engine's `c_flags_no_init` packing.
    /// Engine type: `c_collision_test_flags`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct CollisionTestFlags: u32 {
        const STRUCTURE                                    = 1 <<  0; // _collision_test_structure_bit
        const MEDIA                                        = 1 <<  1; // _collision_test_media_bit
        const SOFT_CEILINGS                                = 1 <<  2; // _collision_test_soft_ceilings_bit
        const INSTANCED_GEOMETRY                           = 1 <<  3; // _collision_test_instanced_geometry_bit
        const RENDER_ONLY_BSPS                             = 1 <<  4; // _collision_test_render_only_bsps_bit
        const IGNORE_CHILD_OBJECTS                         = 1 <<  5; // _collision_test_ignore_child_objects_bit
        const IGNORE_NONPATHFINDABLE_OBJECTS               = 1 <<  6; // _collision_test_ignore_nonpathfindable_objects_bit
        const IGNORE_NON_FLIGHT_BLOCKING                   = 1 <<  7; // _collision_test_ignore_non_flight_blocking_bit
        const IGNORE_CINEMATIC_OBJECTS                     = 1 <<  8; // _collision_test_ignore_cinematic_objects_bit
        const IGNORE_DEAD_BIPEDS                           = 1 <<  9; // _collision_test_ignore_dead_bipeds_bit
        const IGNORE_GEOMETRY_THAT_IGNORES_AOE             = 1 << 10; // _collision_test_ignore_geometry_that_ignores_aoe_bit
        const FRONT_FACING_SURFACES                        = 1 << 11; // _collision_test_front_facing_surfaces_bit
        const BACK_FACING_SURFACES                         = 1 << 12; // _collision_test_back_facing_surfaces_bit
        const IGNORE_TWO_SIDED_SURFACES                    = 1 << 13; // _collision_test_ignore_two_sided_surfaces_bit
        const IGNORE_INVISIBLE_SURFACES                    = 1 << 14; // _collision_test_ignore_invisible_surfaces_bit
        const IGNORE_BREAKABLE_SURFACES                    = 1 << 15; // _collision_test_ignore_breakable_surfaces_bit
        const ALLOW_EARLY_OUT                              = 1 << 16; // _collision_test_allow_early_out_bit
        const TRY_TO_KEEP_LOCATION_VALID                   = 1 << 17; // _collision_test_try_to_keep_location_valid_bit
        const IGNORE_NON_PRT_OBJECTS                       = 1 << 18; // _collision_test_ignore_non_prt_objects_bit
        const IGNORE_NON_COLLISION_MODEL_PHYSICS           = 1 << 19; // _collision_test_ignore_non_collision_model_physics_bit
        const IGNORE_NON_CAMERA_CAMERA_COLLISION           = 1 << 20; // _collision_test_ignore_non_camera_camera_collision_bit
        const IGNORE_INVISIBLE_TO_PROJECTILE_AIMING        = 1 << 21; // _collision_test_ignore_invisible_to_projectile_aiming_bit
    }
}

/// Number of distinct `e_collision_test_flag` values — matches engine's
/// `k_collision_test_flags_count`.
pub const NUMBER_OF_COLLISION_TEST_FLAGS: u32 = 22;

// =============================================================================
// e_collision_test_objects_flag
// =============================================================================
//
// Mirrors:
//
// ```cpp
// enum e_collision_test_objects_flag
// {
//     _collision_test_objects_bit            = 0,  // 0x0000
//     _collision_test_objects_first_type_bit = 1,  // 0x0001 (alias of bipeds)
//     _collision_test_objects_bipeds_bit     = 1,  // 0x0001
//     _collision_test_objects_vehicles_bit   = 2,  // 0x0002
//     _collision_test_objects_weapons_bit    = 3,  // 0x0003
//     _collision_test_objects_equipment_bit  = 4,  // 0x0004
//     _collision_test_objects_terminals_bit  = 5,  // 0x0005
//     _collision_test_objects_projectiles_bit = 6, // 0x0006
//     _collision_test_objects_scenery_bit    = 7,  // 0x0007
//     _collision_test_objects_machines_bit   = 8,  // 0x0008
//     _collision_test_objects_controls_bit   = 9,  // 0x0009
//     _collision_test_objects_sound_scenery_bit = 10, // 0x000A
//     _collision_test_objects_crates_bit     = 11, // 0x000B
//     _collision_test_objects_creatures_bit  = 12, // 0x000C
//     _collision_test_objects_giants_bit     = 13, // 0x000D
//     _collision_test_objects_effect_scenery_bit = 14, // 0x000E
//     _collision_test_objects_last_type_bit  = 14, // 0x000E (alias of effect_scenery)
//     k_collision_test_objects_flags_count   = 15, // 0x000F
// };
// ```
//
// Note bit 0 is the master "test objects" toggle; bits 1-14 select per-type.
// Engine wraps as `c_collision_test_objects_flags` (15-bit no-init flag set).

bitflags! {
    /// Engine type: `c_collision_test_objects_flags`. Bit 0 is the master
    /// objects-on toggle; bits 1-14 enable per-object-type testing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct CollisionTestObjectsFlags: u32 {
        const OBJECTS         = 1 <<  0; // _collision_test_objects_bit (master enable)
        const BIPEDS          = 1 <<  1; // _collision_test_objects_bipeds_bit / first_type
        const VEHICLES        = 1 <<  2;
        const WEAPONS         = 1 <<  3;
        const EQUIPMENT       = 1 <<  4;
        const TERMINALS       = 1 <<  5;
        const PROJECTILES     = 1 <<  6;
        const SCENERY         = 1 <<  7;
        const MACHINES        = 1 <<  8;
        const CONTROLS        = 1 <<  9;
        const SOUND_SCENERY   = 1 << 10;
        const CRATES          = 1 << 11;
        const CREATURES       = 1 << 12;
        const GIANTS          = 1 << 13;
        const EFFECT_SCENERY  = 1 << 14; // last_type
    }
}

pub const FIRST_OBJECT_TYPE_BIT: u32 = 1; // _collision_test_objects_first_type_bit
pub const LAST_OBJECT_TYPE_BIT: u32 = 14; // _collision_test_objects_last_type_bit
pub const NUMBER_OF_COLLISION_TEST_OBJECTS_FLAGS: u32 = 15;

// =============================================================================
// s_collision_test_flags
// =============================================================================
//
// Mirrors:
//
// ```cpp
// struct s_collision_test_flags
// {
//     c_collision_test_flags         collision_flags;  // 0x0
//     c_collision_test_objects_flags object_flags;     // 0x4
// };
// static_assert(sizeof(s_collision_test_flags) == 8);
// ```
//
// Engine passes this by value as the first argument to every
// `collision_test_*` entry point.

/// Engine `s_collision_test_flags` (8 B). Pairs the surface-test mask with
/// the object-type mask so callers can specify "test BSP only / objects only
/// / both" in one parameter.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CollisionTestFlagsPair {
    /// Engine `collision_flags` @ 0x0 — wraps `c_collision_test_flags`.
    pub collision_flags: CollisionTestFlags,
    /// Engine `object_flags` @ 0x4 — wraps `c_collision_test_objects_flags`.
    pub object_flags: CollisionTestObjectsFlags,
}

impl CollisionTestFlagsPair {
    /// `collision_test_flags_build` — construct from raw u32 masks. Mirrors
    /// the engine inline helper of the same name.
    pub const fn build(collision: u32, objects: u32) -> Self {
        Self {
            collision_flags: CollisionTestFlags::from_bits_truncate(collision),
            object_flags: CollisionTestObjectsFlags::from_bits_truncate(objects),
        }
    }

    /// `collision_test_flags_or`.
    pub const fn or(a: Self, b: Self) -> Self {
        Self {
            collision_flags: a.collision_flags.union(b.collision_flags),
            object_flags: a.object_flags.union(b.object_flags),
        }
    }

    /// `collision_test_flags_and`.
    pub const fn and(a: Self, b: Self) -> Self {
        Self {
            collision_flags: a.collision_flags.intersection(b.collision_flags),
            object_flags: a.object_flags.intersection(b.object_flags),
        }
    }

    /// `collision_test_flags_flip`.
    pub const fn flip(a: Self) -> Self {
        Self {
            collision_flags: a.collision_flags.complement(),
            object_flags: a.object_flags.complement(),
        }
    }

    /// `collision_test_flags_empty`.
    pub const fn empty(self) -> bool {
        self.collision_flags.bits() == 0 && self.object_flags.bits() == 0
    }
}

// Per-Ares header `static_assert(sizeof(s_collision_test_flags) == 8)`.
const _: () = assert!(std::mem::size_of::<CollisionTestFlagsPair>() == 8);

// =============================================================================
// Global flag presets
// =============================================================================
//
// Engine declares 28 `s_collision_test_flags` globals at the bottom of the
// header (e.g. `_collision_test_environment_flags`). The actual bit values
// are set by `physics/collision_constants.cpp` initialization — which we
// haven't decompiled yet because no consumer in the decorator-bake chain
// reads them. Adding them here as `OnceCell`-style placeholders would be
// premature; they'll go in alongside whichever later phase first reaches
// for one (search the dllcache decompiles in
// `/Users/camden/Source/protomorph/refs/engine_geometry_sampler/` for
// `_collision_test_*_flags` references when porting).
