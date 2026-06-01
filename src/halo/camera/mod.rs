//! Subset of `Ares/source/camera/` used by the render path.
//!
//! Halo's `camera/` is the observer/camera-state subsystem (director,
//! first-person cam, dead cam, scripted cam, etc.). The full update
//! pipeline is gameplay (input → director → observer command →
//! collision → result). protomorph ports the **result types** + the
//! **host-friendly fast path** for building results from raw vectors
//! (bypassing the gameplay command stream).
//!
//! See `memory/reference_observer.md` for the design + IDA addresses.

pub mod flying_camera;
pub mod observer;

pub use flying_camera::{CameraUniforms, FlyingCamera};
pub use observer::ObserverResult;
