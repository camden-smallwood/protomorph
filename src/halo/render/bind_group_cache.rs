//! Per-frame `wgpu::BindGroup` reuse cache.
//!
//! `device.create_bind_group()` on the Metal backend allocates an
//! `MTLArgumentEncoder`-backed buffer per visible shader stage and
//! takes a global registry mutex (see `gfx-rs/wgpu#5333`). 20-40
//! per-frame creates is a documented FPS bleed in Bevy-class
//! workloads. The engine's D3D11 path has no equivalent — slots are
//! populated directly via `PSSetShaderResources` with no allocation.
//!
//! This cache hashes the addresses of the resources used by a bind
//! group (texture views, samplers, buffers, layout) and reuses an
//! existing `Arc<BindGroup>` when seen again. The cache MUST be
//! cleared on resize because intermediate views are rebuilt and the
//! old keys would collide with new resources.
//!
//! Pointer identity is sufficient because:
//! - Self-owned views/samplers/buffers live on a struct field; the
//!   field's address is stable until the struct is dropped or
//!   resize replaces it.
//! - Externally-passed views (e.g. `&intermediates.depth.view`) live
//!   on `IntermediateTargets` / `GBuffer` and are similarly stable
//!   until resize.

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default)]
pub struct BindGroupCache {
    map: HashMap<u64, Arc<wgpu::BindGroup>>,
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every cached entry. Call from `resize` once the
    /// underlying intermediates have been rebuilt.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Look up by `key` and return the cached bind group; create one
    /// via `create` and insert it on miss. Returned `Arc` keeps the
    /// entry live for the duration of the caller's render pass.
    pub fn get_or_create(
        &mut self,
        key: u64,
        create: impl FnOnce() -> wgpu::BindGroup,
    ) -> Arc<wgpu::BindGroup> {
        self.map
            .entry(key)
            .or_insert_with(|| Arc::new(create()))
            .clone()
    }
}

/// Address of `r` cast to `*const ()`. Stable while the parent struct
/// isn't moved/dropped.
#[inline]
pub fn ptr_addr<T: ?Sized>(r: &T) -> *const () {
    r as *const T as *const ()
}

/// Hash a list of resource pointers into a u64 cache key. Uses
/// `DefaultHasher` (SipHash) — collisions on addresses are
/// astronomically unlikely.
pub fn key_from_ptrs(ptrs: &[*const ()]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in ptrs {
        (*p as usize).hash(&mut h);
    }
    h.finish()
}
