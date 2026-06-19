//! Mirror of `Ares/source/rasterizer/dx11/`. Per-state caches that
//! map (BlendStateCache, DepthStencilStateCache, RasterizerStateCache,
//! InputLayoutCache) to D3D11 state objects in Halo. Our wgpu port
//! caches `wgpu::RenderPipeline` builders' state in the same shape.
//!
//! In Halo's PC build these are LRU caches keyed by the
//! mode-enum/desc; we mirror that exact shape.
//!
//! ## TODO — rename to `wgpu/`
//!
//! The `dx11` directory + the `dx11`-prefixed names match Halo's PC
//! build (which uses D3D11). Once the renderer rebuild is complete
//! and stable, rename this module to `wgpu/` and drop any remaining
//! D3D-specific terminology in identifiers. The internal mapping is
//! already wgpu — only the path/naming needs to change.

pub mod blend_state_cache;
pub mod depth_stencil_state_cache;
pub mod input_layout_cache;
pub mod rasterizer_state_cache;
pub mod sampler_state_cache;
pub mod shader_constants;

pub use blend_state_cache::BlendStateCache;
pub use depth_stencil_state_cache::DepthStencilStateCache;
pub use input_layout_cache::InputLayoutCache;
pub use rasterizer_state_cache::RasterizerStateCache;
pub use sampler_state_cache::{SamplerKey, SamplerStateCache};
pub use shader_constants::CbufferPool;
