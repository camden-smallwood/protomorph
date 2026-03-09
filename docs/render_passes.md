# RenderPass Trait — Implementation Reference

## Status: COMPLETE

All 14 passes implement the `RenderPass` trait. The `Renderer` struct holds
`passes: Vec<Box<dyn RenderPass>>` and orchestrates via three loops (prepare,
record, post_submit).

---

## Core Types (in `src/renderer/shared.rs`)

### SharedBindGroup

```rust
pub type SharedBindGroup = Rc<RefCell<wgpu::BindGroup>>;
```

- Producer passes (shadow, env probe) wrap their bind group in `SharedBindGroup`.
- Consumer passes (lighting) clone the `Rc` at construction time.
- On resize, producers swap the inner value; consumers see updates automatically.

### FrameContext

```rust
pub struct FrameContext<'a> {
    pub shared: &'a SharedResources,
    pub gbuffer: &'a GBuffer,
    pub intermediates: &'a IntermediateTargets,
    pub surface_view: &'a wgpu::TextureView,
    pub models: &'a [GpuModel],
    pub render_list: &'a [(ObjectIndex, usize)],
    pub game: &'a GameState,
}
```

No pre-computed convenience fields — each pass extracts what it needs from `game`.

### RenderPass Trait

```rust
pub trait RenderPass {
    fn name(&self) -> &str;
    fn prepare(&mut self, _ctx: &FrameContext) {}
    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext);
    fn resize(&mut self, _shared: &SharedResources, _gbuffer: &GBuffer, _intermediates: &IntermediateTargets) {}
    fn post_submit(&mut self) {}
    fn is_enabled(&self, _ctx: &FrameContext) -> bool { true }
}
```

`prepare` is `&mut self`, `record` is `&self`.

---

## Renderer Struct

```rust
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    shared: SharedResources,
    gbuffer: GBuffer,
    intermediates: IntermediateTargets,
    passes: Vec<Box<dyn RenderPass>>,
    models: Vec<GpuModel>,
    render_list: Vec<(ObjectIndex, usize)>,
    node_matrix_staging: Vec<u8>,
}
```

---

## Pass Order (vec index)

| # | Pass | File | Notes |
|---|------|------|-------|
| 0 | `ShadowPass` | `shadow_pass.rs` | Produces `SharedBindGroup`. Owns caster lists + shadow assignments. Uploads lighting uniforms in `prepare()` |
| 1 | `EnvProbePass` | `env_probe_pass.rs` | Produces `SharedBindGroup`. Extracts sun params from game in `prepare()`. Skips cubemap re-render via `should_update()` |
| 2 | `DepthPrepass` | `geometry_pass.rs` | Owns its own rigid + skinned pipelines |
| 3 | `GBufferPass` | `geometry_pass.rs` | Owns its own rigid + skinned pipelines |
| 4 | `SsaoPass` | `ssao_pass.rs` | Half-resolution. Implements `resize()` |
| 5 | `SsaoBlurPass` | `ssao_blur_pass.rs` | Bilateral blur. Implements `resize()` |
| 6 | `LightingPass` | `lighting_pass.rs` | Consumes shadow + env probe `SharedBindGroup` via `.borrow()`. Implements `resize()` |
| 7 | `EmissiveForwardPass` | `geometry_pass.rs` | Additive blend onto lighting buffer. Owns its own pipelines |
| 8 | `GodRaysTracePass` | `god_rays_pass.rs` | Computes sun screen position in `prepare()`. Implements `resize()` |
| 9 | `GodRaysCompositePass` | `god_rays_pass.rs` | Additive blend onto lighting buffer. Implements `resize()` |
| 10 | `BloomPass` | `bloom_pass.rs` | 4-phase internally. Implements `resize()` |
| 11 | `FxaaPass` | `fxaa_pass.rs` | Implements `resize()` |
| 12 | `CubemapDebugPass` | `cubemap_debug_pass.rs` | `is_enabled` checks `ctx.game.debug_cubemap` |
| 13 | `TextPass` | `text_pass.rs` | `prepare()` builds glyphs, `post_submit()` trims atlas |

---

## Cross-Pass Dependencies

Resolved at construction time via `Rc::clone`:

- **ShadowPass** `bind_group` + `bgl` -> LightingPass
- **EnvProbePass** `bind_group` + `bgl` -> LightingPass
- **EnvProbePass** `face_view()` + `filtering_sampler()` -> CubemapDebugPass

---

## Render Loop

```rust
fn render(&mut self, game: &GameState) {
    // 1. Build sorted render_list
    // 2. Upload camera, model, bone matrix, atmosphere, sky params uniforms
    // 3. Acquire surface texture
    // 4. Build FrameContext

    for pass in &mut self.passes {
        if pass.is_enabled(&ctx) { pass.prepare(&ctx); }
    }

    let mut encoder = ...;
    for pass in self.passes.iter() {
        if pass.is_enabled(&ctx) { pass.record(&mut encoder, &ctx); }
    }

    queue.submit(...);
    output.present();

    for pass in &mut self.passes {
        pass.post_submit();
    }
}
```

Shared uniform uploads (camera, model, bone matrices, atmosphere, sky params)
happen before the pass loops because multiple passes read from those buffers.

---

## Resize Flow

```rust
fn resize(&mut self, width: u32, height: u32) {
    self.gbuffer = create_gbuffer(...);
    self.intermediates = create_intermediates(...);
    for pass in &mut self.passes {
        pass.resize(&self.shared, &self.gbuffer, &self.intermediates);
    }
}
```

Passes that don't depend on resolution use the default no-op impl.

---

## Implementation Notes

- **No Arc sharing**: The plan originally called for `Arc<GeometryPipelines>` and
  `Arc<GodRaysState>` to share pipelines between split passes. In practice, each
  split pass owns its own pipelines directly — simpler and no measurable overhead
  since pipeline objects are lightweight handles.

- **Shadow pass internalizes state**: `point_shadow_casters`, `spot_shadow_casters`,
  `shadow_assignments`, and `has_directional_shadow` are all fields on `ShadowPass`.
  They get `clear()`ed and reused each frame in `prepare()`. The lighting uniform
  upload (`GpuLightingUniforms::from_scene`) also lives in shadow's `prepare()`
  because it depends on `shadow_assignments`.

- **Env probe prepare/record split**: Uniform uploads (probe data, SH coefficients,
  face cameras, sky params) happen in `prepare()`. GPU command recording (sky
  background, geometry, mip downsample) happens in `record()`. The `should_update()`
  check gates both the camera uploads in prepare and all GPU work in record.

- **`&*` deref pattern**: When a pass holds a `SharedBindGroup` and needs to call
  `rpass.set_bind_group()`, it does `let bg = self.bind_group.borrow();` then
  `rpass.set_bind_group(n, &*bg, &[]);`. The `&*` derefs through `Ref<T>`'s
  `Deref` impl to get `&BindGroup`.
