# Protomorph

A real-time 3D rendering engine written in Rust, targeting Halo 3-era graphical fidelity using modern GPU APIs. Built on **wgpu** and inspired by Bungie's rendering techniques from SIGGRAPH 2008, Protomorph implements a deferred rendering pipeline with physically-based shading, cascaded shadow mapping, skeletal animation, and a full HDR post-processing stack.

## Features

- **Deferred Rendering** with a 4-target G-buffer (position/depth, normals, albedo/specular, material properties)
- **Cook-Torrance BRDF** matching Halo 3's microfacet shading model (GGX NDF, Schlick Fresnel, Smith geometry)
- **Cascaded Shadow Mapping** with 3 cascades, PCSS soft shadows, and cascade blending
- **Point & Spot Light Shadows** via cubemap and 2D depth maps with rotated Poisson disk PCF
- **Environment Probes** with runtime cubemap capture, L2 spherical harmonics for diffuse IBL, and roughness-based mip selection for specular IBL
- **Screen-Space Ambient Occlusion** with 32-sample hemisphere kernel and bilateral depth-aware blur
- **HDR Pipeline** with Rgba16Float intermediates, ACES Narkowicz tone mapping, and configurable exposure
- **Bloom** using Jimenez 13-tap downsample, tent-filter upsample, and soft-knee brightness threshold
- **God Rays** via screen-space radial blur with exponential decay
- **FXAA 3.11** quality anti-aliasing
- **Atmospheric Scattering** with Rayleigh and Mie models, height-based fog, and Beer's law extinction
- **Skeletal Animation** with keyframe interpolation (linear position/scale, SLERP rotation), multi-animation blending, and 4-bone-per-vertex skinning
- **FBX Model Loading** via asset-importer with automatic tangent generation and full node hierarchy
- **DDS Texture Loading** supporting BC1, BC2, and BC3 compressed formats with full mipchain support
- **PBR Material Pipeline** with automatic Blinn-Phong to roughness/F0 conversion
- **Film Grain** and saturation controls in the final composite
- **First-Person Viewmodel** with animation blending (idle/moving/action)
- **FPS Counter** HUD overlay via glyphon text rendering

## Building & Running

```bash
cargo run
```

### Controls

| Input | Action |
|-------|--------|
| WASD | Move |
| R / F | Up / Down |
| Shift | Sprint |
| Mouse | Look (click to capture) |
| Esc | Release cursor |
| H | Toggle flashlight |
| 1 / 2 / 3 | Weapon animations |

## Architecture

Protomorph uses a pass-based deferred rendering architecture. Each frame executes the following pipeline:

```
Shadow Pass          Depth-only rendering to shadow maps
  Point lights         6-face cubemap per light (1024x1024)
  Spot lights          2D depth map per light (2048x2048)
  Directional (CSM)    3-cascade texture array (2048x2048)
        |
Env Probe Pass       Forward-render scene to 128x128 cubemap,
                     compute L2 SH coefficients, generate mips
        |
Geometry Pass        Fill 4-target G-buffer + depth (rigid & skinned)
        |
SSAO Pass            Half-res ambient occlusion (32 samples)
        |
SSAO Blur Pass       Bilateral depth-aware blur
        |
Lighting Pass        Deferred Cook-Torrance shading, shadow lookups,
                     IBL from env probe, SSAO modulation
        |
God Rays Pass        Half-res radial blur from sun screen position
        |
Bloom Pass           Prefilter -> 5-level downsample -> upsample ->
                     ACES tone map + composite + film grain
        |
FXAA Pass            Edge-detect + directional blur
        |
Text Pass            HUD overlay
        |
Present
```

### Project Structure

```
src/
  main.rs               Entry point, window, input handling
  game.rs               Scene setup, camera, animation, update loop
  camera.rs             First-person camera with pitch/yaw
  objects.rs            Object transforms and sparse storage
  lights.rs             Directional, point, and spot lights (max 16)
  materials.rs          Material loading with PBR/Blinn-Phong conversion
  model.rs              FBX loading, node hierarchy, vertex formats
  animation.rs          Skeletal animation, keyframe interpolation, blending
  gpu_types.rs          GPU uniform structs (camera, shadow, SSAO, bloom, etc.)
  dds.rs                BC1/BC2/BC3 compressed texture loading
  renderer/
    mod.rs              Renderer orchestrator, frame submission
    shared.rs           G-buffer, samplers, bind group layouts, fallback textures
    helpers.rs          Texture/buffer creation utilities
    geometry_pass.rs    G-buffer fill (rigid + skinned pipelines)
    shadow_pass.rs      CSM, cubemap, and spot shadow rendering
    ssao_pass.rs        Screen-space ambient occlusion
    ssao_blur_pass.rs   Bilateral blur for SSAO
    lighting_pass.rs    Deferred shading with PBR + shadows + IBL
    env_probe_pass.rs   Cubemap capture, SH computation, mip generation
    bloom_pass.rs       HDR prefilter, mip pyramid, tone map composite
    god_rays_pass.rs    Volumetric light shafts
    fxaa_pass.rs        Fast approximate anti-aliasing
    text_pass.rs        HUD text rendering

assets/
  shaders/              WGSL shaders for each render pass
  models/               FBX models (ground plane, characters, weapons)
  textures/             DDS textures (diffuse, normal, specular, emissive)
```

## Halo 3 Technique Comparison

Protomorph targets the rendering techniques documented in Bungie's **"Lighting and Material of Halo 3"** (SIGGRAPH 2008) and related GDC talks. Below is a breakdown of Halo 3's graphical features and what Protomorph currently implements.

### Lighting & Shading

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Cook-Torrance BRDF | Beckmann NDF, full Fresnel, CT geometry term | GGX NDF, Schlick Fresnel, Smith geometry | Implemented with modern NDF variant |
| Frequency-decomposed specular | 4 levels: diffuse SH, area specular, env map, analytical | Analytical + env map + SH diffuse | Area specular (SH-convolved BRDF) not yet implemented |
| Multiple material models | cook_torrance, two_lobe_phong, glass, organism, foliage | Cook-Torrance only | Single material model currently |
| SH lightmaps (baked GI) | L2 SH per lightmap texel via photon mapping | Runtime SH from env probe only | No offline lightmap baking pipeline |
| Light probes / PRT | Baked PRT with self-shadowing transfer vectors | Runtime env probe with SH | PRT not implemented |
| Rim Fresnel | Configurable rim term with color/power/blend | Not yet implemented | -- |
| Anti-shadow control | Specular clamping near shadow terminator | Not yet implemented | -- |
| Analytical dominant light from SH | Extracts brightest direction from SH for sharp specular | Not yet implemented | -- |

### Shadows

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Baked shadows in lightmaps | Static shadow occlusion in SH lightmaps | Not applicable | No lightmap pipeline |
| Shadow maps (directional) | Shadow mapping for dynamic objects | 3-cascade CSM with PCSS | Exceeds Halo 3 with cascade blending + penumbra estimation |
| Shadow maps (point) | Cubemap shadow maps + PCF | Cubemap shadows + rotated Poisson PCF | Equivalent or better |
| Shadow maps (spot) | 2D shadow maps + PCF | 2D shadows + rotated Poisson PCF | Equivalent or better |
| PRT self-shadowing | Precomputed self-occlusion per object region | Not implemented | -- |

### Post-Processing

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| HDR rendering | Dual RGBA8 framebuffers ("HDR the Bungie Way") | Single Rgba16Float pipeline + ACES tone mapping | Modern approach with superior dynamic range |
| Bloom | Mip-chain blur + additive blend | 13-tap downsample + tent upsample + soft-knee threshold | Modern physically-motivated bloom |
| Motion blur | Per-pixel velocity blur (corners of screen) | Not implemented | -- |
| Depth of field | Cutscene-only post-process blur | Not implemented | -- |
| Film grain | Not present | Hash-based noise in final composite | Protomorph addition |

### Anti-Aliasing

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| MSAA / AA | None (resolution sacrificed for HDR) | FXAA 3.11 Quality | Protomorph exceeds Halo 3 |

### Atmospheric & Volumetric Effects

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Rayleigh scattering | Full implementation for sky color + aerial perspective | Implemented with configurable coefficients | Equivalent |
| Mie scattering | Haze, sun halos, volumetric fog | Implemented with Henyey-Greenstein phase function | Equivalent |
| Height-based fog | Exponential density falloff with reference height | Implemented with max thickness + height scale | Equivalent |
| God rays / light shafts | Atmospheric + bloom + shadow interaction | Screen-space radial blur (GPU Gems 3 approach) | Implemented, different technique |
| Volumetric particles | Particle-based fog, smoke, explosion clouds | Not implemented | -- |

### Screen-Space Effects

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| SSAO | Not present (used baked AO in lightmaps) | 32-sample hemisphere kernel + bilateral blur | Protomorph exceeds Halo 3 |
| Screen-space reflections | Not present | Not implemented | Neither engine has SSR |

### Geometry & Materials

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Normal mapping | Standard + detail bump (dual-layer) | Standard normal mapping | Detail bump layer not yet implemented |
| Parallax / height mapping | Supported as shader option | Not implemented | -- |
| Self-illumination | 6 modes (simple, plasma, 3-channel, from_diffuse, detail) | Emissive map + HDR intensity | Simple mode only |
| Environment cubemaps | Per-cluster dynamic cubemaps + SH modulation | Single runtime env probe + SH | Equivalent core, no per-cluster placement |
| Specular masking | Per-pixel specular maps with tint + albedo blend | Specular texture + roughness + F0 | Equivalent |
| DDS / compressed textures | DXT1/3/5 on Xbox 360 | BC1/BC2/BC3 (DXT1/3/5 equivalent) | Equivalent |

### Animation

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Skeletal animation | FK/IK rigs, ~10,000 gameplay anims | FK with keyframe interpolation + SLERP | Core system implemented |
| Multi-animation blending | Layered animation blending | Weight-based blend with fade in/out | Implemented |
| Facial animation | 50-bone face rig + procedural speech | Not implemented | -- |
| Physics-driven animation | Havok ragdoll + rigid body weapons | Not implemented | -- |

### Effects & Misc

| Technique | Halo 3 | Protomorph | Notes |
|-----------|--------|------------|-------|
| Particle system | Custom engine (fire, plasma, smoke, snow, etc.) | Not implemented | -- |
| Water rendering | Dynamic tessellation + wave sim + refraction | Not implemented | -- |
| Decals | Free-floating surface textures | Not implemented | -- |
| Foliage animation | Multi-group wind-driven branch animation | Not implemented | -- |

### Summary

Protomorph implements the core of Halo 3's rendering philosophy — Cook-Torrance shading, environment probes with spherical harmonics, shadow mapping, atmospheric scattering, and HDR bloom — while substituting modern techniques where appropriate (GGX over Beckmann, ACES tone mapping over dual-buffer HDR, FXAA and SSAO which Halo 3 lacked entirely). The main gaps are in content pipeline features (baked SH lightmaps, PRT), specialized material models (glass, organism, two-lobe phong), and gameplay-facing systems (particles, water, decals, physics, motion blur).

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| wgpu | 28.0 | GPU abstraction (Vulkan/Metal/DX12) |
| winit | 0.30 | Window creation and event handling |
| glam | 0.32 | Vector/matrix math |
| asset-importer | 0.7 | FBX model loading |
| bytemuck | 1.24 | Safe GPU struct casting |
| glyphon | 0.10 | Text rendering |
| pollster | 0.4 | Async runtime |
| rand | 0.9 | Random number generation (SSAO kernel) |
| bitflags | 2 | Bitflag types |

## References

- [Lighting and Material of Halo 3](https://advances.realtimerendering.com/s2008/SIGGRAPH-Lighting%20of%20Halo%203.pdf) — Chen & Liu, SIGGRAPH 2008
- [HDR the Bungie Way](https://halo.bungie.org/news.html?item=16666) — Chris Tchou, Gamefest 2006
- [Lightmap Compression in Halo 3](https://archive.org/details/GDC2008Hu2) — Yaohua Hu, GDC 2008
- [Halo 3 Shader Documentation](https://learn.microsoft.com/en-us/halo-master-chief-collection/h3/shaders/shadershome) — Microsoft Learn
