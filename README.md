# Protomorph

A real-time 3D rendering engine written in Rust, targeting Halo 3-era graphical fidelity using modern GPU APIs. Built on **wgpu** and inspired by Bungie's rendering techniques from SIGGRAPH 2008, Protomorph implements a deferred rendering pipeline with physically-based shading, volumetric clouds, water simulation, cascaded shadow mapping, skeletal animation, and a full HDR post-processing stack.

## Building & Running

```bash
cargo run
```

Requires a GPU with BC texture compression and `Rg11b10Ufloat` render target support (all modern desktop GPUs).

## Controls

### Movement

| Input | Action |
|-------|--------|
| WASD | Move |
| Space | Jump |
| Shift | Sprint (2x speed) |
| Mouse | Look (click to capture) |
| Esc | Release cursor |

### Toggles

| Key | Action | Default |
|-----|--------|---------|
| Tab | Camera mode (Player / Flycam) | Player |
| H | Flashlight | OFF |
| G | Pause grunt animation | Playing |
| T | Detach weapon viewmodel | Attached |
| K | Specular occlusion | ON |
| V | Vignette | ON |
| P | Debug cubemap | OFF |
| O | Cubemap face colors | OFF |

### Weapon Animations

| Key | Animation |
|-----|-----------|
| 1 | Ready |
| 2 | Reload |
| 3 | Melee |

### Camera Modes

- **Player mode**: Grounded movement with gravity, jumping, wall collision, and wall sliding. Movement is projected onto the horizontal plane regardless of camera pitch. Ground height is determined by triangle mesh raycasting against level geometry. Walls use circle-vs-segment collision with height-aware filtering.
- **Flycam mode**: Free-flight noclip camera. WASD follows the full 3D look direction, R/F move vertically. No collision or gravity.

## Rendering Pipeline

Protomorph uses a pass-based deferred rendering architecture. Each frame executes 11 passes in order:

```
Shadow Pass             Depth-only rendering into a 4096x4096 shadow atlas
                        CSM (3x 2048x2048) + spot (4x 1024x1024) + point cubemaps
                            |
Environment Probe       Forward-render scene to 128x128 cubemap (every 8 frames)
                        Compute L2 spherical harmonics, generate roughness mips
                            |
Geometry Pass           Fill 4-target G-buffer + reverse-Z depth
                        32-byte compressed vertices, 4-bone LBS
                            |
Cloud Raymarch          Quarter-res volumetric cloud raymarching (compute)
                        FBM noise, multi-scatter lighting, aerial perspective
                            |
SSAO Pass               Screen-space ambient occlusion (16 samples)
                        Hemisphere kernel with bent normal output
                            |
God Rays Pass           Screen-space radial blur from sun position
                        Cloud transmittance sampling, exponential decay
                            |
Deferred Lighting       Cook-Torrance PBR with Beckmann NDF + exact Fresnel
                        Shadow PCF, IBL, SH diffuse, atmospheric scattering
                            |
Cloud Composite         Blend volumetric clouds into the lit scene
                            |
Water Pass              Forward-rendered Gerstner wave simulation
                        Refraction, caustics, SSS, environment reflection
                            |
Bloom Pass              4-level mip pyramid downsample/upsample
                        Soft-knee threshold, Kawase-style blur
                            |
Final Composite         FXAA, ACES tone mapping, analytical sun disc
                        Vignette, film grain, saturation, HUD text
                            |
                        Present
```

## Rendering Techniques

### Deferred Shading

The G-buffer stores surface attributes in 4 render targets plus depth:

| Target | Format | Contents |
|--------|--------|----------|
| 0 | `Rg16Float` | Octahedral-encoded world normal |
| 1 | `Rgba8UnormSrgb` | RGB albedo, A baked specular amount |
| 2 | `Rgba8Unorm` | R ambient, G metallic, B roughness, A fresnel F0 |
| 3 | `Rg11b10Ufloat` | HDR emissive (clamped to 500.0) |
| Depth | `Depth32Float` | Reverse-Z infinite far plane |

Vertices are compressed to 32 bytes: float32 position, oct-encoded snorm8 normal, float16 texcoords, snorm8 tangent with bitangent sign, and 4-bone indices/weights packed as uint8/unorm8.

### Lighting

Cook-Torrance microfacet BRDF matching Halo 3's shading model:
- **Normal Distribution**: Beckmann
- **Fresnel**: Exact Fresnel equations (not Schlick)
- **Geometry**: Cook-Torrance geometry term
- **Frequency-decomposed specular**: Four levels (SH diffuse, area specular, environment map, analytical) with separate intensity controls for metals vs. dielectrics
- **Rim lighting**: Fresnel-based rim with albedo blend, modulated by SH irradiance
- **Anti-shadow control**: Prevents over-darkening near shadow terminators
- **Dominant light extraction**: Luminance-weighted L1 extraction from SH for sharp specular highlights

Up to 16 simultaneous lights (directional, point, spot) with per-light atmospheric extinction.

### Shadows

Single 4096x4096 depth atlas with tiled layout:
- **Cascaded Shadow Maps**: 3 cascades at 2048x2048 with PCF via `textureGatherCompare`
- **Spot lights**: Up to 4 at 1024x1024 with bilinear PCF
- **Point lights**: Up to 4 cubemaps at 1024x1024 with 4-tap Poisson PCF
- Shadow pancaking to prevent near-plane artifacts

### Environment Probes & IBL

128x128 cubemap captured at runtime (refreshed every 8 frames) with 6 mip levels for roughness-based specular IBL. L2 spherical harmonics (9 basis functions, RGB) computed from the cubemap for diffuse irradiance. Specular occlusion uses a roughness-aware AO attenuation formula.

### Atmospheric Scattering

Full Rayleigh + Mie scattering model with:
- Configurable scattering coefficients
- Height-based density falloff
- Per-light extinction filtering
- Inscatter scaling with sun air mass
- Henyey-Greenstein phase function for Mie

The procedural sky renders a sun disc with limb darkening, inner glow, and soft edge, composited analytically in the final pass with a wider halo and depth masking.

### Screen-Space Ambient Occlusion

16-sample hemisphere kernel with bent normal output, stored as `Rgba8Unorm`. Uses a 4x4 Poisson noise texture scaled to screen resolution. Configurable radius (0.1), strength (0.025), and falloff threshold.

### Volumetric Clouds

IQ-style FBM raymarching at quarter resolution:
- **Noise**: 128x128x128 Perlin-Worley base shape + 32x32x32 Worley detail erosion + 2D weather map
- **Raymarching**: Up to 48 samples with distance-dependent step size and blue noise jitter
- **Lighting**: Cornette-Shanks phase function with 3-octave multi-scattering approximation (each bounce halves extinction and transitions toward isotropic). Beer-Powder transmittance model with height-dependent powder factor
- **Atmosphere**: Day/sunset ambient colors, distance-based aerial perspective fade

### God Rays

Screen-space radial blur from the sun's projected position with exponential illumination decay. Samples the cloud transmittance buffer for volumetric shadowing. Depth-masked so only sky pixels contribute.

### Water

Forward-rendered after deferred lighting:
- **Waves**: 4 Gerstner waves with per-vertex displacement and per-pixel normal evaluation
- **Normal mapping**: Reoriented Normal Mapping (RNM) blend of two scrolling bump maps over the Gerstner base normal
- **Refraction**: Chromatic aberration with per-channel offset and depth-validated sampling
- **Reflection**: Environment cubemap with Fresnel blending (F0 = 0.02)
- **Absorption**: Per-channel Beer's Law (RGB coefficients 3.0, 0.6, 0.3)
- **Caustics**: Procedural pattern projected onto the floor with depth fade and shadow masking
- **Subsurface scattering**: Distortion-based SSS approximation modulated by shadow
- **Specular**: GGX NDF with geometric specular anti-aliasing (Tokuyoshi-Kaplanyan 2019) and hash-based sparkle noise

### Bloom

4-level mip pyramid with soft-knee brightness threshold:
- Downsample: 4-tap Kawase-style blur at each level
- Upsample: 4-tap bilinear with additive blending
- All bloom textures use `Rg11b10Ufloat`

### Final Composite

- **FXAA**: Fast approximate anti-aliasing
- **Tone mapping**: ACES Narkowicz approximation
- **Sun disc**: Post-tonemap analytical disc with core, glow, and halo layers, depth-masked to sky
- **Saturation**: Luma-based color saturation control
- **Vignette**: Radial edge darkening
- **Film grain**: Hash-based procedural noise with luma-dependent intensity
- **HUD**: FPS counter and controls overlay via glyphon text rendering
- **Debug**: Optional cubemap face visualization with labels

### Skeletal Animation

- 4-bone-per-vertex Linear Blend Skinning
- Keyframe interpolation: linear position/scale, SLERP rotation
- Multi-animation blending with per-track weights and fade in/out
- First-person viewmodel with idle/moving/action animation state machine

### Collision Detection

Triangle mesh collision extracted from level geometry at load time:
- **Ground**: Moller-Trumbore ray-triangle intersection casting downward, supporting multiple floor heights
- **Walls**: Circle-vs-segment collision in the horizontal plane with height-aware filtering, iterative push-out for corner handling, and wall sliding
- **Physics**: Gravity, jumping, and ground snapping with configurable eye height, jump velocity, and player radius
