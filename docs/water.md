# Water Rendering Research for Protomorph

## What Halo 3 Actually Did

Halo 3's water was surprisingly straightforward for its era:

- **Geometry:** A flat mesh tessellated dynamically based on camera distance (dense nearby, sparse far away). Vertex heights driven by a "wave texture" updated each frame -- essentially a heightmap animation, not Gerstner waves.
- **Normal maps:** Layered scrolling normal maps for surface detail on top of the geometric displacement.
- **Lighting:** Full lightmap + dynamic light contribution. Large "rough specular" reflections from light sources, visible from both above and below the surface. No shortcuts -- water was lit like any other surface.
- **Refraction:** Screen-space distortion of underwater geometry using the normal map to offset UV reads. Improved over Halo 2 to exclude above-water objects from the refracted image.
- **Reflections:** Cubemap-based environment reflections (no planar camera or SSR -- Xbox 360 couldn't afford it).
- **Interaction:** Dynamic splashes injected into the wave heightmap texture. Objects float with buoyancy forces, carried by flow direction.

**The Halo 3 "look"** comes from: rich specular that sparkles across the surface, visible light scatter from below, strong Fresnel (reflective at grazing, transparent up close), and depth-based color absorption giving that teal/blue gradient.

---

## Recommended Approach for Protomorph

Given your deferred renderer architecture (depth prepass, G-buffer, lighting, emissive forward, god rays, bloom, FXAA), here's what I'd recommend:

### Architecture: Forward-Rendered Water Pass

Water is transparent and needs the lit scene behind it for refraction. It **must** be forward-rendered after the lighting pass. This matches how your emissive forward pass already works. Insertion point:

```
Shadow -> Env Probe -> Depth Prepass -> G-buffer -> SSAO -> Lighting
-> Emissive Forward -> **WATER PASS** -> God Rays -> Bloom -> FXAA
```

Water reads from: depth buffer (scene), lighting output (for refraction), env cubemap (for reflection fallback). Water writes to: the lighting buffer (Rg11b10Ufloat) + depth buffer.

### Component Breakdown

#### 1. Water Mesh -- Gerstner Wave Vertex Displacement

A flat grid (e.g. 64x64 tiles) displaced in the vertex shader using a sum of 4-8 Gerstner waves. This is the GPU Gems Chapter 1 approach and gives you the characteristic sharp crests + flat troughs that look natural:

```wgsl
// Per wave: direction, steepness (Q), wavelength, amplitude, speed
// Vertex displacement (simplified):
for each wave:
    phase = speed * time
    dot_d = dot(direction, vertex.xz)
    x += Q * amplitude * direction.x * cos(dot_d * frequency + phase)
    z += Q * amplitude * direction.z * cos(dot_d * frequency + phase)
    y += amplitude * sin(dot_d * frequency + phase)
```

Normals and tangents are computed analytically from the Gerstner derivative (no finite differences needed). Edge dampening clamps displacement near mesh boundaries to prevent cracks.

**Why Gerstner over heightmap animation (Halo 3 style):** Gerstner is cheaper (no texture update), runs entirely in the vertex shader, gives you analytic normals for free, and produces more convincing motion. Halo 3 used heightmaps because of Xbox 360 vertex texture fetch limitations.

#### 2. Normal Map Detail Layers

Two scrolling normal maps at different scales/speeds, blended in tangent space:

```wgsl
let n1 = textureSample(normal_map, sampler, uv * scale1 + time * scroll1);
let n2 = textureSample(normal_map, sampler, uv * scale2 + time * scroll2);
let detail_normal = normalize(n1.xyz + n2.xyz);  // average + normalize
```

The Gerstner-derived geometric normal provides large-scale shape, the detail normals add fine ripples. This dual-frequency approach is exactly what gives Halo 3 its visual richness.

#### 3. Reflections -- SSPR (Screen Space Planar Reflections)

This is the big win. Your unified plan already mentions SSPR (section 3.12). For water specifically, SSPR is dramatically better than ray-marched SSR:

**Why SSPR over SSR for water:**
- 5-10x cheaper (0.3-0.4ms vs 2-3ms)
- No ray marching -- pure geometric projection
- Perfect for flat/near-flat surfaces (water)
- No step artifacts, no thickness heuristics

**Algorithm (Ghost Recon Wildlands approach):**
1. **Compute pass -- Projection:** For each pixel below water plane, reflect world position across the water plane (`reflected.y = 2 * water_height - pos.y`), project back to screen space, write source pixel coords to a hash buffer using `atomicMax` (Y in high 16 bits for depth sorting)
2. **Fragment pass -- Resolve:** Fullscreen quad reads hash buffer, decodes source pixel location, samples the lighting buffer at that location
3. **Gap filling:** Temporal reprojection from previous frame fills holes at screen edges

**Hash buffer format:** `R32Uint`, encoded as `(src_y << 16) | src_x`. The `atomicMax` ensures pixels closest to the water plane (highest Y in reflected space = closest to camera in reflection) win conflicts.

**Fallback:** Where SSPR has no data (sky, off-screen), fall back to your existing env cubemap sampling. Blend based on SSPR confidence (0 where hash is empty).

#### 4. Refraction

Sample the pre-water lighting buffer with UV offset based on the water normal:

```wgsl
let distorted_uv = screen_uv + normal.xz * refraction_strength;
// Validate: only accept if distorted sample is actually underwater
let distorted_depth = textureSample(depth_buffer, sampler, distorted_uv);
if (distorted_depth is above water) { distorted_uv = screen_uv; }  // reject
let refraction_color = textureSample(lighting_buffer, sampler, distorted_uv);
```

#### 5. Depth-Based Absorption (The "Halo 3 Look")

This is the single most important visual element. Compare water surface depth to scene depth behind it:

```wgsl
let underwater_distance = water_depth - scene_depth;  // linear eye-space
let absorption = saturate(underwater_distance / absorption_depth);
let water_color = mix(refraction_color, deep_water_color, absorption);
```

Use a teal/blue-green `deep_water_color` for that Halo look. The absorption distance controls how quickly shallow water becomes opaque. ~2-5 meters gives a good range.

#### 6. Fresnel + Final Composite

Schlick Fresnel blends reflection and refraction:

```wgsl
let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
let color = mix(water_color, reflection_color, fresnel);
```

With `f0 = 0.02` (water's IOR). This gives you transparent water looking down, reflective water at grazing angles -- the core of the Halo 3 aesthetic.

#### 7. Shore Edge Blend + Foam

**Soft edge:** Compare water surface depth to scene depth. Where they're close (shore intersection), fade alpha:

```wgsl
let edge_factor = saturate(depth_difference / edge_softness);
```

**Foam:** Sample a foam texture, activate based on:
- Shore proximity: `pow(1.0 - edge_factor, 3.0)` -- concentrated at geometry contact
- Wave height: foam appears on wave crests above a threshold
- Surface angle: `pow(dot(normal, UP), 80.0)` -- only on upward-facing surfaces

#### 8. Specular Sparkle

This is what makes Halo 3 water pop. Sample 3 noise textures at different UV scales and multiply them together to break up the specular highlight into scattered sparkles:

```wgsl
let sparkle = noise(uv * s1) * noise(uv * s2) * noise(uv * s3);
specular *= sparkle;
```

Use your existing Schlick + Beckmann (or GGX) with low roughness (~0.08) and high intensity multiplier (~50-125x). The sparkle mask prevents the uniform "plastic sheet" look.

---

## Reflection Method Comparison

| Approach | Cost | Quality for Water | Effort | Notes |
|----------|------|-------------------|--------|-------|
| **SSPR (recommended)** | 0.3-0.4ms | Excellent for flat water | Medium | Perfect match for water planes |
| **Planar camera (mirror render)** | 5-15ms | Perfect | High | Re-renders entire scene. Too expensive. |
| **SSR ray march (existing plan)** | 2-3ms | Decent but artifacts | Medium | Thickness heuristics fail at water edges |
| **Cubemap only (Halo 3 style)** | ~0 | Low -- no scene reflections | Trivial | Your env probe already provides this |
| **SSPR + cubemap fallback** | 0.3-0.5ms | Best practical quality | Medium | Recommended combo |

---

## Implementation Priority

1. **Forward water pass + mesh + Gerstner displacement** -- Get geometry on screen
2. **Depth-based absorption + Fresnel + refraction** -- The core visual (this alone will look "Halo-like")
3. **Dual scrolling normal maps + specular sparkle** -- Surface detail and life
4. **SSPR reflections** -- Screen-space reflections without the SSR cost (reusable for floors later)
5. **Shore foam + edge blending** -- Polish
6. **Caustics** (optional future) -- Projected light patterns on underwater geometry

Steps 1-3 get you 80% of the Halo 3 look. SSPR pushes it beyond what Halo 3 could do.

---

## Key Architectural Notes for Protomorph

- **No G-buffer interaction:** Water is entirely forward-rendered. It reads the depth buffer and lighting buffer but doesn't write to the G-buffer.
- **Depth write:** Water MUST write to the depth buffer so god rays / bloom / FXAA respect it.
- **Copy lighting buffer before water:** Water needs to sample the scene color for refraction, but also writes to it. You need to copy the lighting output to a separate texture before the water pass (or use a second color target). Same pattern as your emissive pass reading from a copied buffer.
- **SSPR compute pass:** Runs before the water fragment pass. Needs a `R32Uint` storage texture the size of the screen (or quarter-res for perf).
- **Blend mode:** Alpha blend (`SrcAlpha, OneMinusSrcAlpha`) for shore edges. Most of the surface will be alpha=1.0 with Fresnel handling the reflection/refraction mix internally.

---

## Sources

- [Alex Tardif - Water Walkthrough](https://alextardif.com/Water.html)
- [SSPR in Ghost Recon Wildlands - Remi Genin](https://remi-genin.github.io/posts/screen-space-planar-reflections-in-ghost-recon-wildlands/)
- [The Graphics of Halo 3 - HaloTupolev](https://halotupolev.wordpress.com/2016/04/21/sixth-and-eighth-the-graphics-of-halo-3/4/)
- [GPU Gems Ch.1 - Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models)
- [Making Halo 3 Shine - CGW](https://www.cgw.com/Publications/CGW/2007/Volume-30-Issue-12-Dec-2007-/Making-Halo-3-Shine.aspx)
- [NVIDIA GPU Gems - Water Caustics](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)
- [Catlike Coding - Waves Tutorial](https://catlikecoding.com/unity/tutorials/flow/waves/)
- [80.lv - Ocean Shader with Gerstner Waves](https://80.lv/articles/tutorial-ocean-shader-with-gerstner-waves)
