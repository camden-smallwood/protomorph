//! `c_decal::write_mesh_fragment` — packs the per-fragment dedup
//! output into rasterizer-ready vertices and indices. Also contains
//! `c_decal::build_tangent_frame`, the per-vertex TBN packer.
//!
//! Anchored against dllcache:
//!   `c_decal::write_mesh_fragment   @ 0x18039D460`
//!   `c_decal::build_tangent_frame   @ 0x18039BD40`
//!
//! Pipeline (per output vertex):
//!   1. Look up the lead work vertex via `mesh_builder.grouper`
//!      (start index in `sorter_order`).
//!   2. Copy texcoord from lead work vertex.
//!   3. Copy/transform position from `bsp.vertices[lead.position]`.
//!      Instance-geom path applies the instance's local_to_world;
//!      main-BSP path is a direct copy.
//!   4. Sum normals + binormals across every work vertex that
//!      collapsed into this output (via the grouper run length);
//!      normalize the normal sum.
//!   5. Call `build_tangent_frame` to derive a tangent perpendicular
//!      to both normal and the summed binormal, orthogonalize the
//!      binormal, then pack all three as 4×i16 XMShortN4.
//!
//! Index pass: remap each `work_index_buffer[i]` through the
//! `mesh_builder.collapser` table.
//!
//! TBN is computed only when (definition.flags & 2) or the fragment's
//! floating-z-bias flag is set. The engine skips the TBN math
//! entirely otherwise — leaves the packed shorts uninitialized.
//! Our port zeros them in the skipped case for cleanliness.

use blam_tags::math::{RealPoint3d, RealVector3d};
use blam_tags::structure_bsp::Bsp3d;

use blam_tags::math::RealMatrix4x3;

use super::types::{DecalFragment, DecalMeshBuilder};

/// GPU-side decal vertex layout. Engine `rasterizer_vertex_world`
/// (44 B) — `effects/decals.cpp` writes 14 floats / 6 shorts per
/// entry: 3 position floats + 2 texcoord floats + 4×i16 normal +
/// 4×i16 tangent + 4×i16 binormal. Each TBN component is
/// XMShortN4-packed (signed normalized in [-1, 1]; engine SSE path
/// is `vmaxps NEG_ONE; vminps ONE; vmul SCALE(=32767); vcvtps2dq;
/// vpackssdw`).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RasterizerVertexWorld {
    pub position: [f32; 3],
    pub texcoord: [f32; 2],
    pub normal: [i16; 4],
    pub tangent: [i16; 4],
    pub binormal: [i16; 4],
}

const _: () = assert!(std::mem::size_of::<RasterizerVertexWorld>() == 44);

impl RasterizerVertexWorld {
    /// wgpu vertex layout for the rmd pipeline. Attribute locations
    /// match `entry_decal.wgsl::DecalVertex`. Snorm16x4 lanes (normal /
    /// tangent / binormal) decode the packed XMShortN4 i16 lanes back
    /// to f32 in `[-1, 1]` at attribute fetch.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBS: [wgpu::VertexAttribute; 5] = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 }, // position
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 }, // texcoord
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm16x4, offset: 20, shader_location: 2 }, // normal
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm16x4, offset: 28, shader_location: 3 }, // tangent
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm16x4, offset: 36, shader_location: 4 }, // binormal
        ];
        wgpu::VertexBufferLayout {
            array_stride: 44,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBS,
        }
    }
}

/// Engine constant — small offset along the (transformed) normal to
/// nudge floating decals off their host surface and avoid z-fighting.
/// The dllcache field `c_decal::x_floating_z_bias` is read at runtime;
/// we hardcode a small value here pending a global-init port. Only
/// applies when the decal definition has `flags & 2` (floating) or
/// the fragment is flagged `requires_floating_z_bias` (instance-geom).
pub(super) const X_FLOATING_Z_BIAS: f32 = 0.005;

/// Mirror of `c_decal::write_mesh_fragment @ 0x18039D460`.
///
/// `vertices` must have at least `fragment.output_intervals.vertex_count`
/// slots starting at offset `fragment.output_intervals.starting_vertex`.
/// `indices` must have at least `output_intervals.index_count` slots
/// starting at `output_intervals.starting_index`. Both buffers are
/// interpreted in their entirety so the caller can hand in fresh
/// slices of the correct length.
pub fn write_mesh_fragment(
    bsp: &Bsp3d,
    mesh_builder: &DecalMeshBuilder,
    fragment: &DecalFragment,
    instance_local_to_world: Option<&RealMatrix4x3>,
    definition_flags: u32,
    vertices: &mut [RasterizerVertexWorld],
    indices: &mut [u16],
) {
    let v_start = fragment.output_intervals.starting_vertex as usize;
    let v_count = fragment.output_intervals.vertex_count as usize;
    let i_start = fragment.output_intervals.starting_index as usize;
    let i_count = fragment.output_intervals.index_count as usize;

    // *** PROTOMORPH WORKAROUND — diverges from engine ***
    //
    // Engine `c_decal::write_mesh_fragment @ 0x18039D460` gates TBN
    // packing on:
    //   `(def_flags & 2) || x_apply_floating_z_bias_always
    //    || requires_floating_z_bias`
    // Static `c_decal::x_apply_floating_z_bias_always @ 0x1819fb380`
    // is zero in our dllcache, so for our riverworld corpus
    // (`def_flags = 0`, main-BSP) the engine SKIPS TBN packing and the
    // vertex buffer slot is left as garbage.
    //
    // Engine compensates by submitting a `s_flat_world_vertex` VS
    // variant (selected by `v14 = 4 * !(def_flags & 2)` in
    // `c_decal::render @ 0x18039B100` — vertex_type 4 =
    // `s_flat_world_vertex` per `macro_hlsl_vertex_type_names @
    // 0x181188400`). The flat VS doesn't write TBN interpolators; the
    // PS's `tangent_frame()` function (`decal_fx.hlsl:298-305`) returns
    // `0.0f` for `IS_FLAT_VERTEX` — but `IS_FLAT_VERTEX` in PS is
    // `TEST_CATEGORY_OPTION(bump_mapping, leave)` (PS-side
    // `decal_fx.hlsl:63-65`), which is FALSE for our `bump_mapping=
    // standard` palettes, so the engine PS also reads garbage TBN. The
    // engine apparently tolerates the resulting NaN/0 in RT1 — DX11
    // HLSL `normalize(0)` is implementation-defined; some toolchains
    // produce (0,0,0) which packs to (0.5, 0.5, 0.5) (neutral packed
    // normal) rather than NaN.
    //
    // Our WGSL pipeline doesn't carry the flat-vertex VS variant —
    // there's one decal entry point with one input layout. WGSL's
    // `normalize(vec3<f32>(0))` produces NaN. To avoid NaN
    // propagating into RT1 and breaking SL specular at decal pixels,
    // we ALWAYS pack a valid tangent frame from the BFS-walker's
    // `fold_normal` + `basis_up` data (already in work_vertex_buffer
    // per `mesh_builder.rs:179-181`). This deviates from engine
    // behavior but produces correct shading; a true engine-faithful
    // fix would add a flat-vertex variant and route non-bump_modulate
    // decals through it.
    //
    // The position Z-bias inside `build_tangent_frame` stays gated on
    // `fragment.requires_floating_z_bias` (instance-geom only).
    let _ = definition_flags;

    // === Vertex pass ===
    for slot in 0..v_count {
        let output_idx = v_start + slot;
        let group = mesh_builder.grouper[output_idx];
        let lead_sorter_slot = group.starting_vertex as usize;
        let lead_work_idx = mesh_builder.sorter_order[lead_sorter_slot] as usize;
        let lead = mesh_builder.work_vertex_buffer[lead_work_idx];

        // `WorkingVertex::world_position_override` carries a pre-clipped
        // polygon vertex from the BFS walker (mesh_builder.rs:253). For
        // INSTANCE-geometry fragments this position is in instance-local
        // space (the walker reads from `bsp.vertices` where `bsp` is the
        // instance's collision BSP), so it still needs the
        // `instance_local_to_world` lift. For main-BSP fragments
        // `instance_local_to_world` is `None` and the override is used
        // verbatim. Floating-quad's 4 vertices bypass the working buffer
        // entirely via `fragment.floating_quad`, so the override here is
        // exclusively from the BFS path.
        //
        // **Bug fix 2026-05-23**: previously the override branch skipped
        // the instance transform, putting BFS-walker non-quad decals at
        // world origin instead of at the instance's actual world position
        // — visible as decals floating in midair on shrine (non-quad
        // decals only; quad-fast-path was unaffected because it pre-
        // applies the transform inside `build_floating_quad`).
        let pre_world_pos = lead.world_position_override.unwrap_or_else(|| {
            let pos_idx = lead.position as usize;
            bsp.vertices.get(pos_idx)
                .map(|v| v.point)
                .unwrap_or(RealPoint3d { x: 0.0, y: 0.0, z: 0.0 })
        });
        let world_pos = match instance_local_to_world {
            Some(m) => matrix4x3_transform_point(m, pre_world_pos),
            None => pre_world_pos,
        };

        let v = &mut vertices[output_idx];
        v.position = [world_pos.x, world_pos.y, world_pos.z];
        v.texcoord = [lead.texcoord.x, lead.texcoord.y];

        // Sum normals/binormals across the run.
        let mut n_sum = RealVector3d { i: 0.0, j: 0.0, k: 0.0 };
        let mut b_sum = RealVector3d { i: 0.0, j: 0.0, k: 0.0 };
        for k in 0..group.vertex_count as usize {
            let sorter_slot = (group.starting_vertex as usize + k) as usize;
            let w_idx = mesh_builder.sorter_order[sorter_slot] as usize;
            let wv = mesh_builder.work_vertex_buffer[w_idx];
            n_sum.i += wv.normal.i;
            n_sum.j += wv.normal.j;
            n_sum.k += wv.normal.k;
            b_sum.i += wv.binormal.i;
            b_sum.j += wv.binormal.j;
            b_sum.k += wv.binormal.k;
        }
        // Normalize the normal sum (engine only normalizes N, leaves
        // B raw — Gram-Schmidt later orthogonalizes it against T).
        let n_len2 = n_sum.i * n_sum.i + n_sum.j * n_sum.j + n_sum.k * n_sum.k;
        if n_len2 >= 1e-8 {
            let inv = n_len2.sqrt().recip();
            n_sum = RealVector3d { i: n_sum.i * inv, j: n_sum.j * inv, k: n_sum.k * inv };
        }

        build_tangent_frame(v, instance_local_to_world, n_sum, b_sum, fragment.requires_floating_z_bias);
    }

    // === Index pass ===
    // Remap work indices through the collapser. Engine simply does
    // `indices[i] = collapser[work_index_buffer[i]]`.
    for slot in 0..i_count {
        let i_idx = i_start + slot;
        let work_index = mesh_builder.work_index_buffer[i_idx] as usize;
        indices[i_idx] = mesh_builder.collapser[work_index];
    }
}

/// Mirror of `c_decal::build_tangent_frame @ 0x18039BD40`.
///
/// `desired_normal` is the summed-and-normalized normal. `desired_binormal`
/// is the raw summed binormal (not yet orthonormal). Steps:
///   1. If instance-geom: transform both N and B from instance-local
///      to world (rotation only — translation is omitted).
///   2. T = N × B; normalize.
///   3. B = N × T; normalize (Gram-Schmidt).
///   4. Pack N, T, B as XMShortN4.
///   5. If apply_floating_z_bias: shift position by `X_FLOATING_Z_BIAS *
///      desired_normal` (uses the post-transform normal direction).
pub(super) fn build_tangent_frame(
    vertex: &mut RasterizerVertexWorld,
    instance_local_to_world: Option<&RealMatrix4x3>,
    desired_normal: RealVector3d,
    desired_binormal: RealVector3d,
    apply_floating_z_bias: bool,
) {
    let (n, b) = match instance_local_to_world {
        Some(m) => (
            matrix4x3_transform_normal(m, desired_normal),
            matrix4x3_transform_normal(m, desired_binormal),
        ),
        None => (desired_normal, desired_binormal),
    };

    // T = N × B
    let mut t = RealVector3d {
        i: n.j * b.k - n.k * b.j,
        j: n.k * b.i - n.i * b.k,
        k: n.i * b.j - n.j * b.i,
    };
    let t_len2 = t.i * t.i + t.j * t.j + t.k * t.k;
    if t_len2 >= 1e-8 {
        let inv = t_len2.sqrt().recip();
        t = RealVector3d { i: t.i * inv, j: t.j * inv, k: t.k * inv };
    }

    // B = N × T (Gram-Schmidt: re-derive B to be perpendicular to N+T)
    let mut b_ortho = RealVector3d {
        i: n.j * t.k - n.k * t.j,
        j: n.k * t.i - n.i * t.k,
        k: n.i * t.j - n.j * t.i,
    };
    let b_len2 = b_ortho.i * b_ortho.i + b_ortho.j * b_ortho.j + b_ortho.k * b_ortho.k;
    if b_len2 >= 1e-8 {
        let inv = b_len2.sqrt().recip();
        b_ortho = RealVector3d { i: b_ortho.i * inv, j: b_ortho.j * inv, k: b_ortho.k * inv };
    }

    vertex.normal = pack_short_n4(n.i, n.j, n.k);
    vertex.tangent = pack_short_n4(t.i, t.j, t.k);
    vertex.binormal = pack_short_n4(b_ortho.i, b_ortho.j, b_ortho.k);

    if apply_floating_z_bias {
        let bias = X_FLOATING_Z_BIAS;
        // Engine uses the pre-transform p_k which is either
        // desired_normal or the instance-transformed copy. Match by
        // applying after the (optional) transform — same vector we
        // packed into vertex.normal.
        let p = vertex.position;
        vertex.position = [
            p[0] + bias * n.i,
            p[1] + bias * n.j,
            p[2] + bias * n.k,
        ];
    }
}

/// XMShortN4 packing: clamp each component to [-1, 1], scale by
/// 32767.5 (engine uses `XMStoreShortN4::Scale` = 32767.0), pack as
/// signed 16-bit saturating. Fourth slot is always 0 — engine packs
/// 4 lanes but the input is a vec3, so the high slot drops out as 0
/// after the `vunpcklps + vmovlhps` shuffle when xmm2 was zero-init.
fn pack_short_n4(x: f32, y: f32, z: f32) -> [i16; 4] {
    fn pack(v: f32) -> i16 {
        let c = v.clamp(-1.0, 1.0);
        // Engine SCALE constant; round-to-nearest via cvtps2dq.
        let scaled = c * 32767.0;
        scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
    [pack(x), pack(y), pack(z), 0]
}

#[inline]
pub(super) fn matrix4x3_transform_point(m: &RealMatrix4x3, p: RealPoint3d) -> RealPoint3d {
    let s = m.scale;
    RealPoint3d {
        x: s * (p.x * m.forward.i + p.y * m.left.i + p.z * m.up.i) + m.position.x,
        y: s * (p.x * m.forward.j + p.y * m.left.j + p.z * m.up.j) + m.position.y,
        z: s * (p.x * m.forward.k + p.y * m.left.k + p.z * m.up.k) + m.position.z,
    }
}

#[inline]
fn matrix4x3_transform_normal(m: &RealMatrix4x3, n: RealVector3d) -> RealVector3d {
    let s = m.scale;
    RealVector3d {
        i: s * (n.i * m.forward.i + n.j * m.left.i + n.k * m.up.i),
        j: s * (n.i * m.forward.j + n.j * m.left.j + n.k * m.up.j),
        k: s * (n.i * m.forward.k + n.j * m.left.k + n.k * m.up.k),
    }
}
