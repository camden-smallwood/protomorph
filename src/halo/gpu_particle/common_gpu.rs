//! Shared GPU particle utilities (engine `common_gpu.cpp`): sprite-frame
//! baking and the blend-mode → wgpu blend-state mapping used by both the
//! particle-block render and the light-volume render.

use blam_tags::render_method::AlphaBlendMode;

/// Max sprite-sheet frames carried per batch in the render's `MatParams`
/// uniform (engine `MAXIMUM_SPRITES_PER_SEQUENCE` is 128; 64 covers every
/// shipped particle sheet — overflow is logged, not silently dropped).
pub const MAX_SPRITE_FRAMES: usize = 64;

/// Bake the default sprite corner (engine
/// `c_particle_definition::postprocess_frame_animation`, reach
/// `0x82ed5c30`) for a particle with NO bitmap sequences — the common
/// case (all riverworld waterfall bitmaps). Sprite rect defaults to the
/// full bitmap `(left,right,top,bottom)=(0,1,0,1)`, registration `(0.5,
/// 0.5)` shifted by `center_offset`; the bitmap's pixel aspect stretches
/// x. Returns `(corner.x, corner.y, corner.z, corner.w)` where the
/// billboard vertex is `shift·corner.zw + corner.xy`. For a square,
/// centered bitmap this is `(-1,-1,2,2)` → a `[-1,1]` quad (2·size).
/// (Sequenced/sprite-sheet bitmaps need the per-sprite rect — future work.)
pub fn default_sprite_corner(bitmap_w: u32, bitmap_h: u32, center_offset: [f32; 2]) -> [f32; 4] {
    // Default full-bitmap sprite rect + centered registration.
    let (left, right, top, bottom) = (0.0f32, 1.0, 0.0, 1.0);
    let reg_x = 0.5 + center_offset[0];
    let reg_y = 0.5 - center_offset[1];
    // Bitmap pixel aspect (only when non-square).
    let v50 = if bitmap_w != bitmap_h && bitmap_h != 0 {
        bitmap_w as f32 / bitmap_h as f32
    } else {
        1.0
    };
    let sw = right - left;
    let sh = bottom - top;
    let mut ax = 1.0f32;
    let mut ay = 1.0f32;
    if sw <= sh {
        ax = sw / sh;
    } else {
        ay = sh / sw;
    }
    ax *= v50;
    let rx = (reg_x / sw) * 2.0 - 1.0;
    let ry = (reg_y / sh) * 2.0 - 1.0;
    [(-1.0 - rx) * ax, (-1.0 - ry) * ay, 2.0 * ax, 2.0 * ay]
}

/// Bake a sprite-sheet sequence's frames into the engine `frame_texcoord`
/// UV form `(left, top, right-left, bottom-top)`. `seq_index` selects the
/// particle's `first sequence index`; out-of-range or empty → no frames
/// (plain bitmap). Logs (does not silently drop) sequences over
/// [`MAX_SPRITE_FRAMES`].
pub fn bake_sprite_frames(
    sequences: &[blam_tags::bitmap::BitmapSequence],
    seq_index: i16,
) -> Vec<[f32; 4]> {
    let Some(seq) = usize::try_from(seq_index).ok().and_then(|i| sequences.get(i)) else {
        return Vec::new();
    };
    if seq.sprites.len() > MAX_SPRITE_FRAMES {
        eprintln!(
            "[particles] WARNING: sprite sequence has {} frames > MAX_SPRITE_FRAMES {} — clamping",
            seq.sprites.len(),
            MAX_SPRITE_FRAMES,
        );
    }
    seq.sprites
        .iter()
        .take(MAX_SPRITE_FRAMES)
        .map(|s| [s.left, s.top, s.right - s.left, s.bottom - s.top])
        .collect()
}

/// Map a particle blend mode to its wgpu blend state. Keyed on the
/// runtime [`AlphaBlendMode`] (`e_alpha_blend_mode`), resolved BY NAME
/// upstream — NOT the authored category option index, which differs per
/// rmdf. Mirrors the engine's `set_alpha_blend_mode_no_cache
/// @0x18066E930` D3D states. The PS pre-converts non-linear modes
/// (multiply → lerp, premult → rgb*=a) so the state here is the straight
/// D3D blend factors.
pub(crate) fn blend_state(mode: AlphaBlendMode) -> Option<wgpu::BlendState> {
    use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
    use AlphaBlendMode as M;
    let comp = |s: F, d: F, op: O| C { src_factor: s, dst_factor: d, operation: op };
    let both = |s: F, d: F, op: O| wgpu::BlendState { color: comp(s, d, op), alpha: comp(s, d, op) };
    // Premultiplied coverage-accumulate alpha (kept from the prior,
    // user-verified waterfall path): dst alpha = src.a + dst.a·(1−src.a).
    let premult_alpha = comp(F::One, F::OneMinusSrcAlpha, O::Add);
    Some(match mode {
        M::Opaque => return None,
        M::Additive => both(F::One, F::One, O::Add),
        M::Multiply => both(F::Dst, F::Zero, O::Add), // PS lerps
        M::AlphaBlend => wgpu::BlendState {
            color: comp(F::SrcAlpha, F::OneMinusSrcAlpha, O::Add),
            alpha: premult_alpha,
        },
        M::DoubleMultiply => both(F::Dst, F::Src, O::Add),
        // pre_multiplied_alpha (5) and multiply_add (7) share the engine
        // blend word; the PS premultiplies rgb·=a for both.
        M::PreMultipliedAlpha | M::MultiplyAdd => wgpu::BlendState {
            color: comp(F::One, F::OneMinusSrcAlpha, O::Add),
            alpha: premult_alpha,
        },
        M::Maximum => both(F::One, F::One, O::Max),
        // FIXED: add_src_times_dstalpha previously fell through to the
        // alpha_blend default because the positional index (category 7)
        // had no arm — it is `(DEST_ALPHA, ONE)`.
        M::AddSrcTimesDstAlpha => both(F::DstAlpha, F::One, O::Add),
        M::AddSrcTimesSrcAlpha => both(F::SrcAlpha, F::One, O::Add),
        M::InvAlphaBlend => both(F::OneMinusSrcAlpha, F::SrcAlpha, O::Add),
        // Motion blur — not implemented; safe alpha_blend fallback.
        M::MotionBlurStatic | M::MotionBlurInhibit => wgpu::BlendState {
            color: comp(F::SrcAlpha, F::OneMinusSrcAlpha, O::Add),
            alpha: premult_alpha,
        },
    })
}
