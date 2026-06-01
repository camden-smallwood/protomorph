//! Serialize a [`MaterialData`] into the per-variant cbuffer bytes
//! that match a [`CbufferLayout`].
//!
//! The bulk of the cbuffer comes from `mat.cbuffer.bytes` — the
//! cache-build replay (`resolve_pixel_user_cbuffer`) packed
//! `rmt2.float_constants[]` slot values from rmsh+rmop. A small set of
//! engine externs (per-unit color schemes, etc.) live in
//! [`crate::halo::loader::EXTERN_SLOTS`] and aren't routed by the rmt2 — those
//! are overlaid here from typed `MaterialData` fields.

use crate::halo::render_methods::cbuffer::CbufferLayout;
use crate::halo::render_methods::materials::MaterialData;

/// Build the per-variant material uniform buffer for `mat` according
/// to `layout`. Returns `Vec<u8>` of size `layout.total_size`.
pub fn serialize(mat: &MaterialData, layout: &CbufferLayout) -> Vec<u8> {
    serialize_bytes(
        &mat.cbuffer.bytes,
        layout,
        mat.primary_change_color,
        mat.secondary_change_color,
    )
}

/// Lower-level serialize that takes the cbuffer bytes + extern values
/// directly instead of pulling them off [`MaterialData`]. Used by the
/// P6.5 per-frame animated-cbuffer update path, which holds the
/// re-resolved bytes and the const-at-load extern colors separately
/// rather than carrying a full [`MaterialData`] forward.
pub fn serialize_bytes(
    cbuffer_bytes: &[u8],
    layout: &CbufferLayout,
    primary_change_color: glam::Vec3,
    secondary_change_color: glam::Vec3,
) -> Vec<u8> {
    let total = layout.total_size as usize;
    let mut buf = vec![0u8; total];

    let n = cbuffer_bytes.len().min(total);
    if n > 0 {
        buf[..n].copy_from_slice(&cbuffer_bytes[..n]);
    }

    overlay_extern(&mut buf, layout, "primary_change_color", primary_change_color);
    overlay_extern(&mut buf, layout, "secondary_change_color", secondary_change_color);

    buf
}

fn overlay_extern(buf: &mut [u8], layout: &CbufferLayout, name: &str, value: glam::Vec3) {
    let Some(member) = layout.find(name) else { return };
    let off = member.byte_offset as usize;
    if off + 16 > buf.len() {
        return;
    }
    let v = [value.x, value.y, value.z, 0.0];
    for (i, c) in v.iter().enumerate() {
        buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&c.to_le_bytes());
    }
}
