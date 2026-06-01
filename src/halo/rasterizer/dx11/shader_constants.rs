//! `d3d11_shader_constants` mirror — 112-slot cbuffer pool.
//!
//! Engine wrappers `c_rasterizer::set_shader_constant @ 0x1806BEA70`
//! and `set_shader_constant_bool @ 0x1806BEB20` are thin pass-throughs
//! to `d3d11_shader_constants::set_shader_constant`, which decodes a
//! 32-bit slot ID and writes into one of 112 internal cbuffers
//! (`e_constantbuffers` enum in Ares).
//!
//! ## Slot ID bit layout (audit-verified 2026-05-13)
//!
//! ```text
//! bits 0-15  entry_index (×16 = byte offset within cbuffer)
//! bits 16-26 cbuffer pool index (max 0x7FF; real pool max = 111 = 0x6F)
//! bit  27    bool-flag (set by `set_shader_constant_bool` via `| 0x8000000`)
//! bits 28-31 sub_byte_offset (0..=15)
//! ```
//!
//! Effective byte_offset = `entry_index * 16 + sub_byte_offset`.
//!
//! `set_shader_constant`: writes `count * 16` bytes per call (vec4s).
//! `set_shader_constant_bool`: writes `count * 4` bytes per call (i32 bools).
//!
//! ## Pool index reference
//!
//! `reference_engine_cbuffer_inventory.md` lists all 112 entries by name.
//! Key slots protomorph touches (engine HLSL `CBUFFER_BEGIN` names):
//! - `0x27 _ViewVS`, `0x28 _ExposureVS`, `0x29 _AtmosphereVS`,
//!   `0x2A _LightingVS`, `0x2B _ShadowProjVS`
//! - `0x2C _ViewPS`, `0x2D _ExposurePS`, `0x2E _LightingPS`,
//!   `0x2F _MiscPS` (incl. LDR_gamma2/HDR_gamma2 at entry 5)
//! - `0x31 _SimpleLightsPS`, `0x33 _Kernel5PS` (incl. `actually_calc_albedo`
//!   at entry 5 sub_byte 8)
//!
//! Cross-references:
//! - `reference_engine_shader_constant_routing.md` — bit-layout + real-world slot IDs.
//! - `reference_engine_cbuffer_inventory.md` — full 112-entry inventory.

/// `e_constantbuffers::_constantbuffer_num_buffers`. Engine pool size.
pub const K_NUMBER_OF_CBUFFERS: usize = 112;

/// One slot in the engine cbuffer pool. Holds CPU staging bytes that
/// `set_shader_constant` mutates, plus a lazily-allocated wgpu uniform
/// buffer that `flush_dirty` writes through to.
#[derive(Debug, Default)]
pub struct CbufferSlot {
    /// CPU-side staging. Grown on first/larger write; never shrinks.
    pub staging: Vec<u8>,
    /// GPU uniform buffer. Allocated on first `flush_dirty`; re-allocated
    /// if staging outgrows it (rare — sizes are stable per pool slot).
    pub gpu: Option<wgpu::Buffer>,
    /// Set on every `set_shader_constant*` write; cleared by `flush_dirty`.
    pub dirty: bool,
}

/// Mirrors the `c_rasterizer::ptr[112]` cbuffer table that
/// `d3d11_shader_constants::set_shader_constant` writes into.
///
/// The engine pre-allocates each slot at startup based on the
/// `e_constantbuffers` enum's per-slot size; we lazy-allocate on
/// first write so unused slots cost nothing.
#[derive(Debug)]
pub struct CbufferPool {
    /// Indexed by pool index (`(slot_id >> 16) & 0x7FF`). Length = 112.
    pub slots: Vec<CbufferSlot>,
}

impl Default for CbufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CbufferPool {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(K_NUMBER_OF_CBUFFERS);
        for _ in 0..K_NUMBER_OF_CBUFFERS {
            slots.push(CbufferSlot::default());
        }
        Self { slots }
    }

    /// Eagerly allocate a slot's GPU buffer at `min_size` (rounded up to
    /// wgpu's 256B UBO alignment). Use this when something needs to bind
    /// the buffer into a bind group before any `set_shader_constant`
    /// has touched the slot — without an eager alloc, `slot.gpu` is `None`
    /// and the BG would have nothing to reference.
    ///
    /// **Re-allocation caveat:** if a later `set_shader_constant` writes
    /// past the eager size, `flush_dirty` re-allocates the GPU buffer,
    /// invalidating any bind groups that referenced the old buffer.
    /// Eagerly size for the slot's HLSL-declared maximum to avoid this.
    pub fn ensure_slot_allocated(
        &mut self,
        device: &wgpu::Device,
        cbuffer_idx: usize,
        min_size: u64,
    ) {
        assert!(cbuffer_idx < K_NUMBER_OF_CBUFFERS);
        let slot = &mut self.slots[cbuffer_idx];
        let padded = min_size.next_multiple_of(256).max(256);
        if slot.gpu.is_none() {
            slot.gpu = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("cbuffer_{cbuffer_idx:#04x}")),
                size: padded,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            // Make sure the staging is at least the same size so the first
            // `flush_dirty` writes a fully-initialized buffer (no UB reading
            // uninitialized cbuffer entries).
            if slot.staging.len() < padded as usize {
                slot.staging.resize(padded as usize, 0);
            }
            slot.dirty = true;
        }
    }

    /// Mirror of `c_rasterizer::set_shader_constant @ 0x1806BEA70` →
    /// `d3d11_shader_constants::set_shader_constant(slot_id, constants, 16 * count)`.
    ///
    /// Writes `count` vec4s at the decoded byte offset within the
    /// cbuffer selected by `slot_id`. Marks the slot dirty for the
    /// next `flush_dirty` pass.
    pub fn set_shader_constant(&mut self, slot_id: u32, count: u32, constants: &[[f32; 4]]) {
        let count = count as usize;
        let bytes: &[u8] = bytemuck::cast_slice(&constants[..count]);
        let byte_size = count * 16;
        let (cbuffer_idx, byte_offset) = decode_slot_id(slot_id);
        self.write(cbuffer_idx, byte_offset, &bytes[..byte_size]);
    }

    /// Mirror of `c_rasterizer::set_shader_constant_bool @ 0x1806BEB20` →
    /// `d3d11_shader_constants::set_shader_constant(slot_id | 0x8000000, constants, 4 * count)`.
    ///
    /// ORs bit 27 into the slot_id before decoding (engine ALWAYS does
    /// this in the wrapper), then writes `count * 4` bytes. Bool writes
    /// occupy 4 bytes each — they pack into a vec4's xyzw lanes via
    /// `sub_byte_offset` to target the right lane.
    pub fn set_shader_constant_bool(&mut self, slot_id: u32, count: u32, constants: &[u32]) {
        let routed = slot_id | 0x0800_0000;
        let count = count as usize;
        let bytes: &[u8] = bytemuck::cast_slice(&constants[..count]);
        let byte_size = count * 4;
        let (cbuffer_idx, byte_offset) = decode_slot_id(routed);
        self.write(cbuffer_idx, byte_offset, &bytes[..byte_size]);
    }

    fn write(&mut self, cbuffer_idx: usize, byte_offset: usize, bytes: &[u8]) {
        assert!(
            cbuffer_idx < K_NUMBER_OF_CBUFFERS,
            "cbuffer index {cbuffer_idx} out of range (max 111)",
        );
        let slot = &mut self.slots[cbuffer_idx];
        let needed = byte_offset + bytes.len();
        if slot.staging.len() < needed {
            slot.staging.resize(needed, 0);
        }
        slot.staging[byte_offset..byte_offset + bytes.len()].copy_from_slice(bytes);
        slot.dirty = true;
    }

    /// Upload dirty CPU staging to GPU buffers, allocating/resizing as needed.
    /// Call once per frame (or before each render pass) prior to binding.
    pub fn flush_dirty(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if !slot.dirty {
                continue;
            }
            // wgpu requires UBOs aligned to 256B; pad up.
            let padded_size = slot.staging.len().next_multiple_of(256).max(256) as u64;
            let needs_realloc = match slot.gpu.as_ref() {
                None => true,
                Some(buf) => buf.size() < padded_size,
            };
            if needs_realloc {
                slot.gpu = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("cbuffer_{idx:#04x}")),
                    size: padded_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(slot.gpu.as_ref().unwrap(), 0, &slot.staging);
            slot.dirty = false;
        }
    }
}

/// Decode a 32-bit slot_id into `(cbuffer_idx, byte_offset)`.
/// Shared by both vec4 and bool write paths.
#[inline]
fn decode_slot_id(slot_id: u32) -> (usize, usize) {
    let cbuffer_idx = ((slot_id >> 16) & 0x7FF) as usize;
    let entry_idx = (slot_id & 0xFFFF) as usize;
    let sub_byte = ((slot_id >> 28) & 0xF) as usize;
    let byte_offset = entry_idx * 16 + sub_byte;
    (cbuffer_idx, byte_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audit-anchor test: the `actually_calc_albedo` slot ID seen in
    /// `set_using_albedo_sampler @ 0x18069EF80` decodes to cbuffer 0x33
    /// (Kernel5PS), byte offset 88 (entry 5 + sub_byte 8). After OR
    /// with 0x8000000 the routed slot is 0x88330005.
    #[test]
    fn actually_calc_albedo_decodes_to_kernel5ps_byte_88() {
        let raw = 0x8033_0005_u32; // value passed to set_shader_constant_bool
        let routed = raw | 0x0800_0000;
        let (cbuf, off) = decode_slot_id(routed);
        assert_eq!(cbuf, 0x33, "cbuffer index");
        assert_eq!(off, 5 * 16 + 8, "byte offset (entry 5 + sub_byte 8)");
    }

    /// ExposureVS at slot 0x280000 → cbuffer 0x28, byte 0.
    #[test]
    fn exposure_vs_decodes_to_cbuffer_28_byte_0() {
        let (cbuf, off) = decode_slot_id(0x0028_0000);
        assert_eq!(cbuf, 0x28);
        assert_eq!(off, 0);
    }

    /// SimpleLightsPS entry 1 → cbuffer 0x31, byte 16.
    #[test]
    fn simple_lights_ps_entry_1_decodes_to_cbuffer_31_byte_16() {
        let (cbuf, off) = decode_slot_id(0x0031_0001);
        assert_eq!(cbuf, 0x31);
        assert_eq!(off, 16);
    }

    /// Round-trip: write a vec4 to cbuffer 0x2F entry 3, read it back.
    #[test]
    fn set_shader_constant_writes_at_correct_offset() {
        let mut pool = CbufferPool::new();
        let val = [[1.0_f32, 2.0, 3.0, 4.0]];
        pool.set_shader_constant(0x002F_0003, 1, &val);
        let slot = &pool.slots[0x2F];
        assert!(slot.dirty);
        // entry 3 = byte 48
        let bytes: &[f32] = bytemuck::cast_slice(&slot.staging[48..64]);
        assert_eq!(bytes, &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Audit-anchor round-trip for the `actually_calc_albedo` write path
    /// engine `set_using_albedo_sampler @ 0x18069EF80` exercises:
    ///   set_shader_constant_bool(0x80330005, 1, &actually_calc_albedo)
    /// → routed slot 0x88330005 → cbuffer 0x33, byte offset 88.
    ///
    /// Engine semantics: `actually_calc_albedo = !using_albedo_sampler`.
    /// Verifies the cbuffer bit lands at the right byte so the WGSL
    /// reads `Kernel5PS.entries[5].z` and gets the engine value.
    #[test]
    fn actually_calc_albedo_round_trip_lands_at_byte_88() {
        let mut pool = CbufferPool::new();
        // Engine pseudocode:
        //   set_using_albedo_sampler(true) →
        //     actually_calc_albedo = !true = 0
        //     set_shader_constant_bool(0x80330005, 1, &0i32)
        pool.set_shader_constant_bool(0x8033_0005, 1, &[0u32]);
        let slot = &pool.slots[0x33];
        assert!(slot.dirty);
        // entry 5 byte 8 (sub_byte) = byte 88
        let v = u32::from_le_bytes(slot.staging[88..92].try_into().unwrap());
        assert_eq!(v, 0, "actually_calc_albedo should be 0 after sampler=TRUE");

        // Now flip to using_albedo_sampler=FALSE → actually_calc_albedo=1.
        pool.set_shader_constant_bool(0x8033_0005, 1, &[1u32]);
        let v = u32::from_le_bytes(pool.slots[0x33].staging[88..92].try_into().unwrap());
        assert_eq!(v, 1, "actually_calc_albedo should be 1 after sampler=FALSE");
    }
}
