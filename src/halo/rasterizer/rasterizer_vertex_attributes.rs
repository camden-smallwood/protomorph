//! Mirror of `Ares/source/rasterizer/rasterizer_vertex_attributes.h`
//! (40 lines).
//!
//! Vertex-attribute slot enum used by Halo's input layout
//! description. The shipping platforms use a fixed mapping
//! (position=0, node_indices=1, node_weights=2, ...) so input
//! layouts can be deduced from the vertex format type alone.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RasterizerVertexAttribute {
    Position = 0,
    NodeIndices = 1,
    NodeWeights = 2,
    Texcoord = 3,
    Normal = 4,
    Binormal = 5,
    Tangent = 6,
    SecondaryTexcoord = 7,
    Color = 8,
    VertColor = 9,
    BillboardOffset = 10,
    BillboardAxis = 11,
    PcaClusterId = 12,
    PcaWeights = 13,
}

impl RasterizerVertexAttribute {
    pub const COUNT: usize = 14;
    /// Halo: `_rasterizer_vertex_attribute_skip_bytes = -2`. Used
    /// in input-layout terminator lists.
    pub const SKIP_BYTES: i32 = -2;
    /// Halo: `_rasterizer_vertex_attribute_terminator = -1`.
    pub const TERMINATOR: i32 = -1;
}
