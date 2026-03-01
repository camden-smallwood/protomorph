pub struct DdsInfo {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub format: wgpu::TextureFormat,
    pub block_size: u32,
    pub data: Vec<u8>,
}

pub fn parse_dds(bytes: &[u8]) -> DdsInfo {
    assert!(&bytes[0..4] == b"DDS ", "not a DDS file");
    let width = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let mip_count = u32::from_le_bytes(bytes[28..32].try_into().unwrap()).max(1);
    let fourcc = &bytes[84..88];

    let (format, block_size) = match fourcc {
        b"DXT1" => (wgpu::TextureFormat::Bc1RgbaUnormSrgb, 8u32),
        b"DXT3" => (wgpu::TextureFormat::Bc2RgbaUnormSrgb, 16u32),
        b"DXT5" => (wgpu::TextureFormat::Bc3RgbaUnormSrgb, 16u32),
        _ => panic!(
            "unsupported DDS fourcc: {:?}",
            std::str::from_utf8(fourcc).unwrap_or("???")
        ),
    };

    DdsInfo {
        width,
        height,
        mip_count,
        format,
        block_size,
        data: bytes[128..].to_vec(),
    }
}

pub fn load_dds_from_file(path: &str) -> DdsInfo {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read DDS file {path}: {e}"));
    parse_dds(&bytes)
}

pub fn create_dds_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dds: &DdsInfo,
    linear: bool,
) -> wgpu::Texture {
    let format = if linear { srgb_to_linear(dds.format) } else { dds.format };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: dds.width,
            height: dds.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: dds.mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut offset = 0usize;
    for mip in 0..dds.mip_count {
        let mip_width = (dds.width >> mip).max(1);
        let mip_height = (dds.height >> mip).max(1);
        let blocks_wide = (mip_width + 3) / 4;
        let blocks_high = (mip_height + 3) / 4;
        let mip_size = (blocks_wide * blocks_high * dds.block_size) as usize;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &dds.data[offset..offset + mip_size],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(blocks_wide * dds.block_size),
                rows_per_image: Some(blocks_high),
            },
            wgpu::Extent3d {
                width: blocks_wide * 4,
                height: blocks_high * 4,
                depth_or_array_layers: 1,
            },
        );
        offset += mip_size;
    }

    texture
}

fn srgb_to_linear(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnorm,
        wgpu::TextureFormat::Bc2RgbaUnormSrgb => wgpu::TextureFormat::Bc2RgbaUnorm,
        wgpu::TextureFormat::Bc3RgbaUnormSrgb => wgpu::TextureFormat::Bc3RgbaUnorm,
        other => other,
    }
}

pub fn create_fallback_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fallback_white"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture
}
