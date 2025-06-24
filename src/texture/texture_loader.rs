//! 纹理加载模块
//! 负责从文件系统加载纹理数据

use anyhow::{Result, anyhow};
use ash::{Device, Instance};

use crate::resources::{copy_buffer_to_image, create_buffer, write_buffer_data};
use crate::texture::create_texture_image;
use crate::types::AppData;

/// 纹理加载器配置
#[derive(Debug, Clone)]
pub struct TextureConfig {
    /// 纹理文件路径
    pub path: String,
    /// 是否生成 Mipmap
    pub generate_mipmaps: bool,
    /// 纹理格式
    pub format: ash::vk::Format,
    /// 预期尺寸（可选，用于验证）
    pub expected_size: Option<(u32, u32)>,
}

impl TextureConfig {
    /// 创建默认纹理配置
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            generate_mipmaps: true,
            format: ash::vk::Format::R8G8B8A8_SRGB,
            expected_size: None,
        }
    }

    /// 设置是否生成 Mipmap
    pub fn with_mipmaps(mut self, generate: bool) -> Self {
        self.generate_mipmaps = generate;
        self
    }

    /// 设置纹理格式
    pub fn with_format(mut self, format: ash::vk::Format) -> Self {
        self.format = format;
        self
    }

    /// 设置预期尺寸
    pub fn with_expected_size(mut self, width: u32, height: u32) -> Self {
        self.expected_size = Some((width, height));
        self
    }
}

/// 加载纹理数据到 GPU
/// 从文件加载纹理并创建 Vulkan 图像资源
pub fn load_texture(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    config: TextureConfig,
) -> Result<()> {
    // 加载图像文件
    let img = image::open(&config.path)
        .map_err(|e| anyhow!("无法打开纹理图像 '{}': {}", config.path, e))?
        .into_rgba8();

    let (width, height) = img.dimensions();

    // 验证尺寸（如果指定了预期尺寸）
    if let Some((expected_width, expected_height)) = config.expected_size {
        if width != expected_width || height != expected_height {
            return Err(anyhow!(
                "纹理尺寸不匹配: 期望 {}x{}，实际 {}x{}",
                expected_width,
                expected_height,
                width,
                height
            ));
        }
    }

    let image_data = img.into_raw();
    let image_size = (width * height * 4) as ash::vk::DeviceSize;

    // 计算 Mipmap 级别数
    data.mip_levels = if config.generate_mipmaps {
        (width.max(height) as f32).log2().floor() as u32 + 1
    } else {
        1
    };

    // 创建暂存缓冲区
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        image_size,
        ash::vk::BufferUsageFlags::TRANSFER_SRC,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传数据到暂存缓冲区
    write_buffer_data(device, staging_buffer_memory, &image_data)?;

    // 创建纹理图像
    let (texture_image, texture_image_memory) = create_texture_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        config.format,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // 复制数据到图像
    copy_buffer_to_image(device, data, staging_buffer, texture_image, width, height)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != ash::vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != ash::vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    log::info!(
        "纹理加载完成: {} ({}x{}, {} mip级别)",
        config.path,
        width,
        height,
        data.mip_levels
    );

    Ok(())
}
