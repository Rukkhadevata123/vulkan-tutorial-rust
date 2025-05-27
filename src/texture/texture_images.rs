//! 纹理图像管理模块
//! 负责纹理图像和视图的创建

use anyhow::Result;
use ash::vk;
use ash::{Device, Instance};

use crate::resources::{
    create_image, create_image_view, generate_mipmaps, transition_image_layout,
};
use crate::types::AppData;

/// 创建纹理图像
/// 为纹理数据创建 Vulkan 图像资源
pub fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let (image, memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        mip_levels,
        vk::SampleCountFlags::TYPE_1,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // 转换图像布局为传输目标
    transition_image_layout(
        device,
        data,
        image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;

    Ok((image, memory))
}

/// 创建纹理图像视图
/// 为纹理图像创建视图供着色器访问
pub fn create_texture_image_view(
    device: &Device,
    data: &mut AppData,
    format: vk::Format,
) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        format,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    log::info!("纹理图像视图创建完成");
    Ok(())
}

/// 生成纹理 Mipmap
/// 为纹理生成多级渐远纹理
pub fn generate_texture_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<()> {
    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        format,
        width,
        height,
        data.mip_levels,
    )?;

    log::info!("纹理 Mipmap 生成完成: {} 级别", data.mip_levels);
    Ok(())
}
