//! 模型描述符管理模块
//! 负责创建和管理模型渲染所需的描述符资源

use anyhow::{Result, anyhow};
use ash::{Device, vk};
use std::mem::size_of;

use crate::resources::{DescriptorPoolConfig, create_descriptor_pool};
use crate::types::{AppData, ModelUBO};

/// 创建模型描述符集布局
/// 定义统一缓冲区和纹理采样器的绑定
pub fn create_model_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let bindings = [
        // 绑定0: 模型统一缓冲区 (视图和投影矩阵)
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        // 绑定1: 纹理采样器
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    data.model_descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| anyhow!("创建模型描述符集布局失败: {}", e))?
    };

    log::info!("模型描述符集布局创建完成");
    Ok(())
}

/// 创建模型描述符池
/// 为模型描述符集分配池空间
pub fn create_model_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的描述符池
    if data.model_descriptor_pool != vk::DescriptorPool::null() {
        unsafe {
            device.destroy_descriptor_pool(data.model_descriptor_pool, None);
        }
        data.model_descriptor_pool = vk::DescriptorPool::null();
        data.model_descriptor_sets.clear();
    }

    let image_count = data.swapchain_images.len() as u32;
    if image_count == 0 {
        return Ok(());
    }

    // 配置描述符池
    let config = DescriptorPoolConfig::new()
        .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, image_count)
        .add_pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, image_count)
        .max_sets(image_count)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    data.model_descriptor_pool = create_descriptor_pool(device, config)?;

    log::info!("模型描述符池创建完成: 最大集合数 {image_count}");
    Ok(())
}

/// 创建并更新模型描述符集
/// 为每个交换链图像分配并配置描述符集
pub fn create_model_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let image_count = data.swapchain_images.len();

    if image_count == 0 || data.model_descriptor_pool == vk::DescriptorPool::null() {
        return Ok(());
    }

    data.model_descriptor_sets.clear();

    // 为每个交换链图像准备相同的布局
    let layouts = vec![data.model_descriptor_set_layout; image_count];

    // 分配描述符集
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(data.model_descriptor_pool)
        .set_layouts(&layouts);

    data.model_descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| anyhow!("分配模型描述符集失败: {}", e))?
    };

    // 更新每个描述符集
    for i in 0..image_count {
        update_model_descriptor_set(device, data, i)?;
    }

    log::info!("模型描述符集创建完成: {image_count} 个");
    Ok(())
}

/// 更新单个模型描述符集
/// 绑定统一缓冲区和纹理资源到描述符集
pub fn update_model_descriptor_set(
    device: &Device,
    data: &AppData,
    image_index: usize,
) -> Result<()> {
    // 统一缓冲区信息
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(data.model_uniform_buffers[image_index])
        .offset(0)
        .range(size_of::<ModelUBO>() as u64);

    // 纹理图像信息
    let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(data.texture_image_view)
        .sampler(data.texture_sampler);

    // 描述符写入操作
    let descriptor_writes = [
        // 写入统一缓冲区
        vk::WriteDescriptorSet::default()
            .dst_set(data.model_descriptor_sets[image_index])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info)),
        // 写入纹理采样器
        vk::WriteDescriptorSet::default()
            .dst_set(data.model_descriptor_sets[image_index])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info)),
    ];

    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(())
}
