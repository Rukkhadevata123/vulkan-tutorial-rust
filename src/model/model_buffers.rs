//! 模型缓冲区管理模块
//! 负责创建和管理模型相关的缓冲区

use anyhow::Result;
use ash::vk;
use ash::{Device, Instance};
use std::mem::size_of;

use crate::resources::{copy_buffer, create_buffer, write_buffer_data};
use crate::types::{AppData, ModelUBO, ModelVertex};

/// 创建模型顶点缓冲区
/// 将顶点数据上传到 GPU 内存
pub fn create_model_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    if data.vertices.is_empty() {
        log::warn!("顶点数据为空，跳过顶点缓冲区创建");
        return Ok(());
    }

    let buffer_size = (size_of::<ModelVertex>() * data.vertices.len()) as vk::DeviceSize;

    // 创建暂存缓冲区
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传顶点数据
    write_buffer_data(device, staging_buffer_memory, &data.vertices)?;

    // 创建设备本地顶点缓冲区
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    // 复制数据
    copy_buffer(device, data, staging_buffer, vertex_buffer, buffer_size)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    log::info!("模型顶点缓冲区创建完成: {} 顶点", data.vertices.len());
    Ok(())
}

/// 创建模型索引缓冲区
/// 将索引数据上传到 GPU 内存
pub fn create_model_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    if data.indices.is_empty() {
        log::warn!("索引数据为空，跳过索引缓冲区创建");
        return Ok(());
    }

    let buffer_size = (size_of::<u32>() * data.indices.len()) as vk::DeviceSize;

    // 创建暂存缓冲区
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传索引数据
    write_buffer_data(device, staging_buffer_memory, &data.indices)?;

    // 创建设备本地索引缓冲区
    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    // 复制数据
    copy_buffer(device, data, staging_buffer, index_buffer, buffer_size)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    log::info!("模型索引缓冲区创建完成: {} 索引", data.indices.len());
    Ok(())
}

/// 创建模型统一缓冲区
/// 为每个交换链图像创建统一缓冲区
pub fn create_model_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // 清理已有缓冲区
    cleanup_model_uniform_buffers(device, data);

    let buffer_size = size_of::<ModelUBO>() as vk::DeviceSize;
    let image_count = data.swapchain_images.len();

    if image_count == 0 {
        log::warn!("交换链图像数量为0，跳过统一缓冲区创建");
        return Ok(());
    }

    // 为每个交换链图像创建统一缓冲区
    for _i in 0..image_count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        data.model_uniform_buffers.push(uniform_buffer);
        data.model_uniform_buffers_memory
            .push(uniform_buffer_memory);
    }

    log::info!("模型统一缓冲区创建完成: {image_count} 个");
    Ok(())
}

/// 清理模型统一缓冲区
pub fn cleanup_model_uniform_buffers(device: &Device, data: &mut AppData) {
    unsafe {
        for &memory in &data.model_uniform_buffers_memory {
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        }
        for &buffer in &data.model_uniform_buffers {
            if buffer != vk::Buffer::null() {
                device.destroy_buffer(buffer, None);
            }
        }
    }
    data.model_uniform_buffers.clear();
    data.model_uniform_buffers_memory.clear();
}
