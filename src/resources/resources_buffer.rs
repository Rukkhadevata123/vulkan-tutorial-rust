//! 缓冲区管理模块
//! 包含缓冲区创建、数据传输和内存管理功能

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Device, Instance};
use std::ptr::copy_nonoverlapping as memcpy;

use crate::resources::memory::find_memory_type;
use crate::types::AppData;
use crate::vulkan::commands::{begin_single_time_commands, end_single_time_commands};

/// 创建缓冲区
/// 统一的缓冲区创建接口
pub fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // 创建缓冲区信息
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // 创建缓冲区
    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| anyhow!("创建缓冲区失败: {}", e))?
    };

    // 获取内存需求
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    // 查找合适的内存类型
    let mem_type_index =
        find_memory_type(instance, data.physical_device, properties, mem_requirements)?;

    // 分配内存
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let buffer_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| anyhow!("分配缓冲区内存失败: {}", e))?
    };

    // 绑定缓冲区和内存
    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| anyhow!("绑定缓冲区内存失败: {}", e))?
    };

    Ok((buffer, buffer_memory))
}

/// 复制缓冲区数据
/// 从源缓冲区复制数据到目标缓冲区
pub fn copy_buffer(
    device: &Device,
    data: &AppData,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let copy_region = vk::BufferCopy::default().size(size);

    unsafe {
        device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);
    }

    end_single_time_commands(device, data, command_buffer)?;
    Ok(())
}

/// 映射并写入缓冲区数据
/// 通用的缓冲区数据上传函数
pub fn write_buffer_data<T>(
    device: &Device,
    buffer_memory: vk::DeviceMemory,
    data: &[T],
) -> Result<()> {
    let size = (size_of_val(data)) as vk::DeviceSize;

    unsafe {
        let memory_ptr = device.map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(data.as_ptr(), memory_ptr.cast(), data.len());
        device.unmap_memory(buffer_memory);
    }

    Ok(())
}
