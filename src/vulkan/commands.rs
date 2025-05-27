//! 命令池和命令缓冲区管理模块
//! 负责创建和管理命令池、主命令缓冲区和计算命令缓冲区

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Device, Entry, Instance};
use log::*;

use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::types::{AppData, QueueFamilyIndices};

/// 创建命令池
/// 为命令缓冲区分配创建命令池
pub fn vulkan_create_command_pools(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    // 全局命令池
    data.command_pool = vulkan_create_command_pool_internal(instance, device, entry, data)?;

    // 为每个交换链图像创建命令池
    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = vulkan_create_command_pool_internal(instance, device, entry, data)?;
        data.command_pools.push(command_pool);
    }

    info!("命令池创建完成: 1 个全局池 + {} 个图像池", num_images);
    Ok(())
}

/// 创建单个命令池
/// 内部辅助函数
fn vulkan_create_command_pool_internal(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &AppData,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    unsafe {
        device
            .create_command_pool(&info, None)
            .map_err(|e| anyhow!("创建命令池失败: {}", e))
    }
}

/// 创建命令缓冲区
/// 为每个交换链图像分配主命令缓冲区
pub fn vulkan_create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let num_images = data.swapchain_images.len();

    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .map_err(|e| anyhow!("分配命令缓冲区失败: {}", e))?[0]
        };
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    info!("命令缓冲区创建完成: {} 个主缓冲区", num_images);
    Ok(())
}

/// 创建计算命令缓冲区
/// 为每个飞行帧分配计算命令缓冲区
pub fn vulkan_create_compute_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的计算命令缓冲区
    if !data.compute_command_buffers.is_empty() {
        unsafe {
            for &command_buffer in &data.compute_command_buffers {
                if command_buffer != vk::CommandBuffer::null()
                    && data.command_pool != vk::CommandPool::null()
                {
                    device.free_command_buffers(data.command_pool, &[command_buffer]);
                }
            }
        }
        data.compute_command_buffers.clear();
    }

    // 为每个飞行帧分配计算命令缓冲区
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .map_err(|e| anyhow!("分配计算命令缓冲区 {} 失败: {}", i, e))?[0]
        };

        data.compute_command_buffers.push(command_buffer);
    }

    info!("计算命令缓冲区创建完成: {} 个", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

/// 开始一次性命令缓冲区
/// 用于短期操作的命令缓冲区
pub fn begin_single_time_commands(device: &Device, data: &AppData) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| anyhow!("分配一次性命令缓冲区失败: {}", e))?[0]
    };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(|e| anyhow!("开始命令缓冲区记录失败: {}", e))?
    };

    Ok(command_buffer)
}

/// 结束并提交一次性命令缓冲区
/// 完成命令记录，提交执行并等待完成
pub fn end_single_time_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .map_err(|e| anyhow!("结束命令缓冲区记录失败: {}", e))?
    };

    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

    unsafe {
        device
            .queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())
            .map_err(|e| anyhow!("提交命令缓冲区失败: {}", e))?;
        device
            .queue_wait_idle(data.graphics_queue)
            .map_err(|e| anyhow!("等待队列空闲失败: {}", e))?;
        device.free_command_buffers(data.command_pool, &[command_buffer]);
    }

    Ok(())
}
