//! 同步对象管理模块
//! 负责创建和管理信号量、围栏等同步原语

use anyhow::{Result, anyhow};
use ash::Device;
use ash::vk;
use log::*;

use crate::constants::MAX_FRAMES_IN_FLIGHT;
use crate::types::AppData;

/// 创建同步对象
/// 为每个飞行帧创建信号量和围栏
pub fn vulkan_create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的同步对象
    vulkan_cleanup_sync_objects(device, data);

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    // 为每个飞行帧创建同步对象
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        unsafe {
            data.image_available_semaphores.push(
                device
                    .create_semaphore(&semaphore_info, None)
                    .map_err(|e| anyhow!("创建图像可用信号量 {} 失败: {}", i, e))?,
            );
            data.render_finished_semaphores.push(
                device
                    .create_semaphore(&semaphore_info, None)
                    .map_err(|e| anyhow!("创建渲染完成信号量 {} 失败: {}", i, e))?,
            );
            data.compute_finished_semaphores.push(
                device
                    .create_semaphore(&semaphore_info, None)
                    .map_err(|e| anyhow!("创建计算完成信号量 {} 失败: {}", i, e))?,
            );
            data.in_flight_fences.push(
                device
                    .create_fence(&fence_info, None)
                    .map_err(|e| anyhow!("创建飞行围栏 {} 失败: {}", i, e))?,
            );
        }
    }

    // 初始化交换链图像的围栏跟踪
    data.images_in_flight = vec![vk::Fence::null(); data.swapchain_images.len()];

    info!("同步对象创建完成: {} 个飞行帧", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

/// 清理同步对象
/// 安全销毁所有同步对象
pub fn vulkan_cleanup_sync_objects(device: &Device, data: &mut AppData) {
    unsafe {
        for semaphore in data.image_available_semaphores.drain(..) {
            if semaphore != vk::Semaphore::null() {
                device.destroy_semaphore(semaphore, None);
            }
        }
        for semaphore in data.render_finished_semaphores.drain(..) {
            if semaphore != vk::Semaphore::null() {
                device.destroy_semaphore(semaphore, None);
            }
        }
        for semaphore in data.compute_finished_semaphores.drain(..) {
            if semaphore != vk::Semaphore::null() {
                device.destroy_semaphore(semaphore, None);
            }
        }
        for fence in data.in_flight_fences.drain(..) {
            if fence != vk::Fence::null() {
                device.destroy_fence(fence, None);
            }
        }
    }
}
