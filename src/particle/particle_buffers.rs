//! 粒子缓冲区管理模块
//! 负责粒子存储缓冲区和统一缓冲区的创建和管理

use anyhow::Result;
use ash::{Device, Instance, vk};
use std::mem::size_of;

use crate::constants::{MAX_FRAMES_IN_FLIGHT, PARTICLE_COUNT, Vec2, Vec4};
use crate::resources::{copy_buffer, create_buffer, write_buffer_data};
use crate::types::{AppData, Particle, ParticleUBO};

/// 创建粒子存储缓冲区
/// 初始化粒子数据并为每个飞行帧创建存储缓冲区
pub fn create_particle_storage_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // 初始化随机数生成器
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut rng = StdRng::seed_from_u64(seed);

    // 初始化粒子数据，在圆形区域内分布
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    for _ in 0..PARTICLE_COUNT {
        let r = 0.25f32 * (rng.random::<f32>()).sqrt();
        let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let height_width_ratio =
            data.swapchain_extent.height as f32 / data.swapchain_extent.width as f32;
        let x = r * theta.cos() * height_width_ratio;
        let y = r * theta.sin();

        // 创建粒子实例
        let position = Vec2::new(x, y);
        let velocity = Vec2::new(x, y).normalize() * 0.00025f32;
        let color = Vec4::new(rng.random(), rng.random(), rng.random(), 1.0);

        particles.push(Particle::new(position, velocity, color));
    }

    let buffer_size = (size_of::<Particle>() * PARTICLE_COUNT) as vk::DeviceSize;

    // 创建暂存缓冲区用于上传初始数据
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传粒子数据到暂存缓冲区
    write_buffer_data(device, staging_buffer_memory, &particles)?;

    // 为每个飞行帧创建粒子存储缓冲区
    data.particle_storage_buffers
        .resize(MAX_FRAMES_IN_FLIGHT, vk::Buffer::null());
    data.particle_storage_buffers_memory
        .resize(MAX_FRAMES_IN_FLIGHT, vk::DeviceMemory::null());

    // 将初始数据复制到所有存储缓冲区
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let (buffer, buffer_memory) = create_buffer(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.particle_storage_buffers[i] = buffer;
        data.particle_storage_buffers_memory[i] = buffer_memory;

        copy_buffer(device, data, staging_buffer, buffer, buffer_size)?;
    }

    // 清理暂存缓冲区
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    log::info!(
        "粒子存储缓冲区创建完成: {MAX_FRAMES_IN_FLIGHT} 个缓冲区，每个包含 {PARTICLE_COUNT} 粒子"
    );
    Ok(())
}

/// 创建粒子统一缓冲区
/// 为每个飞行帧创建统一缓冲区（时间信息）
pub fn create_particle_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let buffer_size = size_of::<ParticleUBO>() as vk::DeviceSize;

    // 清理已有的粒子统一缓冲区
    cleanup_particle_uniform_buffers(device, data);

    // 为每个飞行帧创建统一缓冲区
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let (buffer, buffer_memory) = create_buffer(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        data.particle_uniform_buffers.push(buffer);
        data.particle_uniform_buffers_memory.push(buffer_memory);
    }

    log::info!("粒子统一缓冲区创建完成: {MAX_FRAMES_IN_FLIGHT} 个");
    Ok(())
}

/// 清理粒子统一缓冲区
fn cleanup_particle_uniform_buffers(device: &Device, data: &mut AppData) {
    unsafe {
        for memory in data.particle_uniform_buffers_memory.drain(..) {
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        }
        for buffer in data.particle_uniform_buffers.drain(..) {
            if buffer != vk::Buffer::null() {
                device.destroy_buffer(buffer, None);
            }
        }
    }
}
