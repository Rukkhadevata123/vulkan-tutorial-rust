//! 粒子描述符管理模块
//! 负责创建和管理粒子系统所需的描述符资源

use anyhow::{Result, anyhow};
use ash::{Device, vk};
use std::mem::size_of;

use crate::constants::{MAX_FRAMES_IN_FLIGHT, PARTICLE_COUNT};
use crate::resources::{DescriptorPoolConfig, create_descriptor_pool};
use crate::types::{AppData, Particle, ParticleUBO};

/// 创建粒子描述符集布局
/// 定义计算着色器使用的描述符绑定
pub fn create_particle_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let layout_bindings = [
        // 绑定0: 统一缓冲区 (时间信息)
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 绑定1: 存储缓冲区（当前粒子状态，输入）
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 绑定2: 存储缓冲区（新的粒子状态，输出）
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);

    data.particle_descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| anyhow!("创建粒子描述符集布局失败: {}", e))?
    };

    log::info!("粒子描述符集布局创建完成");
    Ok(())
}

/// 创建粒子描述符池
/// 为粒子描述符集分配池空间
pub fn create_particle_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的描述符池
    if data.particle_descriptor_pool != vk::DescriptorPool::null() {
        unsafe {
            device.destroy_descriptor_pool(data.particle_descriptor_pool, None);
        }
        data.particle_descriptor_pool = vk::DescriptorPool::null();
        data.particle_descriptor_sets.clear();
    }

    // 配置描述符池
    let config = DescriptorPoolConfig::new()
        .add_pool_size(
            vk::DescriptorType::UNIFORM_BUFFER,
            MAX_FRAMES_IN_FLIGHT as u32,
        )
        .add_pool_size(
            vk::DescriptorType::STORAGE_BUFFER,
            MAX_FRAMES_IN_FLIGHT as u32 * 2,
        )
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    data.particle_descriptor_pool = create_descriptor_pool(device, config)?;

    log::info!("粒子描述符池创建完成: 最大集合数 {MAX_FRAMES_IN_FLIGHT}");
    Ok(())
}

/// 创建并更新粒子描述符集
/// 为每个飞行帧分配并配置描述符集
pub fn create_particle_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    if data.particle_descriptor_pool == vk::DescriptorPool::null() {
        return Ok(());
    }

    data.particle_descriptor_sets.clear();

    // 为每个飞行帧准备相同的布局
    let layouts = vec![data.particle_descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];

    // 分配描述符集
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(data.particle_descriptor_pool)
        .set_layouts(&layouts);

    data.particle_descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| anyhow!("分配粒子描述符集失败: {}", e))?
    };

    // 为每个飞行帧更新描述符集
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        update_particle_descriptor_set(device, data, i)?;
    }

    log::info!("粒子描述符集创建完成: {MAX_FRAMES_IN_FLIGHT} 个");
    Ok(())
}

/// 更新单个粒子描述符集
/// 绑定统一缓冲区和存储缓冲区到描述符集
pub fn update_particle_descriptor_set(
    device: &Device,
    data: &AppData,
    frame_index: usize,
) -> Result<()> {
    // 统一缓冲区信息（时间数据）
    let uniform_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(data.particle_uniform_buffers[frame_index])
        .offset(0)
        .range(size_of::<ParticleUBO>() as u64);

    // 输入存储缓冲区（上一帧的粒子状态）
    let prev_frame = (frame_index + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
    let storage_buffer_info_input = vk::DescriptorBufferInfo::default()
        .buffer(data.particle_storage_buffers[prev_frame])
        .offset(0)
        .range((size_of::<Particle>() * PARTICLE_COUNT) as u64);

    // 输出存储缓冲区（当前帧的粒子状态）
    let storage_buffer_info_output = vk::DescriptorBufferInfo::default()
        .buffer(data.particle_storage_buffers[frame_index])
        .offset(0)
        .range((size_of::<Particle>() * PARTICLE_COUNT) as u64);

    // 描述符写入操作
    let descriptor_writes = [
        // 绑定0: 统一缓冲区
        vk::WriteDescriptorSet::default()
            .dst_set(data.particle_descriptor_sets[frame_index])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&uniform_buffer_info)),
        // 绑定1: 输入存储缓冲区
        vk::WriteDescriptorSet::default()
            .dst_set(data.particle_descriptor_sets[frame_index])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&storage_buffer_info_input)),
        // 绑定2: 输出存储缓冲区
        vk::WriteDescriptorSet::default()
            .dst_set(data.particle_descriptor_sets[frame_index])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&storage_buffer_info_output)),
    ];

    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(())
}
