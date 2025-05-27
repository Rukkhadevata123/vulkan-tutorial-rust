//! 粒子渲染逻辑模块
//! 负责粒子的实际渲染操作和数据更新

use anyhow::Result;
use ash::vk;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use crate::constants::PARTICLE_COUNT;
use crate::types::ParticleUBO;
use crate::vulkan_app::VulkanApp;

/// 更新粒子统一缓冲区
/// 上传时间信息到GPU用于粒子物理模拟
pub fn update_particle_uniform_buffer(app: &VulkanApp) -> Result<()> {
    let current_time = app.start.elapsed().as_secs_f32();

    // 计算帧间时间差
    let delta_time = if app.last_time > 0.0 {
        // 保持与34版本的时间计算逻辑一致
        (current_time - app.last_time as f32) * 1000.0 * 2.0
    } else {
        16.0 * 2.0 // 默认60fps的帧时间（毫秒）
    };

    let ubo = ParticleUBO {
        delta_time,
        time: current_time * 1000.0, // 转换为毫秒，匹配着色器中的使用
    };

    let buffer_index = app.frame;

    // 映射内存并更新数据
    unsafe {
        let memory = app.device.map_memory(
            app.data.particle_uniform_buffers_memory[buffer_index],
            0,
            size_of::<ParticleUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(
            &ubo as *const _ as *const u8,
            memory.cast(),
            size_of::<ParticleUBO>(),
        );

        app.device
            .unmap_memory(app.data.particle_uniform_buffers_memory[buffer_index]);
    }

    Ok(())
}

/// 渲染粒子系统
/// 在图形渲染通道中渲染粒子
pub fn render_particles(app: &VulkanApp, command_buffer: vk::CommandBuffer) -> Result<()> {
    unsafe {
        // 绑定粒子图形管线
        app.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app.data.particle_pipeline,
        );

        // 设置视口和裁剪区域
        let viewport = vk::Viewport::default()
            .width(app.data.swapchain_extent.width as f32)
            .height(app.data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        app.device
            .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

        let scissor = vk::Rect2D::default().extent(app.data.swapchain_extent);
        app.device
            .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

        // 绑定粒子顶点缓冲区（存储缓冲区用作顶点缓冲区）
        let vertex_buffers = [app.data.particle_storage_buffers[app.frame]];
        let offsets = [0];
        app.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

        // 绘制粒子
        app.device
            .cmd_draw(command_buffer, PARTICLE_COUNT as u32, 1, 0, 0);
    }

    Ok(())
}
