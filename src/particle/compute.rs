//! 粒子计算着色器模块
//! 负责粒子物理模拟的计算着色器执行

use anyhow::Result;
use ash::{Device, vk};

use crate::constants::PARTICLE_COUNT;
use crate::types::AppData;

/// 录制粒子计算命令缓冲区
/// 将粒子物理模拟命令录制到计算命令缓冲区
pub fn record_particle_compute_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
    current_frame: usize,
) -> Result<()> {
    // 重置命令缓冲区
    unsafe {
        device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
    }

    let begin_info = vk::CommandBufferBeginInfo::default();

    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;

        // 绑定计算管线
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.particle_compute_pipeline,
        );

        // 绑定描述符集
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.particle_compute_pipeline_layout,
            0,
            &[data.particle_descriptor_sets[current_frame]],
            &[],
        );

        // 分派计算工作组
        let workgroup_count = (PARTICLE_COUNT as u32).div_ceil(256); // 向上取整到256的倍数
        device.cmd_dispatch(command_buffer, workgroup_count, 1, 1);

        device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}
