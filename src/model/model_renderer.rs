//! 模型渲染逻辑模块
//! 负责模型的实际渲染操作

use anyhow::Result;
use ash::vk;
use nalgebra::{Point3, Unit};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use crate::types::ModelUBO;
use crate::vulkan_app::VulkanApp;

use crate::constants::{Mat4, Vec3};

/// 更新模型统一缓冲区
/// 计算并上传视图和投影矩阵
pub fn update_model_uniform_buffer(app: &VulkanApp, image_index: usize) -> Result<()> {
    // 相机设置 (视图矩阵)
    let eye_position = Point3::new(0.0, 3.0, 3.0);
    let target_position = Point3::origin();
    let up_vector = Vec3::z_axis();
    let view_matrix = Mat4::look_at_rh(&eye_position, &target_position, &up_vector);

    // 投影矩阵设置
    let aspect = app.data.swapchain_extent.width as f32 / app.data.swapchain_extent.height as f32;
    let near_plane = 0.1;
    let far_plane = 100.0;
    let mut proj_matrix =
        Mat4::new_perspective(aspect, 45.0f32.to_radians(), near_plane, far_plane);

    // Vulkan Y轴翻转校正
    proj_matrix[(1, 1)] *= -1.0;

    // Vulkan深度范围 [0, 1] 校正
    let vk_depth_correction = Mat4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
    );
    proj_matrix = vk_depth_correction * proj_matrix;

    let ubo = ModelUBO {
        view: view_matrix,
        proj: proj_matrix,
    };

    // 映射内存并更新数据
    unsafe {
        let memory = app.device.map_memory(
            app.data.model_uniform_buffers_memory[image_index],
            0,
            size_of::<ModelUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(
            &ubo as *const _ as *const u8,
            memory.cast(),
            size_of::<ModelUBO>(),
        );

        app.device
            .unmap_memory(app.data.model_uniform_buffers_memory[image_index]);
    }

    Ok(())
}

/// 更新模型二级命令缓冲区
/// 为特定模型索引录制渲染命令
pub fn update_model_secondary_command_buffer(
    app: &mut VulkanApp,
    image_index: usize,
    model_index: usize,
) -> Result<vk::CommandBuffer> {
    // 确保有足够的二级命令缓冲区
    let command_buffers = &mut app.data.secondary_command_buffers[image_index];
    while model_index >= command_buffers.len() {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(app.data.command_pools[image_index])
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { app.device.allocate_command_buffers(&allocate_info)?[0] };
        command_buffers.push(command_buffer);
    }

    let command_buffer = command_buffers[model_index];

    // 计算模型变换矩阵
    let model_matrix = calculate_model_transform_matrix(app, model_index);
    let opacity = calculate_model_opacity(app, model_index);

    // 录制命令缓冲区
    record_model_secondary_commands(app, command_buffer, image_index, &model_matrix, opacity)?;

    Ok(command_buffer)
}

/// 计算模型变换矩阵
/// 为每个模型生成不同的位置、旋转和缩放
pub fn calculate_model_transform_matrix(app: &VulkanApp, model_index: usize) -> Mat4 {
    // 在圆形上布置模型
    let radius = 2.5;
    let angle = (model_index as f32) * (2.0 * std::f32::consts::PI / app.models as f32);
    let x = radius * angle.cos();
    let z = radius * angle.sin();
    let y = 0.5 * (model_index % 2) as f32; // 交替高度

    // 旋转动画
    let time = app.start.elapsed().as_secs_f32();
    let rotation_speed = 0.1;
    let individual_rotation = time * rotation_speed + (model_index as f32 * 0.5);

    // 不同的旋转轴
    let rotation_axes = [
        Unit::new_normalize(Vec3::new(0.0, 0.0, 1.0)), // Z轴
        Unit::new_normalize(Vec3::new(0.0, 1.0, 0.0)), // Y轴
        Unit::new_normalize(Vec3::new(1.0, 0.0, 0.0)), // X轴
        Unit::new_normalize(Vec3::new(1.0, 1.0, 1.0)), // 对角线
    ];
    let rotation_axis = rotation_axes[model_index % rotation_axes.len()];

    // 不同的缩放
    let scale_variation = 0.8 + (model_index % 3) as f32 * 0.1;
    let scale_factor = 1.5 * scale_variation;

    // 组合变换: 缩放 -> 旋转 -> 平移
    let scale_matrix = Mat4::new_scaling(scale_factor);
    let rotation_matrix = Mat4::from_axis_angle(&rotation_axis, individual_rotation);
    let translation_matrix = Mat4::new_translation(&Vec3::new(x, y, z));

    translation_matrix * rotation_matrix * scale_matrix
}

/// 计算模型透明度
/// 为每个模型分配不同的透明度值
pub fn calculate_model_opacity(app: &VulkanApp, model_index: usize) -> f32 {
    0.7 + (0.3 * model_index as f32 / app.models.max(1) as f32)
}

/// 录制模型二级命令
/// 将模型渲染命令录制到二级命令缓冲区
pub fn record_model_secondary_commands(
    app: &VulkanApp,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
    model_matrix: &Mat4,
    opacity: f32,
) -> Result<()> {
    // 准备推送常量数据
    let model_bytes = unsafe {
        std::slice::from_raw_parts(model_matrix as *const Mat4 as *const u8, size_of::<Mat4>())
    };
    let opacity_bytes = &opacity.to_ne_bytes()[..];

    // 继承信息
    let inheritance_info = vk::CommandBufferInheritanceInfo::default()
        .render_pass(app.data.render_pass)
        .subpass(0)
        .framebuffer(app.data.framebuffers[image_index]);

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
        .inheritance_info(&inheritance_info);

    unsafe {
        app.device
            .begin_command_buffer(command_buffer, &begin_info)?;

        // 绑定模型管线
        app.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app.data.model_pipeline,
        );

        // 绑定顶点和索引缓冲区
        app.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[app.data.vertex_buffer], &[0]);
        app.device.cmd_bind_index_buffer(
            command_buffer,
            app.data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        // 绑定描述符集
        app.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app.data.model_pipeline_layout,
            0,
            &[app.data.model_descriptor_sets[image_index]],
            &[],
        );

        // 推送常量 - 模型矩阵
        app.device.cmd_push_constants(
            command_buffer,
            app.data.model_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );

        // 推送常量 - 透明度
        app.device.cmd_push_constants(
            command_buffer,
            app.data.model_pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );

        // 绘制索引化模型
        app.device
            .cmd_draw_indexed(command_buffer, app.data.indices.len() as u32, 1, 0, 0, 0);

        app.device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}

/// 渲染所有模型
/// 使用二级命令缓冲区执行模型渲染
pub fn render_all_models(
    app: &mut VulkanApp,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    if app.models == 0 {
        return Ok(());
    }

    // 更新所有模型的二级命令缓冲区
    let secondary_command_buffers = (0..app.models)
        .map(|i| update_model_secondary_command_buffer(app, image_index, i))
        .collect::<Result<Vec<_>, _>>()?;

    // 执行二级命令缓冲区
    unsafe {
        app.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers);
    }

    Ok(())
}
