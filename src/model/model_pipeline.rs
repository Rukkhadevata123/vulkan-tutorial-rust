//! 模型渲染管线模块
//! 负责创建和配置模型渲染的图形管线

use anyhow::{Result, anyhow};
use ash::{Device, vk};

use crate::resources::create_shader_module;
use crate::types::{AppData, ModelVertex};

/// 创建模型图形管线
/// 配置顶点输入、着色器阶段和渲染状态
pub fn create_model_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载着色器字节码
    let vert_shader_spirv = include_bytes!("../../assets/shaders/35_viking_room.vert.spv");
    let frag_shader_spirv = include_bytes!("../../assets/shaders/35_viking_room.frag.spv");

    // 创建着色器模块
    let vert_shader_module = create_shader_module(device, vert_shader_spirv)?;
    let frag_shader_module = create_shader_module(device, frag_shader_spirv)?;

    let main_function_name = c"main";

    // 着色器阶段配置
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(main_function_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(main_function_name),
    ];

    // 顶点输入状态
    let binding_descriptions = [ModelVertex::binding_description()];
    let attribute_descriptions = ModelVertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // 输入组装状态
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // 视口状态
    let viewport = vk::Viewport::default()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    // 光栅化状态
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE) // 不背面剔除，显示模型内部
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // 多重采样状态
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(true)
        .min_sample_shading(0.2)
        .rasterization_samples(data.msaa_samples);

    // 深度测试状态
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false);

    // 颜色混合状态 - 支持alpha混合
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&color_blend_attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // 推送常量范围配置
    let push_constant_ranges = [
        // 顶点着色器: 模型矩阵 (64字节)
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64),
        // 片段着色器: 透明度 (4字节)
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(64)
            .size(4),
    ];

    // 管线布局
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&data.model_descriptor_set_layout))
        .push_constant_ranges(&push_constant_ranges);

    data.model_pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| anyhow!("创建模型管线布局失败: {}", e))?
    };

    // 创建图形管线
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.model_pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.model_pipeline = unsafe {
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                // 清理部分创建的管线
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(anyhow!("创建模型图形管线失败: {}", err));
            }
        }
    };

    // 销毁着色器模块
    unsafe {
        if vert_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(vert_shader_module, None);
        }
        if frag_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(frag_shader_module, None);
        }
    }

    log::info!("模型图形管线创建完成");
    Ok(())
}
