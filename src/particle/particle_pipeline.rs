//! 粒子管线管理模块
//! 负责创建粒子图形管线和计算管线

use anyhow::{Result, anyhow};
use ash::{Device, vk};

use crate::resources::create_shader_module;
use crate::types::{AppData, Particle};

/// 创建粒子图形管线
/// 配置粒子渲染的图形管线
pub fn create_particle_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载粒子着色器字节码
    let vert_shader_spirv = include_bytes!("../../assets/shaders/35_particle.vert.spv");
    let frag_shader_spirv = include_bytes!("../../assets/shaders/35_particle.frag.spv");

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

    // 顶点输入状态 - 使用粒子结构
    let binding_description = Particle::binding_description();
    let attribute_descriptions = Particle::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
        .vertex_attribute_descriptions(&attribute_descriptions);

    // 输入组装状态 - 粒子使用点列表
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::POINT_LIST)
        .primitive_restart_enable(false);

    // 视口状态 - 使用动态状态
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    // 光栅化状态
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // 多重采样状态
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(true)
        .min_sample_shading(0.2)
        .rasterization_samples(data.msaa_samples);

    // 深度测试状态 - 粒子需要深度测试但不写入深度
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false) // 粒子通常不写入深度缓冲
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // 颜色混合状态 - 粒子使用alpha混合
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&color_blend_attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // 动态状态
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // 创建粒子管线布局（空布局，粒子不需要额外描述符）
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&[]);

    data.particle_pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| anyhow!("创建粒子管线布局失败: {}", e))?
    };

    // 创建粒子图形管线
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(data.particle_pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null());

    data.particle_pipeline = unsafe {
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(anyhow!("创建粒子图形管线失败: {}", err));
            }
        }
    };

    // 销毁着色器模块
    unsafe {
        if frag_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(frag_shader_module, None);
        }
        if vert_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(vert_shader_module, None);
        }
    }

    log::info!("粒子图形管线创建完成");
    Ok(())
}

/// 创建粒子计算管线
/// 配置粒子物理模拟的计算管线
pub fn create_particle_compute_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载计算着色器字节码
    let compute_shader_spirv = include_bytes!("../../assets/shaders/35_particle.comp.spv");

    // 创建计算着色器模块
    let compute_shader_module = create_shader_module(device, compute_shader_spirv)?;

    let main_function_name = c"main";

    // 计算着色器阶段配置
    let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module)
        .name(main_function_name);

    // 计算管线布局 - 使用粒子描述符集布局
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&data.particle_descriptor_set_layout));

    data.particle_compute_pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| anyhow!("创建粒子计算管线布局失败: {}", e))?
    };

    // 创建计算管线
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(compute_shader_stage_info)
        .layout(data.particle_compute_pipeline_layout);

    data.particle_compute_pipeline = unsafe {
        match device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(anyhow!("创建粒子计算管线失败: {}", err));
            }
        }
    };

    // 销毁着色器模块
    unsafe {
        if compute_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(compute_shader_module, None);
        }
    }

    log::info!("粒子计算管线创建完成");
    Ok(())
}
