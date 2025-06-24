//! VulkanApp核心结构和方法模块
//! 包含Vulkan应用程序的主要逻辑和生命周期管理

use anyhow::Result;
use ash::vk::Handle;
use ash::{Device, Entry, Instance};
use std::time::Instant;
use winit::window::Window;

use crate::constants::*;
use crate::types::*;

// 模型模块函数导入
use crate::model::{
    ModelConfig, create_model_descriptor_pool, create_model_descriptor_set_layout,
    create_model_descriptor_sets, create_model_index_buffer, create_model_pipeline,
    create_model_uniform_buffers, create_model_vertex_buffer, load_model_data, render_all_models,
    update_model_uniform_buffer,
};

// 粒子系统模块函数导入
use crate::particle::{
    create_particle_compute_pipeline, create_particle_descriptor_pool,
    create_particle_descriptor_set_layout, create_particle_descriptor_sets,
    create_particle_pipeline, create_particle_storage_buffers, create_particle_uniform_buffers,
    record_particle_compute_commands, render_particles, update_particle_uniform_buffer,
};

// 纹理模块函数导入
use crate::texture::{
    SamplerConfig, TextureConfig, create_texture_image_view, create_texture_sampler,
    generate_texture_mipmaps, load_texture,
};

// Vulkan核心模块函数导入
use crate::vulkan::{
    vulkan_create_color_objects, vulkan_create_command_buffers, vulkan_create_command_pools,
    vulkan_create_compute_command_buffers, vulkan_create_depth_objects, vulkan_create_framebuffers,
    vulkan_create_instance, vulkan_create_logical_device, vulkan_create_render_pass,
    vulkan_create_swapchain, vulkan_create_swapchain_image_views, vulkan_create_sync_objects,
    vulkan_pick_physical_device,
};

// 窗口系统函数导入
use crate::create_surface;

/// Vulkan应用程序主类
/// 管理整个应用程序的生命周期和渲染流程
#[derive(Clone)]
pub struct VulkanApp {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub last_time: f64,
    pub models: usize,
}

impl VulkanApp {
    /// 初始化Vulkan应用程序
    /// 按正确顺序创建所有Vulkan对象和资源
    pub fn create(window: &Window) -> Result<Self> {
        let entry = unsafe {
            Entry::load().map_err(|e| anyhow::anyhow!("无法加载Vulkan入口点: {}", e))?
        };
        let mut data = AppData::default();

        // 核心Vulkan初始化
        let instance = vulkan_create_instance(window, &entry, &mut data)?;
        data.surface = unsafe { create_surface(&instance, &entry, &window, &window)? };

        // 设备和队列设置
        vulkan_pick_physical_device(&instance, &entry, &mut data)?;
        let device = vulkan_create_logical_device(&entry, &instance, &mut data)?;

        // 交换链和渲染资源
        vulkan_create_swapchain(window, &instance, &device, &entry, &mut data)?;
        vulkan_create_swapchain_image_views(&device, &mut data)?;
        vulkan_create_render_pass(&instance, &device, &mut data)?;

        // 描述符布局
        create_model_descriptor_set_layout(&device, &mut data)?;
        create_particle_descriptor_set_layout(&device, &mut data)?;

        // 管线创建
        create_model_pipeline(&device, &mut data)?;
        create_particle_pipeline(&device, &mut data)?;
        create_particle_compute_pipeline(&device, &mut data)?;

        // 命令和缓冲区
        vulkan_create_command_pools(&instance, &device, &entry, &mut data)?;
        vulkan_create_color_objects(&instance, &device, &mut data)?;
        vulkan_create_depth_objects(&instance, &device, &mut data)?;
        vulkan_create_framebuffers(&device, &mut data)?;

        // 纹理资源
        let config =
            TextureConfig::new("assets/textures/viking_room.png").with_expected_size(1024, 1024);
        load_texture(&instance, &device, &mut data, config)?;
        generate_texture_mipmaps(
            &instance,
            &device,
            &data,
            1024,
            1024,
            ash::vk::Format::R8G8B8A8_SRGB,
        )?;
        create_texture_image_view(&device, &mut data, ash::vk::Format::R8G8B8A8_SRGB)?;

        let sampler_config = SamplerConfig::new()
            .with_anisotropy(true, 16.0)
            .with_mip_range(0.0, data.mip_levels as f32);
        create_texture_sampler(&device, &instance, &mut data, sampler_config)?;

        // 模型资源
        let model_config = ModelConfig::new("assets/models/viking_room.obj")
            .with_default_color(Vec3::new(1.0, 1.0, 1.0));
        load_model_data(&mut data, model_config)?;
        create_model_vertex_buffer(&instance, &device, &mut data)?;
        create_model_index_buffer(&instance, &device, &mut data)?;
        create_model_uniform_buffers(&instance, &device, &mut data)?;

        // 粒子资源
        create_particle_storage_buffers(&instance, &device, &mut data)?;
        create_particle_uniform_buffers(&instance, &device, &mut data)?;

        // 描述符资源
        create_model_descriptor_pool(&device, &mut data)?;
        create_particle_descriptor_pool(&device, &mut data)?;
        create_model_descriptor_sets(&device, &mut data)?;
        create_particle_descriptor_sets(&device, &mut data)?;

        // 命令缓冲区和同步
        vulkan_create_command_buffers(&device, &mut data)?;
        vulkan_create_compute_command_buffers(&device, &mut data)?;
        vulkan_create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            last_time: 0.0,
            models: 1,
        })
    }

    /// 销毁Vulkan应用程序
    /// 确保按正确顺序销毁所有资源，避免验证层错误
    pub fn destroy(&mut self) {
        unsafe {
            // 等待设备空闲，确保不再有任何操作正在进行
            self.device.device_wait_idle().expect("等待设备空闲失败");

            // 1. 先销毁交换链及其相关资源
            self.cleanup_swapchain_resources();

            // 2. 销毁同步对象
            self.cleanup_sync_objects();

            // 3. 销毁命令池
            self.cleanup_command_pools();

            // 4. 销毁模型相关资源
            self.cleanup_model_resources();

            // 5. 销毁粒子系统资源
            self.cleanup_particle_resources();

            // 6. 销毁纹理资源
            self.cleanup_texture_resources();

            // 7. 销毁描述符集布局
            self.cleanup_descriptor_layouts();

            // 8. 销毁逻辑设备
            self.device.destroy_device(None);

            // 9. 销毁表面
            self.cleanup_surface();

            // 10. 销毁调试信使
            self.cleanup_debug_messenger();

            // 11. 销毁Vulkan实例
            self.instance.destroy_instance(None);
        }
    }

    /// 重新创建交换链（窗口大小改变时调用）
    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };

        self.cleanup_swapchain_resources();

        // 重新创建交换链及其依赖资源
        vulkan_create_swapchain(
            window,
            &self.instance,
            &self.device,
            &self.entry,
            &mut self.data,
        )?;
        vulkan_create_swapchain_image_views(&self.device, &mut self.data)?;
        vulkan_create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_model_pipeline(&self.device, &mut self.data)?;
        create_particle_pipeline(&self.device, &mut self.data)?;
        create_particle_compute_pipeline(&self.device, &mut self.data)?;
        vulkan_create_color_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan_create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan_create_framebuffers(&self.device, &mut self.data)?;
        create_model_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_particle_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_model_descriptor_pool(&self.device, &mut self.data)?;
        create_particle_descriptor_pool(&self.device, &mut self.data)?;
        create_model_descriptor_sets(&self.device, &mut self.data)?;
        create_particle_descriptor_sets(&self.device, &mut self.data)?;
        vulkan_create_command_buffers(&self.device, &mut self.data)?;
        vulkan_create_compute_command_buffers(&self.device, &mut self.data)?;

        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), ash::vk::Fence::null());
        Ok(())
    }

    /// 主渲染函数
    /// 协调整个渲染管线的执行
    pub fn render(&mut self, window: &Window) -> Result<()> {
        let current_time = self.start.elapsed().as_secs_f64();
        self.last_time = current_time;

        let in_flight_fence = self.data.in_flight_fences[self.frame];

        // 等待当前帧的围栏
        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
        }

        // 获取下一个交换链图像
        let image_index = self.acquire_next_swapchain_image(window)?;
        if image_index.is_none() {
            return Ok(()); // 交换链需要重建
        }
        let image_index = image_index.unwrap();

        // 检查图像是否正在使用
        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            unsafe {
                self.device
                    .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
            }
        }
        self.data.images_in_flight[image_index] = in_flight_fence;

        // 更新缓冲区数据
        self.update_frame_data(image_index)?;

        // 提交渲染命令
        self.submit_render_commands(image_index)?;

        // 呈现结果
        self.present_frame(window, image_index)?;

        // 更新帧索引
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}

//==================================================================================================
// 资源清理辅助方法
//==================================================================================================

impl VulkanApp {
    /// 清理交换链相关资源
    fn cleanup_swapchain_resources(&mut self) {
        unsafe {
            // 清理描述符池
            if self.data.model_descriptor_pool != ash::vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.model_descriptor_pool, None);
                self.data.model_descriptor_pool = ash::vk::DescriptorPool::null();
                self.data.model_descriptor_sets.clear();
            }

            if self.data.particle_descriptor_pool != ash::vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.particle_descriptor_pool, None);
                self.data.particle_descriptor_pool = ash::vk::DescriptorPool::null();
                self.data.particle_descriptor_sets.clear();
            }

            // 清理统一缓冲区
            self.cleanup_uniform_buffers();

            // 清理帧缓冲区
            for &framebuffer in &self.data.framebuffers {
                if framebuffer != ash::vk::Framebuffer::null() {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
            }
            self.data.framebuffers.clear();

            // 清理MSAA和深度资源
            self.cleanup_msaa_resources();
            self.cleanup_depth_resources();

            // 清理管线
            self.cleanup_pipelines();

            // 清理渲染通道
            if self.data.render_pass != ash::vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = ash::vk::RenderPass::null();
            }

            // 清理交换链图像视图
            for &image_view in &self.data.swapchain_image_views {
                if image_view != ash::vk::ImageView::null() {
                    self.device.destroy_image_view(image_view, None);
                }
            }
            self.data.swapchain_image_views.clear();

            // 清理交换链
            if self.data.swapchain != ash::vk::SwapchainKHR::null() {
                let swapchain_device =
                    ash::khr::swapchain::Device::new(&self.instance, &self.device);
                swapchain_device.destroy_swapchain(self.data.swapchain, None);
                self.data.swapchain = ash::vk::SwapchainKHR::null();
            }
            self.data.swapchain_images.clear();

            // 清理命令缓冲区
            self.cleanup_command_buffers();
        }
    }

    /// 清理统一缓冲区
    fn cleanup_uniform_buffers(&mut self) {
        unsafe {
            // 模型统一缓冲区
            for &memory in &self.data.model_uniform_buffers_memory {
                if memory != ash::vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.model_uniform_buffers {
                if buffer != ash::vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
            self.data.model_uniform_buffers.clear();
            self.data.model_uniform_buffers_memory.clear();

            // 粒子统一缓冲区
            for &memory in &self.data.particle_uniform_buffers_memory {
                if memory != ash::vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.particle_uniform_buffers {
                if buffer != ash::vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
            self.data.particle_uniform_buffers.clear();
            self.data.particle_uniform_buffers_memory.clear();
        }
    }

    /// 清理MSAA颜色资源
    fn cleanup_msaa_resources(&mut self) {
        unsafe {
            if self.data.color_image_view != ash::vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.color_image_view, None);
                self.data.color_image_view = ash::vk::ImageView::null();
            }
            if self.data.color_image != ash::vk::Image::null() {
                self.device.destroy_image(self.data.color_image, None);
                self.data.color_image = ash::vk::Image::null();
            }
            if self.data.color_image_memory != ash::vk::DeviceMemory::null() {
                self.device.free_memory(self.data.color_image_memory, None);
                self.data.color_image_memory = ash::vk::DeviceMemory::null();
            }
        }
    }

    /// 清理深度缓冲区资源
    fn cleanup_depth_resources(&mut self) {
        unsafe {
            if self.data.depth_image_view != ash::vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.depth_image_view, None);
                self.data.depth_image_view = ash::vk::ImageView::null();
            }
            if self.data.depth_image != ash::vk::Image::null() {
                self.device.destroy_image(self.data.depth_image, None);
                self.data.depth_image = ash::vk::Image::null();
            }
            if self.data.depth_image_memory != ash::vk::DeviceMemory::null() {
                self.device.free_memory(self.data.depth_image_memory, None);
                self.data.depth_image_memory = ash::vk::DeviceMemory::null();
            }
        }
    }

    /// 清理渲染管线
    fn cleanup_pipelines(&mut self) {
        unsafe {
            if self.data.model_pipeline != ash::vk::Pipeline::null() {
                self.device.destroy_pipeline(self.data.model_pipeline, None);
                self.data.model_pipeline = ash::vk::Pipeline::null();
            }
            if self.data.particle_pipeline != ash::vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.particle_pipeline, None);
                self.data.particle_pipeline = ash::vk::Pipeline::null();
            }
            if self.data.particle_compute_pipeline != ash::vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.particle_compute_pipeline, None);
                self.data.particle_compute_pipeline = ash::vk::Pipeline::null();
            }

            // 管线布局
            if self.data.model_pipeline_layout != ash::vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.model_pipeline_layout, None);
                self.data.model_pipeline_layout = ash::vk::PipelineLayout::null();
            }
            if self.data.particle_pipeline_layout != ash::vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.particle_pipeline_layout, None);
                self.data.particle_pipeline_layout = ash::vk::PipelineLayout::null();
            }
            if self.data.particle_compute_pipeline_layout != ash::vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.particle_compute_pipeline_layout, None);
                self.data.particle_compute_pipeline_layout = ash::vk::PipelineLayout::null();
            }
        }
    }

    /// 清理命令缓冲区
    fn cleanup_command_buffers(&mut self) {
        unsafe {
            // 主命令缓冲区
            for i in 0..self.data.command_buffers.len() {
                if self.data.command_buffers[i] != ash::vk::CommandBuffer::null()
                    && i < self.data.command_pools.len()
                    && self.data.command_pools[i] != ash::vk::CommandPool::null()
                {
                    self.device.free_command_buffers(
                        self.data.command_pools[i],
                        &[self.data.command_buffers[i]],
                    );
                }
            }
            self.data.command_buffers.clear();

            // 计算命令缓冲区
            for &command_buffer in &self.data.compute_command_buffers {
                if command_buffer != ash::vk::CommandBuffer::null()
                    && self.data.command_pool != ash::vk::CommandPool::null()
                {
                    self.device
                        .free_command_buffers(self.data.command_pool, &[command_buffer]);
                }
            }
            self.data.compute_command_buffers.clear();

            // 二级命令缓冲区
            for (i, secondary_buffers) in self.data.secondary_command_buffers.iter_mut().enumerate()
            {
                if i < self.data.command_pools.len()
                    && self.data.command_pools[i] != ash::vk::CommandPool::null()
                {
                    for &buffer in secondary_buffers.iter() {
                        if buffer != ash::vk::CommandBuffer::null() {
                            self.device
                                .free_command_buffers(self.data.command_pools[i], &[buffer]);
                        }
                    }
                }
                secondary_buffers.clear();
            }
        }
    }

    /// 清理同步对象
    fn cleanup_sync_objects(&mut self) {
        unsafe {
            for &fence in &self.data.in_flight_fences {
                if fence != ash::vk::Fence::null() {
                    self.device.destroy_fence(fence, None);
                }
            }
            for &semaphore in &self.data.render_finished_semaphores {
                if semaphore != ash::vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            for &semaphore in &self.data.image_available_semaphores {
                if semaphore != ash::vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            for &semaphore in &self.data.compute_finished_semaphores {
                if semaphore != ash::vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
        }
    }

    /// 清理命令池
    fn cleanup_command_pools(&mut self) {
        unsafe {
            for &pool in &self.data.command_pools {
                if pool != ash::vk::CommandPool::null() {
                    self.device.destroy_command_pool(pool, None);
                }
            }
            if self.data.command_pool != ash::vk::CommandPool::null() {
                self.device
                    .destroy_command_pool(self.data.command_pool, None);
            }
        }
    }

    /// 清理模型相关资源
    fn cleanup_model_resources(&mut self) {
        unsafe {
            if self.data.index_buffer_memory != ash::vk::DeviceMemory::null() {
                self.device.free_memory(self.data.index_buffer_memory, None);
            }
            if self.data.index_buffer != ash::vk::Buffer::null() {
                self.device.destroy_buffer(self.data.index_buffer, None);
            }
            if self.data.vertex_buffer_memory != ash::vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.vertex_buffer_memory, None);
            }
            if self.data.vertex_buffer != ash::vk::Buffer::null() {
                self.device.destroy_buffer(self.data.vertex_buffer, None);
            }
        }
    }

    /// 清理粒子系统资源
    fn cleanup_particle_resources(&mut self) {
        unsafe {
            for &memory in &self.data.particle_storage_buffers_memory {
                if memory != ash::vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.particle_storage_buffers {
                if buffer != ash::vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
        }
    }

    /// 清理纹理资源
    fn cleanup_texture_resources(&mut self) {
        unsafe {
            if self.data.texture_sampler != ash::vk::Sampler::null() {
                self.device.destroy_sampler(self.data.texture_sampler, None);
            }
            if self.data.texture_image_view != ash::vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.texture_image_view, None);
            }
            if self.data.texture_image_memory != ash::vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.texture_image_memory, None);
            }
            if self.data.texture_image != ash::vk::Image::null() {
                self.device.destroy_image(self.data.texture_image, None);
            }
        }
    }

    /// 清理描述符集布局
    fn cleanup_descriptor_layouts(&mut self) {
        unsafe {
            if self.data.model_descriptor_set_layout != ash::vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.model_descriptor_set_layout, None);
            }
            if self.data.particle_descriptor_set_layout != ash::vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.particle_descriptor_set_layout, None);
            }
        }
    }

    /// 清理表面
    fn cleanup_surface(&mut self) {
        unsafe {
            if self.data.surface != ash::vk::SurfaceKHR::null() {
                let surface_instance =
                    ash::khr::surface::Instance::new(&self.entry, &self.instance);
                surface_instance.destroy_surface(self.data.surface, None);
            }
        }
    }

    /// 清理调试信使
    fn cleanup_debug_messenger(&mut self) {
        unsafe {
            if VALIDATION_ENABLED && self.data.messenger != ash::vk::DebugUtilsMessengerEXT::null()
            {
                let debug_utils = ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
                debug_utils.destroy_debug_utils_messenger(self.data.messenger, None);
            }
        }
    }
}

//==================================================================================================
// 渲染循环私有方法
//==================================================================================================

impl VulkanApp {
    /// 获取下一个交换链图像
    /// 处理交换链过期情况
    fn acquire_next_swapchain_image(&mut self, window: &Window) -> Result<Option<usize>> {
        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let result = unsafe {
            swapchain_device.acquire_next_image(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame],
                ash::vk::Fence::null(),
            )
        };

        match result {
            Ok((image_index, _)) => Ok(Some(image_index as usize)),
            Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)?;
                Ok(None)
            }
            Err(e) => Err(anyhow::anyhow!("获取交换链图像失败: {}", e)),
        }
    }

    /// 更新帧数据
    /// 更新统一缓冲区和录制命令缓冲区
    fn update_frame_data(&mut self, image_index: usize) -> Result<()> {
        // 更新命令缓冲区
        self.update_command_buffer(image_index)?;

        // 更新模型统一缓冲区
        update_model_uniform_buffer(self, image_index)?;

        // 更新粒子统一缓冲区
        update_particle_uniform_buffer(self)?;

        Ok(())
    }

    /// 更新主命令缓冲区
    /// 录制渲染通道和绘制命令
    fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // 重置命令池
        let command_pool = self.data.command_pools[image_index];
        unsafe {
            self.device
                .reset_command_pool(command_pool, ash::vk::CommandPoolResetFlags::empty())?;
        }

        let command_buffer = self.data.command_buffers[image_index];

        // 开始录制命令
        let info = ash::vk::CommandBufferBeginInfo::default()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(command_buffer, &info)?;
        }

        // 配置渲染通道
        let render_area = ash::vk::Rect2D::default()
            .offset(ash::vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let clear_values = &[
            ash::vk::ClearValue {
                color: ash::vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0], // 黑色背景
                },
            },
            ash::vk::ClearValue {
                depth_stencil: ash::vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_info = ash::vk::RenderPassBeginInfo::default()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                ash::vk::SubpassContents::INLINE,
            );

            // 1. 首先渲染粒子系统
            render_particles(self, command_buffer)?;

            // 2. 然后渲染模型（使用二级命令缓冲区）
            render_all_models(self, command_buffer, image_index)?;

            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    /// 提交渲染命令
    /// 协调计算和图形命令的提交
    fn submit_render_commands(&mut self, image_index: usize) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        // 录制并提交计算命令
        record_particle_compute_commands(
            &self.device,
            &self.data,
            self.data.compute_command_buffers[self.frame],
            self.frame,
        )?;

        // 创建数组以确保生命周期足够长
        let compute_command_buffers = [self.data.compute_command_buffers[self.frame]];
        let compute_signal_semaphores = [self.data.compute_finished_semaphores[self.frame]];

        // 计算命令提交信息
        let compute_submit_info = ash::vk::SubmitInfo::default()
            .command_buffers(&compute_command_buffers)
            .signal_semaphores(&compute_signal_semaphores);

        // 图形命令提交信息 - 等待图像可用和计算完成
        let wait_semaphores = &[
            self.data.image_available_semaphores[self.frame],
            self.data.compute_finished_semaphores[self.frame],
        ];
        let wait_stages = &[
            ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ash::vk::PipelineStageFlags::VERTEX_INPUT, // 等待顶点输入阶段
        ];
        let command_buffers_submit = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];

        let graphics_submit_info = ash::vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers_submit)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device.reset_fences(&[in_flight_fence])?;

            // 先提交计算命令
            self.device.queue_submit(
                self.data.compute_queue,
                &[compute_submit_info],
                ash::vk::Fence::null(),
            )?;

            // 然后提交图形命令
            self.device.queue_submit(
                self.data.graphics_queue,
                &[graphics_submit_info],
                in_flight_fence,
            )?;
        }

        Ok(())
    }

    /// 呈现帧
    /// 将渲染结果呈现到屏幕
    fn present_frame(&mut self, window: &Window, image_index: usize) -> Result<()> {
        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let swapchains = &[self.data.swapchain];
        let image_indices_present = &[image_index as u32];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];

        let present_info = ash::vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices_present);

        let result =
            unsafe { swapchain_device.queue_present(self.data.present_queue, &present_info) };

        let changed = match result {
            Ok(true) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Ok(false) => false,
            Err(e) => return Err(anyhow::anyhow!("呈现帧失败: {}", e)),
        };

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        }

        Ok(())
    }
}
