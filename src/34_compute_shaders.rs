#![allow(unsafe_code)] // Allows unsafe blocks and unsafe fn calls within the crate

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::mem::{offset_of, size_of};
use std::os::raw::{c_char, c_void};
use std::ptr::copy_nonoverlapping as memcpy; // Alias for consistency
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use log::*;

use ash::vk;
use ash::vk::Handle; // Used for vk::DebugUtilsMessengerEXT::null(), vk::Fence::null(), etc.
use ash::{Device, Entry, Instance};

use anyhow::{Result, anyhow};
use thiserror::Error;

mod vk_window;
use vk_window::*; // For get_required_instance_extensions, create_surface

//==================================================================================================
// SECTION: Constants & Type Aliases
//==================================================================================================

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

const VALIDATION_LAYER_NAME: &CStr =
    // SAFETY: Byte string is NUL-terminated and valid UTF-8.
    c"VK_LAYER_KHRONOS_validation";
const DEVICE_EXTENSIONS: &[&CStr] = &[
    // SAFETY: Byte string is NUL-terminated and valid UTF-8.
    c"VK_KHR_swapchain",
];
const MAX_FRAMES_IN_FLIGHT: usize = 3;
const PARTICLE_COUNT: usize = 8192;

type Vec2 = nalgebra::Vector2<f32>;
type Vec4 = nalgebra::Vector4<f32>;

//==================================================================================================
// SECTION: Core Vulkan Data Structures (Vertices, UBOs, Support Structs)
//==================================================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    color: Vec4,
}

impl Particle {
    const fn new(position: Vec2, velocity: Vec2, color: Vec4) -> Self {
        Self {
            position,
            velocity,
            color,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                // SAFETY: `position` 是 `Particle` 的有效字段
                .offset(offset_of!(Particle, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                // SAFETY: `color` 是 `Particle` 的有效字段
                .offset(offset_of!(Particle, color) as u32),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    delta_time: f32,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics_and_compute: u32,
    present: u32,
}

impl QueueFamilyIndices {
    fn get(
        instance: &Instance,
        entry: &Entry,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties =
            // SAFETY: `instance` and `physical_device` are assumed valid.
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let graphics = properties
            .iter()
            .position(|p| {
                p.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map(|i| i as u32);

        let mut present = None;
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
        for (index, _properties) in properties.iter().enumerate() {
            // SAFETY: `surface_instance`, `physical_device`, `index`, and `data.surface` are assumed valid.
            let supported = unsafe {
                surface_instance.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    data.surface,
                )?
            };
            if supported {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics_and_compute), Some(present)) = (graphics, present) {
            Ok(Self {
                graphics_and_compute,
                present,
            })
        } else {
            Err(anyhow!(SuitabilityError::Static(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    fn get(
        instance: &Instance,
        entry: &Entry,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
        // SAFETY: All parameters are assumed valid for these Vulkan calls.
        unsafe {
            Ok(Self {
                capabilities: surface_instance
                    .get_physical_device_surface_capabilities(physical_device, data.surface)?,
                formats: surface_instance
                    .get_physical_device_surface_formats(physical_device, data.surface)?,
                present_modes: surface_instance
                    .get_physical_device_surface_present_modes(physical_device, data.surface)?,
            })
        }
    }
}

//==================================================================================================
// SECTION: Application State Structures (AppData, VulkanApp)
//==================================================================================================

#[derive(Clone, Debug, Default)]
struct AppData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    // Queues
    graphics_queue: vk::Queue,
    compute_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Render Pipeline
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Compute Pipeline
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    // Command Pool & Buffers
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    compute_command_buffers: Vec<vk::CommandBuffer>,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Particle Storage
    shader_storage_buffers: Vec<vk::Buffer>,
    shader_storage_buffers_memory: Vec<vk::DeviceMemory>,
    // Uniform Buffers
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    compute_descriptor_sets: Vec<vk::DescriptorSet>,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    compute_finished_semaphores: Vec<vk::Semaphore>,
    graphics_fences: Vec<vk::Fence>,        // 用于帧同步的栅栏
    compute_fences: Vec<vk::Fence>,         // 用于计算同步的栅栏
    swapchain_image_fences: Vec<vk::Fence>, // 跟踪每个交换链图像的栅栏（引用）
    // State Tracking
    last_frame_time: f32,
}

#[derive(Clone)]
struct VulkanApp {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
    last_time: f64,
}

//==================================================================================================
// SECTION: VulkanApp Implementation (Core Logic)
//==================================================================================================

impl VulkanApp {
    /// Initializes Vulkan application state.
    fn create(window: &Window) -> Result<Self> {
        let entry =
            unsafe { Entry::load().map_err(|e| anyhow!("Failed to load Vulkan entry: {}", e))? };
        let mut data = AppData::default();

        let instance = create_instance(window, &entry, &mut data)?;
        // SAFETY: `create_surface` is from `vk_window` module and marked as `unsafe fn` there.
        // The window and display handles must be valid for the duration of surface use.
        data.surface = unsafe { create_surface(&instance, &entry, &window, &window)? };

        pick_physical_device(&instance, &entry, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &entry, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_compute_descriptor_set_layout(&device, &mut data)?;
        create_graphics_pipeline(&device, &mut data)?;
        create_compute_pipeline(&device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &entry, &mut data)?;
        create_shader_storage_buffers(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_compute_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_compute_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            last_time: 0.0,
        })
    }

    /// 渲染一帧。包含不安全的 Vulkan 调用。
    fn render(&mut self, window: &Window) -> Result<()> {
        // 计算帧时间
        let current_time = self.start.elapsed().as_secs_f64();
        self.data.last_frame_time = ((current_time - self.last_time) * 1000.0) as f32;
        self.last_time = current_time;

        // 准备计算阶段的命令缓冲
        let compute_fence = self.data.compute_fences[self.frame];

        // 等待计算栅栏
        unsafe {
            self.device
                .wait_for_fences(&[compute_fence], true, u64::MAX)?;
        }

        // 更新计算的统一缓冲区
        self.compute_particle_movement()?;

        // 重置计算栅栏并记录计算命令缓冲区
        unsafe {
            self.device.reset_fences(&[compute_fence])?;
            self.device.reset_command_buffer(
                self.data.compute_command_buffers[self.frame],
                vk::CommandBufferResetFlags::empty(),
            )?;
        }

        // 记录计算命令
        record_compute_command_buffer(
            &self.device,
            &self.data,
            self.data.compute_command_buffers[self.frame],
            self.frame,
        )?;

        // 提交计算命令
        let compute_wait_semaphores = &[];
        let compute_wait_stages = &[];
        let compute_command_buffers = &[self.data.compute_command_buffers[self.frame]];
        let compute_signal_semaphores = &[self.data.compute_finished_semaphores[self.frame]];

        let compute_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(compute_wait_semaphores)
            .wait_dst_stage_mask(compute_wait_stages)
            .command_buffers(compute_command_buffers)
            .signal_semaphores(compute_signal_semaphores);

        // 提交计算命令
        unsafe {
            self.device.queue_submit(
                self.data.compute_queue,
                &[compute_submit_info],
                compute_fence,
            )?;
        }

        // 准备渲染阶段
        let in_flight_fence = self.data.graphics_fences[self.frame];

        // 等待图形栅栏
        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
        }

        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        // 获取下一张图像
        let image_index = unsafe {
            match swapchain_device.acquire_next_image(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            ) {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window)?;
                    return Ok(()); // 重建交换链后提前返回
                }
                Err(e) => return Err(anyhow!(e)),
            }
        };

        // 处理正在使用的图像
        let image_in_flight = self.data.swapchain_image_fences[image_index];
        if !image_in_flight.is_null() {
            unsafe {
                self.device
                    .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
            }
        }
        self.data.swapchain_image_fences[image_index] = in_flight_fence;

        // 重置并记录图形命令缓冲区
        unsafe {
            self.device.reset_fences(&[in_flight_fence])?;
            self.device.reset_command_buffer(
                self.data.command_buffers[image_index],
                vk::CommandBufferResetFlags::empty(),
            )?;
        }

        // 记录绘制命令
        self.record_command_buffer(self.data.command_buffers[image_index], image_index)?;

        // 提交图形阶段的命令缓冲
        let wait_semaphores = &[
            self.data.image_available_semaphores[self.frame],
            self.data.compute_finished_semaphores[self.frame],
        ];
        let wait_stages = &[
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::VERTEX_INPUT,
        ];
        let command_buffers_submit = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers_submit)
            .signal_semaphores(signal_semaphores);

        // 提交图形命令
        unsafe {
            self.device
                .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;
        }

        // 最后展示
        let swapchains = &[self.data.swapchain];
        let image_indices_present = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices_present);

        // 展示到屏幕
        let result =
            unsafe { swapchain_device.queue_present(self.data.present_queue, &present_info) };

        let changed = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Ok(false) => false,
            Err(e) => return Err(e.into()),
        };

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// 更新粒子计算着色器的统一缓冲区
    fn compute_particle_movement(&self) -> Result<()> {
        // 更新计算着色器使用的统一缓冲
        let ubo = UniformBufferObject {
            delta_time: self.data.last_frame_time * 2.0f32, // 注意：这里应该是 * 2.0f32 以匹配 C++ 版本
        };

        let current_buffer = self.frame;

        // SAFETY: 将 UBO 数据复制到映射的内存中
        unsafe {
            memcpy(
                &ubo as *const _ as *const u8,                           // Source
                self.data.uniform_buffers_mapped[current_buffer].cast(), // Destination
                size_of::<UniformBufferObject>(),                        // Size
            );
        }

        Ok(())
    }

    /// 记录命令缓冲区
    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        // SAFETY: 开始命令缓冲区是不安全的
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        let render_area = vk::Rect2D::default().extent(self.data.swapchain_extent);

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(std::slice::from_ref(&clear_color));

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.data.pipeline,
            );

            // 设置视口和裁剪区域
            let viewport = vk::Viewport::default()
                .width(self.data.swapchain_extent.width as f32)
                .height(self.data.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);

            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

            let scissor = vk::Rect2D::default().extent(self.data.swapchain_extent);

            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

            // 绑定顶点缓冲区（粒子存储缓冲区）
            let offsets = [0];
            self.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.data.shader_storage_buffers[self.frame]],
                &offsets,
            );

            // 绘制粒子
            self.device
                .cmd_draw(command_buffer, PARTICLE_COUNT as u32, 1, 0, 0);

            self.device.cmd_end_render_pass(command_buffer);

            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    /// Recreates the swapchain and dependent resources when the window is resized or surface becomes outdated.
    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // SAFETY: `device_wait_idle` is an unsafe Vulkan call. Device must be valid.
        unsafe { self.device.device_wait_idle()? };

        // 清理已有资源 - 包括管线和渲染通道
        self.destroy_swapchain_resources();

        // 重建交换链和相关资源
        create_swapchain(
            window,
            &self.instance,
            &self.device,
            &self.entry,
            &mut self.data,
        )?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_graphics_pipeline(&self.device, &mut self.data)?;
        create_compute_pipeline(&self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_compute_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;

        // 重置图像跟踪数组
        self.data
            .swapchain_image_fences
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    /// 销毁交换链相关资源
    // 新增此函数，与C++版本的cleanupSwapChain相同
    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            // 1. 销毁帧缓冲区
            for framebuffer in self.data.framebuffers.drain(..) {
                if framebuffer != vk::Framebuffer::null() {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
            }

            // 2. 销毁图像视图
            for view in self.data.swapchain_image_views.drain(..) {
                if view != vk::ImageView::null() {
                    self.device.destroy_image_view(view, None);
                }
            }

            // 3. 销毁管线和渲染通道
            if self.data.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.data.pipeline, None);
                self.data.pipeline = vk::Pipeline::null();
            }

            if self.data.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.pipeline_layout, None);
                self.data.pipeline_layout = vk::PipelineLayout::null();
            }

            if self.data.compute_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.compute_pipeline, None);
                self.data.compute_pipeline = vk::Pipeline::null();
            }

            if self.data.compute_pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.compute_pipeline_layout, None);
                self.data.compute_pipeline_layout = vk::PipelineLayout::null();
            }

            if self.data.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = vk::RenderPass::null();
            }

            // 4. 销毁交换链
            if self.data.swapchain != vk::SwapchainKHR::null() {
                let swapchain_device =
                    ash::khr::swapchain::Device::new(&self.instance, &self.device);
                swapchain_device.destroy_swapchain(self.data.swapchain, None);
                self.data.swapchain = vk::SwapchainKHR::null();
            }

            // 清除图像句柄列表
            self.data.swapchain_images.clear();
        }
    }

    /// 销毁所有应用程序管理的 Vulkan 资源。
    /// 确保按正确的顺序销毁资源，以避免验证错误。
    fn destroy(&mut self) {
        // SAFETY: 在销毁任何 Vulkan 资源之前，`device_wait_idle` 是关键的，
        // 以确保它们不再被 GPU 使用。
        // 所有后续的 Vulkan 销毁调用都是不安全的，并假设设备处于空闲状态，
        // 资源按有效顺序被销毁。
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait for device to be idle before destruction");

            // 1. 销毁交换链及其依赖资源
            self.destroy_swapchain_resources();

            // 2. 销毁图形和计算管线 (注意顺序变更)
            if self.data.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.data.pipeline, None);
                self.data.pipeline = vk::Pipeline::null();
            }

            if self.data.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.pipeline_layout, None);
                self.data.pipeline_layout = vk::PipelineLayout::null();
            }

            if self.data.compute_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.compute_pipeline, None);
                self.data.compute_pipeline = vk::Pipeline::null();
            }

            if self.data.compute_pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.compute_pipeline_layout, None);
                self.data.compute_pipeline_layout = vk::PipelineLayout::null();
            }

            // 3. 销毁渲染通道 (注意顺序变更)
            if self.data.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = vk::RenderPass::null();
            }

            // 4. 销毁渲染通道
            if self.data.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = vk::RenderPass::null();
            }

            // 5. 销毁统一缓冲区及其内存
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                if i < self.data.uniform_buffers.len()
                    && self.data.uniform_buffers[i] != vk::Buffer::null()
                {
                    self.device
                        .destroy_buffer(self.data.uniform_buffers[i], None);
                }
                if i < self.data.uniform_buffers_memory.len()
                    && self.data.uniform_buffers_memory[i] != vk::DeviceMemory::null()
                {
                    self.device
                        .free_memory(self.data.uniform_buffers_memory[i], None);
                }
            }
            self.data.uniform_buffers.clear();
            self.data.uniform_buffers_memory.clear();
            self.data.uniform_buffers_mapped.clear();

            // 6. 销毁描述符池
            if self.data.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.descriptor_pool, None);
                self.data.descriptor_pool = vk::DescriptorPool::null();
                self.data.compute_descriptor_sets.clear();
            }

            // 7. 销毁计算描述符集布局
            if self.data.compute_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.compute_descriptor_set_layout, None);
                self.data.compute_descriptor_set_layout = vk::DescriptorSetLayout::null();
            }

            // 8. 销毁着色器存储缓冲区及其内存
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                if i < self.data.shader_storage_buffers.len()
                    && self.data.shader_storage_buffers[i] != vk::Buffer::null()
                {
                    self.device
                        .destroy_buffer(self.data.shader_storage_buffers[i], None);
                }
                if i < self.data.shader_storage_buffers_memory.len()
                    && self.data.shader_storage_buffers_memory[i] != vk::DeviceMemory::null()
                {
                    self.device
                        .free_memory(self.data.shader_storage_buffers_memory[i], None);
                }
            }
            self.data.shader_storage_buffers.clear();
            self.data.shader_storage_buffers_memory.clear();

            // 9. 销毁同步对象（信号量和栅栏）
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                if i < self.data.render_finished_semaphores.len()
                    && self.data.render_finished_semaphores[i] != vk::Semaphore::null()
                {
                    self.device
                        .destroy_semaphore(self.data.render_finished_semaphores[i], None);
                }
                if i < self.data.image_available_semaphores.len()
                    && self.data.image_available_semaphores[i] != vk::Semaphore::null()
                {
                    self.device
                        .destroy_semaphore(self.data.image_available_semaphores[i], None);
                }
                if i < self.data.compute_finished_semaphores.len()
                    && self.data.compute_finished_semaphores[i] != vk::Semaphore::null()
                {
                    self.device
                        .destroy_semaphore(self.data.compute_finished_semaphores[i], None);
                }
                if i < self.data.graphics_fences.len()
                    && self.data.graphics_fences[i] != vk::Fence::null()
                {
                    self.device
                        .destroy_fence(self.data.graphics_fences[i], None);
                }
                if i < self.data.compute_fences.len()
                    && self.data.compute_fences[i] != vk::Fence::null()
                {
                    self.device.destroy_fence(self.data.compute_fences[i], None);
                }
            }
            self.data.render_finished_semaphores.clear();
            self.data.image_available_semaphores.clear();
            self.data.compute_finished_semaphores.clear();
            self.data.graphics_fences.clear();
            self.data.compute_fences.clear();

            // 10. 销毁命令池
            if self.data.command_pool != vk::CommandPool::null() {
                self.device
                    .destroy_command_pool(self.data.command_pool, None);
                self.data.command_pool = vk::CommandPool::null();
            }

            // 11. 销毁逻辑设备
            self.device.destroy_device(None);

            // 12. 销毁 Vulkan 表面
            if self.data.surface != vk::SurfaceKHR::null() {
                let surface_instance =
                    ash::khr::surface::Instance::new(&self.entry, &self.instance);
                surface_instance.destroy_surface(self.data.surface, None);
                self.data.surface = vk::SurfaceKHR::null();
            }

            // 13. 销毁调试消息器（如果启用了验证）
            if VALIDATION_ENABLED && self.data.messenger != vk::DebugUtilsMessengerEXT::null() {
                let debug_utils_instance =
                    ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
                debug_utils_instance.destroy_debug_utils_messenger(self.data.messenger, None);
                self.data.messenger = vk::DebugUtilsMessengerEXT::null();
            }

            // 14. 销毁 Vulkan 实例
            self.instance.destroy_instance(None);
        }

        // 注意：self.instance 和 self.entry 不需要置空，因为 VulkanApp 本身将被丢弃
    }
}

//==================================================================================================
// SECTION: Vulkan Initialization and Resource Creation Functions
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Subsection: Instance and Debug Setup
//--------------------------------------------------------------------------------------------------

/// Creates a Vulkan instance and sets up debug messaging if enabled.
fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // anyhow::Result
    let app_name = CString::new("Vulkan Tutorial (Rust)")?;
    let engine_name = CString::new("No Engine")?;

    let application_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    // SAFETY: `enumerate_instance_layer_properties` is unsafe.
    // `CStr::from_ptr` relies on Vulkan providing a valid C string.
    let available_layers = unsafe { entry.enumerate_instance_layer_properties()? }
        .iter()
        .map(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) })
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED
        && !available_layers
            .iter()
            .any(|&layer| layer == VALIDATION_LAYER_NAME)
    {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    // Based on the compiler error, get_required_instance_extensions(window)
    // appears to return &'static [&'static CStr] directly, not a Result.
    // If this function can indeed fail, its signature in `vk_window.rs`
    // or its usage here would need to be adjusted to handle errors appropriately.
    let required_extensions_cstrs: &'static [&'static CStr] =
        get_required_instance_extensions(window);

    let mut extensions_ptrs: Vec<*const c_char> = required_extensions_cstrs
        .iter()
        .map(|e| e.as_ptr())
        .collect();

    if VALIDATION_ENABLED {
        extensions_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    let layers_names_raw_instance = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER_NAME.as_ptr()]
    } else {
        Vec::new()
    };

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layers_names_raw_instance)
        .enabled_extension_names(&extensions_ptrs);

    if VALIDATION_ENABLED {
        create_info = create_info.push_next(&mut debug_info);
    }

    // SAFETY: `create_instance` is an unsafe Vulkan call. All parameters must be valid.
    let instance = unsafe { entry.create_instance(&create_info, None)? };

    if VALIDATION_ENABLED {
        let debug_utils_instance = ash::ext::debug_utils::Instance::new(entry, &instance);
        // SAFETY: `create_debug_utils_messenger` is an unsafe Vulkan call.
        data.messenger =
            unsafe { debug_utils_instance.create_debug_utils_messenger(&debug_info, None)? };
    }

    Ok(instance)
}

/// Vulkan debug callback function.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: `data` is a pointer from Vulkan, assumed valid.
    // `callback_data.p_message` is a C string from Vulkan, assumed valid.
    let callback_data = unsafe { &*data };
    let message = unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() };

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) Validation Layer: {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) Validation Layer: {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) Validation Layer: {}", type_, message);
    } else {
        trace!("({:?}) Validation Layer: {}", type_, message);
    }
    vk::FALSE
}

//--------------------------------------------------------------------------------------------------
// Subsection: Physical Device and Logical Device
//--------------------------------------------------------------------------------------------------

/// Error type for physical device suitability checks.
#[derive(Debug, Error)]
pub enum SuitabilityError {
    #[error("Static error: {0}")]
    Static(&'static str),
    #[error("Dynamic error: {0}")]
    Dynamic(String),
}

/// Picks a suitable physical device (GPU).
fn pick_physical_device(instance: &Instance, entry: &Entry, data: &mut AppData) -> Result<()> {
    // SAFETY: `enumerate_physical_devices` is an unsafe Vulkan call.
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.is_empty() {
        return Err(anyhow!("Failed to find GPUs with Vulkan support."));
    }

    for physical_device in physical_devices {
        // SAFETY: `get_physical_device_properties` is unsafe.
        // `CStr::from_ptr` relies on Vulkan providing a valid C string.
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name =
            unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };

        if let Err(error) =
            check_physical_device_suitability(instance, entry, data, physical_device)
        {
            warn!("Skipping physical device (`{}`): {}", device_name, error);
        } else {
            info!("Selected physical device (`{}`).", device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }
    Err(anyhow!("Failed to find a suitable physical device."))
}

/// Checks if a given physical device meets the application's requirements.
fn check_physical_device_suitability(
    instance: &Instance,
    entry: &Entry,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, entry, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, entry, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError::Static(
            "Insufficient swapchain support."
        )));
    }

    // SAFETY: `get_physical_device_features2` is an unsafe Vulkan call.
    let mut features2_query = vk::PhysicalDeviceFeatures2::default();
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2_query) };

    if features2_query.features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError::Static(
            "Sampler anisotropy not supported."
        )));
    }
    Ok(())
}

/// Checks if a physical device supports all required device extensions.
fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // SAFETY: `enumerate_device_extension_properties` is unsafe.
    // `CStr::from_ptr` relies on Vulkan providing valid C strings.
    let available_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device)? }
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect::<HashSet<_>>();

    for &required_ext in DEVICE_EXTENSIONS.iter() {
        if !available_extensions.contains(required_ext) {
            return Err(anyhow!(SuitabilityError::Dynamic(format!(
                "Missing required device extension: {}",
                required_ext.to_string_lossy()
            ))));
        }
    }
    Ok(())
}

/// Creates a logical Vulkan device from a physical device.
fn create_logical_device(entry: &Entry, instance: &Instance, data: &mut AppData) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics_and_compute);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let extension_ptrs: Vec<*const c_char> =
        DEVICE_EXTENSIONS.iter().map(|ext| ext.as_ptr()).collect();

    let base_features_to_enable = vk::PhysicalDeviceFeatures::default().sample_rate_shading(true);
    let mut vulkan_1_2_features_to_enable = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan_1_3_features_to_enable = vk::PhysicalDeviceVulkan13Features::default();

    let mut features_chain = vk::PhysicalDeviceFeatures2::default()
        .features(base_features_to_enable)
        .push_next(&mut vulkan_1_2_features_to_enable)
        .push_next(&mut vulkan_1_3_features_to_enable);

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_ptrs)
        .push_next(&mut features_chain);

    // SAFETY: `create_device` is an unsafe Vulkan call. Physical device and create_info must be valid.
    let device = unsafe { instance.create_device(data.physical_device, &create_info, None)? };

    // SAFETY: `get_device_queue` is an unsafe Vulkan call. Device and queue indices must be valid.
    unsafe {
        data.graphics_queue = device.get_device_queue(indices.graphics_and_compute, 0);
        data.compute_queue = device.get_device_queue(indices.graphics_and_compute, 0);
        data.present_queue = device.get_device_queue(indices.present, 0);
    }
    Ok(device)
}

//--------------------------------------------------------------------------------------------------
// Subsection: Swapchain and Image Views
//--------------------------------------------------------------------------------------------------

/// Creates the Vulkan swapchain for presenting images to the screen.
fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, entry, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices_vec = vec![];
    let image_sharing_mode = if indices.graphics_and_compute != indices.present {
        queue_family_indices_vec.push(indices.graphics_and_compute);
        queue_family_indices_vec.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(if image_sharing_mode == vk::SharingMode::CONCURRENT {
            &queue_family_indices_vec
        } else {
            &[]
        })
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    // SAFETY: `create_swapchain` and `get_swapchain_images` are unsafe. All parameters must be valid.
    unsafe {
        data.swapchain = swapchain_loader.create_swapchain(&create_info, None)?;
        data.swapchain_images = swapchain_loader.get_swapchain_images(data.swapchain)?;
    }
    Ok(())
}

/// Selects an appropriate surface format for the swapchain.
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

/// Selects an appropriate presentation mode for the swapchain.
fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// Determines the extent (resolution) of the swapchain images.
fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let window_size = window.inner_size();
        let mut actual_extent = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };
        actual_extent.width = actual_extent.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );
        actual_extent.height = actual_extent.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );
        actual_extent
    }
}

/// 为交换链中的每个图像创建图像视图
fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    // 预先分配足够大小的向量
    data.swapchain_image_views.clear();
    data.swapchain_image_views
        .reserve(data.swapchain_images.len());

    for &image in &data.swapchain_images {
        // 创建图像视图信息
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image) // 设置关联的图像
            .view_type(vk::ImageViewType::TYPE_2D) // 2D 图像视图类型
            .format(data.swapchain_format) // 使用交换链格式
            // 默认的组件映射（RGBA -> RGBA）
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            // 设置子资源范围
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR, // 颜色方面
                base_mip_level: 0,                        // 基础 mip 级别
                level_count: 1,                           // mip 级别数量
                base_array_layer: 0,                      // 基础数组层
                layer_count: 1,                           // 层数
            });

        // SAFETY: `create_image_view` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
        let image_view = unsafe {
            device
                .create_image_view(&create_info, None)
                .map_err(|e| anyhow!("无法创建图像视图: {}", e))?
        };

        data.swapchain_image_views.push(image_view);
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Render Pass, Pipeline Layout, Pipeline
//--------------------------------------------------------------------------------------------------

/// 创建渲染通道，定义帧缓冲附件和子通道
fn create_render_pass(_instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // 创建颜色附件
    let color_attachment = vk::AttachmentDescription::default()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1) // 单采样
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // 颜色附件引用
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // 创建子通道
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_ref));

    // 创建子通道依赖
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    // 创建渲染通道
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));

    // SAFETY: `create_render_pass` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.render_pass = unsafe { device.create_render_pass(&create_info, None)? };
    Ok(())
}

/// 创建计算着色器的描述符集布局
fn create_compute_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    // 创建三个绑定：一个统一缓冲区和两个存储缓冲区
    let layout_bindings = [
        // 绑定 0：统一缓冲区 (UBO)
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 绑定 1：存储缓冲区（当前粒子状态）
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 绑定 2：存储缓冲区（新的粒子状态）
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    // 创建描述符集布局
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);

    // SAFETY: `create_descriptor_set_layout` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.compute_descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| anyhow!("无法创建计算描述符集布局: {}", e))?
    };

    Ok(())
}

/// 创建图形管线，包括着色器和固定功能状态
fn create_graphics_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载着色器代码
    let vert_shader_spirv = include_bytes!("../assets/shaders/34_compute_shaders.vert.spv");
    let frag_shader_spirv = include_bytes!("../assets/shaders/34_compute_shaders.frag.spv");

    // 创建着色器模块
    let vert_shader_module = create_shader_module_internal(device, vert_shader_spirv)?;
    let frag_shader_module = create_shader_module_internal(device, frag_shader_spirv)?;

    // 指定入口函数名
    let main_function_name = c"main";

    // 创建着色器阶段信息
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
    let binding_description = Particle::binding_description();
    let attribute_descriptions = Particle::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
        .vertex_attribute_descriptions(&attribute_descriptions);

    // 输入组装状态
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::POINT_LIST) // 粒子系统使用点列表
        .primitive_restart_enable(false);

    // 视口和裁剪状态 - 使用动态状态
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // 不需要提供视口和裁剪矩形，因为它们将作为动态状态设置

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
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // 颜色混合附件状态
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    // 颜色混合状态
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&color_blend_attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // 动态状态
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // 创建管线布局
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&[]);

    // SAFETY: `create_pipeline_layout` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| anyhow!("无法创建管线布局: {}", e))?
    };

    // 创建图形管线
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null());

    // SAFETY: `create_graphics_pipelines` 是不安全的 Vulkan 调用。设备、缓存和管线信息必须有效。
    data.pipeline = unsafe {
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(anyhow!("无法创建图形管线: {}", err));
            }
        }
    };

    // 销毁着色器模块
    unsafe {
        device.destroy_shader_module(frag_shader_module, None);
        device.destroy_shader_module(vert_shader_module, None);
    }

    Ok(())
}

/// 创建计算管线
fn create_compute_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载计算着色器代码
    let compute_shader_spirv = include_bytes!("../assets/shaders/34_compute_shaders.comp.spv");

    // 创建计算着色器模块
    let compute_shader_module = create_shader_module_internal(device, compute_shader_spirv)?;

    // 指定入口函数名
    let main_function_name = c"main";

    // 创建计算着色器阶段信息
    let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module)
        .name(main_function_name);

    // 计算管线布局
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&data.compute_descriptor_set_layout)); // 修改此行

    // SAFETY: `create_pipeline_layout` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.compute_pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| anyhow!("无法创建计算管线布局: {}", e))?
    };

    // 创建计算管线
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(compute_shader_stage_info)
        .layout(data.compute_pipeline_layout);

    // SAFETY: `create_compute_pipelines` 是不安全的 Vulkan 调用。设备、缓存和管线信息必须有效。
    data.compute_pipeline = unsafe {
        match device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(anyhow!("无法创建计算管线: {}", err));
            }
        }
    };

    // 销毁着色器模块
    unsafe {
        device.destroy_shader_module(compute_shader_module, None);
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Framebuffers and Command Pool
//--------------------------------------------------------------------------------------------------

/// 为交换链中的每个图像视图创建帧缓冲
fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    // 调整帧缓冲向量大小
    data.framebuffers
        .resize(data.swapchain_image_views.len(), vk::Framebuffer::null());

    for (i, &image_view) in data.swapchain_image_views.iter().enumerate() {
        // 只有一个附件：交换链图像视图
        let attachments = [image_view];

        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(data.render_pass)
            .attachment_count(1)
            .attachments(&attachments)
            .width(data.swapchain_extent.width)
            .height(data.swapchain_extent.height)
            .layers(1);

        // SAFETY: `create_framebuffer` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
        data.framebuffers[i] = unsafe {
            device
                .create_framebuffer(&create_info, None)
                .map_err(|e| anyhow!("无法创建帧缓冲: {}", e))?
        };
    }

    Ok(())
}

/// 创建用于分配命令缓冲的命令池
fn create_command_pool(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;

    let create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics_and_compute);

    // SAFETY: `create_command_pool` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.command_pool = unsafe {
        device
            .create_command_pool(&create_info, None)
            .map_err(|e| anyhow!("无法创建命令池: {}", e))?
    };

    Ok(())
}

/// 创建着色器存储缓冲区并初始化粒子数据
fn create_shader_storage_buffers(
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

    // 初始化粒子，在圆上分布
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    for _ in 0..PARTICLE_COUNT {
        let r = 0.25f32 * (rng.random::<f32>()).sqrt();
        let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let height_width_ratio =
            data.swapchain_extent.height as f32 / data.swapchain_extent.width as f32;
        let x = r * theta.cos() * height_width_ratio;
        let y = r * theta.sin();

        // 创建粒子
        let position = Vec2::new(x, y);
        let velocity = Vec2::new(x, y).normalize() * 0.00025f32;
        let color = Vec4::new(rng.random(), rng.random(), rng.random(), 1.0);

        particles.push(Particle::new(position, velocity, color));
    }

    let buffer_size = (std::mem::size_of::<Particle>() * PARTICLE_COUNT) as vk::DeviceSize;

    // 创建暂存缓冲区，用于上传数据到 GPU
    let (staging_buffer, staging_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 映射内存并复制粒子数据
    unsafe {
        let mem_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(particles.as_ptr(), mem_ptr.cast(), particles.len());

        device.unmap_memory(staging_buffer_memory);
    }

    // 为每一帧创建着色器存储缓冲区
    data.shader_storage_buffers
        .resize(MAX_FRAMES_IN_FLIGHT, vk::Buffer::null());
    data.shader_storage_buffers_memory
        .resize(MAX_FRAMES_IN_FLIGHT, vk::DeviceMemory::null());

    // 将初始粒子数据复制到所有存储缓冲区
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let (buffer, buffer_memory) = create_buffer_internal(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.shader_storage_buffers[i] = buffer;
        data.shader_storage_buffers_memory[i] = buffer_memory;

        copy_buffer_internal(device, data, staging_buffer, buffer, buffer_size)?;
    }

    // 清理暂存缓冲区
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Buffers (Uniform)
//--------------------------------------------------------------------------------------------------

/// Creates uniform buffers for each frame in flight.
fn create_uniform_buffers(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let buffer_size = size_of::<UniformBufferObject>() as vk::DeviceSize;

    // SAFETY: free_memory and destroy_buffer are unsafe
    unsafe {
        for memory in data.uniform_buffers_memory.drain(..) {
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        }
        for buffer in data.uniform_buffers.drain(..) {
            if buffer != vk::Buffer::null() {
                device.destroy_buffer(buffer, None);
            }
        }
    }

    // 调整数组大小
    data.uniform_buffers.reserve(MAX_FRAMES_IN_FLIGHT);
    data.uniform_buffers_memory.reserve(MAX_FRAMES_IN_FLIGHT);
    data.uniform_buffers_mapped.clear();
    data.uniform_buffers_mapped.reserve(MAX_FRAMES_IN_FLIGHT);

    // 为每一帧创建统一缓冲区
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let (buffer, buffer_memory) = create_buffer_internal(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 映射内存
        let ptr = unsafe {
            device.map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?
        };

        // 存储缓冲区、内存和映射指针
        data.uniform_buffers.push(buffer);
        data.uniform_buffers_memory.push(buffer_memory);
        data.uniform_buffers_mapped.push(ptr);
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Descriptors (Pool, Sets)
//--------------------------------------------------------------------------------------------------

/// 创建描述符池，用于分配描述符集
fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // SAFETY: 销毁描述符池是不安全的
    if data.descriptor_pool != vk::DescriptorPool::null() {
        unsafe {
            device.destroy_descriptor_pool(data.descriptor_pool, None);
        }
        data.descriptor_pool = vk::DescriptorPool::null();
    }

    // 创建两种类型的池大小：统一缓冲区和存储缓冲区
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * 2), // 每帧两个存储缓冲区
    ];

    // 创建描述符池信息
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
        .flags(vk::DescriptorPoolCreateFlags::empty());

    // SAFETY: `create_descriptor_pool` 是不安全的 Vulkan 调用。设备和创建信息必须有效。
    data.descriptor_pool = unsafe {
        device
            .create_descriptor_pool(&create_info, None)
            .map_err(|e| anyhow!("无法创建描述符池: {}", e))?
    };

    Ok(())
}

/// 为计算着色器创建描述符集
fn create_compute_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    // 为每一帧创建统一的描述符集布局
    let layouts = vec![data.compute_descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];

    // 描述符集分配信息
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    // 调整计算描述符集的大小
    data.compute_descriptor_sets = vec![vk::DescriptorSet::null(); MAX_FRAMES_IN_FLIGHT];

    data.compute_descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

    // 为每一帧更新描述符
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        // 统一缓冲区信息
        let uniform_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        // 上一帧的存储缓冲区信息
        let prev_frame = (i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
        let storage_buffer_info_last_frame = vk::DescriptorBufferInfo::default()
            .buffer(data.shader_storage_buffers[prev_frame])
            .offset(0)
            .range((size_of::<Particle>() * PARTICLE_COUNT) as u64);

        // 当前帧的存储缓冲区信息
        let storage_buffer_info_current_frame = vk::DescriptorBufferInfo::default()
            .buffer(data.shader_storage_buffers[i])
            .offset(0)
            .range((size_of::<Particle>() * PARTICLE_COUNT) as u64);

        // 创建三个描述符写入操作
        let descriptor_writes = [
            // 绑定 0：统一缓冲区
            vk::WriteDescriptorSet::default()
                .dst_set(data.compute_descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&uniform_buffer_info)),
            // 绑定 1：上一帧的存储缓冲区（输入）
            vk::WriteDescriptorSet::default()
                .dst_set(data.compute_descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&storage_buffer_info_last_frame)),
            // 绑定 2：当前帧的存储缓冲区（输出）
            vk::WriteDescriptorSet::default()
                .dst_set(data.compute_descriptor_sets[i])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&storage_buffer_info_current_frame)),
        ];

        // SAFETY: `update_descriptor_sets` 是不安全的 Vulkan 调用
        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Command Buffers and Sync Objects
//--------------------------------------------------------------------------------------------------

/// 创建计算命令缓冲区
fn create_compute_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    // 分配计算命令缓冲区
    data.compute_command_buffers
        .resize(MAX_FRAMES_IN_FLIGHT, vk::CommandBuffer::null());

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.compute_command_buffers.len() as u32);

    data.compute_command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };

    Ok(())
}

/// 记录计算命令缓冲区
fn record_compute_command_buffer(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
    current_frame: usize,
) -> Result<()> {
    let begin_info = vk::CommandBufferBeginInfo::default();

    // SAFETY: 开始命令缓冲区是不安全的
    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
    }

    // 绑定计算管线
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.compute_pipeline,
        );

        // 绑定描述符集
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.compute_pipeline_layout,
            0,
            &[data.compute_descriptor_sets[current_frame]],
            &[],
        );

        // 分派计算工作组
        // 假设每个工作组有256个工作项，总共有PARTICLE_COUNT个粒子
        device.cmd_dispatch(command_buffer, PARTICLE_COUNT as u32 / 256, 1, 1);

        device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}

fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    if data.framebuffers.is_empty() {
        return Ok(());
    }

    // 释放旧的命令缓冲区
    if !data.command_buffers.is_empty() {
        unsafe {
            device.free_command_buffers(data.command_pool, &data.command_buffers);
        }
        data.command_buffers.clear();
    }

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    // 分配新的命令缓冲区
    data.command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };

    Ok(())
}

/// 创建同步对象（信号量和栅栏）
fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    // SAFETY: 销毁信号量/栅栏是不安全的。
    unsafe {
        for s in data.image_available_semaphores.drain(..) {
            if s != vk::Semaphore::null() {
                device.destroy_semaphore(s, None);
            }
        }
        for s in data.render_finished_semaphores.drain(..) {
            if s != vk::Semaphore::null() {
                device.destroy_semaphore(s, None);
            }
        }
        for s in data.compute_finished_semaphores.drain(..) {
            if s != vk::Semaphore::null() {
                device.destroy_semaphore(s, None);
            }
        }
        for f in data.graphics_fences.drain(..) {
            if f != vk::Fence::null() {
                device.destroy_fence(f, None);
            }
        }
        for f in data.compute_fences.drain(..) {
            if f != vk::Fence::null() {
                device.destroy_fence(f, None);
            }
        }
    }

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    data.image_available_semaphores.clear();
    data.render_finished_semaphores.clear();
    data.compute_finished_semaphores.clear();
    data.graphics_fences.clear();
    data.compute_fences.clear();

    // 预分配向量大小
    data.image_available_semaphores
        .reserve(MAX_FRAMES_IN_FLIGHT);
    data.render_finished_semaphores
        .reserve(MAX_FRAMES_IN_FLIGHT);
    data.compute_finished_semaphores
        .reserve(MAX_FRAMES_IN_FLIGHT);
    data.graphics_fences.reserve(MAX_FRAMES_IN_FLIGHT);
    data.compute_fences.reserve(MAX_FRAMES_IN_FLIGHT);

    // 创建所有帧的同步对象
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        unsafe {
            // 创建图形渲染同步对象
            let image_available = device.create_semaphore(&semaphore_info, None)?;
            let render_finished = device.create_semaphore(&semaphore_info, None)?;
            let in_flight_fence = device.create_fence(&fence_info, None)?;

            data.image_available_semaphores.push(image_available);
            data.render_finished_semaphores.push(render_finished);
            data.graphics_fences.push(in_flight_fence);

            // 创建计算着色器同步对象
            let compute_finished = device.create_semaphore(&semaphore_info, None)?;
            let compute_fence = device.create_fence(&fence_info, None)?;

            data.compute_finished_semaphores.push(compute_finished);
            data.compute_fences.push(compute_fence);
        }
    }

    // 为交换链图像创建栅栏跟踪数组（值为null的引用）
    data.swapchain_image_fences = vec![vk::Fence::null(); data.swapchain_images.len()];

    Ok(())
}

//==================================================================================================
// SECTION: Internal Vulkan Helper Functions (Buffer/Image Creation, Commands, etc.)
//==================================================================================================

/// Creates a Vulkan buffer and allocates its memory. (Internal Helper)
fn create_buffer_internal(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: `create_buffer` is unsafe. Device and buffer_info must be valid.
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    // SAFETY: `get_buffer_memory_requirements` is unsafe. Device and buffer must be valid.
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_type_index = get_memory_type_index_internal(
        instance,
        data.physical_device,
        properties,
        mem_requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    // SAFETY: `allocate_memory` and `bind_buffer_memory` are unsafe.
    // Device, alloc_info, buffer, and memory must be valid.
    let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

    Ok((buffer, buffer_memory))
}

/// Copies data from a source buffer to a destination buffer. (Internal Helper)
fn copy_buffer_internal(
    device: &Device,
    data: &AppData,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands_internal(device, data)?;

    let copy_region = vk::BufferCopy::default().size(size);
    // SAFETY: `cmd_copy_buffer` is unsafe. Command buffer and buffers must be valid.
    unsafe { device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]) };

    end_single_time_commands_internal(device, data, command_buffer)?;
    Ok(())
}

/// Creates a shader module from SPIR-V bytecode. (Internal Helper)
fn create_shader_module_internal(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = Cursor::new(bytecode);
    let code = ash::util::read_spv(&mut cursor)
        .map_err(|e| anyhow!("Failed to read SPIR-V bytecode: {}", e))?;
    if code.is_empty() {
        return Err(anyhow!("SPIR-V code is empty after reading."));
    }
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: `create_shader_module` is unsafe. Device, create_info must be valid.
    unsafe { Ok(device.create_shader_module(&create_info, None)?) }
}

/// Finds a suitable memory type index for a given memory requirement and properties. (Internal Helper)
fn get_memory_type_index_internal(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_properties: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> Result<u32> {
    // SAFETY: `get_physical_device_memory_properties` is an unsafe Vulkan call.
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..device_memory_properties.memory_type_count {
        let type_filter_met = (memory_requirements.memory_type_bits & (1 << i)) != 0;
        let properties_met = device_memory_properties.memory_types[i as usize]
            .property_flags
            .contains(required_properties);

        if type_filter_met && properties_met {
            return Ok(i);
        }
    }
    Err(anyhow!("Failed to find suitable memory type."))
}

/// Begins a single-time command buffer for short-lived operations. (Internal Helper)
fn begin_single_time_commands_internal(
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    // SAFETY: `allocate_command_buffers` is unsafe. Device and alloc_info must be valid.
    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `begin_command_buffer` is unsafe. Command buffer and begin_info must be valid.
    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

    Ok(command_buffer)
}

/// Ends, submits, and frees a single-time command buffer. (Internal Helper)
fn end_single_time_commands_internal(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    // SAFETY: `end_command_buffer` is unsafe. Command buffer must be valid.
    unsafe { device.end_command_buffer(command_buffer)? };

    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: `queue_submit`, `queue_wait_idle`, `free_command_buffers` are unsafe.
    // All parameters and objects must be valid.
    unsafe {
        device.queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;
        device.free_command_buffers(data.command_pool, &[command_buffer]);
    }
    Ok(())
}

//==================================================================================================
// SECTION: Winit Application Handler
//==================================================================================================
#[derive(Default)]
struct AppHandler {
    window: Option<Window>,
    vulkan_app: Option<VulkanApp>,
    minimized: bool,
}

impl ApplicationHandler for AppHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("AppHandler: Resumed.");
        if self.window.is_some() {
            if let Some(app) = self.vulkan_app.as_mut() {
                if let Some(window) = self.window.as_ref() {
                    if let Err(e) = app.recreate_swapchain(window) {
                        error!("Failed to recreate swapchain on resume: {:?}", e);
                        event_loop.exit();
                    }
                }
            }
            return;
        }

        // 创建窗口，类似于 C++ 中的 initWindow()
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Tutorial (Rust) - 34 Compute Shaders")
            .with_inner_size(LogicalSize::new(1024.0, 768.0));

        let window = match event_loop.create_window(window_attributes) {
            Ok(win) => win,
            Err(e) => {
                error!("Failed to create window: {:?}", e);
                event_loop.exit();
                return;
            }
        };

        // 初始化 Vulkan，类似于 C++ 中的 initVulkan()
        match VulkanApp::create(&window) {
            Ok(app) => {
                self.vulkan_app = Some(app);
                info!("AppHandler: VulkanApp created successfully.");
            }
            Err(e) => {
                error!("Failed to create VulkanApp: {:?}", e);
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
        self.minimized = false;
    }

    // 处理窗口事件，相当于 C++ 中的 framebufferResizeCallback
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("AppHandler: CloseRequested. Exiting.");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                info!("AppHandler: Window resized to {:?}", new_size);
                if new_size.width == 0 || new_size.height == 0 {
                    self.minimized = true;
                } else {
                    self.minimized = false;
                    if let Some(app) = self.vulkan_app.as_mut() {
                        app.resized = true; // 类似于 C++ 中的 framebufferResized = true
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if self.minimized {
                    return;
                }
                // 绘制帧，相当于 C++ 中的 drawFrame()
                if let (Some(app), Some(window)) = (self.vulkan_app.as_mut(), self.window.as_ref())
                {
                    if let Err(e) = app.render(window) {
                        error!("Error during VulkanApp render: {:?}", e);
                        if let Some(vk_err) = e.downcast_ref::<vk::Result>() {
                            match *vk_err {
                                vk::Result::ERROR_DEVICE_LOST => {
                                    error!("Device lost, exiting.");
                                    event_loop.exit();
                                }
                                vk::Result::ERROR_OUT_OF_DATE_KHR => {
                                    warn!(
                                        "Render returned OUT_OF_DATE_KHR, attempting to recreate swapchain."
                                    );
                                    app.resized = true; // Force recreate on next frame
                                }
                                _ => {
                                    error!("Unhandled Vulkan render error: {:?}", vk_err);
                                    event_loop.exit();
                                }
                            }
                        } else {
                            error!("Non-Vulkan error during render: {:?}", e);
                            event_loop.exit();
                        }
                    }
                }
            }
            _ => (),
        }
    }

    // 类似于 C++ 中的主循环，持续请求重绘
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            if !self.minimized && self.vulkan_app.is_some() {
                window.request_redraw();
            }
        }
    }

    // 清理资源
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("AppHandler: Exiting. Cleaning up VulkanApp.");
        if let Some(mut app) = self.vulkan_app.take() {
            app.destroy();
        }
        self.window = None;
        info!("AppHandler: Cleanup complete.");
    }
}

//==================================================================================================
// SECTION: Main Application Entry Point
//==================================================================================================

pub fn main() -> Result<()> {
    // Initialize logger. Ensure RUST_LOG environment variable is set (e.g., RUST_LOG=info).
    pretty_env_logger::init();
    info!("Starting application with winit ApplicationHandler API...");

    let event_loop = EventLoop::new().map_err(|e| anyhow!("Failed to create event loop: {}", e))?;

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app_handler = AppHandler::default();
    let run_result = event_loop.run_app(&mut app_handler);

    println!("DEBUG: event_loop.run_app has returned.");

    if let Err(e) = run_result {
        error!("Event loop error: {}", e);
        return Err(anyhow!("Event loop failed: {}", e));
    }

    info!("Application finished normally.");
    Ok(())
}
