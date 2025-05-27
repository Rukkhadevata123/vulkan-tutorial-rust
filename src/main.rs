// 常量定义和类型别名模块
// 包含应用程序使用的所有常量、类型别名和基础数据结构

#![allow(unsafe_code)]

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::mem::{offset_of, size_of};
use std::os::raw::{c_char, c_void};
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use log::*;

use ash::vk;
use ash::vk::Handle;
use ash::{Device, Entry, Instance};

use anyhow::{Result, anyhow};
use nalgebra::Unit;
use thiserror::Error;

mod vk_window;
use vk_window::*;

//==================================================================================================
// 应用程序常量配置
//==================================================================================================

/// 是否启用验证层（调试模式下自动启用）
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// Vulkan验证层名称
const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// 设备扩展列表
const DEVICE_EXTENSIONS: &[&CStr] = &[c"VK_KHR_swapchain"];

/// 最大并发帧数（用于帧资源管理）
const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// 粒子系统中的粒子数量
const PARTICLE_COUNT: usize = 8192;

//==================================================================================================
// 数学类型别名
//==================================================================================================

/// 二维浮点向量
type Vec2 = nalgebra::Vector2<f32>;

/// 三维浮点向量  
type Vec3 = nalgebra::Vector3<f32>;

/// 四维浮点向量
type Vec4 = nalgebra::Vector4<f32>;

/// 三维浮点点
type Point3 = nalgebra::Point3<f32>;

/// 4x4浮点矩阵
type Mat4 = nalgebra::Matrix4<f32>;

//==================================================================================================
// 顶点数据结构
//==================================================================================================

/// 模型顶点数据结构
/// 包含位置、颜色和纹理坐标信息
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ModelVertex {
    pos: Vec3,       // 顶点位置
    color: Vec3,     // 顶点颜色
    tex_coord: Vec2, // 纹理坐标
}

impl ModelVertex {
    /// 创建新的模型顶点
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    /// 获取顶点输入绑定描述
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<ModelVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// 获取顶点属性描述数组
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            // 位置属性
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(ModelVertex, pos) as u32),
            // 颜色属性
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(ModelVertex, color) as u32),
            // 纹理坐标属性
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(ModelVertex, tex_coord) as u32),
        ]
    }
}

impl PartialEq for ModelVertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for ModelVertex {}

impl Hash for ModelVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

/// 粒子数据结构
/// 包含位置、速度和颜色信息
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Particle {
    position: Vec2, // 粒子位置
    velocity: Vec2, // 粒子速度
    color: Vec4,    // 粒子颜色（包含透明度）
}

impl Particle {
    /// 创建新的粒子
    const fn new(position: Vec2, velocity: Vec2, color: Vec4) -> Self {
        Self {
            position,
            velocity,
            color,
        }
    }

    /// 获取粒子顶点输入绑定描述
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// 获取粒子顶点属性描述数组
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // 位置属性
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Particle, position) as u32),
            // 颜色属性
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Particle, color) as u32),
        ]
    }
}

//==================================================================================================
// 统一缓冲区对象 (UBO)
//==================================================================================================

/// 模型渲染统一缓冲区数据
/// 包含视图和投影矩阵
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ModelUBO {
    view: Mat4, // 视图矩阵
    proj: Mat4, // 投影矩阵
}

/// 粒子系统统一缓冲区数据
/// 包含时间相关信息
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ParticleUBO {
    delta_time: f32, // 帧间时间差（毫秒）
    time: f32,       // 总时间（毫秒）
}

//==================================================================================================
// Vulkan 设备支持查询结构
//==================================================================================================

/// 队列族索引结构
/// 存储图形、计算和呈现队列的索引
#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32, // 图形队列族索引
    compute: u32,  // 计算队列族索引
    present: u32,  // 呈现队列族索引
}

/// 交换链支持信息
/// 包含表面能力、格式和呈现模式
#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR, // 表面能力
    formats: Vec<vk::SurfaceFormatKHR>,       // 支持的格式
    present_modes: Vec<vk::PresentModeKHR>,   // 支持的呈现模式
}

//==================================================================================================
// 错误类型定义
//==================================================================================================

/// 物理设备适用性检查错误类型
#[derive(Debug, Error)]
pub enum SuitabilityError {
    #[error("静态错误: {0}")]
    Static(&'static str),
    #[error("动态错误: {0}")]
    Dynamic(String),
}

// 核心数据结构模块
// 包含AppData和VulkanApp的定义，以及相关的实现方法

//==================================================================================================
// 应用程序数据结构
//==================================================================================================

/// 应用程序状态数据
/// 包含所有Vulkan对象和应用程序状态信息
#[derive(Clone, Debug, Default)]
struct AppData {
    // 调试相关
    messenger: vk::DebugUtilsMessengerEXT,

    // 表面和设备
    surface: vk::SurfaceKHR,
    msaa_samples: vk::SampleCountFlags,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    compute_queue: vk::Queue,
    present_queue: vk::Queue,

    // 交换链资源
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,

    // 渲染通道和管线
    render_pass: vk::RenderPass,

    // 模型系统
    model_descriptor_set_layout: vk::DescriptorSetLayout,
    model_pipeline_layout: vk::PipelineLayout,
    model_pipeline: vk::Pipeline,

    // 粒子系统
    particle_descriptor_set_layout: vk::DescriptorSetLayout,
    particle_pipeline_layout: vk::PipelineLayout,
    particle_pipeline: vk::Pipeline,
    particle_compute_pipeline_layout: vk::PipelineLayout,
    particle_compute_pipeline: vk::Pipeline,

    // 帧缓冲区
    framebuffers: Vec<vk::Framebuffer>,

    // 命令池
    command_pool: vk::CommandPool,

    // 纹理资源
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,

    // 深度缓冲区
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    // MSAA颜色图像
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,

    // 模型数据
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,

    // 模型缓冲区
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    model_uniform_buffers: Vec<vk::Buffer>,
    model_uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // 粒子缓冲区
    particle_storage_buffers: Vec<vk::Buffer>,
    particle_storage_buffers_memory: Vec<vk::DeviceMemory>,
    particle_uniform_buffers: Vec<vk::Buffer>,
    particle_uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // 描述符资源
    model_descriptor_pool: vk::DescriptorPool,
    model_descriptor_sets: Vec<vk::DescriptorSet>,
    particle_descriptor_pool: vk::DescriptorPool,
    particle_descriptor_sets: Vec<vk::DescriptorSet>,

    // 命令缓冲区
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    compute_command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,

    // 同步对象
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    compute_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

/// Vulkan应用程序主类
/// 管理整个应用程序的生命周期和渲染流程
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
    models: usize,
}

//==================================================================================================
// 队列族查询实现
//==================================================================================================

impl QueueFamilyIndices {
    /// 查询物理设备的队列族支持情况
    /// 优先查找同时支持图形和计算的队列族，以减少队列族切换开销
    fn get(
        instance: &Instance,
        entry: &Entry,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        // 优先寻找同时支持图形和计算的队列族
        let graphics_and_compute = properties
            .iter()
            .position(|p| {
                p.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map(|i| i as u32);

        // 如果没有找到同时支持的，分别寻找
        let (graphics, compute) = if let Some(combined) = graphics_and_compute {
            (combined, combined)
        } else {
            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

            let compute = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);

            match (graphics, compute) {
                (Some(g), Some(c)) => (g, c),
                _ => {
                    return Err(anyhow!(SuitabilityError::Static(
                        "缺少必需的图形或计算队列族。"
                    )));
                }
            }
        };

        // 查找支持呈现的队列族
        let mut present = None;
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
        for (index, _properties) in properties.iter().enumerate() {
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

        if let Some(present) = present {
            Ok(Self {
                graphics,
                compute,
                present,
            })
        } else {
            Err(anyhow!(SuitabilityError::Static("缺少必需的呈现队列族。")))
        }
    }
}

//==================================================================================================
// 交换链支持查询实现
//==================================================================================================

impl SwapchainSupport {
    /// 查询物理设备的交换链支持情况
    /// 获取表面能力、支持的格式和呈现模式
    fn get(
        instance: &Instance,
        entry: &Entry,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
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
// VulkanApp核心方法实现
//==================================================================================================

impl VulkanApp {
    /// 初始化Vulkan应用程序
    /// 按正确顺序创建所有Vulkan对象和资源
    fn create(window: &Window) -> Result<Self> {
        let entry =
            unsafe { Entry::load().map_err(|e| anyhow!("无法加载Vulkan入口点: {}", e))? };
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
        model_create_descriptor_set_layout(&device, &mut data)?;
        particle_create_descriptor_set_layout(&device, &mut data)?;

        // 管线创建
        model_create_pipeline(&device, &mut data)?;
        particle_create_pipeline(&device, &mut data)?;
        particle_create_compute_pipeline(&device, &mut data)?;

        // 命令和缓冲区
        vulkan_create_command_pools(&instance, &device, &entry, &mut data)?;
        vulkan_create_color_objects(&instance, &device, &mut data)?;
        vulkan_create_depth_objects(&instance, &device, &mut data)?;
        vulkan_create_framebuffers(&device, &mut data)?;

        // 纹理资源
        texture_create_image(&instance, &device, &mut data)?;
        texture_create_image_view(&device, &mut data)?;
        texture_create_sampler(&device, &instance, &mut data)?;

        // 模型资源
        model_load_data(&mut data)?;
        model_create_vertex_buffer(&instance, &device, &mut data)?;
        model_create_index_buffer(&instance, &device, &mut data)?;
        model_create_uniform_buffers(&instance, &device, &mut data)?;

        // 粒子资源
        particle_create_storage_buffers(&instance, &device, &mut data)?;
        particle_create_uniform_buffers(&instance, &device, &mut data)?;

        // 描述符资源
        model_create_descriptor_pool(&device, &mut data)?;
        particle_create_descriptor_pool(&device, &mut data)?;
        model_create_descriptor_sets(&device, &mut data)?;
        particle_create_descriptor_sets(&device, &mut data)?;

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
    fn destroy(&mut self) {
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
    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
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
        model_create_pipeline(&self.device, &mut self.data)?;
        particle_create_pipeline(&self.device, &mut self.data)?;
        particle_create_compute_pipeline(&self.device, &mut self.data)?;
        vulkan_create_color_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan_create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan_create_framebuffers(&self.device, &mut self.data)?;
        model_create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        particle_create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        model_create_descriptor_pool(&self.device, &mut self.data)?;
        particle_create_descriptor_pool(&self.device, &mut self.data)?;
        model_create_descriptor_sets(&self.device, &mut self.data)?;
        particle_create_descriptor_sets(&self.device, &mut self.data)?;
        vulkan_create_command_buffers(&self.device, &mut self.data)?;
        vulkan_create_compute_command_buffers(&self.device, &mut self.data)?;

        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());
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
            if self.data.model_descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.model_descriptor_pool, None);
                self.data.model_descriptor_pool = vk::DescriptorPool::null();
                self.data.model_descriptor_sets.clear();
            }

            if self.data.particle_descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.particle_descriptor_pool, None);
                self.data.particle_descriptor_pool = vk::DescriptorPool::null();
                self.data.particle_descriptor_sets.clear();
            }

            // 清理统一缓冲区
            self.cleanup_uniform_buffers();

            // 清理帧缓冲区
            for &framebuffer in &self.data.framebuffers {
                if framebuffer != vk::Framebuffer::null() {
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
            if self.data.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = vk::RenderPass::null();
            }

            // 清理交换链图像视图
            for &image_view in &self.data.swapchain_image_views {
                if image_view != vk::ImageView::null() {
                    self.device.destroy_image_view(image_view, None);
                }
            }
            self.data.swapchain_image_views.clear();

            // 清理交换链
            if self.data.swapchain != vk::SwapchainKHR::null() {
                let swapchain_device =
                    ash::khr::swapchain::Device::new(&self.instance, &self.device);
                swapchain_device.destroy_swapchain(self.data.swapchain, None);
                self.data.swapchain = vk::SwapchainKHR::null();
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
                if memory != vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.model_uniform_buffers {
                if buffer != vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
            self.data.model_uniform_buffers.clear();
            self.data.model_uniform_buffers_memory.clear();

            // 粒子统一缓冲区
            for &memory in &self.data.particle_uniform_buffers_memory {
                if memory != vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.particle_uniform_buffers {
                if buffer != vk::Buffer::null() {
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
            if self.data.color_image_view != vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.color_image_view, None);
                self.data.color_image_view = vk::ImageView::null();
            }
            if self.data.color_image != vk::Image::null() {
                self.device.destroy_image(self.data.color_image, None);
                self.data.color_image = vk::Image::null();
            }
            if self.data.color_image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.data.color_image_memory, None);
                self.data.color_image_memory = vk::DeviceMemory::null();
            }
        }
    }

    /// 清理深度缓冲区资源
    fn cleanup_depth_resources(&mut self) {
        unsafe {
            if self.data.depth_image_view != vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.depth_image_view, None);
                self.data.depth_image_view = vk::ImageView::null();
            }
            if self.data.depth_image != vk::Image::null() {
                self.device.destroy_image(self.data.depth_image, None);
                self.data.depth_image = vk::Image::null();
            }
            if self.data.depth_image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.data.depth_image_memory, None);
                self.data.depth_image_memory = vk::DeviceMemory::null();
            }
        }
    }

    /// 清理渲染管线
    fn cleanup_pipelines(&mut self) {
        unsafe {
            if self.data.model_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.data.model_pipeline, None);
                self.data.model_pipeline = vk::Pipeline::null();
            }
            if self.data.particle_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.particle_pipeline, None);
                self.data.particle_pipeline = vk::Pipeline::null();
            }
            if self.data.particle_compute_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.data.particle_compute_pipeline, None);
                self.data.particle_compute_pipeline = vk::Pipeline::null();
            }

            // 管线布局
            if self.data.model_pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.model_pipeline_layout, None);
                self.data.model_pipeline_layout = vk::PipelineLayout::null();
            }
            if self.data.particle_pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.particle_pipeline_layout, None);
                self.data.particle_pipeline_layout = vk::PipelineLayout::null();
            }
            if self.data.particle_compute_pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.particle_compute_pipeline_layout, None);
                self.data.particle_compute_pipeline_layout = vk::PipelineLayout::null();
            }
        }
    }

    /// 清理命令缓冲区
    fn cleanup_command_buffers(&mut self) {
        unsafe {
            // 主命令缓冲区
            for i in 0..self.data.command_buffers.len() {
                if self.data.command_buffers[i] != vk::CommandBuffer::null()
                    && i < self.data.command_pools.len()
                    && self.data.command_pools[i] != vk::CommandPool::null()
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
                if command_buffer != vk::CommandBuffer::null()
                    && self.data.command_pool != vk::CommandPool::null()
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
                    && self.data.command_pools[i] != vk::CommandPool::null()
                {
                    for &buffer in secondary_buffers.iter() {
                        if buffer != vk::CommandBuffer::null() {
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
                if fence != vk::Fence::null() {
                    self.device.destroy_fence(fence, None);
                }
            }
            for &semaphore in &self.data.render_finished_semaphores {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            for &semaphore in &self.data.image_available_semaphores {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            for &semaphore in &self.data.compute_finished_semaphores {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
        }
    }

    /// 清理命令池
    fn cleanup_command_pools(&mut self) {
        unsafe {
            for &pool in &self.data.command_pools {
                if pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(pool, None);
                }
            }
            if self.data.command_pool != vk::CommandPool::null() {
                self.device
                    .destroy_command_pool(self.data.command_pool, None);
            }
        }
    }

    /// 清理模型相关资源
    fn cleanup_model_resources(&mut self) {
        unsafe {
            if self.data.index_buffer_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.data.index_buffer_memory, None);
            }
            if self.data.index_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.data.index_buffer, None);
            }
            if self.data.vertex_buffer_memory != vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.vertex_buffer_memory, None);
            }
            if self.data.vertex_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.data.vertex_buffer, None);
            }
        }
    }

    /// 清理粒子系统资源
    fn cleanup_particle_resources(&mut self) {
        unsafe {
            for &memory in &self.data.particle_storage_buffers_memory {
                if memory != vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for &buffer in &self.data.particle_storage_buffers {
                if buffer != vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
        }
    }

    /// 清理纹理资源
    fn cleanup_texture_resources(&mut self) {
        unsafe {
            if self.data.texture_sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.data.texture_sampler, None);
            }
            if self.data.texture_image_view != vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.texture_image_view, None);
            }
            if self.data.texture_image_memory != vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.texture_image_memory, None);
            }
            if self.data.texture_image != vk::Image::null() {
                self.device.destroy_image(self.data.texture_image, None);
            }
        }
    }

    /// 清理描述符集布局
    fn cleanup_descriptor_layouts(&mut self) {
        unsafe {
            if self.data.model_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.model_descriptor_set_layout, None);
            }
            if self.data.particle_descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.particle_descriptor_set_layout, None);
            }
        }
    }

    /// 清理表面
    fn cleanup_surface(&mut self) {
        unsafe {
            if self.data.surface != vk::SurfaceKHR::null() {
                let surface_instance =
                    ash::khr::surface::Instance::new(&self.entry, &self.instance);
                surface_instance.destroy_surface(self.data.surface, None);
            }
        }
    }

    /// 清理调试信使
    fn cleanup_debug_messenger(&mut self) {
        unsafe {
            if VALIDATION_ENABLED && self.data.messenger != vk::DebugUtilsMessengerEXT::null() {
                let debug_utils = ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
                debug_utils.destroy_debug_utils_messenger(self.data.messenger, None);
            }
        }
    }
}

// 资源管理模块
// 包含缓冲区、图像、内存管理等通用资源操作函数

//==================================================================================================
// 缓冲区管理操作
//==================================================================================================

/// 创建Vulkan缓冲区并分配内存
/// 统一的缓冲区创建接口，处理内存分配和绑定
fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // 创建缓冲区信息
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // 创建缓冲区
    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| anyhow!("创建缓冲区失败: {}", e))?
    };

    // 获取内存需求
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    // 查找合适的内存类型
    let mem_type_index =
        find_memory_type(instance, data.physical_device, properties, mem_requirements)?;

    // 分配内存
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let buffer_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| anyhow!("分配缓冲区内存失败: {}", e))?
    };

    // 绑定缓冲区和内存
    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| anyhow!("绑定缓冲区内存失败: {}", e))?
    };

    Ok((buffer, buffer_memory))
}

/// 复制缓冲区数据
/// 从源缓冲区复制数据到目标缓冲区
fn copy_buffer(
    device: &Device,
    data: &AppData,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let copy_region = vk::BufferCopy::default().size(size);

    unsafe {
        device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);
    }

    end_single_time_commands(device, data, command_buffer)?;
    Ok(())
}

/// 映射并写入缓冲区数据
/// 通用的缓冲区数据上传函数
fn write_buffer_data<T>(
    device: &Device,
    buffer_memory: vk::DeviceMemory,
    data: &[T],
) -> Result<()> {
    let size = (std::mem::size_of_val(data)) as vk::DeviceSize;

    unsafe {
        let memory_ptr = device.map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(data.as_ptr(), memory_ptr.cast(), data.len());
        device.unmap_memory(buffer_memory);
    }

    Ok(())
}

//==================================================================================================
// 图像和纹理管理操作
//==================================================================================================

/// 创建Vulkan图像并分配内存
/// 统一的图像创建接口
#[allow(clippy::too_many_arguments)]
fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // 创建图像信息
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples);

    // 创建图像
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| anyhow!("创建图像失败: {}", e))?
    };

    // 获取内存需求
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

    // 查找合适的内存类型
    let mem_type_index =
        find_memory_type(instance, data.physical_device, properties, mem_requirements)?;

    // 分配内存
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let image_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| anyhow!("分配图像内存失败: {}", e))?
    };

    // 绑定图像和内存
    unsafe {
        device
            .bind_image_memory(image, image_memory, 0)
            .map_err(|e| anyhow!("绑定图像内存失败: {}", e))?
    };

    Ok((image, image_memory))
}

/// 创建图像视图
/// 从图像创建视图用于着色器访问
fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    unsafe {
        device
            .create_image_view(&create_info, None)
            .map_err(|e| anyhow!("创建图像视图失败: {}", e))
    }
}

/// 转换图像布局
/// 使用管线屏障转换图像布局
fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    // 确定访问掩码和管线阶段
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => {
                return Err(anyhow!(
                    "不支持的图像布局转换: {:?} -> {:?}",
                    old_layout,
                    new_layout
                ));
            }
        };

    let command_buffer = begin_single_time_commands(device, data)?;

    // 确定图像方面
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::DEPTH,
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let subresource = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    end_single_time_commands(device, data, command_buffer)?;
    Ok(())
}

/// 从缓冲区复制数据到图像
fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    end_single_time_commands(device, data, command_buffer)?;
    Ok(())
}

/// 生成Mipmap
/// 为纹理生成多级渐远纹理
#[allow(clippy::too_many_arguments)]
fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // 检查物理设备是否支持线性过滤
    unsafe {
        if !instance
            .get_physical_device_format_properties(data.physical_device, format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return Err(anyhow!("纹理图像格式不支持线性blit操作"));
        }
    }

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    // 为每个mip级别生成数据
    for i in 1..mip_levels {
        // 转换上一级mip为传输源
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // 设置blit操作
        let src_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        unsafe {
            device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }

        // 转换上一级mip为着色器可读
        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // 更新mip尺寸
        if mip_width > 1 {
            mip_width /= 2;
        }
        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    // 转换最后一个mip级别为着色器可读
    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    end_single_time_commands(device, data, command_buffer)?;
    Ok(())
}

//==================================================================================================
// 内存管理操作
//==================================================================================================

/// 查找合适的内存类型
/// 根据内存需求和属性查找匹配的内存类型索引
fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_properties: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> Result<u32> {
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

    Err(anyhow!("找不到合适的内存类型"))
}

//==================================================================================================
// 着色器管理操作
//==================================================================================================

/// 从SPIR-V字节码创建着色器模块
fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = Cursor::new(bytecode);
    let code =
        ash::util::read_spv(&mut cursor).map_err(|e| anyhow!("读取SPIR-V字节码失败: {}", e))?;

    if code.is_empty() {
        return Err(anyhow!("读取后SPIR-V代码为空"));
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| anyhow!("创建着色器模块失败: {}", e))
    }
}

//==================================================================================================
// 命令缓冲区管理操作
//==================================================================================================

/// 开始一次性命令缓冲区
/// 用于短期操作的命令缓冲区
fn begin_single_time_commands(device: &Device, data: &AppData) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| anyhow!("分配一次性命令缓冲区失败: {}", e))?[0]
    };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(|e| anyhow!("开始命令缓冲区记录失败: {}", e))?
    };

    Ok(command_buffer)
}

/// 结束并提交一次性命令缓冲区
/// 完成命令记录，提交执行并等待完成
fn end_single_time_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .map_err(|e| anyhow!("结束命令缓冲区记录失败: {}", e))?
    };

    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

    unsafe {
        device
            .queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())
            .map_err(|e| anyhow!("提交命令缓冲区失败: {}", e))?;
        device
            .queue_wait_idle(data.graphics_queue)
            .map_err(|e| anyhow!("等待队列空闲失败: {}", e))?;
        device.free_command_buffers(data.command_pool, &[command_buffer]);
    }

    Ok(())
}

//==================================================================================================
// 格式和能力查询操作
//==================================================================================================

/// 获取支持的格式
/// 从候选格式中选择第一个支持指定特性的格式
fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    unsafe {
        candidates
            .iter()
            .cloned()
            .find(|&format| {
                let properties =
                    instance.get_physical_device_format_properties(data.physical_device, format);
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => {
                        properties.optimal_tiling_features.contains(features)
                    }
                    _ => false,
                }
            })
            .ok_or_else(|| anyhow!("找不到支持的格式"))
    }
}

/// 获取深度格式
/// 查找支持深度模版附件的格式
fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D16_UNORM,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

/// 获取最大MSAA采样数
/// 查找设备支持的最高MSAA采样数
fn get_max_msaa_samples(instance: &Instance, data: &AppData) -> vk::SampleCountFlags {
    let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;

    // 按优先级顺序尝试不同的采样数
    [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ]
    .iter()
    .find(|&&sample_count| counts.contains(sample_count))
    .copied()
    .unwrap_or(vk::SampleCountFlags::TYPE_1)
}

//==================================================================================================
// 描述符管理操作
//==================================================================================================

/// 通用描述符池创建参数
#[derive(Debug)]
struct DescriptorPoolConfig {
    /// 池大小配置
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    /// 最大描述符集数量
    max_sets: u32,
    /// 池创建标志
    flags: vk::DescriptorPoolCreateFlags,
}

impl DescriptorPoolConfig {
    /// 创建新的描述符池配置
    fn new() -> Self {
        Self {
            pool_sizes: Vec::new(),
            max_sets: 0,
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        }
    }

    /// 添加池大小
    fn add_pool_size(mut self, descriptor_type: vk::DescriptorType, count: u32) -> Self {
        self.pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(descriptor_type)
                .descriptor_count(count),
        );
        self
    }

    /// 设置最大描述符集数量
    fn max_sets(mut self, max_sets: u32) -> Self {
        self.max_sets = max_sets;
        self
    }

    /// 设置创建标志
    fn flags(mut self, flags: vk::DescriptorPoolCreateFlags) -> Self {
        self.flags = flags;
        self
    }
}

/// 通用描述符池创建函数
fn create_descriptor_pool(
    device: &Device,
    config: DescriptorPoolConfig,
) -> Result<vk::DescriptorPool> {
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&config.pool_sizes)
        .max_sets(config.max_sets)
        .flags(config.flags);

    unsafe {
        device
            .create_descriptor_pool(&create_info, None)
            .map_err(|e| anyhow!("创建描述符池失败: {}", e))
    }
}

// 模型系统模块
// 包含模型加载、缓冲区管理、描述符和渲染相关功能

//==================================================================================================
// 模型数据加载操作
//==================================================================================================

/// 加载模型数据
/// 从OBJ文件读取顶点和索引数据，去除重复顶点
fn model_load_data(data: &mut AppData) -> Result<()> {
    let mut reader = BufReader::new(File::open("assets/models/viking_room.obj")?);
    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    let mut unique_vertices = HashMap::new();

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = ModelVertex::new(
                Vec3::new(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                Vec3::new(1.0, 1.0, 1.0), // 白色
                Vec2::new(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            );

            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    info!(
        "加载模型完成: {} 顶点, {} 索引",
        data.vertices.len(),
        data.indices.len()
    );
    Ok(())
}

//==================================================================================================
// 模型缓冲区管理操作
//==================================================================================================

/// 创建模型顶点缓冲区
/// 将顶点数据上传到GPU内存
fn model_create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let buffer_size = (size_of::<ModelVertex>() * data.vertices.len()) as vk::DeviceSize;

    // 创建暂存缓冲区用于数据传输
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传顶点数据到暂存缓冲区
    write_buffer_data(device, staging_buffer_memory, &data.vertices)?;

    // 创建设备本地顶点缓冲区
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    // 从暂存缓冲区复制到顶点缓冲区
    copy_buffer(device, data, staging_buffer, vertex_buffer, buffer_size)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    info!("模型顶点缓冲区创建完成");
    Ok(())
}

/// 创建模型索引缓冲区
/// 将索引数据上传到GPU内存
fn model_create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let buffer_size = (size_of::<u32>() * data.indices.len()) as vk::DeviceSize;

    // 创建暂存缓冲区用于数据传输
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传索引数据到暂存缓冲区
    write_buffer_data(device, staging_buffer_memory, &data.indices)?;

    // 创建设备本地索引缓冲区
    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    // 从暂存缓冲区复制到索引缓冲区
    copy_buffer(device, data, staging_buffer, index_buffer, buffer_size)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    info!("模型索引缓冲区创建完成");
    Ok(())
}

/// 创建模型统一缓冲区
/// 为每个交换链图像创建一个统一缓冲区
fn model_create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // 清理已有的统一缓冲区
    model_cleanup_uniform_buffers(device, data);

    let buffer_size = size_of::<ModelUBO>() as vk::DeviceSize;
    let image_count = data.swapchain_images.len();

    // 为每个交换链图像创建统一缓冲区
    for _ in 0..image_count {
        // 使用 _i 表示有意未使用
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        data.model_uniform_buffers.push(uniform_buffer);
        data.model_uniform_buffers_memory
            .push(uniform_buffer_memory);
    }

    info!("模型统一缓冲区创建完成: {} 个", image_count);
    Ok(())
}

/// 清理模型统一缓冲区
fn model_cleanup_uniform_buffers(device: &Device, data: &mut AppData) {
    unsafe {
        for &memory in &data.model_uniform_buffers_memory {
            if memory != vk::DeviceMemory::null() {
                device.free_memory(memory, None);
            }
        }
        for &buffer in &data.model_uniform_buffers {
            if buffer != vk::Buffer::null() {
                device.destroy_buffer(buffer, None);
            }
        }
    }
    data.model_uniform_buffers.clear();
    data.model_uniform_buffers_memory.clear();
}

//==================================================================================================
// 模型描述符系统操作
//==================================================================================================

/// 创建模型描述符集布局
/// 定义统一缓冲区和纹理采样器的绑定
fn model_create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let bindings = [
        // 绑定0: 模型统一缓冲区 (视图和投影矩阵)
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        // 绑定1: 纹理采样器
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    data.model_descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| anyhow!("创建模型描述符集布局失败: {}", e))?
    };

    info!("模型描述符集布局创建完成");
    Ok(())
}

/// 创建模型描述符池
/// 为模型描述符集分配池空间
fn model_create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的描述符池
    if data.model_descriptor_pool != vk::DescriptorPool::null() {
        unsafe {
            device.destroy_descriptor_pool(data.model_descriptor_pool, None);
        }
        data.model_descriptor_pool = vk::DescriptorPool::null();
        data.model_descriptor_sets.clear();
    }

    let image_count = data.swapchain_images.len() as u32;
    if image_count == 0 {
        return Ok(());
    }

    // 配置描述符池
    let config = DescriptorPoolConfig::new()
        .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, image_count)
        .add_pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, image_count)
        .max_sets(image_count)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    data.model_descriptor_pool = create_descriptor_pool(device, config)?;

    info!("模型描述符池创建完成: 最大集合数 {}", image_count);
    Ok(())
}

/// 创建并更新模型描述符集
/// 为每个交换链图像分配并配置描述符集
fn model_create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let image_count = data.swapchain_images.len();

    if image_count == 0 || data.model_descriptor_pool == vk::DescriptorPool::null() {
        return Ok(());
    }

    data.model_descriptor_sets.clear();

    // 为每个交换链图像准备相同的布局
    let layouts = vec![data.model_descriptor_set_layout; image_count];

    // 分配描述符集
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(data.model_descriptor_pool)
        .set_layouts(&layouts);

    data.model_descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| anyhow!("分配模型描述符集失败: {}", e))?
    };

    // 更新每个描述符集
    for i in 0..image_count {
        model_update_descriptor_set(device, data, i)?;
    }

    info!("模型描述符集创建完成: {} 个", image_count);
    Ok(())
}

/// 更新单个模型描述符集
/// 绑定统一缓冲区和纹理资源到描述符集
fn model_update_descriptor_set(device: &Device, data: &AppData, image_index: usize) -> Result<()> {
    // 统一缓冲区信息
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(data.model_uniform_buffers[image_index])
        .offset(0)
        .range(size_of::<ModelUBO>() as u64);

    // 纹理图像信息
    let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(data.texture_image_view)
        .sampler(data.texture_sampler);

    // 描述符写入操作
    let descriptor_writes = [
        // 写入统一缓冲区
        vk::WriteDescriptorSet::default()
            .dst_set(data.model_descriptor_sets[image_index])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info)),
        // 写入纹理采样器
        vk::WriteDescriptorSet::default()
            .dst_set(data.model_descriptor_sets[image_index])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info)),
    ];

    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(())
}

//==================================================================================================
// 模型渲染管线操作
//==================================================================================================

/// 创建模型图形管线
/// 配置顶点输入、着色器阶段和渲染状态
fn model_create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载着色器字节码
    let vert_shader_spirv = include_bytes!("../assets/shaders/35_viking_room.vert.spv");
    let frag_shader_spirv = include_bytes!("../assets/shaders/35_viking_room.frag.spv");

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

    info!("模型图形管线创建完成");
    Ok(())
}

//==================================================================================================
// 模型渲染操作
//==================================================================================================

/// 更新模型统一缓冲区
/// 计算并上传视图和投影矩阵
fn model_update_uniform_buffer(app: &VulkanApp, image_index: usize) -> Result<()> {
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
fn model_update_secondary_command_buffer(
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
    let model_matrix = model_calculate_transform_matrix(app, model_index);
    let opacity = model_calculate_opacity(app, model_index);

    // 录制命令缓冲区
    model_record_secondary_commands(app, command_buffer, image_index, &model_matrix, opacity)?;

    Ok(command_buffer)
}

/// 计算模型变换矩阵
/// 为每个模型生成不同的位置、旋转和缩放
fn model_calculate_transform_matrix(app: &VulkanApp, model_index: usize) -> Mat4 {
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
fn model_calculate_opacity(app: &VulkanApp, model_index: usize) -> f32 {
    0.7 + (0.3 * model_index as f32 / app.models.max(1) as f32)
}

/// 录制模型二级命令
/// 将模型渲染命令录制到二级命令缓冲区
fn model_record_secondary_commands(
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
fn model_render_all(
    app: &mut VulkanApp,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
) -> Result<()> {
    if app.models == 0 {
        return Ok(());
    }

    // 更新所有模型的二级命令缓冲区
    let secondary_command_buffers = (0..app.models)
        .map(|i| model_update_secondary_command_buffer(app, image_index, i))
        .collect::<Result<Vec<_>, _>>()?;

    // 执行二级命令缓冲区
    unsafe {
        app.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers);
    }

    Ok(())
}

// 粒子系统模块
// 包含粒子缓冲区管理、计算管线、描述符和渲染相关功能

//==================================================================================================
// 粒子缓冲区管理操作
//==================================================================================================

/// 创建粒子存储缓冲区
/// 初始化粒子数据并为每个飞行帧创建存储缓冲区
fn particle_create_storage_buffers(
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

    let buffer_size = (std::mem::size_of::<Particle>() * PARTICLE_COUNT) as vk::DeviceSize;

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

    info!(
        "粒子存储缓冲区创建完成: {} 个缓冲区，每个包含 {} 粒子",
        MAX_FRAMES_IN_FLIGHT, PARTICLE_COUNT
    );
    Ok(())
}

/// 创建粒子统一缓冲区
/// 为每个飞行帧创建统一缓冲区（时间信息）
fn particle_create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let buffer_size = size_of::<ParticleUBO>() as vk::DeviceSize;

    // 清理已有的粒子统一缓冲区
    particle_cleanup_uniform_buffers(device, data);

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

    info!("粒子统一缓冲区创建完成: {} 个", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

/// 清理粒子统一缓冲区
fn particle_cleanup_uniform_buffers(device: &Device, data: &mut AppData) {
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

//==================================================================================================
// 粒子描述符系统操作
//==================================================================================================

/// 创建粒子描述符集布局
/// 定义计算着色器使用的描述符绑定
fn particle_create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
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

    info!("粒子描述符集布局创建完成");
    Ok(())
}

/// 创建粒子描述符池
/// 为粒子描述符集分配池空间
fn particle_create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
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

    info!("粒子描述符池创建完成: 最大集合数 {}", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

/// 创建并更新粒子描述符集
/// 为每个飞行帧分配并配置描述符集
fn particle_create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
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
        particle_update_descriptor_set(device, data, i)?;
    }

    info!("粒子描述符集创建完成: {} 个", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

/// 更新单个粒子描述符集
/// 绑定统一缓冲区和存储缓冲区到描述符集
fn particle_update_descriptor_set(
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

//==================================================================================================
// 粒子管线创建操作
//==================================================================================================

/// 创建粒子图形管线
/// 配置粒子渲染的图形管线
fn particle_create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载粒子着色器字节码
    let vert_shader_spirv = include_bytes!("../assets/shaders/35_particle.vert.spv");
    let frag_shader_spirv = include_bytes!("../assets/shaders/35_particle.frag.spv");

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

    info!("粒子图形管线创建完成");
    Ok(())
}

/// 创建粒子计算管线
/// 配置粒子物理模拟的计算管线
fn particle_create_compute_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 加载计算着色器字节码
    let compute_shader_spirv = include_bytes!("../assets/shaders/35_particle.comp.spv");

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

    info!("粒子计算管线创建完成");
    Ok(())
}

//==================================================================================================
// 粒子渲染和计算操作
//==================================================================================================

/// 更新粒子统一缓冲区
/// 上传时间信息到GPU用于粒子物理模拟
fn particle_update_uniform_buffer(app: &VulkanApp) -> Result<()> {
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

/// 录制粒子计算命令缓冲区
/// 将粒子物理模拟命令录制到计算命令缓冲区
fn particle_record_compute_commands(
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
        let workgroup_count = (PARTICLE_COUNT as u32 + 255) / 256; // 向上取整到256的倍数
        device.cmd_dispatch(command_buffer, workgroup_count, 1, 1);

        device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}

/// 渲染粒子系统
/// 在图形渲染通道中渲染粒子
fn particle_render(app: &VulkanApp, command_buffer: vk::CommandBuffer) -> Result<()> {
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

// 纹理系统模块
// 包含纹理加载、图像处理、采样器管理等功能

//==================================================================================================
// 纹理图像管理操作
//==================================================================================================

/// 创建纹理图像
/// 从文件加载纹理数据并创建Vulkan图像资源
fn texture_create_image(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let img_path = "assets/textures/viking_room.png";
    let img = image::open(img_path)
        .map_err(|e| anyhow!("无法打开纹理图像 '{}': {}", img_path, e))?
        .into_rgba8();

    let (width, height) = img.dimensions();
    if width != 1024 || height != 1024 {
        return Err(anyhow!(
            "无效的纹理图像尺寸 {}x{}，应为 1024x1024",
            width,
            height
        ));
    }

    let image_data = img.into_raw();
    let image_size = (width * height * 4) as vk::DeviceSize;

    // 计算mipmap级别数
    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    // 创建暂存缓冲区用于上传纹理数据
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // 上传纹理数据到暂存缓冲区
    write_buffer_data(device, staging_buffer_memory, &image_data)?;

    // 创建纹理图像
    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::TYPE_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // 转换图像布局为传输目标
    transition_image_layout(
        device,
        data,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;

    // 从缓冲区复制数据到图像
    copy_buffer_to_image(device, data, staging_buffer, texture_image, width, height)?;

    // 清理暂存缓冲区
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }

    // 生成mipmap级别
    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    info!(
        "纹理图像创建完成: {}x{}, {} mip级别",
        width, height, data.mip_levels
    );
    Ok(())
}

/// 创建纹理图像视图
/// 为纹理图像创建视图以供着色器访问
fn texture_create_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    info!("纹理图像视图创建完成");
    Ok(())
}

/// 创建纹理采样器
/// 配置纹理采样参数
fn texture_create_sampler(device: &Device, instance: &Instance, data: &mut AppData) -> Result<()> {
    // 获取物理设备属性以确定各向异性过滤支持
    let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };

    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(properties.limits.max_sampler_anisotropy.min(16.0))
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(data.mip_levels as f32);

    data.texture_sampler = unsafe {
        device
            .create_sampler(&create_info, None)
            .map_err(|e| anyhow!("创建纹理采样器失败: {}", e))?
    };

    info!("纹理采样器创建完成");
    Ok(())
}

// Vulkan核心初始化模块
// 包含实例、设备、交换链等核心Vulkan对象的创建

//==================================================================================================
// Vulkan实例和调试设置
//==================================================================================================

/// 创建Vulkan实例并设置调试消息
/// 初始化Vulkan环境和验证层
fn vulkan_create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // 应用程序信息
    let app_name = CString::new("Vulkan Tutorial (Rust)")?;
    let engine_name = CString::new("No Engine")?;

    let application_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    // 检查验证层支持
    let available_layers = unsafe { entry.enumerate_instance_layer_properties()? }
        .iter()
        .map(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) })
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED
        && !available_layers
            .iter()
            .any(|&layer| layer == VALIDATION_LAYER_NAME)
    {
        return Err(anyhow!("请求的验证层不受支持"));
    }

    // 获取所需扩展
    let required_extensions_cstrs = get_required_instance_extensions(window);
    let mut extensions_ptrs: Vec<*const c_char> = required_extensions_cstrs
        .iter()
        .map(|e| e.as_ptr())
        .collect();

    if VALIDATION_ENABLED {
        extensions_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    // 设置验证层
    let layers_names_raw = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER_NAME.as_ptr()]
    } else {
        Vec::new()
    };

    // 调试信息配置
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
        .pfn_user_callback(Some(vulkan_debug_callback));

    // 实例创建信息
    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extensions_ptrs);

    if VALIDATION_ENABLED {
        create_info = create_info.push_next(&mut debug_info);
    }

    // 创建Vulkan实例
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| anyhow!("创建Vulkan实例失败: {}", e))?
    };

    // 设置调试回调
    if VALIDATION_ENABLED {
        let debug_utils_instance = ash::ext::debug_utils::Instance::new(entry, &instance);
        data.messenger = unsafe {
            debug_utils_instance
                .create_debug_utils_messenger(&debug_info, None)
                .map_err(|e| anyhow!("创建调试信使失败: {}", e))?
        };
    }

    info!("Vulkan实例创建完成");
    Ok(instance)
}

/// Vulkan调试回调函数
/// 处理验证层消息并输出到日志
extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { &*data };
    let message = unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() };

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) 验证层: {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) 验证层: {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) 验证层: {}", type_, message);
    } else {
        trace!("({:?}) 验证层: {}", type_, message);
    }
    vk::FALSE
}

//==================================================================================================
// 物理设备选择和逻辑设备创建
//==================================================================================================

/// 选择合适的物理设备
/// 遍历可用GPU并选择最适合的设备
fn vulkan_pick_physical_device(
    instance: &Instance,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| anyhow!("枚举物理设备失败: {}", e))?
    };

    if physical_devices.is_empty() {
        return Err(anyhow!("找不到支持Vulkan的GPU"));
    }

    for physical_device in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name =
            unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };

        if let Err(error) = vulkan_check_device_suitability(instance, entry, data, physical_device)
        {
            warn!("跳过物理设备 ({}): {}", device_name, error);
        } else {
            info!("选择的物理设备: {}", device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            info!("最大MSAA采样数: {:?}", data.msaa_samples);
            return Ok(());
        }
    }

    Err(anyhow!("找不到合适的物理设备"))
}

/// 检查物理设备适用性
/// 验证设备是否满足应用程序需求
fn vulkan_check_device_suitability(
    instance: &Instance,
    entry: &Entry,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // 检查队列族支持
    QueueFamilyIndices::get(instance, entry, data, physical_device)?;

    // 检查设备扩展支持
    vulkan_check_device_extensions(instance, physical_device)?;

    // 检查交换链支持
    let support = SwapchainSupport::get(instance, entry, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError::Static("交换链支持不足")));
    }

    // 检查设备特性
    let mut features2_query = vk::PhysicalDeviceFeatures2::default();
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features2_query);
    }

    if features2_query.features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError::Static("不支持采样器各向异性")));
    }

    Ok(())
}

/// 检查设备扩展支持
/// 验证所需的设备扩展是否可用
fn vulkan_check_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|e| anyhow!("枚举设备扩展失败: {}", e))?
    }
    .iter()
    .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
    .collect::<HashSet<_>>();

    for &required_ext in DEVICE_EXTENSIONS.iter() {
        if !available_extensions.contains(required_ext) {
            return Err(anyhow!(SuitabilityError::Dynamic(format!(
                "缺少必需的设备扩展: {}",
                required_ext.to_string_lossy()
            ))));
        }
    }

    Ok(())
}

/// 创建逻辑设备
/// 从物理设备创建逻辑设备并获取队列句柄
fn vulkan_create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;

    // 创建唯一队列族集合
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.compute);
    unique_indices.insert(indices.present);

    // 队列创建信息
    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|&index| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(index)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // 设备扩展
    let extension_ptrs: Vec<*const c_char> =
        DEVICE_EXTENSIONS.iter().map(|ext| ext.as_ptr()).collect();

    // 设备特性配置
    let base_features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .sample_rate_shading(true);

    let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();

    let mut features_chain = vk::PhysicalDeviceFeatures2::default()
        .features(base_features)
        .push_next(&mut vulkan_1_2_features)
        .push_next(&mut vulkan_1_3_features);

    // 设备创建信息
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_ptrs)
        .push_next(&mut features_chain);

    // 创建逻辑设备
    let device = unsafe {
        instance
            .create_device(data.physical_device, &create_info, None)
            .map_err(|e| anyhow!("创建逻辑设备失败: {}", e))?
    };

    // 获取队列句柄
    unsafe {
        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
        data.compute_queue = device.get_device_queue(indices.compute, 0);
        data.present_queue = device.get_device_queue(indices.present, 0);
    }

    info!(
        "逻辑设备创建完成 - 图形队列: {}, 计算队列: {}, 呈现队列: {}",
        indices.graphics, indices.compute, indices.present
    );
    Ok(device)
}

//==================================================================================================
// 交换链和图像视图创建
//==================================================================================================

/// 创建交换链
/// 配置并创建用于呈现的交换链
fn vulkan_create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, entry, data, data.physical_device)?;

    // 选择交换链配置
    let surface_format = vulkan_choose_swap_surface_format(&support.formats);
    let present_mode = vulkan_choose_swap_present_mode(&support.present_modes);
    let extent = vulkan_choose_swap_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    // 计算图像数量
    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    // 处理队列族共享模式
    let mut queue_family_indices_vec = vec![];
    let image_sharing_mode = {
        let mut unique_families = HashSet::new();
        unique_families.insert(indices.graphics);
        unique_families.insert(indices.compute);
        unique_families.insert(indices.present);

        if unique_families.len() > 1 {
            queue_family_indices_vec.extend(unique_families);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        }
    };

    // 交换链创建信息
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

    // 创建交换链并获取图像
    unsafe {
        data.swapchain = swapchain_loader
            .create_swapchain(&create_info, None)
            .map_err(|e| anyhow!("创建交换链失败: {}", e))?;
        data.swapchain_images = swapchain_loader
            .get_swapchain_images(data.swapchain)
            .map_err(|e| anyhow!("获取交换链图像失败: {}", e))?;
    }

    info!(
        "交换链创建完成: {}x{}, {} 图像, 格式: {:?}",
        extent.width,
        extent.height,
        data.swapchain_images.len(),
        surface_format.format
    );
    Ok(())
}

/// 选择交换链表面格式
/// 优先选择SRGB格式
fn vulkan_choose_swap_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .copied()
        .unwrap_or(formats[0])
}

/// 选择交换链呈现模式
/// 优先选择三重缓冲模式
fn vulkan_choose_swap_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
        .copied()
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// 选择交换链范围
/// 确定交换链图像的分辨率
fn vulkan_choose_swap_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
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

/// 创建交换链图像视图
/// 为每个交换链图像创建图像视图
fn vulkan_create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|&image| {
            create_image_view(
                device,
                image,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    info!(
        "交换链图像视图创建完成: {} 个",
        data.swapchain_image_views.len()
    );
    Ok(())
}

//==================================================================================================
// 渲染通道和帧缓冲区
//==================================================================================================

/// 创建渲染通道
/// 定义渲染目标和子通道配置
fn vulkan_create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // 颜色附件描述
    let color_attachment = vk::AttachmentDescription::default()
        .format(data.swapchain_format)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // 深度附件描述
    let depth_stencil_attachment = vk::AttachmentDescription::default()
        .format(get_depth_format(instance, data)?)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // 颜色解析附件描述
    let color_resolve_attachment = vk::AttachmentDescription::default()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // 附件引用
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment_ref = vk::AttachmentReference::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // 子通道配置
    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_stencil_attachment_ref)
        .resolve_attachments(resolve_attachments);

    // 子通道依赖
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    // 渲染通道创建信息
    let attachments = &[
        color_attachment,
        depth_stencil_attachment,
        color_resolve_attachment,
    ];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = unsafe {
        device
            .create_render_pass(&create_info, None)
            .map_err(|e| anyhow!("创建渲染通道失败: {}", e))?
    };

    info!("渲染通道创建完成");
    Ok(())
}

/// 创建帧缓冲区
/// 为每个交换链图像视图创建帧缓冲区
fn vulkan_create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|&image_view| {
            let attachments = &[data.color_image_view, data.depth_image_view, image_view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
        })
        .collect::<Result<Vec<_>, vk::Result>>()?;

    info!("帧缓冲区创建完成: {} 个", data.framebuffers.len());
    Ok(())
}

//==================================================================================================
// 深度和颜色缓冲区
//==================================================================================================

/// 创建深度缓冲区对象
/// 包括深度图像、内存和图像视图
fn vulkan_create_depth_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let format = get_depth_format(instance, data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        data.msaa_samples,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;
    data.depth_image_view =
        create_image_view(device, depth_image, format, vk::ImageAspectFlags::DEPTH, 1)?;

    transition_image_layout(
        device,
        data,
        depth_image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        1,
    )?;

    info!("深度缓冲区对象创建完成");
    Ok(())
}

/// 创建MSAA颜色对象
/// 包括颜色图像、内存和图像视图
fn vulkan_create_color_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let (color_image, color_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        data.msaa_samples,
        data.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;
    data.color_image_view = create_image_view(
        device,
        data.color_image,
        data.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    info!("MSAA颜色对象创建完成");
    Ok(())
}

//==================================================================================================
// 命令池和命令缓冲区
//==================================================================================================

/// 创建命令池
/// 为命令缓冲区分配创建命令池
fn vulkan_create_command_pools(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    // 全局命令池
    data.command_pool = vulkan_create_command_pool_internal(instance, device, entry, data)?;

    // 为每个交换链图像创建命令池
    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = vulkan_create_command_pool_internal(instance, device, entry, data)?;
        data.command_pools.push(command_pool);
    }

    info!("命令池创建完成: 1 个全局池 + {} 个图像池", num_images);
    Ok(())
}

/// 创建单个命令池
/// 内部辅助函数
fn vulkan_create_command_pool_internal(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &AppData,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    unsafe {
        device
            .create_command_pool(&info, None)
            .map_err(|e| anyhow!("创建命令池失败: {}", e))
    }
}

/// 创建命令缓冲区
/// 为每个交换链图像分配主命令缓冲区
fn vulkan_create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let num_images = data.swapchain_images.len();

    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .map_err(|e| anyhow!("分配命令缓冲区失败: {}", e))?[0]
        };
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    info!("命令缓冲区创建完成: {} 个主缓冲区", num_images);
    Ok(())
}

/// 创建计算命令缓冲区
/// 为每个飞行帧分配计算命令缓冲区
fn vulkan_create_compute_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    // 清理已有的计算命令缓冲区
    if !data.compute_command_buffers.is_empty() {
        unsafe {
            for &command_buffer in &data.compute_command_buffers {
                if command_buffer != vk::CommandBuffer::null()
                    && data.command_pool != vk::CommandPool::null()
                {
                    device.free_command_buffers(data.command_pool, &[command_buffer]);
                }
            }
        }
        data.compute_command_buffers.clear();
    }

    // 为每个飞行帧分配计算命令缓冲区
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .map_err(|e| anyhow!("分配计算命令缓冲区 {} 失败: {}", i, e))?[0]
        };

        data.compute_command_buffers.push(command_buffer);
    }

    info!("计算命令缓冲区创建完成: {} 个", MAX_FRAMES_IN_FLIGHT);
    Ok(())
}

//==================================================================================================
// 同步对象
//==================================================================================================

/// 创建同步对象
/// 为每个飞行帧创建信号量和围栏
fn vulkan_create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
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
fn vulkan_cleanup_sync_objects(device: &Device, data: &mut AppData) {
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

// 渲染循环模块
// 包含主渲染循环、命令缓冲区录制和帧渲染逻辑

impl VulkanApp {
    /// 主渲染函数
    /// 协调整个渲染管线的执行
    fn render(&mut self, window: &Window) -> Result<()> {
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

    /// 获取下一个交换链图像
    /// 处理交换链过期情况
    fn acquire_next_swapchain_image(&mut self, window: &Window) -> Result<Option<usize>> {
        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let result = unsafe {
            swapchain_device.acquire_next_image(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            )
        };

        match result {
            Ok((image_index, _)) => Ok(Some(image_index as usize)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)?;
                Ok(None)
            }
            Err(e) => Err(anyhow!("获取交换链图像失败: {}", e)),
        }
    }

    /// 更新帧数据
    /// 更新统一缓冲区和录制命令缓冲区
    fn update_frame_data(&mut self, image_index: usize) -> Result<()> {
        // 更新命令缓冲区
        self.update_command_buffer(image_index)?;

        // 更新模型统一缓冲区
        model_update_uniform_buffer(self, image_index)?;

        // 更新粒子统一缓冲区
        particle_update_uniform_buffer(self)?;

        Ok(())
    }

    /// 更新主命令缓冲区
    /// 录制渲染通道和绘制命令
    fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // 重置命令池
        let command_pool = self.data.command_pools[image_index];
        unsafe {
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
        }

        let command_buffer = self.data.command_buffers[image_index];

        // 开始录制命令
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(command_buffer, &info)?;
        }

        // 配置渲染通道
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let clear_values = &[
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0], // 黑色背景
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            // 1. 首先渲染粒子系统
            particle_render(self, command_buffer)?;

            // 2. 然后渲染模型（使用二级命令缓冲区）
            model_render_all(self, command_buffer, image_index)?;

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
        particle_record_compute_commands(
            &self.device,
            &self.data,
            self.data.compute_command_buffers[self.frame],
            self.frame,
        )?;

        // 创建数组以确保生命周期足够长
        let compute_command_buffers = [self.data.compute_command_buffers[self.frame]];
        let compute_signal_semaphores = [self.data.compute_finished_semaphores[self.frame]];

        // 计算命令提交信息
        let compute_submit_info = vk::SubmitInfo::default()
            .command_buffers(&compute_command_buffers)
            .signal_semaphores(&compute_signal_semaphores);

        // 图形命令提交信息 - 等待图像可用和计算完成
        let wait_semaphores = &[
            self.data.image_available_semaphores[self.frame],
            self.data.compute_finished_semaphores[self.frame],
        ];
        let wait_stages = &[
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::VERTEX_INPUT, // 等待顶点输入阶段
        ];
        let command_buffers_submit = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];

        let graphics_submit_info = vk::SubmitInfo::default()
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
                vk::Fence::null(),
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

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices_present);

        let result =
            unsafe { swapchain_device.queue_present(self.data.present_queue, &present_info) };

        let changed = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Ok(false) => false,
            Err(e) => return Err(anyhow!("呈现帧失败: {}", e)),
        };

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        }

        Ok(())
    }
}

// Winit应用程序处理器模块
// 包含窗口事件处理和应用程序生命周期管理

//==================================================================================================
// 应用程序事件处理器
//==================================================================================================

/// Winit应用程序处理器
/// 管理窗口生命周期和事件处理
#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan_app: Option<VulkanApp>,
    minimized: bool,
}

impl ApplicationHandler for App {
    /// 应用程序恢复处理
    /// 当应用程序重新获得焦点时调用
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // 创建窗口
            match self.create_window(event_loop) {
                Ok(window) => {
                    info!("窗口创建成功");

                    // 初始化Vulkan应用程序
                    match VulkanApp::create(&window) {
                        Ok(vulkan_app) => {
                            info!("Vulkan应用程序初始化成功");
                            self.vulkan_app = Some(vulkan_app);
                            self.window = Some(window);
                        }
                        Err(e) => {
                            error!("Vulkan应用程序初始化失败: {}", e);
                            self.exit_with_error(event_loop, &e);
                        }
                    }
                }
                Err(e) => {
                    error!("窗口创建失败: {}", e);
                    self.exit_with_error(event_loop, &e);
                }
            }
        }
    }

    /// 窗口事件处理
    /// 处理所有窗口相关事件
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // 窗口关闭请求
            WindowEvent::CloseRequested => {
                info!("接收到窗口关闭请求");
                self.cleanup_and_exit(event_loop);
            }

            // 键盘输入事件
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key_press(&event.physical_key, event_loop);
                }
            }

            // 窗口大小改变事件
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    info!("窗口最小化");
                    self.minimized = true;
                } else {
                    if self.minimized {
                        info!("窗口恢复显示: {}x{}", size.width, size.height);
                        self.minimized = false;
                    } else {
                        info!("窗口大小改变: {}x{}", size.width, size.height);
                    }
                    self.handle_resize();
                }
            }

            // 重绘请求事件
            WindowEvent::RedrawRequested => {
                self.handle_redraw(event_loop);
            }

            _ => {} // 忽略其他事件
        }
    }

    /// 应用程序退出处理
    /// 在应用程序完全退出前进行清理
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("应用程序正在退出");
        self.cleanup_vulkan();
    }
}

//==================================================================================================
// 窗口管理方法
//==================================================================================================

impl App {
    /// 创建应用程序窗口
    /// 配置窗口属性并创建窗口实例
    fn create_window(&self, event_loop: &ActiveEventLoop) -> Result<Window> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Tutorial (Rust) - 多模型 + 粒子系统")
            .with_inner_size(LogicalSize::new(1024, 768))
            .with_resizable(true);

        event_loop
            .create_window(window_attributes)
            .map_err(|e| anyhow!("创建窗口失败: {}", e))
    }

    /// 处理窗口大小改变
    /// 标记Vulkan应用程序需要重建交换链
    fn handle_resize(&mut self) {
        if let Some(ref mut vulkan_app) = self.vulkan_app {
            vulkan_app.resized = true;
            debug!("标记交换链需要重建");
        }
    }

    /// 处理重绘请求
    /// 执行Vulkan渲染循环
    fn handle_redraw(&mut self, event_loop: &ActiveEventLoop) {
        // 跳过最小化状态的渲染
        if self.minimized {
            return;
        }

        match (&mut self.vulkan_app, &self.window) {
            (Some(vulkan_app), Some(window)) => {
                // 执行渲染
                if let Err(e) = vulkan_app.render(window) {
                    error!("渲染失败: {}", e);
                    self.exit_with_error(event_loop, &e);
                    return;
                }

                // 请求下一帧
                window.request_redraw();
            }
            _ => {
                warn!("渲染跳过: Vulkan应用程序或窗口未初始化");
            }
        }
    }
}

//==================================================================================================
// 输入处理方法
//==================================================================================================

impl App {
    /// 处理按键事件
    /// 响应用户键盘输入
    fn handle_key_press(&mut self, key: &PhysicalKey, event_loop: &ActiveEventLoop) {
        match key {
            // ESC键退出应用程序
            PhysicalKey::Code(KeyCode::Escape) => {
                info!("按下ESC键，退出应用程序");
                self.cleanup_and_exit(event_loop);
            }

            // 左箭头键减少模型数量
            PhysicalKey::Code(KeyCode::ArrowLeft) => {
                if let Some(ref mut vulkan_app) = self.vulkan_app {
                    if vulkan_app.models > 1 {
                        vulkan_app.models -= 1;
                        info!("减少模型数量至: {}", vulkan_app.models);
                    }
                }
            }

            // 右箭头键增加模型数量
            PhysicalKey::Code(KeyCode::ArrowRight) => {
                if let Some(ref mut vulkan_app) = self.vulkan_app {
                    if vulkan_app.models < 10 {
                        vulkan_app.models += 1;
                        info!("增加模型数量至: {}", vulkan_app.models);
                    }
                }
            }

            // F1键显示帮助信息
            PhysicalKey::Code(KeyCode::F1) => {
                self.show_help();
            }

            // F11键切换全屏模式
            PhysicalKey::Code(KeyCode::F11) => {
                self.toggle_fullscreen();
            }

            _ => {} // 忽略其他按键
        }
    }

    /// 显示帮助信息
    /// 输出控制说明到日志
    fn show_help(&self) {
        info!("=== 控制说明 ===");
        info!("ESC       - 退出应用程序");
        info!("←/→       - 减少/增加模型数量 (1-10)");
        info!("F1        - 显示此帮助信息");
        info!("F11       - 切换全屏模式");
        info!(
            "当前模型数量: {}",
            self.vulkan_app.as_ref().map_or(0, |app| app.models)
        );
    }

    /// 切换全屏模式
    /// 在窗口模式和全屏模式之间切换
    fn toggle_fullscreen(&mut self) {
        // 注意这里改为 &mut self
        if let Some(ref window) = self.window {
            let is_fullscreen = window.fullscreen().is_some();

            if is_fullscreen {
                info!("退出全屏模式");
                window.set_fullscreen(None);

                // 恢复窗口大小
                window.set_min_inner_size(Some(LogicalSize::new(1024.0, 768.0)));
            } else {
                info!("进入全屏模式");

                // 获取主显示器
                if let Some(monitor) = window
                    .primary_monitor()
                    .or_else(|| window.current_monitor())
                {
                    let monitor_name = monitor.name().unwrap_or_else(|| "Unknown".to_string());
                    let monitor_size = monitor.size();

                    info!(
                        "目标显示器: {} ({}x{})",
                        monitor_name, monitor_size.width, monitor_size.height
                    );

                    // 首先尝试无边框全屏
                    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(
                        monitor.clone(),
                    ))));

                    // 标记需要重建交换链（重要！）
                    if let Some(ref mut vulkan_app) = self.vulkan_app {
                        vulkan_app.resized = true;
                    }
                } else {
                    error!("无法获取任何显示器信息");
                }
            }

            // 请求重绘
            window.request_redraw();
        }
    }
}

//==================================================================================================
// 清理和错误处理方法
//==================================================================================================

impl App {
    /// 清理并退出应用程序
    /// 正常退出流程
    fn cleanup_and_exit(&mut self, event_loop: &ActiveEventLoop) {
        info!("开始清理应用程序资源");
        self.cleanup_vulkan();
        event_loop.exit();
    }

    /// 错误退出
    /// 发生不可恢复错误时的退出流程
    fn exit_with_error(&mut self, event_loop: &ActiveEventLoop, error: &anyhow::Error) {
        error!("应用程序遇到严重错误: {}", error);

        // 输出详细错误信息
        let mut source = error.source();
        let mut level = 1;
        while let Some(err) = source {
            error!("  原因 {}: {}", level, err);
            source = err.source();
            level += 1;
        }

        self.cleanup_vulkan();
        event_loop.exit();
    }

    /// 清理Vulkan资源
    /// 安全销毁所有Vulkan对象
    fn cleanup_vulkan(&mut self) {
        if let Some(mut vulkan_app) = self.vulkan_app.take() {
            info!("清理Vulkan资源");
            vulkan_app.destroy();
            debug!("Vulkan资源清理完成");
        }

        if self.window.take().is_some() {
            debug!("窗口句柄已清理");
        }
    }
}

//==================================================================================================
// 工具函数
//==================================================================================================

/// 获取所需的实例扩展
/// 根据平台和配置返回必需的Vulkan实例扩展
fn get_required_instance_extensions(window: &Window) -> Vec<CString> {
    let mut extensions: Vec<CString> = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|&ext| CString::from(ext))
        .collect();

    if VALIDATION_ENABLED {
        extensions.push(CString::new("VK_EXT_debug_utils").unwrap());
    }

    debug!(
        "所需实例扩展: {:?}",
        extensions
            .iter()
            .map(|e| e.to_string_lossy())
            .collect::<Vec<_>>()
    );

    extensions
}

//==================================================================================================
// 主程序入口
//==================================================================================================

/// 应用程序主入口点
/// 初始化日志系统并启动事件循环
fn main() -> Result<()> {
    // 初始化日志系统 - 默认使用debug级别
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string());

    pretty_env_logger::formatted_builder()
        .filter_level(log_level.parse().unwrap_or(log::LevelFilter::Debug))
        .filter_module("sctk", log::LevelFilter::Warn) // 只显示 sctk 的警告和错误
        .filter_module("wayland", log::LevelFilter::Warn) // 可选: 同时过滤 wayland 相关日志
        .format_timestamp_secs()
        .init();

    info!("=== Vulkan教程应用程序启动 ===");
    info!("版本: 多模型渲染 + 粒子系统 + 计算着色器");
    info!("日志级别: {}", log::max_level());

    // 显示控制说明
    info!("控制说明:");
    info!("  ESC       - 退出应用程序");
    info!("  ←/→       - 减少/增加模型数量");
    info!("  F1        - 显示帮助信息");
    info!("  F11       - 切换全屏模式");

    // 创建事件循环
    let event_loop = EventLoop::new().map_err(|e| anyhow!("创建事件循环失败: {}", e))?;

    // 设置控制流为等待模式（节能）
    event_loop.set_control_flow(ControlFlow::Wait);

    // 创建应用程序实例
    let mut app = App::default();

    // 启动事件循环
    info!("启动事件循环");
    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("事件循环运行失败: {}", e))?;

    info!("应用程序正常退出");
    Ok(())
}
