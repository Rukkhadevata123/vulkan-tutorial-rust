//! 数据结构定义模块
//! 包含顶点、UBO、队列族等核心数据结构

use std::hash::{Hash, Hasher};
use std::mem::{offset_of, size_of};

use ash::vk;

use crate::constants::{Mat4, Vec2, Vec3, Vec4};

//==================================================================================================
// 顶点数据结构
//==================================================================================================

/// 模型顶点数据结构
/// 包含位置、颜色和纹理坐标信息
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    pub pos: Vec3,       // 顶点位置
    pub color: Vec3,     // 顶点颜色
    pub tex_coord: Vec2, // 纹理坐标
}

impl ModelVertex {
    /// 创建新的模型顶点
    pub const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    /// 获取顶点输入绑定描述
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<ModelVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// 获取顶点属性描述数组
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
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
pub struct Particle {
    pub position: Vec2, // 粒子位置
    pub velocity: Vec2, // 粒子速度
    pub color: Vec4,    // 粒子颜色（包含透明度）
}

impl Particle {
    /// 创建新的粒子
    pub const fn new(position: Vec2, velocity: Vec2, color: Vec4) -> Self {
        Self {
            position,
            velocity,
            color,
        }
    }

    /// 获取粒子顶点输入绑定描述
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// 获取粒子顶点属性描述数组
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
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
pub struct ModelUBO {
    pub view: Mat4, // 视图矩阵
    pub proj: Mat4, // 投影矩阵
}

/// 粒子系统统一缓冲区数据
/// 包含时间相关信息
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParticleUBO {
    pub delta_time: f32, // 帧间时间差（毫秒）
    pub time: f32,       // 总时间（毫秒）
}

//==================================================================================================
// Vulkan 设备支持查询结构
//==================================================================================================

/// 队列族索引结构
/// 存储图形、计算和呈现队列的索引
#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32, // 图形队列族索引
    pub compute: u32,  // 计算队列族索引
    pub present: u32,  // 呈现队列族索引
}

/// 交换链支持信息
/// 包含表面能力、格式和呈现模式
#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR, // 表面能力
    pub formats: Vec<vk::SurfaceFormatKHR>,       // 支持的格式
    pub present_modes: Vec<vk::PresentModeKHR>,   // 支持的呈现模式
}

//==================================================================================================
// 应用程序数据结构
//==================================================================================================

/// 应用程序状态数据
/// 包含所有Vulkan对象和应用程序状态信息
#[derive(Clone, Debug, Default)]
pub struct AppData {
    // 调试相关
    pub messenger: vk::DebugUtilsMessengerEXT,

    // 表面和设备
    pub surface: vk::SurfaceKHR,
    pub msaa_samples: vk::SampleCountFlags,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub present_queue: vk::Queue,

    // 交换链资源
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,

    // 渲染通道和管线
    pub render_pass: vk::RenderPass,

    // 模型系统
    pub model_descriptor_set_layout: vk::DescriptorSetLayout,
    pub model_pipeline_layout: vk::PipelineLayout,
    pub model_pipeline: vk::Pipeline,

    // 粒子系统
    pub particle_descriptor_set_layout: vk::DescriptorSetLayout,
    pub particle_pipeline_layout: vk::PipelineLayout,
    pub particle_pipeline: vk::Pipeline,
    pub particle_compute_pipeline_layout: vk::PipelineLayout,
    pub particle_compute_pipeline: vk::Pipeline,

    // 帧缓冲区
    pub framebuffers: Vec<vk::Framebuffer>,

    // 命令池
    pub command_pool: vk::CommandPool,

    // 纹理资源
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,

    // 深度缓冲区
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,

    // MSAA颜色图像
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,

    // 模型数据
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,

    // 模型缓冲区
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub model_uniform_buffers: Vec<vk::Buffer>,
    pub model_uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // 粒子缓冲区
    pub particle_storage_buffers: Vec<vk::Buffer>,
    pub particle_storage_buffers_memory: Vec<vk::DeviceMemory>,
    pub particle_uniform_buffers: Vec<vk::Buffer>,
    pub particle_uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // 描述符资源
    pub model_descriptor_pool: vk::DescriptorPool,
    pub model_descriptor_sets: Vec<vk::DescriptorSet>,
    pub particle_descriptor_pool: vk::DescriptorPool,
    pub particle_descriptor_sets: Vec<vk::DescriptorSet>,

    // 命令缓冲区
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub compute_command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,

    // 同步对象
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub compute_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}
