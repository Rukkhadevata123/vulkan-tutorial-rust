#![allow(unsafe_code)] // Allows unsafe blocks and unsafe fn calls within the crate

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::mem::{offset_of, size_of};
use std::os::raw::{c_char, c_void};
use std::ptr::copy_nonoverlapping as memcpy; // Alias for consistency
use std::time::Instant;

use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;

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
use nalgebra::Unit;
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

type Vec2 = nalgebra::Vector2<f32>;
type Vec3 = nalgebra::Vector3<f32>;
type Point3 = nalgebra::Point3<f32>;
type Mat4 = nalgebra::Matrix4<f32>;

//==================================================================================================
// SECTION: Core Vulkan Data Structures (Vertices, UBOs, Support Structs)
//==================================================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                // SAFETY: `pos` is a valid field of `Vertex`.
                .offset(offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                // SAFETY: `color` is a valid field of `Vertex`.
                .offset(offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                // SAFETY: `tex_coord` is a valid field of `Vertex`.
                .offset(offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
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
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
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

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
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
// SECTION: Load Model Data (Vertices, Indices)
//==================================================================================================

fn load_model(data: &mut AppData) -> Result<()> {
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

            let vertex = Vertex::new(
                Vec3::new(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                Vec3::new(1.0, 1.0, 1.0),
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
    Ok(())
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
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    command_pool: vk::CommandPool,
    // Texture
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    // Depth Buffer
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Model
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    command_buffers: Vec<vk::CommandBuffer>,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
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
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_command_pool(&instance, &device, &entry, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &instance, &mut data)?;
        load_model(&mut data)?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
        })
    }

    /// Renders a frame. Contains unsafe Vulkan calls.
    fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        // SAFETY: `wait_for_fences` is an unsafe Vulkan call. Device and fence must be valid.
        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
        }

        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        // SAFETY: `acquire_next_image` is an unsafe Vulkan call. Swapchain, semaphore must be valid.
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
                    return Ok(()); // Early return after recreating swapchain
                }
                Err(e) => return Err(anyhow!(e)),
            }
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            // SAFETY: `wait_for_fences` is an unsafe Vulkan call. Device and fence must be valid.
            unsafe {
                self.device
                    .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
            }
        }
        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers_submit = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers_submit)
            .signal_semaphores(signal_semaphores);

        // SAFETY: `reset_fences` and `queue_submit` are unsafe Vulkan calls.
        //         Device, queue, fence, and submit_info must be valid.
        unsafe {
            self.device.reset_fences(&[in_flight_fence])?;
            self.device
                .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;
        }

        let swapchains = &[self.data.swapchain];
        let image_indices_present = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices_present);

        // SAFETY: `queue_present` is an unsafe Vulkan call. Queue and present_info must be valid.
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

    /// Updates the uniform buffer for the current frame.
    fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();
        let rotation_speed = 0.2; // 稍微减慢旋转以便观察

        // 1. 定义缩放因子
        let scale_factor = 2.0; // 例如，放大5倍。根据viking_room模型调整这个值。
        // viking_room.obj 可能很大，你可能需要一个很小的值如 0.01 或 0.05
        // 如果是简单的立方体，5.0 可能太大。从 1.0 开始，然后调整。

        // 2. 创建缩放矩阵
        let scale_matrix = Mat4::new_scaling(scale_factor);

        // 3. 创建旋转矩阵
        let rotation_axis = Unit::new_normalize(*Vec3::z_axis()); // 单位化Y轴
        let rotation_matrix = Mat4::from_axis_angle(&rotation_axis, rotation_speed * time);

        // 4. 合并变换：先缩放，后旋转 (SRT)
        // 如果模型不是以原点为中心，你可能还需要一个平移矩阵。
        // 对于 viking_room.obj，它通常是以原点为中心的。
        let model_matrix = rotation_matrix * scale_matrix;
        // 或者，如果你想让它稍微向上/向下移动一点以更好地观察：
        // let translation_matrix = Mat4::from_translation(&Vec3::new(0.0, -1.0, 0.0)); // 向下移动一点（如果Y是上的话）
        // let model_matrix = translation_matrix * rotation_matrix * scale_matrix;

        // 相机设置 (View Matrix)
        // 让相机离远一点，或者调整观察目标
        let eye_position = Point3::new(0.0, 3.0, 3.0); // 稍微远一点，或者根据模型大小调整
        // 对于viking_room，可能需要 (10.0, 10.0, 10.0) 或更大
        let target_position = Point3::origin(); // 看着原点
        let up_vector = Vec3::z_axis(); // Y 轴朝上
        let view_matrix = Mat4::look_at_rh(&eye_position, &target_position, &up_vector);

        // 投影矩阵 (Projection Matrix)
        let aspect =
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32;
        // 对于较大的模型，你可能需要调整近平面和远平面
        let near_plane = 0.1;
        let far_plane = 100.0; // 增加远平面距离，以防模型太大被裁剪
        let mut proj_matrix =
            Mat4::new_perspective(aspect, 45.0f32.to_radians(), near_plane, far_plane);
        proj_matrix[(1, 1)] *= -1.0; // Vulkan Y-flip

        // Vulkan 深度范围 [0, 1] 的校正
        let vk_depth_correction = Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5,
            0.0, // z_ndc = z_clip * 0.5 / w_clip
            0.0, 0.0, 0.5, 1.0, // z_ndc = (z_clip * 0.5 + w_clip * 0.5) / w_clip
        );
        proj_matrix = vk_depth_correction * proj_matrix;

        let ubo = UniformBufferObject {
            model: model_matrix,
            view: view_matrix,
            proj: proj_matrix,
        };

        // SAFETY: `map_memory`, `memcpy`, and `unmap_memory` are unsafe.
        // Assumes `uniform_buffers_memory[image_index]` is valid and not in use by GPU for writing.
        // The mapped memory region must be valid.
        unsafe {
            let memory = self.device.map_memory(
                self.data.uniform_buffers_memory[image_index],
                0,
                size_of::<UniformBufferObject>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            memcpy(
                &ubo as *const _ as *const u8,    // Source
                memory.cast(),                    // Destination
                size_of::<UniformBufferObject>(), // Size
            );

            self.device
                .unmap_memory(self.data.uniform_buffers_memory[image_index]);
        }
        Ok(())
    }

    /// Recreates the swapchain and dependent resources when the window is resized or surface becomes outdated.
    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // SAFETY: `device_wait_idle` is an unsafe Vulkan call. Device must be valid.
        unsafe { self.device.device_wait_idle()? };
        self.destroy_swapchain_internal(); // Destroys old swapchain resources

        // Recreate swapchain and dependent resources
        create_swapchain(
            window,
            &self.instance,
            &self.device,
            &self.entry,
            &mut self.data,
        )?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?; // Render pass might depend on format
        create_pipeline(&self.device, &mut self.data)?; // Pipeline depends on extent and render pass
        create_depth_objects(&self.instance, &self.device, &mut self.data)?; // Depth image depends on swapchain
        create_framebuffers(&self.device, &mut self.data)?; // Framebuffers depend on image views and render pass
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?; // Uniform buffers per frame image
        create_descriptor_pool(&self.device, &mut self.data)?; // Descriptor pool might need resizing
        create_descriptor_sets(&self.device, &mut self.data)?; // Descriptor sets depend on pool and buffers
        create_command_buffers(&self.device, &mut self.data)?; // Command buffers record drawing with new resources

        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroys all Vulkan resources managed by the application.
    fn destroy(&mut self) {
        // SAFETY: `device_wait_idle` and all subsequent Vulkan destruction calls are unsafe.
        // It's crucial that these resources are not in use when destroyed.
        // Order of destruction matters.
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle before destruction");

            self.destroy_swapchain_internal(); // Destroys swapchain and related resources first

            // Sync objects
            for fence in self.data.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for semaphore in self.data.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.data.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }

            // Command pool (command buffers were freed in destroy_swapchain_internal)
            if self.data.command_pool != vk::CommandPool::null() {
                self.device
                    .destroy_command_pool(self.data.command_pool, None);
            }

            // Buffers and their memory
            if self.data.vertex_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.data.vertex_buffer, None);
            }
            if self.data.vertex_buffer_memory != vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.vertex_buffer_memory, None);
            }
            if self.data.index_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.data.index_buffer, None);
            }
            if self.data.index_buffer_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.data.index_buffer_memory, None);
            }

            // Texture resources
            if self.data.texture_sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.data.texture_sampler, None);
            }
            if self.data.texture_image_view != vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.texture_image_view, None);
            }
            if self.data.texture_image != vk::Image::null() {
                self.device.destroy_image(self.data.texture_image, None);
            }
            if self.data.texture_image_memory != vk::DeviceMemory::null() {
                self.device
                    .free_memory(self.data.texture_image_memory, None);
            }

            // Descriptor set layout
            if self.data.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
            }

            // Logical device
            self.device.destroy_device(None);
        }

        // Surface destruction must happen after logical device, before instance.
        let surface_instance = ash::khr::surface::Instance::new(&self.entry, &self.instance);
        // SAFETY: `destroy_surface` is an unsafe Vulkan call. Surface must be valid.
        unsafe { surface_instance.destroy_surface(self.data.surface, None) };

        // Debug messenger (if enabled)
        if VALIDATION_ENABLED && self.data.messenger != vk::DebugUtilsMessengerEXT::null() {
            let debug_utils_instance =
                ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
            // SAFETY: `destroy_debug_utils_messenger` is an unsafe Vulkan call. Messenger must be valid.
            unsafe {
                debug_utils_instance.destroy_debug_utils_messenger(self.data.messenger, None);
            }
        }
        // Instance
        // SAFETY: `destroy_instance` is an unsafe Vulkan call. Instance must be valid.
        unsafe { self.instance.destroy_instance(None) };
    }

    /// Destroys swapchain-related resources. Internal helper for `destroy` and `recreate_swapchain`.
    fn destroy_swapchain_internal(&mut self) {
        // SAFETY: All Vulkan destruction calls are unsafe. Assumes resources are not in use.
        // Order of destruction for swapchain-dependent resources.
        // SAFETY: All Vulkan destruction calls are unsafe. Assumes resources are not in use.
        // Order of destruction for swapchain-dependent resources.
        unsafe {
            // 1. 释放命令缓冲区
            if !self.data.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.data.command_pool, &self.data.command_buffers);
                self.data.command_buffers.clear();
            }

            // 2. 销毁描述符池（会自动释放关联的描述符集）
            if self.data.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.data.descriptor_pool, None);
                self.data.descriptor_pool = vk::DescriptorPool::null();
                self.data.descriptor_sets.clear();
            }

            // 3. 销毁Uniform缓冲区
            for memory in self.data.uniform_buffers_memory.drain(..) {
                if memory != vk::DeviceMemory::null() {
                    self.device.free_memory(memory, None);
                }
            }
            for buffer in self.data.uniform_buffers.drain(..) {
                if buffer != vk::Buffer::null() {
                    self.device.destroy_buffer(buffer, None);
                }
            }

            // 4. 销毁帧缓冲区（必须在深度图像视图之前销毁）
            for framebuffer in self.data.framebuffers.drain(..) {
                if framebuffer != vk::Framebuffer::null() {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
            }

            // 5. 销毁深度图像视图（新增）
            if self.data.depth_image_view != vk::ImageView::null() {
                self.device
                    .destroy_image_view(self.data.depth_image_view, None);
                self.data.depth_image_view = vk::ImageView::null();
            }

            // 6. 销毁深度图像（新增）
            if self.data.depth_image != vk::Image::null() {
                self.device.destroy_image(self.data.depth_image, None);
                self.data.depth_image = vk::Image::null();
            }

            // 7. 释放深度图像内存（新增）
            if self.data.depth_image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.data.depth_image_memory, None);
                self.data.depth_image_memory = vk::DeviceMemory::null();
            }

            // 8. 销毁图形管线
            if self.data.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.data.pipeline, None);
                self.data.pipeline = vk::Pipeline::null();
            }

            // 9. 销毁管线布局
            if self.data.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.data.pipeline_layout, None);
                self.data.pipeline_layout = vk::PipelineLayout::null();
            }

            // 10. 销毁渲染通道
            if self.data.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.data.render_pass, None);
                self.data.render_pass = vk::RenderPass::null();
            }

            // 11. 销毁交换链图像视图
            for view in self.data.swapchain_image_views.drain(..) {
                if view != vk::ImageView::null() {
                    self.device.destroy_image_view(view, None);
                }
            }

            // 12. 销毁交换链（最后）
            if self.data.swapchain != vk::SwapchainKHR::null() {
                let swapchain_device =
                    ash::khr::swapchain::Device::new(&self.instance, &self.device);
                swapchain_device.destroy_swapchain(self.data.swapchain, None);
                self.data.swapchain = vk::SwapchainKHR::null();
            }
            self.data.swapchain_images.clear();
        }
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
    unique_indices.insert(indices.graphics);
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

    let base_features_to_enable = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
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
        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
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
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices_vec.push(indices.graphics);
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

/// Creates image views for each image in the swapchain.
fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|&image| {
            create_image_view_internal(
                device,
                image,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Render Pass, Pipeline Layout, Pipeline
//--------------------------------------------------------------------------------------------------

/// Creates the render pass defining the framebuffer attachments and subpasses.
fn create_render_pass(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment = vk::AttachmentDescription::default()
        .format(get_depth_format(instance, data)?)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_ref))
        .depth_stencil_attachment(&depth_stencil_attachment_ref);

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

    let attachments = &[color_attachment, depth_stencil_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    // SAFETY: `create_render_pass` is unsafe. Device and create_info must be valid.
    data.render_pass = unsafe { device.create_render_pass(&create_info, None)? };
    Ok(())
}

/// Creates the descriptor set layout for uniform buffers and samplers.
fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let sampler_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[ubo_binding, sampler_binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

    // SAFETY: `create_descriptor_set_layout` is unsafe. Device and create_info must be valid.
    data.descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&create_info, None)? };
    Ok(())
}

/// Creates the graphics pipeline, including shaders and fixed-function state.
fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    let vert_shader_spirv = include_bytes!("../assets/shaders/27_depth_buffering.vert.spv");
    let frag_shader_spirv = include_bytes!("../assets/shaders/26_shader_textures.frag.spv");

    let vert_shader_module = create_shader_module_internal(device, vert_shader_spirv)?;
    let frag_shader_module = create_shader_module_internal(device, frag_shader_spirv)?;

    // SAFETY: "main\0" is a NUL-terminated C-style string.
    let main_function_name = c"main";

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

    let binding_descriptions = [Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

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

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&color_blend_attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let set_layouts = &[data.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    // SAFETY: `create_pipeline_layout` is unsafe. Device and layout_info must be valid.
    data.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0) // Optional
        .max_depth_bounds(1.0) // Optional
        .stencil_test_enable(false)
        .front(vk::StencilOpState::default()) // Optional
        .back(vk::StencilOpState::default()); //

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    // SAFETY: `create_graphics_pipelines` is unsafe. Device, cache, and pipeline_info must be valid.
    data.pipeline = unsafe {
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) {
            Ok(pipelines) => pipelines[0],
            Err((mut pipelines, err)) => {
                for pipeline in pipelines.drain(..) {
                    if pipeline != vk::Pipeline::null() {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(err.into());
            }
        }
    };

    // SAFETY: `destroy_shader_module` is unsafe. Shader modules must be valid.
    unsafe {
        if vert_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(vert_shader_module, None);
        }
        if frag_shader_module != vk::ShaderModule::null() {
            device.destroy_shader_module(frag_shader_module, None);
        }
    }
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Framebuffers and Command Pool
//--------------------------------------------------------------------------------------------------

/// Creates framebuffers for each swapchain image view.
fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|&image_view| {
            let attachments = &[image_view, data.depth_image_view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);
            // SAFETY: `create_framebuffer` is unsafe. Device, create_info must be valid.
            unsafe { device.create_framebuffer(&create_info, None) }
        })
        .collect::<Result<Vec<_>, vk::Result>>()?;
    Ok(())
}

/// Creates the command pool for allocating command buffers.
fn create_command_pool(
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;
    let create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);
    // SAFETY: `create_command_pool` is unsafe. Device and create_info must be valid.
    data.command_pool = unsafe { device.create_command_pool(&create_info, None)? };
    Ok(())
}

/// Create depth objects
fn create_depth_objects(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let format = get_depth_format(instance, data)?;
    let (depth_image, depth_image_memory) = create_image_internal(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;
    data.depth_image_view =
        create_image_view_internal(device, depth_image, format, vk::ImageAspectFlags::DEPTH)?;
    transition_image_layout_internal(
        device,
        data,
        depth_image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )?;
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Texture Resources (Image, View, Sampler)
//--------------------------------------------------------------------------------------------------

/// Creates the texture image, its memory, and uploads pixel data.
fn create_texture_image(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let img_path = "assets/textures/viking_room.png";
    let img = image::open(img_path)
        .map_err(|e| anyhow!("Failed to open texture image '{}': {}", img_path, e))?
        .into_rgba8();

    let (width, height) = img.dimensions();
    if width != 1024 || height != 1024 {
        panic!(
            "Invalid texture image (use https://kylemayes.github.io/vulkanalia/images/viking_room.png)."
        );
    }
    let image_data = img.into_raw();
    let image_size = (width * height * 4) as vk::DeviceSize;

    let (staging_buffer, staging_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // SAFETY: `map_memory`, `memcpy`, `unmap_memory` are unsafe.
    // Memory, pointer, and size must be valid.
    unsafe {
        let memory_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            image_size,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(image_data.as_ptr(), memory_ptr.cast(), image_data.len());
        device.unmap_memory(staging_buffer_memory);
    }

    let (texture_image, texture_image_memory) = create_image_internal(
        instance,
        device,
        data,
        width,
        height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    transition_image_layout_internal(
        device,
        data,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;
    copy_buffer_to_image_internal(device, data, staging_buffer, texture_image, width, height)?;
    transition_image_layout_internal(
        device,
        data,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    // SAFETY: `destroy_buffer` and `free_memory` are unsafe. Buffer and memory must be valid.
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }
    Ok(())
}

/// Creates an image view for the texture image.
fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    data.texture_image_view = create_image_view_internal(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
    )?;
    Ok(())
}

/// Creates a sampler for accessing the texture in shaders.
fn create_texture_sampler(device: &Device, instance: &Instance, data: &mut AppData) -> Result<()> {
    // SAFETY: `get_physical_device_properties` is an unsafe Vulkan call.
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
        .max_lod(0.0); // Set to actual mip levels if using mipmapping

    // SAFETY: `create_sampler` is unsafe. Device and create_info must be valid.
    data.texture_sampler = unsafe { device.create_sampler(&create_info, None)? };
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Buffers (Vertex, Index, Uniform)
//--------------------------------------------------------------------------------------------------

/// Creates the vertex buffer and uploads vertex data.
fn create_vertex_buffer(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let buffer_size = (size_of::<Vertex>() * data.vertices.len()) as vk::DeviceSize;

    let (staging_buffer, staging_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // SAFETY: `map_memory`, `memcpy`, `unmap_memory` are unsafe.
    unsafe {
        let memory_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(
            data.vertices.as_ptr(),
            memory_ptr.cast(),
            data.vertices.len(),
        );
        device.unmap_memory(staging_buffer_memory);
    }

    let (vertex_buffer, vertex_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer_internal(device, data, staging_buffer, vertex_buffer, buffer_size)?;

    // SAFETY: `destroy_buffer` and `free_memory` are unsafe.
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }
    Ok(())
}

/// Creates the index buffer and uploads index data.
fn create_index_buffer(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let buffer_size = (size_of::<u32>() * data.indices.len()) as vk::DeviceSize;

    let (staging_buffer, staging_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // SAFETY: `map_memory`, `memcpy`, `unmap_memory` are unsafe.
    unsafe {
        let memory_ptr = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(data.indices.as_ptr(), memory_ptr.cast(), data.indices.len());
        device.unmap_memory(staging_buffer_memory);
    }

    let (index_buffer, index_buffer_memory) = create_buffer_internal(
        instance,
        device,
        data,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer_internal(device, data, staging_buffer, index_buffer, buffer_size)?;

    // SAFETY: `destroy_buffer` and `free_memory` are unsafe.
    unsafe {
        if staging_buffer != vk::Buffer::null() {
            device.destroy_buffer(staging_buffer, None);
        }
        if staging_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(staging_buffer_memory, None);
        }
    }
    Ok(())
}

/// Creates uniform buffers for each frame in flight.
fn create_uniform_buffers(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
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

    data.uniform_buffers.reserve(data.swapchain_images.len());
    data.uniform_buffers_memory
        .reserve(data.swapchain_images.len());

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer_internal(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Descriptors (Pool, Sets)
//--------------------------------------------------------------------------------------------------

/// Creates the descriptor pool for allocating descriptor sets.
fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // SAFETY: destroy_descriptor_pool is unsafe.
    if data.descriptor_pool != vk::DescriptorPool::null() {
        unsafe {
            device.destroy_descriptor_pool(data.descriptor_pool, None);
        }
        data.descriptor_pool = vk::DescriptorPool::null();
    }

    let num_swapchain_images = data.swapchain_images.len() as u32;
    if num_swapchain_images == 0 {
        return Ok(());
    }

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(num_swapchain_images),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(num_swapchain_images),
    ];

    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(num_swapchain_images)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    // SAFETY: `create_descriptor_pool` is unsafe. Device and create_info must be valid.
    data.descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None)? };
    Ok(())
}

/// Allocates and updates descriptor sets for each uniform buffer and texture.
fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    if data.swapchain_images.is_empty() || data.descriptor_pool == vk::DescriptorPool::null() {
        return Ok(());
    }
    // Descriptor sets are implicitly freed when the pool is destroyed,
    // but we clear the vector here to avoid dangling references if this function is called multiple times.
    data.descriptor_sets.clear();

    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    // SAFETY: `allocate_descriptor_sets` is unsafe. Device, alloc_info must be valid.
    data.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

    for i in 0..data.swapchain_images.len() {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as vk::DeviceSize);

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info)),
        ];
        // SAFETY: `update_descriptor_sets` is unsafe. Device and descriptor_writes must be valid.
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Command Buffers and Sync Objects
//--------------------------------------------------------------------------------------------------

/// Creates and records command buffers for drawing.
fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    if data.framebuffers.is_empty() {
        return Ok(());
    }
    // SAFETY: free_command_buffers is unsafe.
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

    // SAFETY: `allocate_command_buffers` is unsafe. Device and alloc_info must be valid.
    data.command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };

    for (i, &command_buffer) in data.command_buffers.iter().enumerate() {
        // SAFETY: `begin_command_buffer` is unsafe. Command buffer and begin_info must be valid.
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(command_buffer, &begin_info)?;
        }

        let render_area = vk::Rect2D::default().extent(data.swapchain_extent);
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = &[color_clear_value, depth_clear_value];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(data.render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        // SAFETY: All `cmd_` functions are unsafe. Command buffer and parameters must be valid.
        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.pipeline,
            );
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[data.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                data.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.pipeline_layout,
                0,
                &[data.descriptor_sets[i]],
                &[],
            );
            device.cmd_draw_indexed(command_buffer, data.indices.len() as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }
    }
    Ok(())
}

/// Creates synchronization objects (semaphores and fences) for frame rendering.
fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    // SAFETY: destroy semaphore/fence is unsafe.
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
        for f in data.in_flight_fences.drain(..) {
            if f != vk::Fence::null() {
                device.destroy_fence(f, None);
            }
        }
    }

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    data.image_available_semaphores
        .reserve(MAX_FRAMES_IN_FLIGHT);
    data.render_finished_semaphores
        .reserve(MAX_FRAMES_IN_FLIGHT);
    data.in_flight_fences.reserve(MAX_FRAMES_IN_FLIGHT);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        // SAFETY: `create_semaphore` and `create_fence` are unsafe. Device and infos must be valid.
        unsafe {
            data.image_available_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            data.render_finished_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            data.in_flight_fences
                .push(device.create_fence(&fence_info, None)?);
        }
    }
    data.images_in_flight = vec![vk::Fence::null(); data.swapchain_images.len()];
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Subsection: Format
//--------------------------------------------------------------------------------------------------

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
            .find(|f| {
                let properties =
                    instance.get_physical_device_format_properties(data.physical_device, *f);
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => {
                        properties.optimal_tiling_features.contains(features)
                    }
                    _ => false,
                }
            })
            .ok_or_else(|| anyhow!("Failed to find supported format!"))
    }
}
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

/// Creates a Vulkan image and allocates its memory. (Internal Helper)
#[allow(clippy::too_many_arguments)]
fn create_image_internal(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    // SAFETY: `create_image` is unsafe. Device and image_info must be valid.
    let image = unsafe { device.create_image(&image_info, None)? };

    // SAFETY: `get_image_memory_requirements` is unsafe. Device and image must be valid.
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

    let mem_type_index = get_memory_type_index_internal(
        instance,
        data.physical_device,
        properties,
        mem_requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    // SAFETY: `allocate_memory` and `bind_image_memory` are unsafe.
    let image_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, image_memory, 0)? };

    Ok((image, image_memory))
}

/// Creates a Vulkan image view from an image. (Internal Helper)
fn create_image_view_internal(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    // SAFETY: `create_image_view` is unsafe. Device, create_info must be valid.
    unsafe { Ok(device.create_image_view(&create_info, None)?) }
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

/// Transitions the layout of an image using a pipeline barrier. (Internal Helper)
fn transition_image_layout_internal(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
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
                    "Unsupported image layout transition: {:?} to {:?}",
                    old_layout,
                    new_layout
                ));
            }
        };
    let command_buffer = begin_single_time_commands_internal(device, data)?;

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
        .level_count(1)
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

    // SAFETY: `cmd_pipeline_barrier` is unsafe. Command buffer and barrier must be valid.
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

    end_single_time_commands_internal(device, data, command_buffer)?;
    Ok(())
}

/// Copies data from a buffer to an image. (Internal Helper)
fn copy_buffer_to_image_internal(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands_internal(device, data)?;

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

    // SAFETY: `cmd_copy_buffer_to_image` is unsafe. Command buffer, buffer, image, and region must be valid.
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    end_single_time_commands_internal(device, data, command_buffer)?;
    Ok(())
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

        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Tutorial (Rust) - 28 Model Loading")
            .with_inner_size(LogicalSize::new(1024.0, 768.0));

        let window = match event_loop.create_window(window_attributes) {
            Ok(win) => win,
            Err(e) => {
                error!("Failed to create window: {:?}", e);
                event_loop.exit();
                return;
            }
        };

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
                        app.resized = true;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if self.minimized {
                    return;
                }
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            if !self.minimized && self.vulkan_app.is_some() {
                window.request_redraw();
            }
        }
    }

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
