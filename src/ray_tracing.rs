use nalgebra as na;
use std::{sync::Arc, time::Instant};
use vulkano::Packed24_8;

use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::IndexBuffer,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderingAttachmentInfo,
        RenderingAttachmentResolveInfo, RenderingInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::layout::DescriptorBindingFlags,
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::{ClearValue, Format},
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
        view::ImageView,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex as VertexTrait, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp, ResolveMode},
    shader::{ShaderModule, ShaderModuleCreateInfo},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::GpuFuture,
};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// --- Group 2: Constants & Data Structures ---

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
// Add VertexTrait derive to automatically generate input state definition
#[derive(Debug, Clone, Copy, BufferContents, VertexTrait)]
struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    #[name("input.inPosition")]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    #[name("input.inColor")]
    color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    #[name("input.inTexCoord")]
    tex_coord: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    #[name("input.inNormal")]
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
struct UniformBufferObject {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32, // Padding to ensure 16-byte alignment if needed
}

#[derive(Debug, Clone)]
struct SubMesh {
    index_offset: u32,
    index_count: u32,
    material_id: i32,
    max_vertex: u32,
    alpha_cut: bool,
    reflective: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
struct InstanceLUT {
    material_id: u32,
    index_buffer_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
struct PushConstant {
    material_index: u32,
    reflective: u32,
}

// --- Group 3: App Struct Definition ---

struct App {
    // Winit window
    window: Option<Arc<Window>>,

    // Core Vulkan objects
    instance: Option<Arc<Instance>>,
    physical_device: Option<Arc<PhysicalDevice>>, // Keep track of physical device
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,

    // Swapchain & Presentation
    surface: Option<Arc<Surface>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,

    // Pipeline (No RenderPass object needed for Dynamic Rendering)
    pipeline: Option<Arc<GraphicsPipeline>>,

    // Allocators
    memory_allocator: Option<Arc<StandardMemoryAllocator>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,

    // Resources
    vertex_buffer: Option<Subbuffer<[Vertex]>>,
    index_buffer: Option<Subbuffer<[u32]>>,
    uv_buffer: Option<Subbuffer<[[f32; 2]]>>,
    instance_lut_buffer: Option<Subbuffer<[InstanceLUT]>>,
    texture_images: Vec<Arc<Image>>,
    texture_image_views: Vec<Arc<ImageView>>,
    texture_sampler: Option<Arc<Sampler>>,

    // Resources (Continued)
    uniform_buffers: Vec<Subbuffer<UniformBufferObject>>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    material_descriptor_set: Option<Arc<DescriptorSet>>,

    // Acceleration Structures
    blas: Vec<Arc<AccelerationStructure>>,
    tlas: Vec<Arc<AccelerationStructure>>,
    tlas_buffer: Vec<Subbuffer<[u8]>>,
    tlas_scratch_buffer: Vec<Subbuffer<[u8]>>,
    // We also need the instance buffer to update transforms
    instance_buffers: Vec<Subbuffer<[AccelerationStructureInstance]>>,

    // MSAA & Depth Resources (Used as attachments in dynamic rendering)
    width: u32,
    height: u32,
    msaa_samples: vulkano::image::SampleCount,
    color_images: Vec<Arc<Image>>, // MSAA color images
    color_image_views: Vec<Arc<ImageView>>,
    depth_images: Vec<Arc<Image>>,
    depth_image_views: Vec<Arc<ImageView>>,

    // Model Data (CPU side cache before uploading)
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    submeshes: Vec<SubMesh>,
    materials: Vec<tobj::Material>,

    // Synchronization
    // Instead of manual Semaphores/Fences, Vulkano uses GpuFuture to track frame state.
    // We store the future of the last submission for each frame slot.
    fences: Vec<Option<Box<dyn GpuFuture>>>,

    // Runtime State
    frame_index: usize,
    recreate_swapchain: bool,
    is_initialized: bool,

    // Time tracking
    start_time: Instant,
    last_fps_update: Instant,
    frame_count: u32,

    // Current Model Matrix (stored to be reused for TLAS update)
    current_model_matrix: na::Matrix4<f32>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            instance: None,
            physical_device: None,
            device: None,
            queue: None,
            surface: None,
            swapchain: None,
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            pipeline: None,
            memory_allocator: None,
            command_buffer_allocator: None,
            descriptor_set_allocator: None,
            vertex_buffer: None,
            index_buffer: None,
            uv_buffer: None,
            instance_lut_buffer: None,
            texture_images: Vec::new(),
            texture_image_views: Vec::new(),
            texture_sampler: None,
            uniform_buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            material_descriptor_set: None,
            blas: Vec::new(),
            tlas: Vec::new(),
            tlas_buffer: Vec::new(),
            tlas_scratch_buffer: Vec::new(),
            instance_buffers: Vec::new(),
            width: 800,
            height: 600,
            msaa_samples: vulkano::image::SampleCount::Sample1,
            color_images: Vec::new(),
            color_image_views: Vec::new(),
            depth_images: Vec::new(),
            depth_image_views: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            submeshes: Vec::new(),
            materials: Vec::new(),
            fences: (0..MAX_FRAMES_IN_FLIGHT).map(|_| None).collect(),
            frame_index: 0,
            recreate_swapchain: false,
            is_initialized: false,
            start_time: Instant::now(),
            last_fps_update: Instant::now(),
            frame_count: 0,
            current_model_matrix: na::Matrix4::identity(),
        }
    }
}

// --- Group 4: ApplicationHandler Implementation ---

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.is_initialized {
            self.init_window(event_loop);
            self.init_vulkan(event_loop);
            self.is_initialized = true;
            self.start_time = Instant::now(); // Reset start time
            self.last_fps_update = Instant::now();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => self.recreate_swapchain = true,
            WindowEvent::RedrawRequested => self.draw_frame(),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Equivalent to device.waitIdle()
        if let Some(device) = &self.device {
            let _ = unsafe { device.wait_idle() };
        }
    }
}

// --- Group 5: Vulkan Initialization Helpers ---

impl App {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        println!("Initializing Window...");
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Vulkan Tutorial (Rust)"))
                .unwrap(),
        );
        self.window = Some(window);
    }

    fn init_vulkan(&mut self, event_loop: &ActiveEventLoop) {
        println!("Initializing Vulkan with Dynamic Rendering...");

        self.create_instance(event_loop);
        self.create_surface_and_device();
        self.create_allocators();
        self.create_swapchain();

        // Initialize resources
        self.load_model();
        self.create_texture_resources();
        self.create_vertex_buffer();
        self.create_uv_buffer();
        self.create_instance_lut_buffer();
        self.create_index_buffer();

        self.create_bottom_level_acceleration_structure();
        self.create_top_level_acceleration_structure();

        self.create_uniform_buffers();

        self.create_color_resources();
        self.create_depth_resources();

        self.create_graphics_pipeline();

        self.create_descriptor_pool();
        self.create_descriptor_sets();

        self.create_sync_objects();

        println!("Vulkan Initialization Phase 1 Complete.");
    }

    fn create_instance(&mut self, event_loop: &ActiveEventLoop) {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        let mut enabled_layers = Vec::new();

        #[cfg(debug_assertions)]
        {
            let layer = "VK_LAYER_KHRONOS_validation";
            let layers: Vec<_> = library.layer_properties().unwrap().collect();
            if layers.iter().any(|p| p.name() == layer) {
                enabled_layers.push(layer.to_owned());
                println!("Validation layer enabled: {}", layer);
            } else {
                println!("Validation layer {} not found", layer);
            }
        }

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                enabled_layers,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        self.instance = Some(instance);
    }

    fn create_surface_and_device(&mut self) {
        let instance = self.instance.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();

        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("failed to create surface");
        self.surface = Some(surface.clone());

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_dynamic_rendering: true,
            khr_acceleration_structure: true,
            khr_ray_query: true,
            khr_deferred_host_operations: true,
            khr_buffer_device_address: true,
            khr_spirv_1_4: true,
            khr_shader_float_controls: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().sampler_anisotropy
                    && p.supported_features().buffer_device_address
                    && p.supported_features().acceleration_structure
                    && p.supported_features().ray_query
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                _ => 3,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        self.physical_device = Some(physical_device.clone());

        let features = DeviceFeatures {
            dynamic_rendering: true,
            sampler_anisotropy: true,
            buffer_device_address: true,
            acceleration_structure: true,
            ray_query: true,
            shader_float64: true,
            runtime_descriptor_array: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            descriptor_binding_sampled_image_update_after_bind: true,
            ..DeviceFeatures::empty()
        };

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        self.device = Some(device.clone());
        self.queue = Some(queues.next().unwrap());
    }

    fn create_allocators(&mut self) {
        let device = self.device.as_ref().unwrap();
        self.memory_allocator = Some(Arc::new(StandardMemoryAllocator::new_default(
            device.clone(),
        )));
        self.command_buffer_allocator = Some(Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        )));
        self.descriptor_set_allocator = Some(Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        )));
    }

    fn create_swapchain(&mut self) {
        let physical_device = self.physical_device.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();

        let caps = physical_device
            .surface_capabilities(surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

        let surface_formats = physical_device
            .surface_formats(surface, Default::default())
            .unwrap();
        let image_format = surface_formats
            .iter()
            .find(|(fmt, color_space)| {
                *fmt == Format::B8G8R8A8_SRGB
                    && *color_space == vulkano::swapchain::ColorSpace::SrgbNonLinear
            })
            .or_else(|| {
                surface_formats.iter().find(|(fmt, color_space)| {
                    *fmt == Format::R8G8B8A8_SRGB
                        && *color_space == vulkano::swapchain::ColorSpace::SrgbNonLinear
                })
            })
            .map(|(fmt, _)| *fmt)
            .unwrap_or(surface_formats[0].0);

        println!("Swapchain Format: {:?}", image_format);

        // Look for Mailbox present mode, fallback to Fifo
        let present_modes = physical_device
            .surface_present_modes(surface, Default::default())
            .unwrap();
        let present_mode = present_modes
            .into_iter()
            .find(|&mode| mode == PresentMode::Mailbox)
            .unwrap_or(PresentMode::Fifo);

        println!("Present Mode: {:?}", present_mode);

        let mut min_image_count = caps.min_image_count + 1;
        if let Some(max_image_count) = caps.max_image_count
            && min_image_count > max_image_count
        {
            min_image_count = max_image_count;
        }

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                present_mode,
                ..Default::default()
            },
        )
        .unwrap();

        self.swapchain = Some(swapchain);
        self.swapchain_images = images;
        self.width = dimensions.width;
        self.height = dimensions.height;

        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect();
    }

    fn create_graphics_pipeline(&mut self) {
        println!("Creating Graphics Pipeline...");
        let device = self.device.as_ref().unwrap();

        let vs_bytes = std::fs::read("assets/shaders/ray_tracing.spv")
            .expect("Failed to read vertex shader spv");
        let shader_words = vulkano::shader::spirv::bytes_to_words(&vs_bytes)
            .expect("Failed to create shader module");
        let shader_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&shader_words))
                .expect("Failed to create shader module")
        };
        let vs_entry_point = shader_module
            .entry_point("vertMain")
            .expect("Missing vertMain entry point");
        let fs_entry_point = shader_module
            .entry_point("fragMain")
            .expect("Missing fragMain entry point");

        let vertex_input_state = Vertex::per_vertex().definition(&vs_entry_point).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry_point),
            PipelineShaderStageCreateInfo::new(fs_entry_point),
        ];

        // Retrieve reflection info regarding descriptor sets
        let mut pipeline_descriptor_set_layout_create_info =
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

        if let Some(set_layout) = pipeline_descriptor_set_layout_create_info
            .set_layouts
            .get_mut(1)
            && let Some(binding) = set_layout.bindings.get_mut(&1)
        {
            binding.descriptor_count = self.texture_images.len().max(1) as u32;
            binding.binding_flags -= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
        }

        let pipeline_layout_create_info = pipeline_descriptor_set_layout_create_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap();

        let depth_format = self.depth_images[0].format();

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(self.swapchain.as_ref().unwrap().image_format())],
            depth_attachment_format: Some(depth_format),
            ..Default::default()
        };

        let layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)
            .expect("Failed to create pipeline layout");

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face:
                        vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: self.msaa_samples,
                    ..Default::default()
                }),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .expect("Failed to create graphics pipeline");

        self.pipeline = Some(pipeline);
    }

    fn create_color_resources(&mut self) {
        let physical_device = self.physical_device.as_ref().unwrap();
        let limits = physical_device.properties();
        let counts =
            limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts;

        self.msaa_samples = if counts.contains(vulkano::image::SampleCount::Sample4.into()) {
            vulkano::image::SampleCount::Sample4
        } else if counts.contains(vulkano::image::SampleCount::Sample2.into()) {
            vulkano::image::SampleCount::Sample2
        } else {
            vulkano::image::SampleCount::Sample1
        };

        self.color_images.clear();
        self.color_image_views.clear();

        if self.msaa_samples == vulkano::image::SampleCount::Sample1 {
            return;
        }

        let format = self.swapchain.as_ref().unwrap().image_format();
        let image_count = MAX_FRAMES_IN_FLIGHT;

        for _ in 0..image_count {
            let image = Image::new(
                self.memory_allocator.as_ref().unwrap().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [self.width, self.height, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                    samples: self.msaa_samples,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("failed to create MSAA color image");

            self.color_images.push(image.clone());
            self.color_image_views
                .push(ImageView::new_default(image).unwrap());
        }
    }

    fn create_depth_resources(&mut self) {
        let physical_device = self.physical_device.as_ref().unwrap();

        let candidates = [
            Format::D32_SFLOAT,
            Format::D32_SFLOAT_S8_UINT,
            Format::D24_UNORM_S8_UINT,
        ];

        let format = candidates
            .into_iter()
            .find(|&format| {
                let properties = physical_device.format_properties(format);
                properties
                    .unwrap()
                    .optimal_tiling_features
                    .contains(vulkano::format::FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
            })
            .expect("failed to find supported depth format");

        self.depth_images.clear();
        self.depth_image_views.clear();

        let image_count = MAX_FRAMES_IN_FLIGHT;

        for _ in 0..image_count {
            let image = Image::new(
                self.memory_allocator.as_ref().unwrap().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [self.width, self.height, 1],
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    samples: self.msaa_samples,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .expect("failed to create depth image");

            self.depth_images.push(image.clone());
            self.depth_image_views
                .push(ImageView::new_default(image).unwrap());
        }
    }
}

// --- Group 6: Resource Creation & Model Loading ---

impl App {
    fn load_model(&mut self) {
        println!("Loading OBJ model...");
        let (models, materials) = tobj::load_obj(
            "assets/models/plant_on_table.obj",
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )
        .expect("Failed to load OBJ model");

        // We assume materials are successfully loaded
        self.materials = materials.expect("Failed to load materials");

        self.vertices.clear();
        self.indices.clear();
        self.submeshes.clear();

        for model in models {
            let mesh = model.mesh;
            let material_id = mesh.material_id.map(|id| id as i32).unwrap_or(-1);

            let first_vertex = self.vertices.len() as u32;
            let index_offset = self.indices.len() as u32;
            let index_count = mesh.indices.len() as u32;

            // Load vertices
            let positions = mesh.positions;
            let normals = mesh.normals;
            let tex_coords = mesh.texcoords;
            let vertex_count = positions.len() / 3;

            let mut max_vertex = 0;

            for i in 0..vertex_count {
                let position = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];

                // Defaults if missing
                let normal = if !normals.is_empty() {
                    [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]]
                } else {
                    [0.0, 0.0, 0.0]
                };

                let tex_coord = if !tex_coords.is_empty() {
                    [tex_coords[i * 2], 1.0 - tex_coords[i * 2 + 1]] // Flip V
                } else {
                    [0.0, 0.0]
                };

                self.vertices.push(Vertex {
                    position,
                    color: [1.0, 1.0, 1.0], // White default
                    tex_coord,
                    normal,
                });
                max_vertex = max_vertex.max(i as u32);
            }

            // Load indices
            let model_indices: Vec<u32> = mesh.indices.iter().map(|&i| i + first_vertex).collect();
            self.indices.extend_from_slice(&model_indices);

            // Create SubMesh info
            let submesh = SubMesh {
                index_offset,
                index_count,
                material_id,
                max_vertex: first_vertex + vertex_count as u32 - 1,
                alpha_cut: model.name.contains("nettle_plant"),
                reflective: model.name.contains("table"),
            };
            self.submeshes.push(submesh);
        }

        println!(
            "Model Loaded. Vertices: {}, Indices: {}, Submeshes: {}",
            self.vertices.len(),
            self.indices.len(),
            self.submeshes.len()
        );
    }

    fn create_texture_resources(&mut self) {
        println!("Creating texture resources...");

        self.texture_images.clear();

        let mut images_to_load = Vec::new();

        if self.materials.is_empty() {
            images_to_load.push(None);
        } else {
            for material in &self.materials {
                images_to_load.push(material.diffuse_texture.clone());
            }
        }

        let command_buffer_allocator = self.command_buffer_allocator.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        let mut uploads = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        for texture_path_opt in images_to_load {
            let mut width = 1;
            let mut height = 1;
            let mut image_data = vec![255u8, 255, 255, 255]; // RGBA

            if let Some(path_str) = texture_path_opt
                && !path_str.is_empty()
            {
                let full_path = format!("assets/models/{}", path_str);
                let path = std::path::Path::new(&full_path);

                let loaded_image = if path.exists() {
                    image::ImageReader::open(path)
                        .ok()
                        .and_then(|r| r.with_guessed_format().ok())
                        .and_then(|r| r.decode().ok())
                } else {
                    let full_path_2 = format!("assets/textures/{}", path_str);
                    let path_2 = std::path::Path::new(&full_path_2);
                    if path_2.exists() {
                        image::ImageReader::open(path_2)
                            .ok()
                            .and_then(|r| r.with_guessed_format().ok())
                            .and_then(|r| r.decode().ok())
                    } else {
                        println!("Texture not found: {} or {}", full_path, full_path_2);
                        None
                    }
                };

                if let Some(decoder) = loaded_image {
                    let rgba = decoder.into_rgba8();
                    width = rgba.width();
                    height = rgba.height();
                    image_data = rgba.into_raw();
                }
            }

            let staging_buffer = Buffer::from_iter(
                self.memory_allocator.as_ref().unwrap().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                image_data,
            )
            .expect("failed to create staging buffer");

            let image = Image::new(
                self.memory_allocator.as_ref().unwrap().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_SRGB,
                    extent: [width, height, 1],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    mip_levels: 1,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("failed to create image");

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    staging_buffer,
                    image.clone(),
                ))
                .unwrap();

            self.texture_images.push(image);
        }

        let command_buffer = uploads.build().unwrap();

        command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.create_texture_image_view();
        self.create_texture_sampler();
    }

    fn create_texture_image_view(&mut self) {
        self.texture_image_views = self
            .texture_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect();
    }

    fn create_texture_sampler(&mut self) {
        let properties = self.physical_device.as_ref().unwrap().properties();

        let sampler = Sampler::new(
            self.device.as_ref().unwrap().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                anisotropy: Some(properties.max_sampler_anisotropy),
                lod: 0.0..=1.0,
                ..Default::default()
            },
        )
        .expect("failed to create sampler");
        self.texture_sampler = Some(sampler);
    }

    fn create_vertex_buffer(&mut self) {
        self.vertex_buffer = Some(
            Buffer::from_iter(
                self.memory_allocator.as_ref().unwrap().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                self.vertices.clone(),
            )
            .expect("failed to create vertex buffer"),
        );
    }
    fn create_index_buffer(&mut self) {
        self.index_buffer = Some(
            Buffer::from_iter(
                self.memory_allocator.as_ref().unwrap().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                        | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                self.indices.clone(),
            )
            .expect("failed to create index buffer"),
        );
    }

    fn create_uv_buffer(&mut self) {
        let uvs: Vec<[f32; 2]> = self.vertices.iter().map(|v| v.tex_coord).collect();

        self.uv_buffer = Some(
            Buffer::from_iter(
                self.memory_allocator.as_ref().unwrap().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                uvs,
            )
            .expect("failed to create UV buffer"),
        );
    }

    fn create_instance_lut_buffer(&mut self) {
        let instance_luts: Vec<InstanceLUT> = self
            .submeshes
            .iter()
            .map(|submesh| InstanceLUT {
                material_id: submesh.material_id as u32,
                index_buffer_offset: submesh.index_offset,
            })
            .collect();

        self.instance_lut_buffer = Some(
            Buffer::from_iter(
                self.memory_allocator.as_ref().unwrap().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                instance_luts,
            )
            .expect("failed to create Instance LUT buffer"),
        );
    }

    fn create_bottom_level_acceleration_structure(&mut self) {
        println!("Creating Bottom Level Acceleration Structures...");

        let mut blas_list = Vec::new();

        for submesh in &self.submeshes {
            let primitive_count = submesh.index_count / 3;

            let vertex_buffer_bytes = self.vertex_buffer.as_ref().unwrap().clone().into_bytes();
            let index_buffer_slice = self.index_buffer.as_ref().unwrap().clone().slice(
                (submesh.index_offset as u64)
                    ..((submesh.index_offset + submesh.index_count) as u64),
            );

            let geometry_flags = if submesh.alpha_cut {
                vulkano::acceleration_structure::GeometryFlags::empty()
            } else {
                vulkano::acceleration_structure::GeometryFlags::OPAQUE
            };

            let mut geometry_data =
                AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
            geometry_data.vertex_data = Some(vertex_buffer_bytes);
            geometry_data.vertex_stride = std::mem::size_of::<Vertex>() as u32;
            geometry_data.max_vertex = submesh.max_vertex;
            geometry_data.index_data = Some(IndexBuffer::U32(index_buffer_slice));
            geometry_data.flags = geometry_flags;

            let geometry = AccelerationStructureGeometries::Triangles(vec![geometry_data]);

            let blas = self.build_acceleration_structure(
                geometry,
                primitive_count,
                AccelerationStructureType::BottomLevel,
            );

            blas_list.push(blas);
        }
        self.blas = blas_list;
    }

    fn create_top_level_acceleration_structure(&mut self) {
        println!("Creating Top Level Acceleration Structure...");

        // Initial Identity Matrix
        let transform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];

        let instances: Vec<AccelerationStructureInstance> = self
            .blas
            .iter()
            .enumerate()
            .map(|(i, blas)| AccelerationStructureInstance {
                transform,
                instance_custom_index_and_mask: Packed24_8::new(i as u32, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
                acceleration_structure_reference: blas.device_address().into(),
            })
            .collect();

        // Create Instance Buffer - must be HOST_VISIBLE for updates
        self.instance_buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::from_iter(
                    self.memory_allocator.as_ref().unwrap().clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::SHADER_DEVICE_ADDRESS
                            | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                            | BufferUsage::TRANSFER_DST
                            | BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    instances.clone(),
                )
                .expect("failed to create instance buffer")
            })
            .collect();

        // Clear existing vectors
        self.tlas.clear();
        self.tlas_buffer.clear();
        self.tlas_scratch_buffer.clear();

        let primitive_count = self.blas.len() as u32;

        // Build a TLAS for each frame
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let instance_buffer_slice = self.instance_buffers[i].clone().slice(..);
            let geometry_data = AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer_slice)),
            );

            let geometry = AccelerationStructureGeometries::Instances(geometry_data);

            // Build with AllowUpdate flag
            let (tlas, buffer, scratch) = self.build_tlas_internal(
                geometry,
                primitive_count,
                BuildAccelerationStructureFlags::ALLOW_UPDATE,
            );

            self.tlas.push(tlas);
            self.tlas_buffer.push(buffer);
            self.tlas_scratch_buffer.push(scratch);
        }
    }

    fn build_tlas_internal(
        &self,
        geometries: AccelerationStructureGeometries,
        primitive_count: u32,
        flags: BuildAccelerationStructureFlags,
    ) -> (Arc<AccelerationStructure>, Subbuffer<[u8]>, Subbuffer<[u8]>) {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let memory_allocator = self.memory_allocator.as_ref().unwrap();
        let command_buffer_allocator = self.command_buffer_allocator.as_ref().unwrap();

        let mut build_info = AccelerationStructureBuildGeometryInfo::new(geometries);
        build_info.flags = flags;
        build_info.mode = BuildAccelerationStructureMode::Build;

        let build_sizes = device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &build_info,
                &[primitive_count],
            )
            .expect("failed to get build sizes");

        let buffer = Buffer::new_slice::<u8>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            build_sizes.acceleration_structure_size,
        )
        .expect("failed to create acceleration structure storage buffer");

        let mut create_info = AccelerationStructureCreateInfo::new(buffer.clone());
        create_info.ty = AccelerationStructureType::TopLevel;

        let acc_struct = unsafe {
            AccelerationStructure::new(device.clone(), create_info)
                .expect("failed to create acceleration structure")
        };

        let scratch_buffer = Buffer::new_slice::<u8>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            build_sizes.build_scratch_size,
        )
        .expect("failed to create scratch buffer");

        build_info.dst_acceleration_structure = Some(acc_struct.clone());
        build_info.scratch_data = Some(scratch_buffer.clone());

        let build_range_info = AccelerationStructureBuildRangeInfo {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .build_acceleration_structure(
                    build_info,
                    std::iter::once(build_range_info).collect(),
                )
                .unwrap();
        }

        builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        (acc_struct, buffer, scratch_buffer)
    }

    fn build_acceleration_structure(
        &self,
        geometries: AccelerationStructureGeometries,
        primitive_count: u32,
        ty: AccelerationStructureType,
    ) -> Arc<AccelerationStructure> {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let memory_allocator = self.memory_allocator.as_ref().unwrap();
        let command_buffer_allocator = self.command_buffer_allocator.as_ref().unwrap();

        let mut build_info = AccelerationStructureBuildGeometryInfo::new(geometries);
        build_info.flags = BuildAccelerationStructureFlags::PREFER_FAST_TRACE;

        let build_sizes = device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &build_info,
                &[primitive_count],
            )
            .expect("failed to get build sizes");

        let buffer = Buffer::new_slice::<u8>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            build_sizes.acceleration_structure_size,
        )
        .expect("failed to create acceleration structure storage buffer");

        let mut create_info = AccelerationStructureCreateInfo::new(buffer);
        create_info.ty = ty;

        let acc_struct = unsafe {
            AccelerationStructure::new(device.clone(), create_info)
                .expect("failed to create acceleration structure")
        };

        let scratch_buffer = Buffer::new_slice::<u8>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            build_sizes.build_scratch_size,
        )
        .expect("failed to create scratch buffer");

        build_info.dst_acceleration_structure = Some(acc_struct.clone());
        build_info.scratch_data = Some(scratch_buffer);

        let build_range_info = AccelerationStructureBuildRangeInfo {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .build_acceleration_structure(
                    build_info,
                    std::iter::once(build_range_info).collect(),
                )
                .unwrap();
        }

        builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        acc_struct
    }

    fn create_uniform_buffers(&mut self) {
        self.uniform_buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::from_data(
                    self.memory_allocator.as_ref().unwrap().clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    UniformBufferObject {
                        model: na::Matrix4::identity().into(),
                        view: na::Matrix4::identity().into(),
                        proj: na::Matrix4::identity().into(),
                        camera_pos: [2.0, 2.0, 2.0],
                        _padding: 0.0,
                    },
                )
                .expect("failed to create uniform buffer")
            })
            .collect();
    }

    fn create_descriptor_pool(&mut self) {
        // We do not need to create a descriptor pool explicitly, as vulkano handles it internally.
    }

    fn create_descriptor_sets(&mut self) {
        println!("Creating Descriptor Sets...");
        let pipeline = self.pipeline.as_ref().unwrap();
        let set_layouts = pipeline.layout().set_layouts();

        let global_layout = set_layouts
            .first()
            .expect("failed to get global descriptor set layout (set=0)");
        let material_layout = set_layouts
            .get(1)
            .expect("failed to get material descriptor set layout (set=1)");

        self.descriptor_sets.clear();

        // --- Create Set 1: Material (Static) ---
        let sampler = self.texture_sampler.as_ref().unwrap().clone();

        self.material_descriptor_set = Some(
            DescriptorSet::new(
                self.descriptor_set_allocator.as_ref().unwrap().clone(),
                material_layout.clone(),
                [
                    WriteDescriptorSet::sampler(0, sampler),
                    WriteDescriptorSet::image_view_array(1, 0, self.texture_image_views.clone()),
                ],
                [],
            )
            .expect("failed to create material descriptor set"),
        );

        // --- Create Set 0: Global (Per Frame) ---
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let ubo_buffer = self.uniform_buffers[i].clone();
            // Use the TLAS corresponding to this frame
            let tlas = self.tlas[i].clone();
            let index_buffer = self.index_buffer.as_ref().unwrap().clone();
            let uv_buffer = self.uv_buffer.as_ref().unwrap().clone();
            let instance_lut_buffer = self.instance_lut_buffer.as_ref().unwrap().clone();

            let set = DescriptorSet::new(
                self.descriptor_set_allocator.as_ref().unwrap().clone(),
                global_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, ubo_buffer),
                    WriteDescriptorSet::acceleration_structure(1, tlas),
                    WriteDescriptorSet::buffer(2, index_buffer),
                    WriteDescriptorSet::buffer(3, uv_buffer),
                    WriteDescriptorSet::buffer(4, instance_lut_buffer),
                ],
                [],
            )
            .expect("failed to create global descriptor set");

            self.descriptor_sets.push(set);
        }
    }

    fn create_sync_objects(&mut self) {
        let device = self.device.as_ref().unwrap();
        self.fences.clear();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            self.fences
                .push(Some(vulkano::sync::now(device.clone()).boxed()));
        }
    }
}

// --- Group 7: Rendering & Runtime ---

impl App {
    fn update_top_level_acceleration_structure(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        // Use resources for the current frame
        if self.frame_index < self.tlas.len() {
            let tlas = &self.tlas[self.frame_index];
            let scratch_buf = &self.tlas_scratch_buffer[self.frame_index];
            let instance_buf = &self.instance_buffers[self.frame_index];

            // Update Instance Buffer with new transform on host-visible buffer
            // First we need new instance data

            // Do NOT transpose. Vulkan expects rows, and nalgebra (row, col) gives us the elements we need.
            // Transposing was inverting rotation and destroying translation.
            let m = self.current_model_matrix;

            let transform = [
                [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]],
                [m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]],
                [m[(2, 0)], m[(2, 1)], m[(2, 2)], m[(2, 3)]],
            ];

            // Reconstruct instances
            let instances: Vec<AccelerationStructureInstance> = self
                .blas
                .iter()
                .enumerate()
                .map(|(i, blas)| AccelerationStructureInstance {
                    transform,
                    instance_custom_index_and_mask: Packed24_8::new(i as u32, 0xFF),
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
                    acceleration_structure_reference: blas.device_address().into(),
                })
                .collect();

            // Host update of instance buffer via mapping
            // This avoids recording a conflicting copy command in the same command buffer.
            if let Ok(mut content) = instance_buf.write() {
                content.copy_from_slice(&instances);
            }

            // Build Info for Update
            let instance_buffer_slice = instance_buf.clone().slice(..);
            let geometry_data = AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer_slice)),
            );
            let geometry = AccelerationStructureGeometries::Instances(geometry_data);

            let mut build_info = AccelerationStructureBuildGeometryInfo::new(geometry);
            build_info.flags = BuildAccelerationStructureFlags::ALLOW_UPDATE;
            build_info.mode = BuildAccelerationStructureMode::Update(tlas.clone());
            build_info.dst_acceleration_structure = Some(tlas.clone());
            build_info.scratch_data = Some(scratch_buf.clone());

            let build_range_info = AccelerationStructureBuildRangeInfo {
                primitive_count: self.blas.len() as u32,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            };

            unsafe {
                builder
                    .build_acceleration_structure(
                        build_info,
                        std::iter::once(build_range_info).collect(),
                    )
                    .unwrap();
            }
        }
    }

    fn draw_frame(&mut self) {
        let now = Instant::now();
        // Update FPS once per second
        if now.duration_since(self.last_fps_update).as_secs() >= 1 {
            let fps =
                self.frame_count as f64 / now.duration_since(self.last_fps_update).as_secs_f64();
            if let Some(window) = &self.window {
                window.set_title(&format!("Vulkan Tutorial (Rust) - FPS: {:.1}", fps));
            }
            self.last_fps_update = now;
            self.frame_count = 0;
        }
        self.frame_count += 1;

        if self.recreate_swapchain {
            self.recreate_swapchain_impl();
        }

        if let Some(mut fence) = self.fences[self.frame_index].take() {
            if fence.queue().is_some() {
                match fence.then_signal_fence_and_flush() {
                    Ok(f) => {
                        f.wait(None).unwrap();
                        self.fences[self.frame_index] = Some(f.boxed());
                    }
                    Err(e) => {
                        println!("Failed to signal fence and flush: {:?}", e);
                        self.fences[self.frame_index] =
                            Some(vulkano::sync::now(self.device.as_ref().unwrap().clone()).boxed());
                    }
                }
            } else {
                fence.cleanup_finished();
                self.fences[self.frame_index] = Some(fence);
            }
        }

        let (image_index, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(
            self.swapchain.as_ref().unwrap().clone(),
            None,
        ) {
            Ok(r) => r,
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        // Update main UB and stored model matrix
        self.update_uniform_buffer(self.frame_index);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref().unwrap().clone(),
            self.queue.as_ref().unwrap().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // TASK06: Update TLAS for animation
        self.update_top_level_acceleration_structure(&mut builder);

        let swapchain_view = self.swapchain_image_views[image_index as usize].clone();

        let color_attachment = if self.msaa_samples != vulkano::image::SampleCount::Sample1 {
            let mut info = RenderingAttachmentInfo::image_view(
                self.color_image_views[self.frame_index].clone(),
            );
            info.image_layout = ImageLayout::ColorAttachmentOptimal;
            info.load_op = AttachmentLoadOp::Clear;
            info.store_op = AttachmentStoreOp::DontCare; // DontCare for MSAA usually
            info.clear_value = Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0]));
            info.resolve_info = Some(RenderingAttachmentResolveInfo {
                mode: ResolveMode::Average,
                image_view: swapchain_view,
                image_layout: ImageLayout::ColorAttachmentOptimal,
            });
            info
        } else {
            let mut info = RenderingAttachmentInfo::image_view(swapchain_view);
            info.image_layout = ImageLayout::ColorAttachmentOptimal;
            info.load_op = AttachmentLoadOp::Clear;
            info.store_op = AttachmentStoreOp::Store;
            info.clear_value = Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])); // Black background
            info
        };

        let mut depth_attachment =
            RenderingAttachmentInfo::image_view(self.depth_image_views[self.frame_index].clone());
        depth_attachment.image_layout = ImageLayout::DepthStencilAttachmentOptimal;
        depth_attachment.load_op = AttachmentLoadOp::Clear;
        depth_attachment.store_op = AttachmentStoreOp::DontCare;
        depth_attachment.clear_value = Some(ClearValue::Depth(1.0));

        builder
            .begin_rendering(RenderingInfo {
                render_area_offset: [0, 0],
                render_area_extent: [self.width, self.height],
                layer_count: 1,
                color_attachments: vec![Some(color_attachment)],
                depth_attachment: Some(depth_attachment),
                ..Default::default()
            })
            .unwrap();

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [self.width as f32, self.height as f32],
            depth_range: 0.0..=1.0,
        };
        builder
            .set_viewport(0, [viewport].into_iter().collect())
            .unwrap();

        let scissor = Scissor {
            offset: [0, 0],
            extent: [self.width, self.height],
        };
        builder
            .set_scissor(0, [scissor].into_iter().collect())
            .unwrap();

        let pipeline = self.pipeline.as_ref().unwrap();
        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

        let pipeline_layout = pipeline.layout().clone();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_layout.clone(),
                0,
                vec![
                    self.descriptor_sets[self.frame_index].clone(),
                    self.material_descriptor_set.as_ref().unwrap().clone(),
                ],
            )
            .unwrap();

        builder
            .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())
            .unwrap();

        builder
            .bind_index_buffer(self.index_buffer.as_ref().unwrap().clone())
            .unwrap();

        // Iterate over submeshes to draw
        for submesh in &self.submeshes {
            let material_index = if submesh.material_id < 0 {
                0
            } else {
                submesh.material_id as u32
            };
            let push_constants = PushConstant {
                material_index,
                reflective: if submesh.reflective { 1 } else { 0 },
            };

            builder
                .push_constants(pipeline_layout.clone(), 0, push_constants)
                .unwrap();

            unsafe {
                builder
                    .draw_indexed(
                        submesh.index_count,
                        1,
                        submesh.index_offset,
                        0, // vertex_offset is 0 because indices are absolute
                        0,
                    )
                    .unwrap();
            }
        }

        builder.end_rendering().unwrap();

        let command_buffer = builder.build().unwrap();

        let previous_future = self.fences[self.frame_index].take().unwrap();

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.as_ref().unwrap().clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.as_ref().unwrap().clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.as_ref().unwrap().clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.fences[self.frame_index] = Some(future.boxed());
            }
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.recreate_swapchain = true;
                self.fences[self.frame_index] =
                    Some(vulkano::sync::now(self.device.as_ref().unwrap().clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.fences[self.frame_index] =
                    Some(vulkano::sync::now(self.device.as_ref().unwrap().clone()).boxed());
            }
        }

        self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn update_uniform_buffer(&mut self, frame_index: usize) {
        let time = self.start_time.elapsed().as_secs_f32();

        let model =
            na::Matrix4::from_axis_angle(&na::Vector3::z_axis(), time * 10.0_f32.to_radians());

        self.current_model_matrix = model; // Store for TLAS update

        let view = na::Matrix4::look_at_rh(
            &na::Point3::new(2.0, 2.0, 2.0),
            &na::Point3::new(0.0, 0.0, 0.0),
            &na::Vector3::new(0.0, 0.0, 1.0),
        );

        let mut proj = na::Perspective3::new(
            self.width as f32 / self.height as f32,
            45.0_f32.to_radians(),
            0.1,
            10.0,
        )
        .to_homogeneous();

        proj[(1, 1)] *= -1.0;

        let ubo = UniformBufferObject {
            model: model.into(),
            view: view.into(),
            proj: proj.into(),
            camera_pos: [2.0, 2.0, 2.0],
            _padding: 0.0,
        };

        if let Ok(mut content) = self.uniform_buffers[frame_index].write() {
            *content = ubo;
        }
    }

    fn recreate_swapchain_impl(&mut self) {
        self.recreate_swapchain = false;

        let window = self.window.as_ref().unwrap();
        let dimensions = window.inner_size();

        if dimensions.width == 0 || dimensions.height == 0 {
            return;
        }

        self.width = dimensions.width;
        self.height = dimensions.height;

        let (new_swapchain, new_images) = self
            .swapchain
            .as_ref()
            .unwrap()
            .recreate(SwapchainCreateInfo {
                image_extent: dimensions.into(),
                ..self.swapchain.as_ref().unwrap().create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = Some(new_swapchain);
        self.swapchain_images = new_images;

        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect();

        self.create_color_resources();
        self.create_depth_resources();
    }
}
// --- Group 8: Main Entry Point ---

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
