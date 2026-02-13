// --- Group 1: Imports ---
use nalgebra as na;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
        ImageBlit, PrimaryCommandBufferAbstract, RenderingAttachmentInfo,
        RenderingAttachmentResolveInfo, RenderingInfo, allocator::StandardCommandBufferAllocator,
    },
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
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType,
        ImageUsage,
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
}

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
struct ModelUBO {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
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
    texture_image: Option<Arc<Image>>,
    texture_image_view: Option<Arc<ImageView>>,
    texture_sampler: Option<Arc<Sampler>>,

    // Resources (Continued)
    uniform_buffers: Vec<Subbuffer<ModelUBO>>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,

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

    // Synchronization
    // Instead of manual Semaphores/Fences, Vulkano uses GpuFuture to track frame state.
    // We store the future of the last submission for each frame slot.
    fences: Vec<Option<Box<dyn GpuFuture>>>,

    // Runtime State
    frame_index: usize,
    recreate_swapchain: bool,
    is_initialized: bool,

    // FPS Counter
    last_fps_update: Instant,
    frame_count: u32,
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
            texture_image: None,
            texture_image_view: None,
            texture_sampler: None,
            uniform_buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            width: 800,
            height: 600,
            msaa_samples: vulkano::image::SampleCount::Sample1,
            color_images: Vec::new(),
            color_image_views: Vec::new(),
            depth_images: Vec::new(),
            depth_image_views: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            fences: (0..MAX_FRAMES_IN_FLIGHT).map(|_| None).collect(),
            frame_index: 0,
            recreate_swapchain: false,
            is_initialized: false,
            last_fps_update: Instant::now(),
            frame_count: 0,
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
        self.create_index_buffer();
        self.create_uniform_buffers();

        // 1. Create attachments (Color/Depth) FIRST so we know their formats.
        self.create_color_resources();
        self.create_depth_resources();

        // 2. Create Pipeline SECOND, using the correct formats from above.
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
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().sampler_anisotropy
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
            Default::default(),
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

        let vs_bytes = std::fs::read("assets/shaders/viking_room.spv")
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

        let pipeline_layout_create_info =
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap();

        // 3. Dynamic Rendering Configuration
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
                    cull_mode: CullMode::None,
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
        let image_count = self.swapchain_images.len();

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

        let image_count = self.swapchain_images.len();

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
                AllocationCreateInfo::default(),
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
        println!("Loading glTF model...");
        // Use gltf crate to import the .glb file
        // Ensure properties are loaded (buffers)
        let (document, buffers, _images) =
            gltf::import("assets/models/viking_room.glb").expect("Failed to load glTF model");

        self.vertices.clear();
        self.indices.clear();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // 1. Read Positions (Mandatory in glTF)
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .expect("Failed to read positions")
                    .collect();
                let vertex_count = positions.len();

                // 2. Read Colors (Optional) using default if missing
                let colors: Vec<[f32; 3]> = if let Some(iter) = reader.read_colors(0) {
                    iter.into_rgb_f32().collect()
                } else {
                    vec![[1.0, 1.0, 1.0]; vertex_count]
                };

                // 3. Read TexCoords (Optional)
                let tex_coords: Vec<[f32; 2]> = if let Some(iter) = reader.read_tex_coords(0) {
                    iter.into_f32().collect()
                } else {
                    vec![[0.0, 0.0]; vertex_count]
                };

                // 4. Construct Vertices
                let mut local_vertices = Vec::with_capacity(vertex_count);
                for i in 0..vertex_count {
                    local_vertices.push(Vertex {
                        position: positions[i],
                        color: colors[i],
                        // Vulkan UV: (0,0) is top-left. glTF/OpenGL UV: (0,0) is bottom-left.
                        // Should we flip the V coordinate here?
                        tex_coord: [tex_coords[i][0], tex_coords[i][1]],
                    });
                }

                // 5. Read Indices
                // We need to offset indices by the number of vertices already in the global buffer
                let base_vertex_index = self.vertices.len() as u32;

                if let Some(indices_iter) = reader.read_indices() {
                    let indices_u32: Vec<u32> = indices_iter.into_u32().collect();
                    self.indices
                        .extend(indices_u32.iter().map(|i| i + base_vertex_index));
                } else {
                    // Primitive is not indexed; generate sequential indices
                    self.indices
                        .extend((0..vertex_count as u32).map(|i| i + base_vertex_index));
                }

                // Append local vertices to the global vertex buffer
                self.vertices.extend(local_vertices);
            }
        }

        println!(
            "Model Loaded. Vertices: {}, Indices: {}",
            self.vertices.len(),
            self.indices.len()
        );
    }

    fn create_texture_resources(&mut self) {
        println!("Creating texture resources...");

        let image_bytes =
            std::fs::read("assets/textures/viking_room.png").expect("Failed to load texture file");
        let cursor = Cursor::new(image_bytes);
        let decoder = image::ImageReader::new(cursor)
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        let rgba = decoder.into_rgba8();
        let dimensions = rgba.dimensions();
        let width = dimensions.0;
        let height = dimensions.1;
        let image_data = rgba.into_raw();

        // Calculate mip levels: floor(log2(max(w, h))) + 1
        let mip_levels = ((width.max(height) as f32).log2().floor() as u32) + 1;

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
                // TRANSFER_SRC is required for blitting from a mip level
                // TRANSFER_DST is required for copying buffer to image and blitting to a mip level
                usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                mip_levels,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to create image");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref().unwrap().clone(),
            self.queue.as_ref().unwrap().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Copy buffer to the first mip level (level 0)
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                image.clone(),
            ))
            .unwrap();

        // Generate mipmaps using blit
        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            let src_width = mip_width;
            let src_height = mip_height;
            let dst_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let dst_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            let blit = ImageBlit {
                src_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: i - 1,
                    array_layers: 0..1,
                },
                src_offsets: [[0, 0, 0], [src_width as u32, src_height as u32, 1]],
                dst_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: i,
                    array_layers: 0..1,
                },
                dst_offsets: [[0, 0, 0], [dst_width as u32, dst_height as u32, 1]],
                ..Default::default()
            };

            let mut blit_info = BlitImageInfo::images(image.clone(), image.clone());
            blit_info.regions = [blit].into_iter().collect();
            blit_info.filter = Filter::Linear;

            builder.blit_image(blit_info).unwrap();

            mip_width = dst_width;
            mip_height = dst_height;
        }

        let command_buffer = builder.build().unwrap();

        command_buffer
            .execute(self.queue.as_ref().unwrap().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.texture_image = Some(image.clone());
        self.create_texture_image_view();
        self.create_texture_sampler();
    }

    fn create_texture_image_view(&mut self) {
        // Create view for all mip levels
        self.texture_image_view =
            Some(ImageView::new_default(self.texture_image.as_ref().unwrap().clone()).unwrap());
    }

    fn create_texture_sampler(&mut self) {
        let mip_levels = self.texture_image.as_ref().unwrap().mip_levels();

        let properties = self.physical_device.as_ref().unwrap().properties();

        let sampler = Sampler::new(
            self.device.as_ref().unwrap().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                anisotropy: Some(properties.max_sampler_anisotropy),
                lod: 0.0..=mip_levels as f32,
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
                    usage: BufferUsage::VERTEX_BUFFER,
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
                    usage: BufferUsage::INDEX_BUFFER,
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
    fn create_uniform_buffers(&mut self) {
        self.uniform_buffers.clear();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = Buffer::from_data(
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
                ModelUBO {
                    model: na::Matrix4::identity().into(),
                    view: na::Matrix4::identity().into(),
                    proj: na::Matrix4::identity().into(),
                },
            )
            .expect("failed to create uniform buffer");
            self.uniform_buffers.push(buffer);
        }
    }
    fn create_descriptor_pool(&mut self) {
        // Vulkano handles pools internally.
    }
    fn create_descriptor_sets(&mut self) {
        println!("Creating Descriptor Sets...");
        let pipeline = self.pipeline.as_ref().unwrap();
        let layout = pipeline.layout().set_layouts().first().unwrap();

        self.descriptor_sets.clear();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let ubo_buffer = self.uniform_buffers[i].clone();
            let sampler = self.texture_sampler.as_ref().unwrap().clone();
            let image_view = self.texture_image_view.as_ref().unwrap().clone();

            let set = DescriptorSet::new(
                self.descriptor_set_allocator.as_ref().unwrap().clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, ubo_buffer),
                    WriteDescriptorSet::image_view_sampler(1, image_view, sampler),
                ],
                [],
            )
            .expect("failed to create descriptor set");

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
    fn draw_frame(&mut self) {
        // Calculate FPS
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_update);

        if elapsed.as_secs() >= 1 {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            if let Some(window) = &self.window {
                window.set_title(&format!("Vulkan Tutorial (Rust) - FPS: {:.1}", fps));
            }
            self.last_fps_update = now;
            self.frame_count = 0;
        }

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

        self.update_uniform_buffer(self.frame_index);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref().unwrap().clone(),
            self.queue.as_ref().unwrap().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let swapchain_view = self.swapchain_image_views[image_index as usize].clone();

        let color_attachment = if self.msaa_samples != vulkano::image::SampleCount::Sample1 {
            let mut info = RenderingAttachmentInfo::image_view(
                self.color_image_views[image_index as usize].clone(),
            );
            info.image_layout = ImageLayout::ColorAttachmentOptimal;
            info.load_op = AttachmentLoadOp::Clear;
            info.store_op = AttachmentStoreOp::DontCare; // DontCare for MSAA usually
            info.clear_value = Some(ClearValue::Float([0.2, 0.2, 0.2, 1.0]));
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
            info.clear_value = Some(ClearValue::Float([0.2, 0.2, 0.2, 1.0]));
            info
        };

        let mut depth_attachment = RenderingAttachmentInfo::image_view(
            self.depth_image_views[image_index as usize].clone(),
        );
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

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                self.descriptor_sets[self.frame_index].clone(),
            )
            .unwrap();

        builder
            .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())
            .unwrap();

        builder
            .bind_index_buffer(self.index_buffer.as_ref().unwrap().clone())
            .unwrap();

        unsafe {
            builder
                .draw_indexed(self.indices.len() as u32, 1, 0, 0, 0)
                .unwrap()
        };

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

    fn update_uniform_buffer(&self, image_index: usize) {
        let aspect_ratio = self.width as f32 / self.height as f32;

        let duration = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let time = (duration.as_millis() % 100000) as f32 / 1000.0;

        // Orientation: Rotate 90 degrees around X to lay the model flat (align Y-up to Z-up)
        // Then apply the rotation animation around the new Up axis (Z)
        let rotation_x =
            na::Matrix4::new_rotation(na::Vector3::new(90.0f32.to_radians(), 0.0, 0.0));
        let rotation_z = na::Matrix4::new_rotation(na::Vector3::new(0.0, 0.0, time));

        // Combine rotations: Apply X then Z
        let model = rotation_z * rotation_x;

        let view = na::Matrix4::look_at_rh(
            &na::Point3::new(1.75, 1.75, 1.75),
            &na::Point3::new(0.0, 0.0, 0.0),
            &na::Vector3::new(0.0, 0.0, 1.0),
        );

        let mut proj =
            na::Perspective3::new(aspect_ratio, 45.0f32.to_radians(), 0.1, 10.0).to_homogeneous();
        proj[(1, 1)] *= -1.0;

        let ubo = ModelUBO {
            model: model.into(),
            view: view.into(),
            proj: proj.into(),
        };

        *self.uniform_buffers[image_index].write().unwrap() = ubo;
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

        // We might also need to recreate the pipeline if depth format changes (unlikely)
        // or just rely on dynamic rendering to handle resolution changes (Viewport is dynamic).
        // We assume the pipeline is compatible with the new swapchain, but if not, we would need to recreate it here as well.
    }
}
// --- Group 8: Main Entry Point ---

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
