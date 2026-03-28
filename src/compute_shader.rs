use rand::RngExt;
use std::{sync::Arc, time::Instant};

use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        RenderingAttachmentInfo, RenderingInfo, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::ClearValue,
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipeline,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
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

// --- Constants & Data Structures ---

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const PARTICLE_COUNT: usize = 8192;
const MAX_FRAMES_IN_FLIGHT: usize = 2; // Double buffering

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents, Vertex)]
struct Particle {
    #[format(R32G32_SFLOAT)]
    #[name("input.inPosition")]
    position: [f32; 2],

    #[format(R32G32_SFLOAT)]
    #[name("input.inVelocity")]
    velocity: [f32; 2],

    #[format(R32G32B32A32_SFLOAT)]
    #[name("input.inColor")]
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
struct PushConstants {
    delta_time: f32,
}

// --- App Struct Definition ---

struct App {
    // Winit window
    window: Option<Arc<Window>>,

    // Core Vulkan objects
    instance: Option<Arc<Instance>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,

    // Swapchain & Presentation
    surface: Option<Arc<Surface>>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,

    // Allocators
    memory_allocator: Option<Arc<StandardMemoryAllocator>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,

    // Pipelines
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    compute_pipeline: Option<Arc<ComputePipeline>>,

    // Resources
    // Use one buffer for both storage (compute) and vertex input (graphics)
    storage_buffers: Vec<Subbuffer<[Particle]>>,

    // Descriptor Sets
    compute_descriptor_sets: Vec<Arc<DescriptorSet>>,

    // Synchronization
    fences: Vec<Option<Box<dyn GpuFuture>>>,

    // Runtime State
    frame_index: usize,
    recreate_swapchain: bool,
    is_initialized: bool,

    // Time tracking
    last_time: Instant,
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
            memory_allocator: None,
            command_buffer_allocator: None,
            descriptor_set_allocator: None,
            graphics_pipeline: None,
            compute_pipeline: None,
            storage_buffers: Vec::new(),
            compute_descriptor_sets: Vec::new(),
            fences: (0..MAX_FRAMES_IN_FLIGHT).map(|_| None).collect(),
            frame_index: 0,
            recreate_swapchain: false,
            is_initialized: false,
            last_time: Instant::now(), // will be reset in resumed
        }
    }
}

// --- ApplicationHandler Implementation ---

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.is_initialized {
            self.init_window(event_loop);
            self.init_vulkan(event_loop);
            self.is_initialized = true;
            self.last_time = Instant::now();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Closing...");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
                self.draw_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// --- Vulkan Implementation ---

impl App {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        println!("Initializing Window...");
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vulkan Compute Shader Particles")
                        .with_inner_size(winit::dpi::LogicalSize::new(WIDTH as f64, HEIGHT as f64)),
                )
                .unwrap(),
        );
        self.window = Some(window);
    }

    fn init_vulkan(&mut self, event_loop: &ActiveEventLoop) {
        println!("Initializing Vulkan...");

        self.create_instance(event_loop);
        self.create_surface_and_device();
        self.create_allocators();
        self.create_swapchain();

        self.create_buffers(); // Create SSBO and Uniform buffers

        // NOTE: Order matters here. Compute Pipeline defines the layout used for Descriptor Sets.
        self.create_compute_pipeline();
        self.create_descriptor_sets(); // For compute shader

        self.create_graphics_pipeline();

        self.last_time = Instant::now();

        println!("Vulkan initialized.");
    }

    fn create_instance(&mut self, event_loop: &ActiveEventLoop) {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
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
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags
                            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| {
                // Prefer discrete GPUs
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    _ => 4,
                }
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        self.physical_device = Some(physical_device.clone());

        // Dynamic Rendering feature is required if we want to skip RenderPass objects
        let features = DeviceFeatures {
            dynamic_rendering: true,
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
        let device = self.device.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();

        let caps = self
            .physical_device
            .as_ref()
            .unwrap()
            .surface_capabilities(surface, Default::default())
            .expect("failed to get surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = self
            .physical_device
            .as_ref()
            .unwrap()
            .surface_formats(surface, Default::default())
            .unwrap()[0]
            .0;

        let window_size = window.inner_size();
        let image_extent: [u32; 2] = if let Some(extent) = caps.current_extent {
            extent
        } else {
            [window_size.width, window_size.height]
        };

        let min_image_count = caps.min_image_count.max(2);

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format,
                image_extent,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                present_mode: PresentMode::Fifo, // V-Sync
                ..Default::default()
            },
        )
        .expect("failed to create swapchain");

        self.swapchain = Some(swapchain);
        self.swapchain_images = images;

        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect();
    }

    fn recreate_swapchain_impl(&mut self) {
        let window = self.window.as_ref().unwrap();
        let new_dimensions = window.inner_size();

        if new_dimensions.width == 0 || new_dimensions.height == 0 {
            return;
        }

        let (new_swapchain, new_images) = self
            .swapchain
            .as_ref()
            .unwrap()
            .recreate(SwapchainCreateInfo {
                image_extent: [new_dimensions.width, new_dimensions.height],
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

        self.recreate_swapchain = false;
    }

    fn create_buffers(&mut self) {
        let mut rng = rand::rng();
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);

        for _ in 0..PARTICLE_COUNT {
            let r = 0.25f32 * (rng.random::<f32>()).sqrt();
            let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let x = r * theta.cos();
            let y = r * theta.sin();

            particles.push(Particle {
                position: [x, y],
                velocity: [theta.cos() * 0.00025, theta.sin() * 0.00025],
                color: [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    1.0,
                ],
            });
        }

        let memory_allocator = self.memory_allocator.as_ref().unwrap();

        // 1. Create a Staging Buffer (Host Visible) to upload data
        let staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            particles.clone(),
        )
        .expect("failed to create staging buffer");

        // 2. Create Storage Buffers (Device Local) for each frame in flight
        // Usage: STORAGE_BUFFER (for Compute), VERTEX_BUFFER (for Graphics), TRANSFER_DST (for upload)
        self.storage_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let command_buffer_allocator = self.command_buffer_allocator.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // We need to copy data to these buffers immediately
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let storage_buffer = Buffer::new_slice::<Particle>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::VERTEX_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                PARTICLE_COUNT as u64,
            )
            .expect("failed to create storage buffer");

            // Schedule copy from staging to device local buffer
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    staging_buffer.clone(),
                    storage_buffer.clone(),
                ))
                .unwrap();

            self.storage_buffers.push(storage_buffer);
        }

        // Submit the upload commands
        let command_buffer = builder.build().unwrap();
        let future = command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap(); // Wait for upload to finish
    }

    fn create_descriptor_sets(&mut self) {
        let pipeline_layout = self.compute_pipeline.as_ref().unwrap().layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();
        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        self.compute_descriptor_sets = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            // Mimic C++:
            // Input: Frame (i - 1) -> (i + 1) % 2
            // Output: Frame i
            let storage_buffer_input = self.storage_buffers[(i + 1) % MAX_FRAMES_IN_FLIGHT].clone();
            let storage_buffer_output = self.storage_buffers[i].clone();

            let descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.as_ref().unwrap().clone(),
                descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, storage_buffer_input),
                    WriteDescriptorSet::buffer(1, storage_buffer_output), // Output
                ],
                [],
            )
            .expect("failed to create descriptor set");

            self.compute_descriptor_sets.push(descriptor_set);
        }
    }

    fn create_compute_pipeline(&mut self) {
        let device = self.device.as_ref().unwrap();

        let bytes = std::fs::read("assets/shaders/shader_compute.spv")
            .expect("failed to read shader_compute.spv");
        let words = vulkano::shader::spirv::bytes_to_words(&bytes).expect("failed to convert spv");

        let shader_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)).unwrap()
        };

        // Entry point "compMain" from Slang
        let entry_point = shader_module
            .entry_point("compMain")
            .expect("Could not find entry point 'compMain'");

        let stage = PipelineShaderStageCreateInfo::new(entry_point);

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        self.compute_pipeline = Some(pipeline);
    }

    fn create_graphics_pipeline(&mut self) {
        let device = self.device.as_ref().unwrap();
        let swapchain = self.swapchain.as_ref().unwrap();

        let bytes = std::fs::read("assets/shaders/shader_compute.spv")
            .expect("failed to read shader_compute.spv");
        let words = vulkano::shader::spirv::bytes_to_words(&bytes).expect("failed to convert spv");

        let shader_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)).unwrap()
        };

        let vs = shader_module
            .entry_point("vertMain")
            .expect("vertMain missing");
        let fs = shader_module
            .entry_point("fragMain")
            .expect("fragMain missing");

        let vertex_input_state = Particle::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        };

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology:
                        vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::PointList,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::SrcAlpha,
                            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            dst_alpha_blend_factor: BlendFactor::Zero,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        self.graphics_pipeline = Some(pipeline);
    }

    fn draw_frame(&mut self) {
        if self.recreate_swapchain {
            self.recreate_swapchain_impl();
        }

        // Wait for fence
        if let Some(mut fence) = self.fences[self.frame_index].take() {
            if fence.queue().is_some() {
                match fence.then_signal_fence_and_flush() {
                    Ok(f) => {
                        f.wait(None).unwrap(); // Wait for GPU to finish frame i
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

        let now = Instant::now();
        let delta_time = now.duration_since(self.last_time).as_secs_f32() * 40.0;
        self.last_time = now;

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

        let command_buffer_allocator = self.command_buffer_allocator.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // Compute Command Buffer
        let mut compute_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let compute_layout = self.compute_pipeline.as_ref().unwrap().layout().clone();

        unsafe {
            compute_builder
                .bind_pipeline_compute(self.compute_pipeline.as_ref().unwrap().clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_layout.clone(),
                    0,
                    self.compute_descriptor_sets[self.frame_index].clone(),
                )
                .unwrap()
                .push_constants(compute_layout.clone(), 0, PushConstants { delta_time })
                .unwrap()
                .dispatch([PARTICLE_COUNT as u32 / 256, 1, 1])
                .unwrap()
        };

        let compute_command_buffer = compute_builder.build().unwrap();

        // Graphics Command Buffer
        let mut graphics_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let attachment_info = RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            ..RenderingAttachmentInfo::image_view(
                self.swapchain_image_views[image_index as usize].clone(),
            )
        };

        unsafe {
            graphics_builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(attachment_info)],
                    // dynamic rendering without depth
                    ..RenderingInfo::default()
                })
                .unwrap()
                .bind_pipeline_graphics(self.graphics_pipeline.as_ref().unwrap().clone())
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [
                            self.swapchain.as_ref().unwrap().image_extent()[0] as f32,
                            self.swapchain.as_ref().unwrap().image_extent()[1] as f32,
                        ],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .set_scissor(
                    0,
                    [Scissor {
                        offset: [0, 0],
                        extent: self.swapchain.as_ref().unwrap().image_extent(),
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .bind_vertex_buffers(0, self.storage_buffers[self.frame_index].clone())
                .unwrap()
                .draw(PARTICLE_COUNT as u32, 1, 0, 0)
                .unwrap()
                .end_rendering()
                .unwrap()
        };

        let graphics_command_buffer = graphics_builder.build().unwrap();

        // Sync & Submit
        // Execute Compute -> Then Execute Graphics -> Then Present
        let future = self.fences[self.frame_index]
            .take()
            .unwrap_or_else(|| vulkano::sync::now(self.device.as_ref().unwrap().clone()).boxed())
            .join(acquire_future)
            .then_execute(queue.clone(), compute_command_buffer)
            .unwrap()
            .then_execute(queue.clone(), graphics_command_buffer)
            .unwrap()
            .then_swapchain_present(
                queue.clone(),
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
}

// --- Main Entry ---

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
