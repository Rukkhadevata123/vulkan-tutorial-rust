use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 1. Create the window
        let window = Arc::new(
            event_loop
                // In winit 0.30, window creation should happen in the resumed event
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.window = Some(window.clone());
        println!("Window created");

        // 2. Load the Vulkan library
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        // 3. Get the required instance extensions for creating a surface from the window.
        // Surface::required_extensions automatically handles platform detection based on the event_loop.
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        // 4. Create the Vulkan instance
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                // Enable the required extensions
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        // 5. Create a surface from the window
        // Surface::from_window handles the platform-specific logic internally
        let _surface = Surface::from_window(instance.clone(), window.clone())
            .expect("failed to create surface");

        println!("Surface created successfully");

        // --- Device Creation Logic ---

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .next()
            .expect("no device available");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|queue_family_properties| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        let (_device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        println!("Vulkan initialized using queue: {:?}", queue);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if event == WindowEvent::CloseRequested {
            println!("The close button was pressed; stopping");
            event_loop.exit();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
