//! Window integration.

use ash::prelude::*;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

/// Gets the required instance extensions for window integration.
pub fn get_required_instance_extensions(
    window: &dyn HasWindowHandle,
) -> &'static [&'static std::ffi::CStr] {
    match window.window_handle().map(|handle| handle.as_raw()) {
        // Linux
        #[cfg(target_os = "linux")]
        Ok(RawWindowHandle::Wayland(_)) => &[vk::KHR_SURFACE_NAME, vk::KHR_WAYLAND_SURFACE_NAME],

        #[cfg(target_os = "linux")]
        Ok(RawWindowHandle::Xcb(_)) => &[vk::KHR_SURFACE_NAME, vk::KHR_XCB_SURFACE_NAME],

        #[cfg(target_os = "linux")]
        Ok(RawWindowHandle::Xlib(_)) => &[vk::KHR_SURFACE_NAME, vk::KHR_XLIB_SURFACE_NAME],

        _ => {
            unimplemented!(
                "Unsupported window handle type for Linux, or error obtaining window handle."
            )
        }
    }
}

/// Creates a surface for a window.
///
/// # Safety
///
/// The returned `SurfaceKHR` will only be valid while the supplied window is
/// valid so the supplied window must not be destroyed before the returned
/// `SurfaceKHR` is destroyed.
pub unsafe fn create_surface(
    instance: &ash::Instance, // Assuming instance has methods like create_wayland_surface_khr directly
    entry: &ash::Entry,
    display: &dyn HasDisplayHandle,
    window: &dyn HasWindowHandle,
) -> VkResult<vk::SurfaceKHR> {
    match (
        display.display_handle().map(|handle| handle.as_raw()),
        window.window_handle().map(|handle| handle.as_raw()),
    ) {
        // Linux
        #[cfg(target_os = "linux")]
        (Ok(RawDisplayHandle::Wayland(display)), Ok(RawWindowHandle::Wayland(window))) => {
            let info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr())
                .surface(window.surface.as_ptr());
            let wayland_instance = ash::khr::wayland_surface::Instance::new(entry, instance);
            unsafe { wayland_instance.create_wayland_surface(&info, None) }
        }

        #[cfg(target_os = "linux")]
        (Ok(RawDisplayHandle::Xcb(display)), Ok(RawWindowHandle::Xcb(window))) => {
            let connection_ptr = display
                .connection
                .map(|connection| connection.as_ptr())
                .unwrap_or(std::ptr::null_mut());
            let info = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(connection_ptr)
                .window(window.window.get()); // window.window is NonZeroU32, .get() is u32
            let xcb_instance = ash::khr::xcb_surface::Instance::new(entry, instance);
            unsafe { xcb_instance.create_xcb_surface(&info, None) }
        }

        #[cfg(target_os = "linux")]
        (Ok(RawDisplayHandle::Xlib(display)), Ok(RawWindowHandle::Xlib(window))) => {
            let display_ptr = display
                .display
                .map(|display| display.as_ptr())
                .unwrap_or(std::ptr::null_mut()); // This is *mut c_void
            let info = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(display_ptr as *mut _) // Cast *mut c_void to *mut vk::Display
                .window(window.window); // window.window is u64 (XID)
            let xlib_instance = ash::khr::xlib_surface::Instance::new(entry, instance);
            unsafe { xlib_instance.create_xlib_surface(&info, None) }
        }

        _ => {
            unimplemented!(
                "Unsupported window or display handle type for Linux, or error obtaining handles."
            )
        }
    }
}
