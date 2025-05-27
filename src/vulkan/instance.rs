//! Vulkan实例和调试设置模块
//! 负责创建Vulkan实例、设置验证层和调试回调

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Entry, Instance};
use log::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use winit::window::Window;

use crate::constants::*;
use crate::types::AppData;
use crate::vk_window;

/// 创建Vulkan实例并设置调试消息
/// 初始化Vulkan环境和验证层
pub fn vulkan_create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData,
) -> Result<Instance> {
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
