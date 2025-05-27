//! 物理设备选择和逻辑设备创建模块
//! 负责选择合适的GPU并创建逻辑设备

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Device, Entry, Instance};
use log::*;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::constants::*;
use crate::errors::SuitabilityError;
use crate::types::{AppData, QueueFamilyIndices, SwapchainSupport};

//==================================================================================================
// 队列族查询实现
//==================================================================================================

impl QueueFamilyIndices {
    /// 查询物理设备的队列族支持情况
    /// 优先查找同时支持图形和计算的队列族，以减少队列族切换开销
    pub fn get(
        instance: &ash::Instance,
        entry: &ash::Entry,
        data: &AppData,
        physical_device: ash::vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        // 优先寻找同时支持图形和计算的队列族
        let graphics_and_compute = properties
            .iter()
            .position(|p| {
                p.queue_flags
                    .contains(ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE)
            })
            .map(|i| i as u32);

        // 如果没有找到同时支持的，分别寻找
        let (graphics, compute) = if let Some(combined) = graphics_and_compute {
            (combined, combined)
        } else {
            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

            let compute = properties
                .iter()
                .position(|p| p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);

            match (graphics, compute) {
                (Some(g), Some(c)) => (g, c),
                _ => {
                    return Err(anyhow::anyhow!(SuitabilityError::Static(
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
            Err(anyhow::anyhow!(SuitabilityError::Static(
                "缺少必需的呈现队列族。"
            )))
        }
    }
}

//==================================================================================================
// 物理设备选择和逻辑设备创建
//==================================================================================================

/// 选择合适的物理设备
/// 遍历可用GPU并选择最适合的设备
pub fn vulkan_pick_physical_device(
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

        if let Err(error) = check_device_suitability(instance, entry, data, physical_device) {
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
fn check_device_suitability(
    instance: &Instance,
    entry: &Entry,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // 检查队列族支持
    QueueFamilyIndices::get(instance, entry, data, physical_device)?;

    // 检查设备扩展支持
    check_device_extensions(instance, physical_device)?;

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
fn check_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
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
pub fn vulkan_create_logical_device(
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

/// 获取最大MSAA采样数
/// 查找设备支持的最高MSAA采样数
pub fn get_max_msaa_samples(instance: &Instance, data: &AppData) -> vk::SampleCountFlags {
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
