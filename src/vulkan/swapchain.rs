//! 交换链管理模块
//! 负责创建和配置交换链以及图像视图

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Device, Entry, Instance};
use log::*;
use std::collections::HashSet;
use winit::window::Window;

use crate::resources::resources_images::create_image_view;
use crate::types::{AppData, QueueFamilyIndices, SwapchainSupport};

//==================================================================================================
// 交换链支持查询实现
//==================================================================================================

impl SwapchainSupport {
    /// 查询物理设备的交换链支持情况
    /// 获取表面能力、支持的格式和呈现模式
    pub fn get(
        instance: &ash::Instance,
        entry: &ash::Entry,
        data: &AppData,
        physical_device: ash::vk::PhysicalDevice,
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
// 交换链创建和管理
//==================================================================================================

/// 创建交换链
/// 配置并创建用于呈现的交换链
pub fn vulkan_create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    entry: &Entry,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, entry, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, entry, data, data.physical_device)?;

    // 选择交换链配置
    let surface_format = choose_swap_surface_format(&support.formats);
    let present_mode = choose_swap_present_mode(&support.present_modes);
    let extent = choose_swap_extent(window, support.capabilities);

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
fn choose_swap_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
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
fn choose_swap_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
        .copied()
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// 选择交换链范围
/// 确定交换链图像的分辨率
fn choose_swap_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
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
pub fn vulkan_create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
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
