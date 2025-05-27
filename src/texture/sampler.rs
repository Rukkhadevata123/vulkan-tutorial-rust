//! 纹理采样器管理模块
//! 负责创建和配置纹理采样器

use anyhow::{Result, anyhow};
use ash::vk;
use ash::{Device, Instance};

use crate::types::AppData;

/// 采样器配置
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 放大过滤器
    pub mag_filter: vk::Filter,
    /// 缩小过滤器
    pub min_filter: vk::Filter,
    /// U 轴寻址模式
    pub address_mode_u: vk::SamplerAddressMode,
    /// V 轴寻址模式
    pub address_mode_v: vk::SamplerAddressMode,
    /// W 轴寻址模式
    pub address_mode_w: vk::SamplerAddressMode,
    /// 是否启用各向异性过滤
    pub anisotropy_enable: bool,
    /// 最大各向异性值
    pub max_anisotropy: f32,
    /// 边框颜色
    pub border_color: vk::BorderColor,
    /// Mipmap 模式
    pub mipmap_mode: vk::SamplerMipmapMode,
    /// Mip LOD 偏差
    pub mip_lod_bias: f32,
    /// 最小 LOD
    pub min_lod: f32,
    /// 最大 LOD
    pub max_lod: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: true,
            max_anisotropy: 16.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            mip_lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: f32::MAX,
        }
    }
}

impl SamplerConfig {
    /// 创建新的采样器配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置过滤器
    pub fn with_filters(mut self, mag: vk::Filter, min: vk::Filter) -> Self {
        self.mag_filter = mag;
        self.min_filter = min;
        self
    }

    /// 设置寻址模式
    pub fn with_address_mode(mut self, mode: vk::SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    /// 设置各向异性过滤
    pub fn with_anisotropy(mut self, enable: bool, max_anisotropy: f32) -> Self {
        self.anisotropy_enable = enable;
        self.max_anisotropy = max_anisotropy;
        self
    }

    /// 设置 Mipmap 范围
    pub fn with_mip_range(mut self, min_lod: f32, max_lod: f32) -> Self {
        self.min_lod = min_lod;
        self.max_lod = max_lod;
        self
    }
}

/// 创建纹理采样器
/// 根据配置创建 Vulkan 采样器
pub fn create_texture_sampler(
    device: &Device,
    instance: &Instance,
    data: &mut AppData,
    config: SamplerConfig,
) -> Result<()> {
    // 获取物理设备属性
    let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };

    // 限制各向异性值
    let max_anisotropy = if config.anisotropy_enable {
        config
            .max_anisotropy
            .min(properties.limits.max_sampler_anisotropy)
    } else {
        1.0
    };

    // 设置最大 LOD
    let max_lod = if config.max_lod == f32::MAX {
        data.mip_levels as f32
    } else {
        config.max_lod
    };

    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(config.mag_filter)
        .min_filter(config.min_filter)
        .address_mode_u(config.address_mode_u)
        .address_mode_v(config.address_mode_v)
        .address_mode_w(config.address_mode_w)
        .anisotropy_enable(config.anisotropy_enable)
        .max_anisotropy(max_anisotropy)
        .border_color(config.border_color)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(config.mipmap_mode)
        .mip_lod_bias(config.mip_lod_bias)
        .min_lod(config.min_lod)
        .max_lod(max_lod);

    data.texture_sampler = unsafe {
        device
            .create_sampler(&create_info, None)
            .map_err(|e| anyhow!("创建纹理采样器失败: {}", e))?
    };

    log::info!("纹理采样器创建完成 (各向异性: {:.1}x)", max_anisotropy);
    Ok(())
}
