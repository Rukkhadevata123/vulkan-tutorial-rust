//! 描述符管理模块
//! 包含描述符池创建和管理功能

use anyhow::{Result, anyhow};
use ash::Device;
use ash::vk;

/// 通用描述符池创建参数
#[derive(Debug)]
pub struct DescriptorPoolConfig {
    /// 池大小配置
    pub pool_sizes: Vec<vk::DescriptorPoolSize>,
    /// 最大描述符集数量
    pub max_sets: u32,
    /// 池创建标志
    pub flags: vk::DescriptorPoolCreateFlags,
}

impl DescriptorPoolConfig {
    /// 创建新的描述符池配置
    pub fn new() -> Self {
        Self {
            pool_sizes: Vec::new(),
            max_sets: 0,
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        }
    }

    /// 添加池大小
    pub fn add_pool_size(mut self, descriptor_type: vk::DescriptorType, count: u32) -> Self {
        self.pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(descriptor_type)
                .descriptor_count(count),
        );
        self
    }

    /// 设置最大描述符集数量
    pub fn max_sets(mut self, max_sets: u32) -> Self {
        self.max_sets = max_sets;
        self
    }

    /// 设置创建标志
    pub fn flags(mut self, flags: vk::DescriptorPoolCreateFlags) -> Self {
        self.flags = flags;
        self
    }
}

impl Default for DescriptorPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// 通用描述符池创建函数
pub fn create_descriptor_pool(
    device: &Device,
    config: DescriptorPoolConfig,
) -> Result<vk::DescriptorPool> {
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&config.pool_sizes)
        .max_sets(config.max_sets)
        .flags(config.flags);

    unsafe {
        device
            .create_descriptor_pool(&create_info, None)
            .map_err(|e| anyhow!("创建描述符池失败: {}", e))
    }
}
