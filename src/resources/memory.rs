//! 内存管理模块
//! 包含内存类型查找等功能

use anyhow::{Result, anyhow};
use ash::Instance;
use ash::vk;

/// 查找合适的内存类型
/// 根据内存需求和属性查找匹配的内存类型索引
pub fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_properties: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..device_memory_properties.memory_type_count {
        let type_filter_met = (memory_requirements.memory_type_bits & (1 << i)) != 0;
        let properties_met = device_memory_properties.memory_types[i as usize]
            .property_flags
            .contains(required_properties);

        if type_filter_met && properties_met {
            return Ok(i);
        }
    }

    Err(anyhow!("找不到合适的内存类型"))
}
