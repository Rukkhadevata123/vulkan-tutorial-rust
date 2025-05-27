//! 应用程序常量配置和类型别名模块
//! 包含应用程序使用的所有常量、类型别名和基础配置

use std::ffi::CStr;

//==================================================================================================
// 应用程序常量配置
//==================================================================================================

/// 是否启用验证层（调试模式下自动启用）
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// Vulkan验证层名称
pub const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// 设备扩展列表
pub const DEVICE_EXTENSIONS: &[&CStr] = &[c"VK_KHR_swapchain"];

/// 最大并发帧数（用于帧资源管理）
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// 粒子系统中的粒子数量
pub const PARTICLE_COUNT: usize = 8192;

//==================================================================================================
// 数学类型别名
//==================================================================================================

/// 二维浮点向量
pub type Vec2 = nalgebra::Vector2<f32>;

/// 三维浮点向量  
pub type Vec3 = nalgebra::Vector3<f32>;

/// 四维浮点向量
pub type Vec4 = nalgebra::Vector4<f32>;

/// 4x4浮点矩阵
pub type Mat4 = nalgebra::Matrix4<f32>;
