//! 粒子系统管理模块
//! 包含粒子缓冲区管理、描述符管理、渲染管线、计算着色器和渲染逻辑

pub mod compute;
pub mod particle_buffers;
pub mod particle_descriptors;
pub mod particle_pipeline;
pub mod particle_renderer;

// 重新导出公共接口
pub use compute::*;
pub use particle_buffers::*;
pub use particle_descriptors::*;
pub use particle_pipeline::*;
pub use particle_renderer::*;
