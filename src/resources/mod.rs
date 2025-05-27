//! 资源管理模块
//! 包含缓冲区、图像、内存、着色器和描述符管理功能

pub mod memory;
pub mod resources_buffer;
pub mod resources_descriptor;
pub mod resources_images;
pub mod shader;

// 重新导出公共接口
pub use memory::*;
pub use resources_buffer::*;
pub use resources_descriptor::*;
pub use resources_images::*;
pub use shader::*;
