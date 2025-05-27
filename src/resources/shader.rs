//! 着色器管理模块
//! 负责SPIR-V着色器模块的创建和管理

use anyhow::{Result, anyhow};
use ash::Device;
use ash::vk;
use std::io::Cursor;

/// 从SPIR-V字节码创建着色器模块
pub fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = Cursor::new(bytecode);
    let code =
        ash::util::read_spv(&mut cursor).map_err(|e| anyhow!("读取SPIR-V字节码失败: {}", e))?;

    if code.is_empty() {
        return Err(anyhow!("读取后SPIR-V代码为空"));
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| anyhow!("创建着色器模块失败: {}", e))
    }
}
