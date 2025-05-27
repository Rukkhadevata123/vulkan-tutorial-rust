//! 主程序文件
//! 重构后的模块化结构

#![allow(unsafe_code)]

use anyhow::Result;
use log::*;

mod vk_window;
use vk_window::*;

// 基础模块
mod app;
mod constants;
mod errors;
mod types;
mod vulkan_app;

// Vulkan核心模块
mod resources;
mod vulkan;

// 业务逻辑模块
mod model;
mod particle;
mod texture;

// 重新导出 Vulkan 函数
pub use model::*;
pub use particle::*;
pub use resources::*;
pub use texture::*;
pub use vulkan::*;

//==================================================================================================
// 主程序入口
//==================================================================================================

/// 应用程序主入口点
/// 初始化日志系统并启动事件循环
fn main() -> Result<()> {
    // 初始化日志系统 - 默认使用debug级别
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string());

    pretty_env_logger::formatted_builder()
        .filter_level(log_level.parse().unwrap_or(log::LevelFilter::Debug))
        .filter_module("sctk", log::LevelFilter::Warn)
        .filter_module("wayland", log::LevelFilter::Warn)
        .format_timestamp_secs()
        .init();

    info!("=== Vulkan教程应用程序启动 ===");
    info!("版本: 多模型渲染 + 粒子系统 + 计算着色器");
    info!("日志级别: {}", log::max_level());

    // 显示控制说明
    info!("控制说明:");
    info!("  ESC       - 退出应用程序");
    info!("  ←/→       - 减少/增加模型数量");
    info!("  F1        - 显示帮助信息");
    info!("  F11       - 切换全屏模式");

    // 运行应用程序
    app::run()
}
