//! Winit应用程序处理器模块
//! 包含窗口事件处理和应用程序生命周期管理

use anyhow::{Result, anyhow};
use log::*;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::vulkan_app::VulkanApp;

//==================================================================================================
// 应用程序事件处理器
//==================================================================================================

/// Winit应用程序处理器
/// 管理窗口生命周期和事件处理
#[derive(Default)]
pub struct App {
    window: Option<Window>,
    vulkan_app: Option<VulkanApp>,
    minimized: bool,
}

impl ApplicationHandler for App {
    /// 应用程序恢复处理
    /// 当应用程序重新获得焦点时调用
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // 创建窗口
            match self.create_window(event_loop) {
                Ok(window) => {
                    info!("窗口创建成功");

                    // 初始化Vulkan应用程序
                    match VulkanApp::create(&window) {
                        Ok(vulkan_app) => {
                            info!("Vulkan应用程序初始化成功");
                            self.vulkan_app = Some(vulkan_app);
                            self.window = Some(window);
                        }
                        Err(e) => {
                            error!("Vulkan应用程序初始化失败: {e}");
                            self.exit_with_error(event_loop, &e);
                        }
                    }
                }
                Err(e) => {
                    error!("窗口创建失败: {e}");
                    self.exit_with_error(event_loop, &e);
                }
            }
        }
    }

    /// 窗口事件处理
    /// 处理所有窗口相关事件
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // 窗口关闭请求
            WindowEvent::CloseRequested => {
                info!("接收到窗口关闭请求");
                self.cleanup_and_exit(event_loop);
            }

            // 键盘输入事件
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key_press(&event.physical_key, event_loop);
                }
            }

            // 窗口大小改变事件
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    info!("窗口最小化");
                    self.minimized = true;
                } else {
                    if self.minimized {
                        info!("窗口恢复显示: {}x{}", size.width, size.height);
                        self.minimized = false;
                    } else {
                        info!("窗口大小改变: {}x{}", size.width, size.height);
                    }
                    self.handle_resize();
                }
            }

            // 重绘请求事件
            WindowEvent::RedrawRequested => {
                self.handle_redraw(event_loop);
            }

            _ => {} // 忽略其他事件
        }
    }

    /// 应用程序退出处理
    /// 在应用程序完全退出前进行清理
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("应用程序正在退出");
        self.cleanup_vulkan();
    }
}

//==================================================================================================
// 窗口管理方法
//==================================================================================================

impl App {
    /// 创建应用程序窗口
    /// 配置窗口属性并创建窗口实例
    fn create_window(&self, event_loop: &ActiveEventLoop) -> Result<Window> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Tutorial (Rust) - Multi-Model + Particle System")
            .with_inner_size(LogicalSize::new(1024, 768))
            .with_resizable(true);

        event_loop
            .create_window(window_attributes)
            .map_err(|e| anyhow!("创建窗口失败: {}", e))
    }

    /// 处理窗口大小改变
    /// 标记Vulkan应用程序需要重建交换链
    fn handle_resize(&mut self) {
        if let Some(ref mut vulkan_app) = self.vulkan_app {
            vulkan_app.resized = true;
            debug!("标记交换链需要重建");
        }
    }

    /// 处理重绘请求
    /// 执行Vulkan渲染循环
    fn handle_redraw(&mut self, event_loop: &ActiveEventLoop) {
        // 跳过最小化状态的渲染
        if self.minimized {
            return;
        }

        match (&mut self.vulkan_app, &self.window) {
            (Some(vulkan_app), Some(window)) => {
                // 执行渲染
                if let Err(e) = vulkan_app.render(window) {
                    error!("渲染失败: {e}");
                    self.exit_with_error(event_loop, &e);
                    return;
                }

                // 请求下一帧
                window.request_redraw();
            }
            _ => {
                warn!("渲染跳过: Vulkan应用程序或窗口未初始化");
            }
        }
    }
}

//==================================================================================================
// 输入处理方法
//==================================================================================================

impl App {
    /// 处理按键事件
    /// 响应用户键盘输入
    fn handle_key_press(&mut self, key: &PhysicalKey, event_loop: &ActiveEventLoop) {
        match key {
            // ESC键退出应用程序
            PhysicalKey::Code(KeyCode::Escape) => {
                info!("按下ESC键，退出应用程序");
                self.cleanup_and_exit(event_loop);
            }

            // 左箭头键减少模型数量
            PhysicalKey::Code(KeyCode::ArrowLeft) => {
                if let Some(ref mut vulkan_app) = self.vulkan_app {
                    if vulkan_app.models > 1 {
                        vulkan_app.models -= 1;
                        info!("减少模型数量至: {}", vulkan_app.models);
                    }
                }
            }

            // 右箭头键增加模型数量
            PhysicalKey::Code(KeyCode::ArrowRight) => {
                if let Some(ref mut vulkan_app) = self.vulkan_app {
                    if vulkan_app.models < 10 {
                        vulkan_app.models += 1;
                        info!("增加模型数量至: {}", vulkan_app.models);
                    }
                }
            }

            // F1键显示帮助信息
            PhysicalKey::Code(KeyCode::F1) => {
                self.show_help();
            }

            // F11键切换全屏模式
            PhysicalKey::Code(KeyCode::F11) => {
                self.toggle_fullscreen();
            }

            _ => {} // 忽略其他按键
        }
    }

    /// 显示帮助信息
    /// 输出控制说明到日志
    fn show_help(&self) {
        info!("=== 控制说明 ===");
        info!("ESC       - 退出应用程序");
        info!("←/→       - 减少/增加模型数量 (1-10)");
        info!("F1        - 显示此帮助信息");
        info!("F11       - 切换全屏模式");
        info!(
            "当前模型数量: {}",
            self.vulkan_app.as_ref().map_or(0, |app| app.models)
        );
    }

    /// 切换全屏模式
    /// 在窗口模式和全屏模式之间切换
    fn toggle_fullscreen(&mut self) {
        if let Some(ref window) = self.window {
            let is_fullscreen = window.fullscreen().is_some();

            if is_fullscreen {
                info!("退出全屏模式");
                window.set_fullscreen(None);

                // 恢复窗口大小
                window.set_min_inner_size(Some(LogicalSize::new(1024.0, 768.0)));
            } else {
                info!("进入全屏模式");

                // 获取主显示器
                if let Some(monitor) = window
                    .primary_monitor()
                    .or_else(|| window.current_monitor())
                {
                    let monitor_name = monitor.name().unwrap_or_else(|| "Unknown".to_string());
                    let monitor_size = monitor.size();

                    info!(
                        "目标显示器: {} ({}x{})",
                        monitor_name, monitor_size.width, monitor_size.height
                    );

                    // 首先尝试无边框全屏
                    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(
                        monitor.clone(),
                    ))));

                    // 标记需要重建交换链（重要！）
                    if let Some(ref mut vulkan_app) = self.vulkan_app {
                        vulkan_app.resized = true;
                    }
                } else {
                    error!("无法获取任何显示器信息");
                }
            }

            // 请求重绘
            window.request_redraw();
        }
    }
}

//==================================================================================================
// 清理和错误处理方法
//==================================================================================================

impl App {
    /// 清理并退出应用程序
    /// 正常退出流程
    fn cleanup_and_exit(&mut self, event_loop: &ActiveEventLoop) {
        info!("开始清理应用程序资源");
        self.cleanup_vulkan();
        event_loop.exit();
    }

    /// 错误退出
    /// 发生不可恢复错误时的退出流程
    fn exit_with_error(&mut self, event_loop: &ActiveEventLoop, error: &anyhow::Error) {
        error!("应用程序遇到严重错误: {error}");

        // 输出详细错误信息
        let mut source = error.source();
        let mut level = 1;
        while let Some(err) = source {
            error!("  原因 {level}: {err}");
            source = err.source();
            level += 1;
        }

        self.cleanup_vulkan();
        event_loop.exit();
    }

    /// 清理Vulkan资源
    /// 安全销毁所有Vulkan对象
    fn cleanup_vulkan(&mut self) {
        if let Some(mut vulkan_app) = self.vulkan_app.take() {
            info!("清理Vulkan资源");
            vulkan_app.destroy();
            debug!("Vulkan资源清理完成");
        }

        if self.window.take().is_some() {
            debug!("窗口句柄已清理");
        }
    }
}

//==================================================================================================
// 公共接口
//==================================================================================================

/// 创建并运行应用程序
/// 应用程序的主入口点
pub fn run() -> Result<()> {
    // 创建事件循环
    let event_loop = EventLoop::new().map_err(|e| anyhow!("创建事件循环失败: {}", e))?;

    // 设置控制流为等待模式（节能）
    event_loop.set_control_flow(ControlFlow::Wait);

    // 创建应用程序实例
    let mut app = App::default();

    // 启动事件循环
    info!("启动事件循环");
    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("事件循环运行失败: {}", e))?;

    info!("应用程序正常退出");
    Ok(())
}
