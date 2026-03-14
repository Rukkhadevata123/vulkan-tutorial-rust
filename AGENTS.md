# Vulkan Tutorial Rust - 项目上下文文档

## 项目概述

这是一个使用 Rust 和 Vulkano 包装器实现的 Vulkan 应用程序教程项目。该项目提供清晰、实用的现代 Vulkan 图形和计算流水线示例，从基本初始化逐步过渡到功能完整的渲染和模拟。

### 核心技术栈

- **编程语言**: Rust (Edition 2024)
- **Vulkan 包装器**: Vulkano 0.35
- **窗口管理**: winit 0.30
- **模型加载**: gltf 1.4, tobj 4.0
- **图像处理**: image 0.25
- **数学运算**: nalgebra 0.34
- **随机数生成**: rand 0.10
- **着色器语言**: Slang

### 项目架构

项目采用模块化设计，每个示例都是一个独立的二进制目标，展示了不同的 Vulkan 概念和技术：

1. **example1** - 基础示例（未详细说明）
2. **viking_room** - 完整的 3D 渲染流水线
3. **compute_shader** - GPU 计算着色器粒子系统
4. **multithreading** - 多线程粒子系统
5. **ray_tracing** - 硬件光线追踪混合渲染

### 代码组织模式

所有源文件遵循一致的结构化模式：

```rust
// --- Group 1: Imports ---
// 所有外部依赖导入，按类别分组

// --- Group 2: Constants & Data Structures ---
// 常量定义和数据结构（Vertex, UniformBuffer 等）

// --- Group 3: App Struct Definition ---
// 应用程序主结构体，包含所有 Vulkan 对象

// --- Group 4: Implementation ---
// App trait 的实现方法
```

### Vulkan 核心概念

项目涵盖以下 Vulkan 概念：

- **初始化**: Instance, Physical Device, Logical Device, Queues
- **资源管理**: Buffers, Images, Descriptors, Descriptor Sets
- **流水线**: Graphics Pipeline, Compute Pipeline, Ray Query
- **同步**: Fences, Semaphores, Double Buffering (Frames in Flight)
- **呈现**: Surface, Swapchain, VSync

## 构建和运行

### 前置要求

- **Rust 工具链**: 安装最新的稳定版 Rust
- **Vulkan SDK**: 用于验证层和着色器编译工具
- **Slang** (可选): 如果需要手动编译 `.slang` 着色器

### 着色器编译

将 Slang 着色器编译为 Vulkan 兼容的 SPIR-V 格式：

```bash
slangc assets/shaders/shader_compute.slang \
    -target spirv \
    -profile spirv_1_4 \
    -emit-spirv-directly \
    -fvk-use-entrypoint-name \
    -entry vertMain \
    -entry fragMain \
    -entry compMain \
    -o assets/shaders/shader_compute.spv
```

**注意**: 编译后的 `.spv` 文件被 `.gitignore` 忽略，运行时会自动查找已编译的着色器。

### 构建项目

```bash
# Debug 模式构建
cargo build

# Release 模式构建（推荐用于性能）
cargo build --release
```

### 运行示例

每个示例都是独立的二进制目标，可以单独运行：

```bash
# 运行计算着色器粒子系统
cargo run --release --bin compute_shader

# 运行 Viking Room 渲染器
cargo run --release --bin viking_room

# 运行多线程粒子系统
cargo run --release --bin multithreading

# 运行光线追踪演示
cargo run --release --bin ray_tracing
```

### 测试

当前项目没有包含自动化测试。每个示例通过手动运行和视觉验证来测试。

## 开发规范

### 代码风格

1. **导入组织**: 使用分组注释（如 `// --- Group 1: Imports ---`）组织导入语句
2. **常量定义**: 使用 `const` 定义常量，命名使用 SCREAMING_SNAKE_CASE
3. **数据结构**:
   - 使用 `#[repr(C)]` 确保与着色器的内存布局兼容
   - 使用 `#[derive(Debug, Clone, Copy, BufferContents)]` 用于缓冲区数据
   - 使用 `#[derive(Vertex)]` 或 `#[derive(VertexTrait)]` 用于顶点数据
   - 使用属性 `#[format(...)]` 和 `#[name(...)]` 指定顶点输入格式和着色器位置
4. **命名约定**:
   - 结构体: PascalCase
   - 变量和函数: snake_case
   - 常量: SCREAMING_SNAKE_CASE
   - 类型别名: PascalCase

### Vulkan 特定模式

1. **双缓冲**: 使用 `MAX_FRAMES_IN_FLIGHT: usize = 2` 实现双缓冲，避免竞态条件
2. **资源分配**:
   - 使用 `StandardMemoryAllocator` 进行内存分配
   - 使用 `StandardCommandBufferAllocator` 进行命令缓冲区分配
   - 使用 `StandardDescriptorSetAllocator` 进行描述符集分配
3. **同步**:
   - 使用 `GpuFuture` 管理 GPU 操作同步
   - 使用 Fences 进行 CPU-GPU 同步
   - 使用 Semaphores 进行 GPU 内部同步
4. **动态渲染**: 使用 Vulkan 1.3 的动态渲染特性，避免显式的 Render Pass 对象

### 多线程模式

`multithreading` 示例展示了以下多线程模式：

- **并行命令录制**: 使用多个工作线程并行录制命令缓冲区
- **线程本地分配器**: 每个线程使用独立的 `StandardCommandBufferAllocator` 避免锁竞争
- **Push Constants 分区**: 使用 Push Constants 为每个线程分配特定的粒子范围
- **同步提交**: 主线程收集所有录制的命令缓冲区并批量提交

### 着色器开发

- **着色器语言**: 使用 Slang 而非 GLSL
- **入口点**: 使用明确的入口点名称（如 `vertMain`, `fragMain`, `compMain`）
- **着色器阶段**: 支持 vertex、fragment 和 compute 阶段
- **资源绑定**: 使用 ConstantBuffer 和 Sampler 进行资源绑定

### 文件结构

```
src/
├── example1.rs           # 基础示例
├── viking_room.rs        # 3D 渲染示例（约 1229 行）
├── compute_shader.rs     # 计算着色器示例（约 849 行）
├── multithreading.rs     # 多线程示例（约 968 行）
└── ray_tracing.rs        # 光线追踪示例（约 1852 行）

assets/
├── models/               # 3D 模型文件
│   ├── plant_on_table.mtl
│   ├── plant_on_table.obj
│   └── viking_room.glb
├── outputs/              # 渲染输出图像
├── shaders/              # Slang 着色器源文件
│   ├── multithreading.slang
│   ├── ray_tracing.slang
│   ├── shader_compute.slang
│   └── viking_room.slang
└── textures/             # 纹理图像
```

### Git 忽略规则

以下文件和目录被忽略：
- `/target` - Rust 编译输出
- `*.spv` - 编译后的着色器文件（assets/shaders/*.spv）
- `/legacy` - 旧代码
- `tmp.rs`, `tmp.cpp`, `tmpp.cpp` - 临时文件

### 贡献指南

1. **添加新示例**: 在 `src/` 目录中创建新的 `.rs` 文件
2. **更新 Cargo.toml**: 在 `[[bin]]` 部分添加新的二进制目标配置
3. **着色器文件**: 将新着色器放在 `assets/shaders/` 目录中
4. **文档更新**: 更新 README.md 以包含新示例的说明
5. **代码风格**: 遵循现有的代码组织模式和命名约定

### 常见问题

1. **着色器编译失败**: 确保安装了 Vulkan SDK 和 Slang 编译器
2. **性能问题**: 使用 `--release` 模式运行以获得最佳性能
3. **内存问题**: 检查双缓冲实现，确保 CPU 和 GPU 不会同时访问同一资源
4. **多线程问题**: 确保每个线程使用独立的分配器，避免锁竞争

### 参考资源

- [Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html)
- [Vulkano](https://github.com/vulkano-rs/vulkano)
- [Slang Shading Language](https://shader-slang.com/)

## 项目状态

这是一个持续更新的项目。新示例、架构改进和高级功能将随时间添加，以涵盖 Vulkan API 的更广泛方面。当前的实现展示了从基础 3D 渲染到高级光线追踪的完整技术栈。