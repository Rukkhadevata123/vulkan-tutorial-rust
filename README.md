# Vulkan Tutorial in Rust

这个项目是 Vulkan 教程（<https://vulkan-tutorial.com/>）的 Rust 实现版本。

它旨在提供一个清晰、现代的示例，展示如何在 Rust 中使用 Ash 箱（crate）进行 Vulkan 绑定。

## 已实现示例

该项目已完整实现 Vulkan 教程的全部示例：

- `26_texture_mapping`: 展示纹理映射
- `27_depth_buffering`: 实现深度缓冲
- `28_model_loading`: 演示 3D 模型加载
- `29_mipmapping`: 实现 mipmap 生成
- `30_multisampling`: 演示多重采样抗锯齿
- `31_push_constants`: 实现推送常量
- `32_recycling_command_buffers`: 演示命令缓冲区回收
- `33_secondary_command_buffers`: 使用次级命令缓冲区渲染多个模型
- `34_compute_shaders`: 实现计算着色器处理粒子系统

## 如何运行

1. **确保已安装并配置 Vulkan SDK**

   你需要将 `glslc`（SPIR-V 编译器）添加到 PATH 中，用于编译着色器。它通常包含在 Vulkan SDK 中。

2. **编译和运行示例**

   要运行特定示例并查看详细的调试日志，请从项目根目录执行以下命令：

   ```bash
   RUST_LOG=debug cargo run --bin 34_compute_shaders
   ```

   你可以根据需要将 `debug` 更改为其他日志级别，如 `info`、`warn` 或 `error`。
   例如，仅查看信息级别的消息：

   ```bash
   RUST_LOG=info cargo run --bin 34_compute_shaders
   ```

   如果未设置 `RUST_LOG` 环境变量，`pretty_env_logger` 可能会默认使用更高的日志级别（例如，仅显示错误），此时你可能看不到应用程序的 `info!` 或 `debug!` 消息。

3. **交互控制**

   - 在 `33_secondary_command_buffers` 示例中，使用左右箭头键可以控制显示的模型数量。
   - 在 `34_compute_shaders` 示例中，粒子系统会自动模拟物理运动。

## 编译着色器

着色器（`assets/shaders/` 目录中扩展名为 `.vert`、`.frag` 和 `.comp` 的文件）在构建过程中会通过 `build.rs` 脚本使用 `glslc` 自动编译为 SPIR-V 格式（`.spv`）。如果修改这些着色器源文件，Cargo 会自动重新运行 `build.rs` 来重新编译它们。

## 主要示例介绍

### 次级命令缓冲区 (33_secondary_command_buffers)

`33_secondary_command_buffers.rs` 展示了如何使用次级命令缓冲区来并行渲染多个模型实例，并通过键盘交互控制实例数量。

**核心功能**：

- 支持渲染多达 4 个独立的 3D 模型实例
- 每个模型具有不同的位置、旋转、缩放和透明度
- 使用次级命令缓冲区并行记录渲染命令
- 通过键盘左右箭头键控制模型数量

**次级命令缓冲区的优势**：

1. **并行记录**：多个次级命令缓冲区可以同时记录，提高 CPU 利用率
2. **重用**：次级命令缓冲区可以重复使用，减少记录开销
3. **模块化**：将不同的渲染任务分离到不同的次级命令缓冲区中
4. **效率**：可以在多个渲染过程中引用相同的次级命令缓冲区

### 计算着色器 (34_compute_shaders)

`34_compute_shaders.rs` 是本项目的最终示例，它展示了如何使用 Vulkan 的计算着色器功能实现高性能的粒子系统。

**核心功能**：

- 基于 GPU 加速的粒子系统实现
- 使用计算着色器进行粒子位置和速度更新
- 计算管线与图形管线并行工作
- 使用着色器存储缓冲区 (SSBO) 存储粒子数据
- 高效的帧间数据共享机制

**计算着色器的优势**：

1. **并行计算**：利用 GPU 的并行计算能力处理大量粒子
2. **与图形管线分离**：计算工作可与渲染工作并行执行
3. **灵活性**：可用于各种通用计算问题，不限于图形处理
4. **高性能**：适合大规模数据处理，如粒子系统、物理模拟等

## Vulkan 核心概念及拓扑关系

以下是 Vulkan 中的核心概念及其逻辑关系：

```mermaid
graph TD
    subgraph 物理设备和逻辑设备
        Instance-->|物理设备选择|PhysicalDevice
        PhysicalDevice-->|创建|LogicalDevice
        LogicalDevice-->|获取|Queue
    end
    
    subgraph 呈现
        Surface-->|能力查询|SwapchainSupport
        SwapchainSupport-->|创建|Swapchain
        Swapchain-->|获取|SwapchainImages
        SwapchainImages-->|创建|ImageViews
    end
    
    subgraph 渲染管线
        ShaderModules-->|构建|PipelineLayout
        DescriptorSetLayout-->|影响|PipelineLayout
        PushConstants-->|添加到|PipelineLayout
        PipelineLayout-->|用于|GraphicsPipeline
        PipelineLayout-->|用于|ComputePipeline
        RenderPass-->|配置|GraphicsPipeline
    end
    
    subgraph 资源
        Buffers-->|顶点数据|VertexBuffer
        Buffers-->|索引数据|IndexBuffer
        Buffers-->|统一数据|UniformBuffers
        Buffers-->|存储数据|StorageBuffers
        Images-->|加载|TextureImage
        TextureImage-->|生成|Mipmaps
        TextureImage-->|创建|TextureImageView
        TextureImageView-->|使用|Sampler
        TextureImageView-->|绑定到|DescriptorSets
        UniformBuffers-->|绑定到|DescriptorSets
        StorageBuffers-->|绑定到|DescriptorSets
    end
    
    subgraph 命令和同步
        CommandPool-->|分配|PrimaryCommandBuffers
        CommandPool-->|分配|SecondaryCommandBuffers
        CommandPool-->|分配|ComputeCommandBuffers
        SecondaryCommandBuffers-->|执行于|PrimaryCommandBuffers
        PrimaryCommandBuffers-->|提交到|GraphicsQueue
        ComputeCommandBuffers-->|提交到|ComputeQueue
        Semaphores-->|同步|Queue之间的工作
        Fences-->|同步|CPU与GPU之间的工作
    end
    
    subgraph 多重采样
        MSAA-->|创建|ColorImage
        MSAA-->|创建|DepthImage
        ColorImage-->|绑定到|Framebuffers
        DepthImage-->|绑定到|Framebuffers
        ImageViews-->|绑定到|Framebuffers
    end
    
    RenderPass-->|使用|Framebuffers
    GraphicsPipeline-->|在|PrimaryCommandBuffers
    ComputePipeline-->|在|ComputeCommandBuffers
    DescriptorSets-->|绑定到|CommandBuffers
    PushConstants-->|推送到|CommandBuffers
    VertexBuffer-->|绑定到|CommandBuffers
    IndexBuffer-->|绑定到|CommandBuffers
```

## Vulkan 渲染流程

1. **初始化**：创建 Instance、PhysicalDevice 和 LogicalDevice
2. **窗口集成**：创建 Surface 并设置 Swapchain
3. **资源准备**：加载纹理和 3D 模型，创建各种缓冲区
4. **渲染设置**：创建 RenderPass、Pipeline 和 DescriptorSets
5. **渲染循环**：
   - 获取下一个 Swapchain 图像
   - 执行计算着色器工作（在计算示例中）
   - 更新命令缓冲区
   - 更新统一缓冲区（矩阵变换）
   - 提交命令缓冲区到图形队列
   - 呈现到屏幕
