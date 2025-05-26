#version 450

// 特化常量用于区分渲染类型
layout(constant_id = 0) const uint RENDER_TYPE = 0; // 0=模型, 1=粒子

// 模型绘制所需的统一缓冲区
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// 模型矩阵的推送常量
layout(push_constant) uniform PushConstants {
    mat4 model;
    float opacity;  // 偏移量为64 (用于片段着色器)
} pcs;

// 模型网格的顶点属性
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

// 粒子数据 (inPosition复用于粒子位置, inColor的rgb用于粒子颜色)

// 输出到片段着色器
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out float isParticle; // 告知片段着色器渲染类型

void main() {
    if (RENDER_TYPE == 0) {
        // 模型渲染路径
        gl_Position = ubo.proj * ubo.view * pcs.model * vec4(inPosition, 1.0);
        fragColor = inColor;
        fragTexCoord = inTexCoord;
        isParticle = 0.0;
    } else {
        // 粒子渲染路径
        gl_PointSize = 14.0;
        // 使用inPosition的前两个分量作为粒子的2D位置
        gl_Position = vec4(inPosition.xy, 0.0, 1.0);
        fragColor = inColor;
        fragTexCoord = vec2(0.0); // 粒子不使用纹理坐标
        isParticle = 1.0;
    }
}