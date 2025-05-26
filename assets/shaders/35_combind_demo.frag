#version 450

// 特化常量用于区分渲染类型（与顶点着色器相同）
layout(constant_id = 0) const uint RENDER_TYPE = 0; // 0=模型, 1=粒子

// 纹理采样器（用于模型）
layout(binding = 1) uniform sampler2D texSampler;

// 来自顶点着色器的输入
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in float isParticle;

// 推送常量
layout(push_constant) uniform PushConstants {
    mat4 model;     // 偏移0 (给顶点着色器)
    float opacity;  // 偏移64
} pcs;

// 输出颜色
layout(location = 0) out vec4 outColor;

void main() {
    if (isParticle > 0.5) {
        // 粒子渲染路径
        vec2 coord = gl_PointCoord - vec2(0.5);
        float alpha = 0.5 - length(coord);
        
        // 添加发光效果
        vec3 glow = fragColor * (1.0 + 0.5 * sin(length(coord) * 15.0));
        
        outColor = vec4(glow, alpha);
    } else {
        // 模型渲染路径
        vec4 texColor = texture(texSampler, fragTexCoord);
        outColor = vec4(texColor.rgb * fragColor, texColor.a * pcs.opacity);
    }
}