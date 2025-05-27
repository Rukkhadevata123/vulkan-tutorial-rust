#version 450

layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // 创建圆形粒子效果
    vec2 coord = gl_PointCoord - vec2(0.5);
    float distance = length(coord);
    
    if (distance > 0.5) {
        discard;
    }
    
    // 边缘淡化效果
    float alpha = 1.0 - smoothstep(0.0, 0.5, distance);
    outColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
