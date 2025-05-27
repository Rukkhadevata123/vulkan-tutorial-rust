// filepath: /home/yoimiya/vulkan-tutorial-rust/assets/shaders/35_viking_room.frag
#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
} pcs;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);
    outColor = vec4(texColor.rgb * fragColor, texColor.a * pcs.opacity);
}
