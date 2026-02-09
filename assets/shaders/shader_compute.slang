// shader_compute.slang
// Vertex: point sprite input -> set point size and forward color
// Fragment: circular, smooth (Gaussian) falloff for point sprite
// Compute: particle integration with attraction + tangential (vortex) force

struct VSInput
{
    [[vk::location(0)]] float2 inPosition;
    [[vk::location(1)]] float2 inVelocity;
    [[vk::location(2)]] float4 inColor;
};
struct VSOutput
{
    float4 pos : SV_Position;
    float pointSize : SV_PointSize;
    float3 fragColor : COLOR0;
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float3 fragColor : COLOR0;
    float2 pointCoord : SV_PointCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    // Set a fixed screen-space point size and pass color to fragment shader
    VSOutput output;
    output.pointSize = 14.0;
    output.pos = float4(input.inPosition, 1.0, 1.0);
    output.fragColor = input.inColor.rgb;
    return output;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_TARGET
{
    // Compute circular mask centered in point sprite and apply Gaussian alpha
    float2 coord = input.pointCoord - float2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;     // ensure circular shape

    float r = dist * 2.0;
    float alpha = exp(-4.0 * r * r); // Gaussian-like falloff

    return float4(input.fragColor, alpha);
}

struct Particle
{
    float2 position;
    float2 velocity;
    float4 color;
};

struct PushConstants
{
    float deltaTime;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct ParticleSSBO
{
    Particle particles;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleSSBO> particlesIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<ParticleSSBO> particlesOut;

[shader("compute")]
[numthreads(256, 1, 1)]
void compMain(uint3 threadId: SV_DispatchThreadID)
{
    uint index = threadId.x;

    // Load current particle state
    float2 position = particlesIn[index].particles.position;
    float2 velocity = particlesIn[index].particles.velocity;

    // --- Vortex-like motion (attraction + tangential force) ---
    // Center is screen origin (0,0). Use 'toCenter' vector to compute forces.
    float2 center = float2(0.0, 0.0);
    float2 toCenter = center - position;
    float dist = length(toCenter);

    // Prevent division by zero / extreme forces near origin
    float softDist = max(dist, 0.05);

    // Attraction: normalized toward center scaled by a small strength / distance
    float2 gravity = normalize(toCenter) * (0.0002 / softDist);

    // Tangential (rotational) force perpendicular to toCenter -> creates vortex
    float2 tangent = float2(-toCenter.y, toCenter.x);
    float2 rotation = normalize(tangent) * (0.0005 / softDist);

    // Integrate velocity with small damping to keep simulation stable
    velocity += (gravity + rotation) * pc.deltaTime;
    velocity *= 0.995; // damping

    // Integrate position
    position += velocity * pc.deltaTime;

    // Boundary handling: bounce with energy loss
    if ((position.x <= -1.0) || (position.x >= 1.0))
    {
        velocity.x = -velocity.x * 0.8;
        position.x = clamp(position.x, -1.0, 1.0);
    }
    if ((position.y <= -1.0) || (position.y >= 1.0))
    {
        velocity.y = -velocity.y * 0.8;
        position.y = clamp(position.y, -1.0, 1.0);
    }

    // Store back to output SSBO
    particlesOut[index].particles.position = position;
    particlesOut[index].particles.velocity = velocity;
}