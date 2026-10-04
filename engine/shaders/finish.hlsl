// Frame.finish: light * exposure + text, bloom (8-level pyramid), soft-knee tonemap per layer,
// white + red, inverted rects, dither. The lift of the output is the one thing here the reference does
// not have: it is the engine's own, set on site (0 = the picture of the reference).
#include "common.hlsl"

Texture2D<float2> gA : register(t0);
Texture2D<float2> gB : register(t1);

float4 VSFull(uint vid : SV_VertexID) : SV_Position
{
    return float4((vid == 1) ? 3.0 : -1.0, (vid == 2) ? -3.0 : 1.0, 0.0, 1.0);
}

// base = light * exposure + text * gain        (gA = light, gB = text)
float4 PSBase(float4 pos : SV_Position) : SV_Target
{
    int3 p = int3(pos.xy, 0);
    return float4(gA.Load(p) * gExposure + gB.Load(p) * gTextGain, 0.0, 0.0);
}

// 2 x 2 average; an odd size repeats its last row / column.   dP0.xy = size of the source - 1
float4 PSDown(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy) * 2;
    int2 m = int2(dP0.xy);
    float2 v = gA.Load(int3(min(p, m), 0)) + gA.Load(int3(min(p + int2(1, 0), m), 0))
             + gA.Load(int3(min(p + int2(0, 1), m), 0)) + gA.Load(int3(min(p + int2(1, 1), m), 0));
    return float4(0.25 * v, 0.0, 0.0);
}

float2 up2(Texture2D<float2> t, int2 p, int2 m)      // bilinear 2x, half-pixel centres, edges clamped
{
    int2 i = p >> 1;
    int2 j = clamp(i + ((p & 1) * 2 - 1), int2(0, 0), m);
    float2 a = t.Load(int3(i, 0)), b = t.Load(int3(j.x, i.y, 0));
    float2 c = t.Load(int3(i.x, j.y, 0)), d = t.Load(int3(j, 0));
    return 0.5625 * a + 0.1875 * (b + c) + 0.0625 * d;
}

// one level of the pyramid: blur(level) * weight + upsampled sum of the smaller levels
// gA = the level, gB = the sum so far (half size).  dP0.xy = size of gA - 1, dP0.zw = size of gB - 1,
// dP1.x = weight, dP1.y = 1 if there is a gB
float4 PSBlurUp(float4 pos : SV_Position) : SV_Target
{
    static const float k[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    int2 p = int2(pos.xy);
    int2 m = int2(dP0.xy);
    float2 acc = float2(0.0, 0.0);
    for (int y = -2; y <= 2; y++)
        for (int x = -2; x <= 2; x++)
            acc += gA.Load(int3(clamp(p + int2(x, y), int2(0, 0), m), 0)) * (k[x + 2] * k[y + 2]);
    acc *= dP1.x;
    if (dP1.y > 0.5)
        acc += up2(gB, p, int2(dP0.zw));
    return float4(acc, 0.0, 0.0);
}

float tonemap(float x)
{
    const float knee = 0.72;
    return x < knee ? x : knee + (1.0 - knee) * (1.0 - exp(-(max(x, knee) - knee) / (1.0 - knee)));
}

float hash(uint2 p, uint seed)
{
    uint h = p.x * 0x9E3779B1u ^ (p.y * 0x85EBCA77u) ^ (seed * 0xC2B2AE3Du);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (float)(h >> 8) / 16777216.0;
}

// gA = base, gB = bloom sum of level 1.   dP0.zw = size of gB - 1, dA = 1: inverted (white field, black lines)
float4 PSFinal(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    float2 light = gA.Load(int3(p, 0));
    if (gBloomGain != 0.0)
        light += up2(gB, p, int2(dP0.zw)) * gBloomGain;
    float wt = tonemap(light.x), rt = tonemap(light.y);
    float3 rgb = dA == 1 ? float3(1.0 - wt, 1.0 - wt - rt, 1.0 - wt - rt) : wt + rt * float3(1.0, 0.045, 0.035);
    rgb = saturate(rgb);
    if (gLift > 0.0) {                    // the lift of the OUTPUT panel: v -> v (1 + lift) / (1 + lift v), on the
        float m = max(rgb.r, max(rgb.g, rgb.b));       // strongest channel so that a colour keeps its hue
        rgb *= (1.0 + gLift) / (1.0 + gLift * m);
    }
    if (gDither > 0.5)
        rgb = floor(clamp(rgb * 255.0 + hash(uint2(p), gDitherSeed) - 0.5, 0.0, 255.0)) / 255.0;
    return float4(rgb, 1.0);
}

struct Box { int x0, y0, x1, y1; };
StructuredBuffer<Box> gBoxes : register(t4);

float4 VSBox(uint vid : SV_VertexID, uint iid : SV_InstanceID) : SV_Position
{
    Box r = gBoxes[dBase + iid];
    return clipFromWindow(float2((vid & 1) ? r.x1 : r.x0, (vid & 2) ? r.y1 : r.y0));
}
