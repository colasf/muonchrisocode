// The one pixel-space effect of the show: the corruption of scenes/glitch.py (make_post), applied to the
// light stack before the bloom. Each operation of the draw list (drawlist.py: post_ops / apply_post) is a
// few passes over its rect: the part it reads is first copied to a scratch texture, then written back.
#include "common.hlsl"

Texture2D<float2> gA : register(t0);
Texture2D<float2> gB : register(t1);

// dP0 = the rect being written (x0, y0, x1, y1), in pixels
float4 VSRect(uint vid : SV_VertexID) : SV_Position
{
    return clipFromWindow(float2((vid & 1) ? dP0.z : dP0.x, (vid & 2) ? dP0.w : dP0.y));
}

// the rect rolled by dP1.xy pixels (it wraps inside the rect)
float4 PSRoll(float4 pos : SV_Position) : SV_Target
{
    int2 o = int2(dP0.xy);
    int2 n = int2(dP0.zw) - o;
    int2 q = ((int2(pos.xy) - o - int2(dP1.xy)) % n + n) % n;
    return float4(gA.Load(int3(o + q, 0)), 0.0, 0.0);
}

// a copy of what is dP1.xy pixels away, times dP1.z
float4 PSStamp(float4 pos : SV_Position) : SV_Target
{
    return float4(gA.Load(int3(int2(pos.xy) + int2(dP1.xy), 0)) * dP1.z, 0.0, 0.0);
}

// one layer poured into the other: dP1.x = 0 white -> red, 1 red -> white
float4 PSPour(float4 pos : SV_Position) : SV_Target
{
    float2 a = gA.Load(int3(pos.xy, 0));
    float v = a.x + a.y;
    return dP1.x < 0.5 ? float4(0.0, v, 0.0, 0.0) : float4(v, 0.0, 0.0, 0.0);
}

// smear, step 1: what is bright enough to drag (dP1.x = threshold)
float4 PSBright(float4 pos : SV_Position) : SV_Target
{
    float2 a = gA.Load(int3(pos.xy, 0));
    return float4(a.x > dP1.x ? a.x : 0.0, a.y > dP1.x ? a.y : 0.0, 0.0, 0.0);
}

// smear, step 2 (repeated with a doubling distance): the running maximum of the bright pixels behind,
// decaying with the distance.   dP1.xy = where to look (pixels), dP1.z = decay over that distance
float4 PSDrag(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    int2 q = p + int2(dP1.xy);
    float2 m = gA.Load(int3(p, 0));
    if (q.x >= (int)dP0.x && q.x < (int)dP0.z && q.y >= (int)dP0.y && q.y < (int)dP0.w)
        m = max(m, gA.Load(int3(q, 0)) * dP1.z);
    return float4(m, 0.0, 0.0);
}

// smear, step 3: the picture (gA), or the smear (gB) times dP1.x where it is brighter
float4 PSSmear(float4 pos : SV_Position) : SV_Target
{
    int3 p = int3(pos.xy, 0);
    return float4(max(gA.Load(p), gB.Load(p) * dP1.x), 0.0, 0.0);
}
