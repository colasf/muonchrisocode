// Single bilinear splats in pixel space (Frame.points): the lattices and point clouds.
#include "common.hlsl"

struct Splat { float x, y, w, layer; };
StructuredBuffer<Splat> gSplats : register(t0);

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation float4 c : C;
};

V2P VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    Splat s = gSplats[dBase + iid];
    float2 f = floor(float2(s.x, s.y));
    bool live = s.w != 0.0 && f.x >= 0.0 && f.x < gSize.x - 1.0 && f.y >= 0.0 && f.y < gSize.y - 1.0;
    float2 corner = f + float2((vid & 1) ? 2.0 : 0.0, (vid & 2) ? 2.0 : 0.0);
    o.pos = live ? clipFromWindow(corner) : NOWHERE;
    o.c = float4(s.x, s.y, s.w, s.layer);
    return o;
}

LightOut PS(V2P i)
{
    float2 u = abs(i.pos.xy - 0.5 - i.c.xy);
    return addLight(i.c.z * max(0.0, 1.0 - u.x) * max(0.0, 1.0 - u.y), i.c.w);
}
