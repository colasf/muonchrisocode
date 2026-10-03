// The finished picture copied into the preview window (scaled), and the rects the tower placement tool
// lays over it.
#include "common.hlsl"

Texture2D<float4> gPicture : register(t0);
SamplerState gLinear : register(s0);

float4 VSFull(uint vid : SV_VertexID) : SV_Position
{
    return float4((vid == 1) ? 3.0 : -1.0, (vid == 2) ? -3.0 : 1.0, 0.0, 1.0);
}

// dP0 = where the picture goes in the window (x0, y0, x1, y1), in pixels
float4 PSBlit(float4 pos : SV_Position) : SV_Target
{
    float2 uv = (pos.xy - dP0.xy) / (dP0.zw - dP0.xy);
    if (any(uv < 0.0) || any(uv > 1.0)) return float4(0.02, 0.02, 0.02, 1.0);
    return float4(gPicture.SampleLevel(gLinear, uv, 0.0).rgb, 1.0);
}

struct Over { float x0, y0, x1, y1, r, g, b, a; };
StructuredBuffer<Over> gOver : register(t4);

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation float4 col : COL;
};

V2P VSOver(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    Over r = gOver[dBase + iid];
    o.pos = clipFromWindow(float2((vid & 1) ? r.x1 : r.x0, (vid & 2) ? r.y1 : r.y0));
    o.col = float4(r.r, r.g, r.b, r.a);
    return o;
}

float4 PSOver(V2P i) : SV_Target
{
    return i.col;
}
