// Filled anti-aliased discs. The reference builds a disc from sub-pixel splats (engine._disc): the picture
// of one disc is a fixed kernel around its centre, tabulated once per radius (the radius is rounded to a
// quarter of a pixel) in gDisc and looked up here.
#include "common.hlsl"

struct Dot { float x, y, r, i, meta; };
StructuredBuffer<Dot> gDots : register(t0);
StructuredBuffer<State> gStates : register(t2);
Texture2DArray<float> gDisc : register(t3);
SamplerState gLinear : register(s0);

static const float LUT_MAX = 30.0;       // largest tabulated radius (renderer.cpp: DISC_MAX)
static const float LUT_T = 128.0;        // texels across a slice

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation float4 c : C;        // centre x, y, quantised radius, brightness
    nointerpolation float layer : LAYER;
};

V2P VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    Dot d = gDots[dBase + iid];
    uint meta = (uint)d.meta;
    State st = gStates[meta >> 1];
    float2 c = viewPoint(st, float2(d.x, d.y));
    float r = d.r * gS * sqrt(st.vz);
    bool live = d.i > 1e-4 && c.x >= st.cx0 && c.x <= st.cx1 && c.y >= st.cy0 && c.y <= st.cy1;
    float rq = max(0.5, round(r * 4.0) / 4.0);
    float E = rq + 1.75;
    float2 corner = float2((vid & 1) ? E : -E, (vid & 2) ? E : -E);
    o.pos = live ? clipFromIndex(c + corner) : NOWHERE;
    o.c = float4(c, rq, d.i);
    o.layer = (float)(meta & 1);
    return o;
}

// Twice the integral of clamp(x, 0, 1): with it, the edge of a disc (a ramp one pixel wide) blurred by the
// splat kernel (a tent two pixels wide) is a second difference.
float ramp2(float x)
{
    return x <= 0.0 ? 0.0 : x < 1.0 ? x * x * x / 6.0 : 1.0 / 6.0 + 0.5 * x * (x - 1.0);
}

LightOut PS(V2P i)
{
    float2 delta = i.pos.xy - 0.5 - i.c.xy;
    float rq = i.c.z;
    float v;
    if (rq <= LUT_MAX)
    {
        float E = rq + 1.75;
        float n4 = 4.0 * rq + 7.0;                      // half-pixel steps across the kernel
        float tm1 = n4 * floor((LUT_T - 1.0) / n4);     // texels - 1 used by this slice
        float2 uv = ((delta + E) / (2.0 * E) * tm1 + 0.5) / LUT_T;
        v = 2.0 * gDisc.SampleLevel(gLinear, float3(uv, 4.0 * rq - 2.0), 0.0);
    }
    else
    {
        // very large discs (no table): the same edge, blurred by the splat kernel along the radius
        float x = rq + 0.5 - length(delta);
        v = ramp2(x + 1.0) - 2.0 * ramp2(x) + ramp2(x - 1.0);
    }
    return addLight(v * i.c.w, i.layer);
}
