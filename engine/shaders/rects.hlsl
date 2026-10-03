// Filled rectangles snapped to whole pixels (Frame._rects), and the rects that multiply what was drawn
// before them (occlude, dim, scale_rect, the boxes of the tags).
#include "common.hlsl"

struct Rect { float x0, y0, x1, y1, i, meta; };
StructuredBuffer<Rect> gRects : register(t0);
StructuredBuffer<State> gStates : register(t2);

struct LightOp { int x0, y0, x1, y1; float fw, fr; uint nseg, ndot, nsplat, nrect; };
StructuredBuffer<LightOp> gOps : register(t4);

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation float2 c : C;
};

V2P VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    Rect r = gRects[dBase + iid];
    uint meta = (uint)r.meta;
    State st = gStates[meta >> 1];
    float2 lo = max(float2(st.cx0, st.cy0), 0.0);
    float2 hi = min(float2(st.cx1, st.cy1), gSize);
    float2 a = trunc(clamp(round(viewPoint(st, float2(r.x0, r.y0))), lo, hi));
    float2 b = trunc(clamp(round(viewPoint(st, float2(r.x1, r.y1))), lo, hi));
    bool live = b.x > a.x && b.y > a.y && r.i != 0.0;
    o.pos = live ? clipFromWindow(float2((vid & 1) ? b.x : a.x, (vid & 2) ? b.y : a.y)) : NOWHERE;
    o.c = float2(r.i, (float)(meta & 1));
    return o;
}

LightOut PS(V2P i)
{
    return addLight(i.c.x, i.c.y);
}

V2P VSOp(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    LightOp r = gOps[dBase + iid];
    o.pos = clipFromWindow(float2((vid & 1) ? r.x1 : r.x0, (vid & 2) ? r.y1 : r.y0));
    o.c = float2(r.fw, r.fr);
    return o;
}

LightOut PSOp(V2P i)
{
    LightOut o;
    o.c = float4(0.0, 0.0, 0.0, 0.0);
    o.f = float4(i.c.x, i.c.y, 1.0, 1.0);
    return o;
}
