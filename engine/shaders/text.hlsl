// The two 8-bit text layers. Every instance paints  dst = dst + (colour - dst) * amount  on its pixels:
// a solid rect, or a glyph (PIL's own bitmap, from the atlas, on whole pixels), upright or turned 90 degrees.
#include "common.hlsl"

struct Inst { int x0, y0, x1, y1; int u, v; uint mode, pad; float cw, cr, fw, fr; };
StructuredBuffer<Inst> gInst : register(t0);
Texture2D<float> gAtlas : register(t1);

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation int4 rect : RECT;
    nointerpolation int3 uvm : UVM;
    nointerpolation float4 col : COL;
};

struct TextOut
{
    float4 c : SV_Target0;
    float4 f : SV_Target1;
};

V2P VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    Inst r = gInst[dBase + iid];
    o.pos = clipFromWindow(float2((vid & 1) ? r.x1 : r.x0, (vid & 2) ? r.y1 : r.y0));
    o.rect = int4(r.x0, r.y0, r.x1, r.y1);
    o.uvm = int3(r.u, r.v, (int)r.mode);
    o.col = float4(r.cw, r.cr, r.fw, r.fr);
    return o;
}

TextOut PS(V2P i)
{
    int2 px = (int2)i.pos.xy;
    float cov = 1.0;
    if (i.uvm.z == 1)
        cov = gAtlas.Load(int3(i.uvm.xy + px - i.rect.xy, 0));
    else if (i.uvm.z == 2)
        cov = gAtlas.Load(int3(i.uvm.x + (i.rect.w - 1 - px.y), i.uvm.y + (px.x - i.rect.x), 0));
    TextOut o;
    o.c = float4(i.col.x, i.col.y, 0.0, 0.0);
    o.f = float4(i.col.z * cov, i.col.w * cov, 0.0, 0.0);
    return o;
}
