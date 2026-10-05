// Anti-aliased lines. The reference drops a bilinear splat every `spacing` pixels along the line.
//   spacing <= 0.5 (nearly all of the show): every pixel gets the integral of that splat kernel along the
//     segment, exactly (the kernel is piecewise linear, the intensity is linear: Simpson's rule on each
//     piece is exact). That is the reference with its sampling made infinitely fine.
//   spacing > 0.5 (the streak field of the opening, the whirl of the outro): the splats are far enough
//     apart to show as a grain, which is part of the look: the same splats are summed here, one by one.
// In both cases a splat whose 2 x 2 footprint is not wholly inside the picture is left out, as the
// reference leaves it out (engine.Frame.flush).
#include "common.hlsl"

struct Seg { float x0, y0, x1, y1, i0, i1, width, spacing, meta; };
StructuredBuffer<Seg> gSegs : register(t0);
Buffer<uint> gExpand : register(t1);          // segment index | pass << 24: a wide line is several passes
StructuredBuffer<State> gStates : register(t2);

struct V2P
{
    float4 pos : SV_Position;
    nointerpolation float4 ab : AB;
    nointerpolation float4 ii : II;           // intensity at a, at b (gain included), layer, splats (0: integrate)
};

V2P VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    V2P o;
    uint e = gExpand[dBase + iid];
    Seg g = gSegs[e & 0xFFFFFF];
    float k = (float)(e >> 24);
    uint meta = (uint)g.meta;
    State st = gStates[meta >> 1];
    float2 p0 = viewPoint(st, float2(g.x0, g.y0));
    float2 p1 = viewPoint(st, float2(g.x1, g.y1));
    float2 d = p1 - p0;
    // Liang-Barsky against the clip rect
    float t0 = 0.0, t1 = 1.0;
    float pp[4] = { -d.x, d.x, -d.y, d.y };
    float qq[4] = { p0.x - st.cx0, st.cx1 - p0.x, p0.y - st.cy0, st.cy1 - p0.y };
    for (int j = 0; j < 4; j++)
    {
        if (pp[j] < 0.0) t0 = max(t0, qq[j] / pp[j]);
        else if (pp[j] > 0.0) t1 = min(t1, qq[j] / pp[j]);
        else if (qq[j] < 0.0) t1 = -1.0;
    }
    bool live = (g.i0 > 1e-4 || g.i1 > 1e-4) && t0 <= t1;
    float2 a = p0 + d * t0;
    float2 b = p0 + d * t1;
    float ia = g.i0 + (g.i1 - g.i0) * t0;
    float ib = g.i0 + (g.i1 - g.i0) * t1;
    float2 dd = b - a;
    float L = length(dd);
    float2 dir = dd / max(L, 1e-6);
    if (L <= 0.0) { dir = float2(1.0, 0.0); live = false; }
    float2 n = float2(-dir.y, dir.x);
    float gain = gS035;
    float width = g.width + gWeight;              // the WEIGHT of the OUTPUT panel
    if (width > 1.25)
    {
        float ws = width * gS;
        float reps = clamp(ceil(ws / 0.6), 2.0, 255.0);       // as many passes as renderer.cpp made instances
        float off = (k / (reps - 1.0) - 0.5) * (ws - 1.0);
        a += n * off;
        b += n * off;
        gain *= ws / reps;
    }
    float ext = abs(dir.x) + abs(dir.y);          // reach of the splat kernel along the line and across it
    float along = (vid & 1) ? L + ext : -ext;
    float across = (vid & 2) ? ext : -ext;
    o.pos = live ? clipFromIndex(a + dir * along + n * across) : NOWHERE;
    o.ab = float4(a, b);
    o.ii = float4(ia * gain, ib * gain, (float)(meta & 1), g.spacing > 0.5001 ? max(1.0, ceil(L / g.spacing)) : 0.0);
    return o;
}

float kern(float t, float2 u0, float2 D, float ia, float ib)
{
    float2 u = abs(u0 - t * D);
    return (ia + (ib - ia) * t) * max(0.0, 1.0 - u.x) * max(0.0, 1.0 - u.y);
}

LightOut PS(V2P i)
{
    float2 p = i.pos.xy - 0.5;
    float2 A = i.ab.xy;
    float2 D = i.ab.zw - A;
    float2 u0 = p - A;
    float ia = i.ii.x, ib = i.ii.y;

    // the part of the line whose splats are kept: 0 <= x < W - 1, 0 <= y < H - 1
    float flo = 0.0, fhi = 1.0;
    float2 lim = gSize - 1.0;
    if (abs(D.x) > 1e-7)
    {
        float a = -A.x / D.x, b = (lim.x - A.x) / D.x;
        flo = max(flo, min(a, b));
        fhi = min(fhi, max(a, b));
    }
    else if (A.x < 0.0 || A.x >= lim.x) discard;
    if (abs(D.y) > 1e-7)
    {
        float a = -A.y / D.y, b = (lim.y - A.y) / D.y;
        flo = max(flo, min(a, b));
        fhi = min(fhi, max(a, b));
    }
    else if (A.y < 0.0 || A.y >= lim.y) discard;
    if (fhi <= flo) discard;

    float n = i.ii.w;
    if (n > 0.5)
    {
        // the splats of the reference, one by one: number k sits at t = (k + 0.5) / n
        float L2 = max(dot(D, D), 1e-12);
        float tc = dot(u0, D) / L2;                // the point of the line nearest to this pixel
        float r = 1.5 / sqrt(L2);                  // a splat further than that along the line cannot reach it
        float k0 = max(ceil((tc - r) * n - 0.5), 0.0);
        float k1 = min(floor((tc + r) * n - 0.5), n - 1.0);
        float sum = 0.0;
        for (int j = 0; j < 12; j++)
        {
            float k = k0 + (float)j;
            if (k > k1) break;
            float t = (k + 0.5) / n;
            if (t >= flo && t < fhi) sum += kern(t, u0, D, ia, ib);
        }
        if (sum == 0.0) discard;
        return addLight(sum * (sqrt(L2) / n), i.ii.z);
    }

    float ta = flo, tb = fhi;
    float t1 = 0.0, t2 = 0.0;
    if (abs(D.x) > 1e-7)
    {
        float a = (u0.x - 1.0) / D.x, b = (u0.x + 1.0) / D.x;
        ta = max(ta, min(a, b));
        tb = min(tb, max(a, b));
        t1 = u0.x / D.x;
    }
    else if (abs(u0.x) >= 1.0) discard;
    if (abs(D.y) > 1e-7)
    {
        float a = (u0.y - 1.0) / D.y, b = (u0.y + 1.0) / D.y;
        ta = max(ta, min(a, b));
        tb = min(tb, max(a, b));
        t2 = u0.y / D.y;
    }
    else if (abs(u0.y) >= 1.0) discard;
    if (tb <= ta) discard;
    t1 = clamp(t1, ta, tb);
    t2 = clamp(t2, ta, tb);
    float lo = min(t1, t2), hi = max(t1, t2);
    float f0 = kern(ta, u0, D, ia, ib);
    float f1 = kern(0.5 * (ta + lo), u0, D, ia, ib);
    float f2 = kern(lo, u0, D, ia, ib);
    float f3 = kern(0.5 * (lo + hi), u0, D, ia, ib);
    float f4 = kern(hi, u0, D, ia, ib);
    float f5 = kern(0.5 * (hi + tb), u0, D, ia, ib);
    float f6 = kern(tb, u0, D, ia, ib);
    float sum = (lo - ta) * (f0 + 4.0 * f1 + f2) + (hi - lo) * (f2 + 4.0 * f3 + f4) + (tb - hi) * (f4 + 4.0 * f5 + f6);
    return addLight(sum * (length(D) / 6.0), i.ii.z);
}
