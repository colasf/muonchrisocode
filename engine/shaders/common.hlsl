// Shared by every shader of the Muon Bloom renderer.
// Pixel coordinates are the ones of the Python reference: pixel i is centred on the integer i (a splat at
// x = 3.25 gives 0.75 to pixel 3 and 0.25 to pixel 4). On the GPU the centre of pixel i is at i + 0.5.

cbuffer FrameCB : register(b0)
{
    float2 gSize;        // W, H in pixels
    float gS;            // W / 2978
    float gS035;         // s ** 0.35: the gain of a line with the resolution
    float gExposure;
    float gTextGain;
    float gBloomGain;
    float gDither;       // 1: dither and round to 8 bits, 0: leave the picture in floats (for comparisons)
    uint gDitherSeed;
    float gLift;         // lift of the output (0: none): the mid levels raised, black and white kept
    uint2 gPad;
};

cbuffer DrawCB : register(b1)
{
    uint dBase;          // first instance of the draw (SV_InstanceID starts at 0)
    uint dA;
    uint dB;
    uint dC;
    float4 dP0;
    float4 dP1;
};

struct State             // view and clip a primitive was drawn under
{
    float vz, vcx, vcy, vsx, vsy, pad0;
    float cx0, cy0, cx1, cy1, pad1, pad2;
};

float2 viewPoint(State st, float2 p)      // design space -> pixel (Frame.tx / Frame.ty)
{
    return ((p - float2(st.vcx, st.vcy)) * st.vz + float2(st.vsx, st.vsy)) * gS;
}

float4 clipFromIndex(float2 p)            // pixel-index coordinates -> clip space
{
    float2 w = p + 0.5;
    return float4(w.x / gSize.x * 2.0 - 1.0, 1.0 - w.y / gSize.y * 2.0, 0.0, 1.0);
}

float4 clipFromWindow(float2 w)           // window coordinates (pixel i covers [i, i + 1)) -> clip space
{
    return float4(w.x / gSize.x * 2.0 - 1.0, 1.0 - w.y / gSize.y * 2.0, 0.0, 1.0);
}

static const float4 NOWHERE = float4(-4.0, -4.0, 0.0, 1.0);

// The light target takes  out = c + dst * f : additive geometry gives f = 1, a rect that multiplies gives c = 0.
struct LightOut
{
    float4 c : SV_Target0;
    float4 f : SV_Target1;
};

LightOut addLight(float v, float layer)
{
    LightOut o;
    o.c = layer > 0.5 ? float4(0.0, v, 0.0, 0.0) : float4(v, 0.0, 0.0, 0.0);
    o.f = float4(1.0, 1.0, 1.0, 1.0);
    return o;
}
