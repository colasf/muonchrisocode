#include "renderer.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace {

const int ATLAS = 4096;                   // glyph atlas, pixels
const int DISC_T = 128;                   // texels across a slice of the disc table (dots.hlsl: LUT_T)
const float DISC_MAX = 30.0f;             // largest tabulated radius (dots.hlsl: LUT_MAX)
const int DISC_N = 119;                   // radii 0.5, 0.75 ... 30

std::string hrText(const char* what, HRESULT hr)
{
    char b[160];
    snprintf(b, sizeof b, "%s failed (0x%08lX)", what, (unsigned long)hr);
    return b;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// set-up
// ------------------------------------------------------------------------------------------------

bool Renderer::compile(const wchar_t* file, const char* vsEntry, const char* psEntry, Shader& sh, std::string& err)
{
    std::wstring path = mShaderDir + L"/" + file;
    const char* entry[2] = { vsEntry, psEntry };
    const char* prof[2] = { "vs_5_0", "ps_5_0" };
    for (int k = 0; k < 2; k++) {
        ComPtr<ID3DBlob> code, msg;
        HRESULT hr = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry[k], prof[k],
                                        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &code, &msg);
        if (FAILED(hr)) {
            std::string f;
            for (wchar_t ch : path) f += (char)(ch < 128 ? ch : '?');
            err = "shader " + f + " " + entry[k] + ": " + (msg ? std::string((const char*)msg->GetBufferPointer(), msg->GetBufferSize()) : hrText("compile", hr));
            return false;
        }
        if (k == 0) hr = mDev->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &sh.vs);
        else hr = mDev->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &sh.ps);
        if (FAILED(hr)) { err = hrText("CreateShader", hr); return false; }
    }
    return true;
}

bool Renderer::init(ID3D11Device* dev, ID3D11DeviceContext* ctx, const std::wstring& shaderDir, std::string& err)
{
    mDev = dev;
    mCtx = ctx;
    mShaderDir = shaderDir;
    if (!compile(L"segments.hlsl", "VS", "PS", mSeg, err)) return false;
    if (!compile(L"dots.hlsl", "VS", "PS", mDot, err)) return false;
    if (!compile(L"splats.hlsl", "VS", "PS", mSplat, err)) return false;
    if (!compile(L"rects.hlsl", "VS", "PS", mRect, err)) return false;
    if (!compile(L"rects.hlsl", "VSOp", "PSOp", mOp, err)) return false;
    if (!compile(L"text.hlsl", "VS", "PS", mText, err)) return false;
    if (!compile(L"finish.hlsl", "VSFull", "PSBase", mBase, err)) return false;
    if (!compile(L"finish.hlsl", "VSFull", "PSDown", mDown, err)) return false;
    if (!compile(L"finish.hlsl", "VSFull", "PSBlurUp", mBlurUp, err)) return false;
    if (!compile(L"finish.hlsl", "VSFull", "PSFinal", mFinal, err)) return false;
    if (!compile(L"finish.hlsl", "VSBox", "PSFinal", mFinalBox, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSRoll", mRoll, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSStamp", mStamp, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSPour", mPour, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSBright", mBright, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSDrag", mDrag, err)) return false;
    if (!compile(L"post.hlsl", "VSRect", "PSSmear", mSmear, err)) return false;
    if (!compile(L"blit.hlsl", "VSFull", "PSBlit", mBlit, err)) return false;
    if (!compile(L"blit.hlsl", "VSOver", "PSOver", mOver, err)) return false;

    // light: out = src0 + dst * src1.   text: out = src0 * src1 + dst * (1 - src1).
    D3D11_BLEND_DESC bd = {};
    auto& rt = bd.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    rt.BlendOp = rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    rt.SrcBlend = rt.SrcBlendAlpha = D3D11_BLEND_ONE;
    rt.DestBlend = D3D11_BLEND_SRC1_COLOR;
    rt.DestBlendAlpha = D3D11_BLEND_SRC1_ALPHA;
    HRESULT hr = mDev->CreateBlendState(&bd, &mBlendLight);
    if (FAILED(hr)) { err = hrText("CreateBlendState", hr); return false; }
    rt.SrcBlend = D3D11_BLEND_SRC1_COLOR;
    rt.SrcBlendAlpha = D3D11_BLEND_SRC1_ALPHA;
    rt.DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    rt.DestBlendAlpha = D3D11_BLEND_INV_SRC1_ALPHA;
    mDev->CreateBlendState(&bd, &mBlendText);
    rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
    rt.DestBlendAlpha = D3D11_BLEND_ONE;
    mDev->CreateBlendState(&bd, &mBlendAlpha);
    rt.BlendEnable = FALSE;
    mDev->CreateBlendState(&bd, &mBlendNone);

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    mDev->CreateRasterizerState(&rd, &mRaster);

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    mDev->CreateSamplerState(&sd, &mLinear);

    D3D11_BUFFER_DESC cb = {};
    cb.Usage = D3D11_USAGE_DEFAULT;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb.ByteWidth = sizeof(FrameCB);
    mDev->CreateBuffer(&cb, nullptr, &mFrameCB);
    cb.ByteWidth = sizeof(DrawCB);
    mDev->CreateBuffer(&cb, nullptr, &mDrawCB);

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = ATLAS;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    std::vector<uint8_t> zero((size_t)ATLAS * ATLAS, 0);
    D3D11_SUBRESOURCE_DATA init = { zero.data(), (UINT)ATLAS, 0 };
    hr = mDev->CreateTexture2D(&td, &init, &mAtlas);
    if (FAILED(hr)) { err = hrText("glyph atlas", hr); return false; }
    mDev->CreateShaderResourceView(mAtlas.Get(), nullptr, &mAtlasSrv);

    buildDisc();
    return mDisc != nullptr;
}

// The picture of one disc of the reference (engine._disc: sub-pixel samples, each one a bilinear splat),
// tabulated around its centre for every radius the reference can ask for (quarters of a pixel).
// A slice covers [-E, E] with E = radius + 1.75; its texels sit on a lattice that divides the half pixel,
// so that bilinear filtering gives back the splats exactly (radius > 1.2: samples every half pixel).
void Renderer::buildDisc()
{
    std::vector<uint16_t> data((size_t)DISC_N * DISC_T * DISC_T, 0);
    std::vector<double> img((size_t)DISC_T * DISC_T);
    for (int s = 0; s < DISC_N; s++) {
        float rq = 0.5f + 0.25f * s;
        float R = rq + 0.75f;
        float step = rq > 1.2f ? 0.5f : 0.34f;
        int n = (int)std::ceil((2.0 * R + 1e-6) / step);
        double E = rq + 1.75;
        int n4 = (int)std::lround(4.0 * rq + 7.0);
        int tm1 = n4 * ((DISC_T - 1) / n4);
        double h = 2.0 * E / tm1;
        std::fill(img.begin(), img.end(), 0.0);
        for (int gy = 0; gy < n; gy++) {
            float oy = -R + gy * step;
            for (int gx = 0; gx < n; gx++) {
                float ox = -R + gx * step;
                float cov = std::clamp(rq + 0.5f - std::hypot(ox, oy), 0.0f, 1.0f);
                if (cov <= 0.0f) continue;
                double w = (double)cov * step * step;
                int jx0 = std::max(0, (int)std::ceil((ox - 1.0 + E) / h)), jx1 = std::min(tm1, (int)std::floor((ox + 1.0 + E) / h));
                int jy0 = std::max(0, (int)std::ceil((oy - 1.0 + E) / h)), jy1 = std::min(tm1, (int)std::floor((oy + 1.0 + E) / h));
                for (int jy = jy0; jy <= jy1; jy++) {
                    double wy = w * std::max(0.0, 1.0 - std::abs(-E + jy * h - oy));
                    for (int jx = jx0; jx <= jx1; jx++)
                        img[(size_t)jy * DISC_T + jx] += wy * std::max(0.0, 1.0 - std::abs(-E + jx * h - ox));
                }
            }
        }
        uint16_t* out = &data[(size_t)s * DISC_T * DISC_T];
        for (size_t k = 0; k < img.size(); k++)
            out[k] = (uint16_t)std::lround(std::clamp(img[k] * 0.5, 0.0, 1.0) * 65535.0);
    }
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = DISC_T;
    td.MipLevels = 1;
    td.ArraySize = DISC_N;
    td.Format = DXGI_FORMAT_R16_UNORM;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    std::vector<D3D11_SUBRESOURCE_DATA> init(DISC_N);
    for (int s = 0; s < DISC_N; s++)
        init[s] = { &data[(size_t)s * DISC_T * DISC_T], (UINT)(DISC_T * sizeof(uint16_t)), 0 };
    if (SUCCEEDED(mDev->CreateTexture2D(&td, init.data(), &mDisc)))
        mDev->CreateShaderResourceView(mDisc.Get(), nullptr, &mDiscSrv);
}

bool Renderer::target(Target& t, int w, int h, DXGI_FORMAT fmt)
{
    t = Target();
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = w;
    td.Height = h;
    td.MipLevels = td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(mDev->CreateTexture2D(&td, nullptr, &t.tex))) return false;
    if (FAILED(mDev->CreateRenderTargetView(t.tex.Get(), nullptr, &t.rtv))) return false;
    if (FAILED(mDev->CreateShaderResourceView(t.tex.Get(), nullptr, &t.srv))) return false;
    t.w = w;
    t.h = h;
    return true;
}

bool Renderer::resize(int w, int h)
{
    if (w == mW && h == mH) return true;
    bool ok = target(mLight, w, h, DXGI_FORMAT_R32G32_FLOAT) && target(mTextT, w, h, DXGI_FORMAT_R8G8_UNORM)
           && target(mBaseT, w, h, DXGI_FORMAT_R32G32_FLOAT) && target(mOut, w, h, DXGI_FORMAT_B8G8R8A8_UNORM) && target(mShow, w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
           && target(mOutF, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT) && target(mScratch[0], w, h, DXGI_FORMAT_R32G32_FLOAT)
           && target(mScratch[1], w, h, DXGI_FORMAT_R32G32_FLOAT) && target(mScratch[2], w, h, DXGI_FORMAT_R32G32_FLOAT);
    int lw = w, lh = h;
    for (int k = 1; k <= 8 && ok; k++) {
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
        ok = target(mDownT[k], lw, lh, DXGI_FORMAT_R32G32_FLOAT) && target(mAcc[k], lw, lh, DXGI_FORMAT_R32G32_FLOAT);
    }
    mW = ok ? w : 0;
    mH = ok ? h : 0;
    return ok;
}

bool Renderer::upload(GpuBuffer& b, const void* data, size_t count, UINT stride, bool structured)
{
    if (count > b.cap || !b.buf) {
        size_t cap = std::max<size_t>(count + count / 2, 1024);
        D3D11_BUFFER_DESC d = {};
        d.ByteWidth = (UINT)(cap * stride);
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        d.MiscFlags = structured ? D3D11_RESOURCE_MISC_BUFFER_STRUCTURED : 0;
        d.StructureByteStride = structured ? stride : 0;
        b = GpuBuffer();
        if (FAILED(mDev->CreateBuffer(&d, nullptr, &b.buf))) return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = structured ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_UINT;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)cap;
        if (FAILED(mDev->CreateShaderResourceView(b.buf.Get(), &sv, &b.srv))) return false;
        b.cap = cap;
    }
    if (count) {
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(mCtx->Map(b.buf.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
        memcpy(m.pData, data, count * stride);
        mCtx->Unmap(b.buf.Get(), 0);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// glyphs
// ------------------------------------------------------------------------------------------------

void Renderer::addGlyphs(const dl::View& v)
{
    const uint8_t* p = v.p + v.sec[dl::GLYPHS].offset;
    const uint8_t* end = p + v.sec[dl::GLYPHS].bytes;
    for (uint32_t k = 0; k < v.count(dl::GLYPHS) && p + sizeof(dl::GlyphDef) <= end; k++) {
        dl::GlyphDef g;
        memcpy(&g, p, sizeof g);
        p += sizeof g;
        if (g.w > (uint32_t)ATLAS || g.h > (uint32_t)ATLAS) break;         // not a glyph: a damaged blob
        size_t n = (size_t)g.w * g.h;
        const uint8_t* bits = p;
        if (((n + 3) & ~(size_t)3) > (size_t)(end - p)) break;
        p += (n + 3) & ~(size_t)3;
        uint64_t key = ((uint64_t)g.font << 32) | g.code;
        if (mGlyphs.count(key)) continue;
        Glyph e = { 0, 0, (int)g.w, (int)g.h, g.ox, g.oy, g.adv };
        if (n) {
            if (mAtlasX + (int)g.w > ATLAS) {
                mAtlasX = 0;
                mAtlasY += mAtlasRow;
                mAtlasRow = 0;
            }
            if (mAtlasY + (int)g.h > ATLAS || (int)g.w > ATLAS) continue;      // atlas full: the glyph is not drawn
            e.ax = mAtlasX;
            e.ay = mAtlasY;
            D3D11_BOX box = { (UINT)e.ax, (UINT)e.ay, 0, (UINT)(e.ax + e.w), (UINT)(e.ay + e.h), 1 };
            mCtx->UpdateSubresource(mAtlas.Get(), 0, &box, bits, g.w, 0);
            mAtlasX += e.w;
            mAtlasRow = std::max(mAtlasRow, e.h);
        }
        mGlyphs.emplace(key, e);
    }
}

bool Renderer::ingest(const uint8_t* blob, size_t size)
{
    dl::View v;
    if (!v.open(blob, size)) return false;
    addGlyphs(v);
    return true;
}

// ------------------------------------------------------------------------------------------------
// drawing
// ------------------------------------------------------------------------------------------------

void Renderer::bind(Target& t)
{
    ID3D11RenderTargetView* r = t.rtv.Get();
    mCtx->OMSetRenderTargets(1, &r, nullptr);
    D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)t.w, (float)t.h, 0.0f, 1.0f };
    mCtx->RSSetViewports(1, &vp);
}

void Renderer::srv(UINT slot, ID3D11ShaderResourceView* v)
{
    mCtx->VSSetShaderResources(slot, 1, &v);
    mCtx->PSSetShaderResources(slot, 1, &v);
}

void Renderer::unbind()
{
    ID3D11ShaderResourceView* none[5] = {};
    mCtx->VSSetShaderResources(0, 5, none);
    mCtx->PSSetShaderResources(0, 5, none);
}

void Renderer::draw(const Shader& sh, UINT instances, UINT base, const DrawCB* cb)
{
    if (!instances) return;
    DrawCB c = cb ? *cb : DrawCB{};
    c.base = base;
    mCtx->UpdateSubresource(mDrawCB.Get(), 0, nullptr, &c, 0, 0);
    mCtx->VSSetShader(sh.vs.Get(), nullptr, 0);
    mCtx->PSSetShader(sh.ps.Get(), nullptr, 0);
    mCtx->DrawInstanced(4, instances, 0, 0);
}

void Renderer::full(const Shader& sh, const DrawCB& cb)
{
    mCtx->UpdateSubresource(mDrawCB.Get(), 0, nullptr, &cb, 0, 0);
    mCtx->VSSetShader(sh.vs.Get(), nullptr, 0);
    mCtx->PSSetShader(sh.ps.Get(), nullptr, 0);
    mCtx->Draw(3, 0);
}

// One pass of the post-process: `sh` over a rect of `dst` (the sources are already bound).
void Renderer::pass(const Shader& sh, Target& dst, int x0, int y0, int x1, int y1, float a, float b, float c)
{
    bind(dst);
    DrawCB cb = {};
    cb.p0[0] = (float)x0;
    cb.p0[1] = (float)y0;
    cb.p0[2] = (float)x1;
    cb.p0[3] = (float)y1;
    cb.p1[0] = a;
    cb.p1[1] = b;
    cb.p1[2] = c;
    mCtx->UpdateSubresource(mDrawCB.Get(), 0, nullptr, &cb, 0, 0);
    mCtx->VSSetShader(sh.vs.Get(), nullptr, 0);
    mCtx->PSSetShader(sh.ps.Get(), nullptr, 0);
    mCtx->Draw(4, 0);
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
}

void Renderer::copy(Target& dst, Target& src, int x0, int y0, int x1, int y1)
{
    D3D11_BOX box = { (UINT)x0, (UINT)y0, 0, (UINT)x1, (UINT)y1, 1 };
    mCtx->CopySubresourceRegion(dst.tex.Get(), 0, x0, y0, 0, src.tex.Get(), 0, &box);
}

// The corruption of the glitch scene on the light stack (mBaseT), one operation after the other
// (drawlist.py: apply_post is the same thing in numpy).
void Renderer::post(const dl::View& v)
{
    const dl::PostOp* ops = v.at<dl::PostOp>(dl::POSTOPS);
    const int W = mW, H = mH;
    Target& A = mBaseT;
    Target& C = mScratch[0];
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
    for (uint32_t k = 0; k < v.count(dl::POSTOPS); k++) {
        const dl::PostOp& o = ops[k];
        int x0 = std::clamp(o.x0, 0, W), x1 = std::clamp(o.x1, 0, W), y0 = std::clamp(o.y0, 0, H), y1 = std::clamp(o.y1, 0, H);
        if (x1 <= x0 || y1 <= y0) continue;
        copy(C, A, x0, y0, x1, y1);                 // what the operation reads
        switch (o.kind) {
        case dl::P_ROLL_X:
        case dl::P_ROLL_Y:
            srv(0, C.srv.Get());
            pass(mRoll, A, x0, y0, x1, y1, o.kind == dl::P_ROLL_X ? o.p0 : 0.0f, o.kind == dl::P_ROLL_Y ? o.p0 : 0.0f);
            break;
        case dl::P_REPEAT: {
            int w = x1 - x0, sgn = o.p1 < 0 ? -1 : 1;
            float gain = 1.0f;
            for (int r = 1; r <= (int)o.p0; r++) {
                int x = x0 + sgn * r * w;
                if (x < 0 || x + w > W) break;
                gain *= o.p2;
                srv(0, C.srv.Get());
                pass(mStamp, A, x, y0, x + w, y1, (float)(x0 - x), 0.0f, gain);
            }
            break;
        }
        case dl::P_TO_RED:
        case dl::P_TO_WHITE:
            srv(0, C.srv.Get());
            pass(mPour, A, x0, y0, x1, y1, o.kind == dl::P_TO_RED ? 0.0f : 1.0f);
            break;
        case dl::P_SMEAR_X:
        case dl::P_SMEAR_Y: {
            bool alongX = o.kind == dl::P_SMEAR_X;
            int n = alongX ? x1 - x0 : y1 - y0;
            int back = (alongX && o.p1 < 0) ? 1 : -1;          // where the light comes from
            Target* m = &mScratch[1];
            Target* m2 = &mScratch[2];
            srv(0, C.srv.Get());
            pass(mBright, *m, x0, y0, x1, y1, o.p0);
            for (int d = 1; d < n; d *= 2) {
                srv(0, m->srv.Get());
                pass(mDrag, *m2, x0, y0, x1, y1, alongX ? (float)(back * d) : 0.0f, alongX ? 0.0f : (float)(back * d), std::pow(o.p2, (float)d));
                std::swap(m, m2);
            }
            srv(0, C.srv.Get());
            srv(1, m->srv.Get());
            pass(mSmear, A, x0, y0, x1, y1, std::abs(o.p1));
            break;
        }
        default:
            break;
        }
    }
}

void Renderer::overlay(const std::vector<Over>& rects)
{
    mHasOver = mW && (mCardOn || !rects.empty());
    if (!mHasOver) return;
    unbind();
    if (mCardOn) {                                  // the part of the card where the picture of the show sits
        D3D11_BOX box = { (UINT)mCardX, (UINT)mCardY, 0, (UINT)(mCardX + mW), (UINT)(mCardY + mH), 1 };
        mCtx->CopySubresourceRegion(mShow.tex.Get(), 0, 0, 0, 0, mCard.tex.Get(), 0, &box);
    } else {
        mCtx->CopyResource(mShow.tex.Get(), mOut.tex.Get());
    }
    if (rects.empty() || !upload(mOverBuf, rects.data(), rects.size(), sizeof(Over), true)) return;
    bind(mShow);
    mCtx->OMSetBlendState(mBlendAlpha.Get(), nullptr, 0xFFFFFFFF);
    mCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    mCtx->RSSetState(mRaster.Get());
    ID3D11Buffer* cbs[2] = { mFrameCB.Get(), mDrawCB.Get() };
    mCtx->VSSetConstantBuffers(0, 2, cbs);
    mCtx->PSSetConstantBuffers(0, 2, cbs);
    srv(4, mOverBuf.srv.Get());
    draw(mOver, (UINT)rects.size(), 0);
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
}

void Renderer::windowRects(ID3D11RenderTargetView* rtv, int w, int h, const std::vector<Over>& rects)
{
    if (rects.empty() || w <= 0 || h <= 0 || !upload(mWinBuf, rects.data(), rects.size(), sizeof(Over), true)) return;
    unbind();
    mCtx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    mCtx->RSSetViewports(1, &vp);
    mCtx->OMSetBlendState(mBlendAlpha.Get(), nullptr, 0xFFFFFFFF);
    mCtx->IASetInputLayout(nullptr);
    mCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    mCtx->RSSetState(mRaster.Get());
    FrameCB fc = mFrame;                            // the shader places the rects with the size of its target
    fc.size[0] = (float)w;
    fc.size[1] = (float)h;
    mCtx->UpdateSubresource(mFrameCB.Get(), 0, nullptr, &fc, 0, 0);
    ID3D11Buffer* cbs[2] = { mFrameCB.Get(), mDrawCB.Get() };
    mCtx->VSSetConstantBuffers(0, 2, cbs);
    mCtx->PSSetConstantBuffers(0, 2, cbs);
    srv(4, mWinBuf.srv.Get());
    draw(mOver, (UINT)rects.size(), 0);
    mCtx->UpdateSubresource(mFrameCB.Get(), 0, nullptr, &mFrame, 0, 0);    // back to the picture's (overlay() uses them)
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
}

// The picture `src` scaled into a rect of a target (filtered). The rest of the target takes the colour
// `around`, or is left as it is (nullptr).
void Renderer::place(ID3D11RenderTargetView* rtv, int w, int h, Target& src, float x0, float y0, float x1, float y1, const float* around)
{
    unbind();
    mCtx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    mCtx->RSSetViewports(1, &vp);
    mCtx->OMSetBlendState(mBlendNone.Get(), nullptr, 0xFFFFFFFF);
    mCtx->IASetInputLayout(nullptr);
    mCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    mCtx->RSSetState(mRaster.Get());
    ID3D11Buffer* cbs[2] = { mFrameCB.Get(), mDrawCB.Get() };
    mCtx->VSSetConstantBuffers(0, 2, cbs);
    mCtx->PSSetConstantBuffers(0, 2, cbs);
    ID3D11SamplerState* smp = mLinear.Get();
    mCtx->PSSetSamplers(0, 1, &smp);
    srv(0, src.srv.Get());
    DrawCB c = {};
    c.p0[0] = x0;
    c.p0[1] = y0;
    c.p0[2] = x1;
    c.p0[3] = y1;
    if (around) for (int k = 0; k < 3; k++) c.p1[k] = around[k];
    else c.a = 1;
    full(mBlit, c);
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
}

void Renderer::blit(ID3D11RenderTargetView* rtv, int w, int h, int below)
{
    if (w <= 0 || h <= 0) return;
    const float dark[4] = { 0.02f, 0.02f, 0.02f, 1.0f };
    if (!mW) {                                      // no picture yet: an empty window, not the garbage of a new buffer
        mCtx->ClearRenderTargetView(rtv, dark);
        return;
    }
    const int hp = std::max(1, h - below);                      // the part of the window the picture may use
    float k = std::min((float)w / mW, (float)hp / mH);          // fit the picture, keep its shape
    float pw = mW * k, ph = mH * k;
    place(rtv, w, h, shown(), 0.5f * (w - pw), 0.5f * (hp - ph), 0.5f * (w + pw), 0.5f * (hp + ph), dark);
}

bool Renderer::output(ID3D11Texture2D* back, ID3D11RenderTargetView* rtv, int w, int h, int rw, int rh, int ox, int oy, int px, int py)
{
    const float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const bool exact = ox >= 0 && oy >= 0 && ox + rw <= w && oy + rh <= h;
    unbind();
    mCtx->ClearRenderTargetView(rtv, black);
    if (!mW) return exact;
    if (exact) {                                    // copies: not one pixel is filtered on the way
        if (mCardOn) mCtx->CopySubresourceRegion(back, 0, (UINT)ox, (UINT)oy, 0, mCard.tex.Get(), 0, nullptr);
        mCtx->CopySubresourceRegion(back, 0, (UINT)(ox + px), (UINT)(oy + py), 0, shown().tex.Get(), 0, nullptr);
        return true;
    }
    const float k = std::min((float)w / rw, (float)h / rh);     // a display smaller than the raster: all of it, as large as it fits
    const float x0 = 0.5f * (w - rw * k), y0 = 0.5f * (h - rh * k);
    if (mCardOn) place(rtv, w, h, mCard, x0, y0, x0 + rw * k, y0 + rh * k, nullptr);
    place(rtv, w, h, shown(), x0 + px * k, y0 + py * k, x0 + (px + mW) * k, y0 + (py + mH) * k, nullptr);
    return false;
}

void Renderer::lift(float v)
{
    mLift = std::max(0.0f, v);
    if (mW) finish(false);
}

bool Renderer::loadCard(const std::wstring& file, int w, int h)
{
    std::vector<uint8_t> px((size_t)w * h * 4);
    std::ifstream f(file, std::ios::binary);
    if (!f.read((char*)px.data(), (std::streamsize)px.size()) || f.peek() != EOF) return false;       // not a card of this raster
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = w;
    td.Height = h;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA init = { px.data(), (UINT)w * 4, 0 };
    Target t;
    if (FAILED(mDev->CreateTexture2D(&td, &init, &t.tex)) || FAILED(mDev->CreateShaderResourceView(t.tex.Get(), nullptr, &t.srv))) return false;
    t.w = w;
    t.h = h;
    mCard = t;
    return true;
}

bool Renderer::card(bool on, int x, int y)
{
    mCardOn = on && mCard.tex && mW && x >= 0 && y >= 0 && x + mW <= mCard.w && y + mH <= mCard.h;
    mCardX = x;
    mCardY = y;
    return mCardOn == on;
}

// From the light of the last frame drawn (mBaseT, its bloom in mAcc[1], its inverted rects in mBoxes) to
// the picture: tonemap, colour, the lift of the output, inverted rects, dither.
void Renderer::finish(bool floatOut)
{
    mFrame.lift = mLift;
    mCtx->UpdateSubresource(mFrameCB.Get(), 0, nullptr, &mFrame, 0, 0);
    ID3D11Buffer* cbs[2] = { mFrameCB.Get(), mDrawCB.Get() };
    mCtx->VSSetConstantBuffers(0, 2, cbs);
    mCtx->PSSetConstantBuffers(0, 2, cbs);
    mCtx->IASetInputLayout(nullptr);
    mCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    mCtx->RSSetState(mRaster.Get());
    mCtx->OMSetBlendState(mBlendNone.Get(), nullptr, 0xFFFFFFFF);
    Target& o = floatOut ? mOutF : mOut;
    unbind();
    bind(o);
    srv(0, mBaseT.srv.Get());
    srv(1, mAcc[1].srv.Get());
    srv(4, mBoxes.srv.Get());
    DrawCB c = {};
    c.p0[2] = (float)(mAcc[1].w - 1);
    c.p0[3] = (float)(mAcc[1].h - 1);
    full(mFinal, c);
    c.a = 1;
    draw(mFinalBox, (UINT)mBoxesCpu.size(), 0, &c);
    unbind();
    ID3D11RenderTargetView* none = nullptr;
    mCtx->OMSetRenderTargets(1, &none, nullptr);
}

bool Renderer::render(const uint8_t* blob, size_t size, bool floatOut, std::string& err)
{
    dl::View v;
    if (!v.open(blob, size)) { err = "not a draw list (or another version of the format)"; return false; }
    const dl::Header& h = *v.h;
    if (!resize((int)h.W, (int)h.H)) { err = "cannot create the render targets"; return false; }
    addGlyphs(v);
    mInfo = FrameInfo();
    mInfo.t = h.t;
    mInfo.flags = h.flags;

    const dl::Seg* segs = v.at<dl::Seg>(dl::SEGS);
    const dl::LightOp* ops = v.at<dl::LightOp>(dl::LIGHTOPS);
    const uint32_t nseg = v.count(dl::SEGS), ndot = v.count(dl::DOTS), nsplat = v.count(dl::SPLATS), nrect = v.count(dl::RECTS);
    const uint32_t nops = v.count(dl::LIGHTOPS), nocc = v.count(dl::OCCL);

    // segments -> passes (a line wider than 1.25 is several parallel passes); where each light op falls
    mExpandCpu.clear();
    std::vector<uint32_t> cut(nops + 1);
    {
        uint32_t i = 0;
        for (uint32_t e = 0; e <= nops; e++) {
            uint32_t upto = e < nops ? std::min(ops[e].nseg, nseg) : nseg;
            for (; i < upto; i++) {
                const dl::Seg& g = segs[i];
                if (!(g.i0 > 1e-4f || g.i1 > 1e-4f)) continue;
                uint32_t reps = 1;
                if (g.width > 1.25f) reps = (uint32_t)std::clamp(std::ceil(g.width * h.s / 0.6f), 2.0f, 255.0f);
                for (uint32_t k = 0; k < reps; k++) mExpandCpu.push_back(i | (k << 24));
            }
            cut[e] = (uint32_t)mExpandCpu.size();
        }
    }

    // light ops, then the boxes of the tags (they win over all the geometry: applied last)
    mOpsCpu.assign(ops, ops + nops);
    const dl::Box* occ = v.at<dl::Box>(dl::OCCL);
    for (uint32_t k = 0; k < nocc; k++)
        mOpsCpu.push_back({ std::max(occ[k].x0, 0), std::max(occ[k].y0, 0), std::max(occ[k].x1, 0), std::max(occ[k].y1, 0), 0.0f, 0.0f, 0, 0, 0, 0 });
    // A box given backwards (x1 < x0) is an empty slice for the reference: it must not be drawn at all here
    // either (the quads are not culled, so it would come out as a full rect).
    for (auto& o : mOpsCpu)
        if (o.x1 <= o.x0 || o.y1 <= o.y0) o.x1 = o.x0, o.y1 = o.y0;
    const uint32_t ninv = v.count(dl::INVERT);
    mBoxesCpu.assign(v.at<dl::Box>(dl::INVERT), v.at<dl::Box>(dl::INVERT) + ninv);
    for (auto& o : mBoxesCpu)
        if (o.x1 <= o.x0 || o.y1 <= o.y0) o.x1 = o.x0, o.y1 = o.y0;

    // text ops -> one instance per rect and per glyph, in order
    mInstCpu.clear();
    {
        const dl::TextOp* t = v.at<dl::TextOp>(dl::TEXTOPS);
        const uint32_t* chars = v.at<uint32_t>(dl::CHARS);
        const uint32_t nchar = v.count(dl::CHARS);
        const int W = (int)h.W, H = (int)h.H;
        for (uint32_t k = 0; k < v.count(dl::TEXTOPS); k++) {
            const dl::TextOp& o = t[k];
            if (o.kind == dl::T_RECT) {
                TextInst r = { std::max(o.a, 0), std::max(o.b, 0), std::min(o.c, W), std::min(o.d, H), 0, 0, 0, 0, o.cw, o.cr, o.fw, o.fr };
                if (r.x1 > r.x0 && r.y1 > r.y0) mInstCpu.push_back(r);
                continue;
            }
            if ((uint64_t)o.off + o.cnt > nchar) continue;
            for (uint32_t n = 0; n < o.cnt; n++) {
                auto it = mGlyphs.find(((uint64_t)(uint32_t)o.d << 32) | chars[o.off + n]);
                if (it == mGlyphs.end()) { mInfo.missing_glyphs++; continue; }
                const Glyph& g = it->second;
                if (!g.w || !g.h) continue;
                TextInst r;
                if (o.kind == dl::T_RUN) {
                    int x = o.a + (int)n * o.c + g.ox, y = o.b + g.oy;
                    r = { x, y, x + g.w, y + g.h, g.ax, g.ay, 1, 0, o.cw, o.cr, o.fw, o.fr };
                } else {
                    int x = o.a + g.oy, y = o.b - (int)n * o.c - g.ox - g.w;
                    r = { x, y, x + g.h, y + g.w, g.ax, g.ay, 2, 0, o.cw, o.cr, o.fw, o.fr };
                }
                mInstCpu.push_back(r);
            }
        }
    }

    bool ok = upload(mStates, v.at<dl::State>(dl::STATES), v.count(dl::STATES), sizeof(dl::State), true)
           && upload(mSegs, segs, nseg, sizeof(dl::Seg), true)
           && upload(mExpand, mExpandCpu.data(), mExpandCpu.size(), 4, false)
           && upload(mDots, v.at<dl::Dot>(dl::DOTS), ndot, sizeof(dl::Dot), true)
           && upload(mSplats, v.at<dl::Splat>(dl::SPLATS), nsplat, sizeof(dl::Splat), true)
           && upload(mRects, v.at<dl::Rect>(dl::RECTS), nrect, sizeof(dl::Rect), true)
           && upload(mOps, mOpsCpu.data(), mOpsCpu.size(), sizeof(dl::LightOp), true)
           && upload(mInst, mInstCpu.data(), mInstCpu.size(), sizeof(TextInst), true)
           && upload(mBoxes, mBoxesCpu.data(), ninv, sizeof(dl::Box), true);
    if (!ok) { err = "cannot create the geometry buffers"; return false; }

    mInfo.segs = nseg;
    mInfo.passes = (uint32_t)mExpandCpu.size();
    mInfo.dots = ndot;
    mInfo.splats = nsplat;
    mInfo.rects = nrect;
    mInfo.lightops = nops;
    mInfo.textinst = (uint32_t)mInstCpu.size();
    mInfo.postops = v.count(dl::POSTOPS);

    FrameCB fc = { { (float)h.W, (float)h.H }, h.s, std::pow(h.s, 0.35f), h.exposure, h.text_gain, h.bloom_gain,
                   floatOut ? 0.0f : 1.0f, h.dither_seed, mLift, { 0, 0 } };
    mFrame = fc;
    mCtx->UpdateSubresource(mFrameCB.Get(), 0, nullptr, &fc, 0, 0);
    ID3D11Buffer* cbs[2] = { mFrameCB.Get(), mDrawCB.Get() };
    mCtx->VSSetConstantBuffers(0, 2, cbs);
    mCtx->PSSetConstantBuffers(0, 2, cbs);
    ID3D11SamplerState* smp = mLinear.Get();
    mCtx->PSSetSamplers(0, 1, &smp);
    mCtx->IASetInputLayout(nullptr);
    mCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    mCtx->RSSetState(mRaster.Get());
    const float zero[4] = { 0, 0, 0, 0 };

    // 1 - the two layers of light: additive geometry, cut by the rects that multiply what is under them
    unbind();
    bind(mLight);
    mCtx->ClearRenderTargetView(mLight.rtv.Get(), zero);
    mCtx->OMSetBlendState(mBlendLight.Get(), nullptr, 0xFFFFFFFF);
    srv(2, mStates.srv.Get());
    srv(3, mDiscSrv.Get());
    srv(4, mOps.srv.Get());
    uint32_t done[4] = { 0, 0, 0, 0 };
    for (uint32_t e = 0; e <= nops; e++) {
        uint32_t upto[4] = { cut[e], e < nops ? std::min(ops[e].ndot, ndot) : ndot, e < nops ? std::min(ops[e].nsplat, nsplat) : nsplat,
                             e < nops ? std::min(ops[e].nrect, nrect) : nrect };
        if (upto[0] > done[0]) {
            srv(0, mSegs.srv.Get());
            srv(1, mExpand.srv.Get());
            draw(mSeg, upto[0] - done[0], done[0]);
        }
        if (upto[1] > done[1]) {
            srv(0, mDots.srv.Get());
            draw(mDot, upto[1] - done[1], done[1]);
        }
        if (upto[2] > done[2]) {
            srv(0, mSplats.srv.Get());
            draw(mSplat, upto[2] - done[2], done[2]);
        }
        if (upto[3] > done[3]) {
            srv(0, mRects.srv.Get());
            draw(mRect, upto[3] - done[3], done[3]);
        }
        for (int k = 0; k < 4; k++) done[k] = std::max(done[k], upto[k]);
        if (e < nops) draw(mOp, 1, e);
    }
    draw(mOp, nocc, nops);

    // 2 - the two text layers
    unbind();
    bind(mTextT);
    mCtx->ClearRenderTargetView(mTextT.rtv.Get(), zero);
    mCtx->OMSetBlendState(mBlendText.Get(), nullptr, 0xFFFFFFFF);
    srv(0, mInst.srv.Get());
    srv(1, mAtlasSrv.Get());
    draw(mText, (UINT)mInstCpu.size(), 0);

    // 3 - base = light * exposure + text
    mCtx->OMSetBlendState(mBlendNone.Get(), nullptr, 0xFFFFFFFF);
    unbind();
    bind(mBaseT);
    srv(0, mLight.srv.Get());
    srv(1, mTextT.srv.Get());
    full(mBase, DrawCB{});

    if (v.count(dl::POSTOPS)) post(v);

    // 4 - bloom: 8 levels down, then blurred and summed on the way up
    if (h.bloom_gain != 0.0f) {
        for (int k = 1; k <= 8; k++) {
            Target& src = k == 1 ? mBaseT : mDownT[k - 1];
            unbind();
            bind(mDownT[k]);
            srv(0, src.srv.Get());
            DrawCB c = {};
            c.p0[0] = (float)(src.w - 1);
            c.p0[1] = (float)(src.h - 1);
            full(mDown, c);
        }
        for (int k = 8; k >= 1; k--) {
            unbind();
            bind(mAcc[k]);
            srv(0, mDownT[k].srv.Get());
            DrawCB c = {};
            c.p0[0] = (float)(mDownT[k].w - 1);
            c.p0[1] = (float)(mDownT[k].h - 1);
            c.p1[0] = h.bloom_w[k - 1];
            if (k < 8) {
                srv(1, mAcc[k + 1].srv.Get());
                c.p0[2] = (float)(mAcc[k + 1].w - 1);
                c.p0[3] = (float)(mAcc[k + 1].h - 1);
                c.p1[1] = 1.0f;
            }
            full(mBlurUp, c);
        }
    }

    // 5 - tonemap, colour, the lift of the output, inverted rects, dither
    finish(floatOut);
    return true;
}
