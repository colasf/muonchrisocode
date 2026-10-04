// GPU renderer for the draw lists of the Muon Bloom show (Direct3D 11).
// It does what muonbloom/engine.py Frame does on the CPU: two additive scalar layers of light (white, red),
// two 8-bit text layers, occlusion, bloom, tonemap. See engine/BRIEF.md, section 5.
#pragma once
#include <d3d11_1.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "drawlist.h"

using Microsoft::WRL::ComPtr;

struct Target {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11ShaderResourceView> srv;
    int w = 0, h = 0;
};

struct GpuBuffer {
    ComPtr<ID3D11Buffer> buf;
    ComPtr<ID3D11ShaderResourceView> srv;
    size_t cap = 0;
};

struct FrameInfo {
    double t = 0;
    uint32_t flags = 0;
    uint32_t segs = 0, passes = 0, dots = 0, splats = 0, rects = 0, lightops = 0, textinst = 0, missing_glyphs = 0;
    uint32_t postops = 0;
};

class Renderer {
public:
    // shaderDir: folder of the .hlsl files (compiled here, at start-up)
    bool init(ID3D11Device* dev, ID3D11DeviceContext* ctx, const std::wstring& shaderDir, std::string& err);

    // Take the glyphs a blob brings. render() does it too; call this for a frame that is skipped, because
    // every glyph travels only once.
    bool ingest(const uint8_t* blob, size_t size);

    // Draw a blob. floatOut: no dither, picture left in floats in outF() (to compare with the reference),
    // else 8 bits in out().
    bool render(const uint8_t* blob, size_t size, bool floatOut, std::string& err);

    // Rects laid over the finished picture (pixels of the picture, colour with alpha): the tower placement
    // tool. The picture with them is shown(); with no rect, shown() is out() - or, while the test card is
    // on, the part of the card where the picture of the show sits.
    struct Over { float x0, y0, x1, y1, r, g, b, a; };
    void overlay(const std::vector<Over>& rects);
    Target& shown() { return mHasOver ? mShow : mOut; }
    // Copy shown() into a window (letterboxed, filtered), leaving `below` pixels free under it.
    void blit(ID3D11RenderTargetView* rtv, int w, int h, int below = 0);

    // The lift of the output (0 = none): the mid levels of the picture are raised, black and white stay
    // where they are. Changing it draws the last picture again (colour, inverted rects, dither).
    void lift(float v);
    float lift() const { return mLift; }

    // The test card: a raw BGRA picture of the whole raster (w x h, rows from the top, as tools/test_card.py
    // writes it). card(true, x, y) shows it in place of the show, the picture of the show sitting at (x, y)
    // of the raster; false when there is no card, or no picture yet to take its size from. Call overlay()
    // after it, as after a new picture.
    bool loadCard(const std::wstring& file, int w, int h);
    bool card(bool on, int x, int y);
    bool cardOn() const { return mCardOn; }

    // The raster on an output (the back buffer of a window that fills a display): a rw x rh raster with its
    // top left corner at (ox, oy), black but for shown() at (px, py) of it - or the whole test card. Pixel
    // for pixel when the raster fits in the target (true); else scaled to fit, filtered (false).
    bool output(ID3D11Texture2D* back, ID3D11RenderTargetView* rtv, int w, int h, int rw, int rh, int ox, int oy, int px, int py);
    // Rects drawn straight into a window (pixels of the window): the time bar. They are not in the picture.
    void windowRects(ID3D11RenderTargetView* rtv, int w, int h, const std::vector<Over>& rects);

    Target& out() { return mOut; }
    Target& outF() { return mOutF; }
    const FrameInfo& info() const { return mInfo; }

private:
    struct Shader {
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
    };
    struct Glyph { int ax, ay, w, h, ox, oy, adv; };
    struct TextInst { int32_t x0, y0, x1, y1, u, v; uint32_t mode, pad; float cw, cr, fw, fr; };
    struct FrameCB { float size[2], s, s035, exposure, textGain, bloomGain, dither; uint32_t seed; float lift; uint32_t pad[2]; };
    struct DrawCB { uint32_t base, a, b, c; float p0[4], p1[4]; };

    bool compile(const wchar_t* file, const char* vsEntry, const char* psEntry, Shader& sh, std::string& err);
    bool target(Target& t, int w, int h, DXGI_FORMAT fmt);
    bool resize(int w, int h);
    bool upload(GpuBuffer& b, const void* data, size_t count, UINT stride, bool structured);
    void buildDisc();
    void addGlyphs(const dl::View& v);
    void bind(Target& t);
    void draw(const Shader& sh, UINT instances, UINT base, const DrawCB* cb = nullptr);
    void full(const Shader& sh, const DrawCB& cb);
    void srv(UINT slot, ID3D11ShaderResourceView* v);
    void unbind();
    void post(const dl::View& v);
    void finish(bool floatOut);
    void place(ID3D11RenderTargetView* rtv, int w, int h, Target& src, float x0, float y0, float x1, float y1, const float* around);
    void pass(const Shader& sh, Target& dst, int x0, int y0, int x1, int y1, float a = 0, float b = 0, float c = 0);
    void copy(Target& dst, Target& src, int x0, int y0, int x1, int y1);
    void mute(Target& t);

    ID3D11Device* mDev = nullptr;
    ID3D11DeviceContext* mCtx = nullptr;
    std::wstring mShaderDir;

    Shader mSeg, mDot, mSplat, mRect, mOp, mText, mBase, mDown, mBlurUp, mFinal, mFinalBox, mMuteBox;
    Shader mRoll, mStamp, mPour, mBright, mDrag, mSmear, mBlit, mOver;
    ComPtr<ID3D11BlendState> mBlendLight, mBlendText, mBlendNone, mBlendAlpha;
    ComPtr<ID3D11RasterizerState> mRaster;
    ComPtr<ID3D11SamplerState> mLinear;
    ComPtr<ID3D11Buffer> mFrameCB, mDrawCB;

    int mW = 0, mH = 0;
    Target mLight, mTextT, mBaseT, mOut, mOutF, mShow;
    bool mHasOver = false;
    float mLift = 0.0f;
    Target mCard;                              // the test card (texture and view only: nothing is drawn into it)
    bool mCardOn = false;
    int mCardX = 0, mCardY = 0;                // where the picture of the show sits in it
    GpuBuffer mOverBuf, mWinBuf;
    FrameCB mFrame = {};                       // the frame constants of the last picture drawn
    Target mScratch[3];                        // for the post-process: what an operation reads
    Target mDownT[9], mAcc[9];                 // bloom pyramid, levels 1..8

    GpuBuffer mStates, mSegs, mExpand, mDots, mSplats, mRects, mOps, mInst, mBoxes, mMute;
    std::vector<uint32_t> mExpandCpu;
    std::vector<dl::LightOp> mOpsCpu;
    std::vector<dl::Box> mBoxesCpu;
    std::vector<dl::Box> mMuteCpu;             // the boxes without glow, at the half size of the bloom
    std::vector<TextInst> mInstCpu;

    ComPtr<ID3D11Texture2D> mAtlas;
    ComPtr<ID3D11ShaderResourceView> mAtlasSrv;
    int mAtlasX = 0, mAtlasY = 0, mAtlasRow = 0;
    std::unordered_map<uint64_t, Glyph> mGlyphs;

    ComPtr<ID3D11Texture2D> mDisc;
    ComPtr<ID3D11ShaderResourceView> mDiscSrv;

    FrameInfo mInfo;
};
