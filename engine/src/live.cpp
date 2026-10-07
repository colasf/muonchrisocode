// muonengine live: the show in real time, with sound, in a preview window and as a Spout sender.
#include "osc.h"                            // first: winsock2.h has to come before windows.h

#include <windows.h>
#include <shellapi.h>
#include <timeapi.h>
#include <windowsx.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "SpoutDX.h"
#include "audio.h"
#include "gui.h"
#include "live.h"
#include "player.h"

namespace fs = std::filesystem;

namespace {

const int PIC_W = 2978, PIC_H = 1400;       // the picture of the show (layout.W, layout.H)
const float FRAME_X0 = 28.0f, FRAME_X1 = 2950.0f, FRAME_Y1 = 1354.0f, HEAD_Y = 220.0f;
const int BAR_H = 36;                       // height of the time bar under the picture, window pixels
const float BAR_PAD = 12.0f;                // its margin left and right
const float BTN_W = 44.0f;                  // the play / pause button, at its left
const float TC_W = 114.0f;                  // the time code MM:SS:FF, after the button
const float BAR_X0 = BTN_W + TC_W + BAR_PAD; // where the time line starts
const float TWR_W = 90.0f, OUT_W = 90.0f, SND_W = 78.0f, SNP_W = 114.0f, CMT_W = 102.0f, ASK_W = 90.0f;      // the TOWERS, OUTPUT, SOUND,
const float RIGHT_W = TWR_W + OUT_W + SND_W + SNP_W + CMT_W + ASK_W;                          // SNAPSHOT, COMMENT and CLAUDE buttons, at its right
const float BOX_H = 40.0f;                  // the line a comment is typed in, above the bar
const float PANEL_H = 134.0f;               // the OUTPUT panel and the TOWERS panel, above the bar
const float GLOW_MAX = 2.0f, RED_MAX = 3.0f, WEIGHT_MAX = 1.5f;      // the ends of its GLOW W, GLOW R, RED and WEIGHT sliders
const float TUNE_X0 = 176.0f, TUNE_W = 180.0f, TUNE_PITCH = 380.0f;  // where those sliders are
const float LIFT_MAX = 3.0f;                // the end of the LIFT slider
const int MOVE_MAX = 400;                   // how far MOVE can take the output from its place, pixels
const float SLIDER_X0 = 150.0f, SLIDER_W = 300.0f;

// ------------------------------------------------------------------------------------------------
// messages: the console, and a file that is still there the morning after
// ------------------------------------------------------------------------------------------------

FILE* gLog = nullptr;

void logLine(FILE* console, const char* fmt, va_list ap)
{
    char b[1200];
    vsnprintf(b, sizeof b, fmt, ap);
    fprintf(console, "%s\n", b);
    fflush(console);
    if (gLog) {
        time_t now = time(nullptr);
        tm lt;
        localtime_s(&lt, &now);
        char ts[32];
        strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &lt);
        fprintf(gLog, "%s  %s\n", ts, b);
        fflush(gLog);
    }
}
void say(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    logLine(stdout, fmt, ap);
    va_end(ap);
}
void warn(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    logLine(stderr, fmt, ap);
    va_end(ap);
}

std::string narrow(const std::wstring& w)
{
    std::string s;
    for (wchar_t c : w) s += (char)(c < 128 ? c : '?');
    return s;
}

// ------------------------------------------------------------------------------------------------
// window
// ------------------------------------------------------------------------------------------------

struct Window {
    HWND hwnd = nullptr;
    ComPtr<IDXGISwapChain1> swap;
    ComPtr<ID3D11RenderTargetView> rtv;
    int w = 0, h = 0;
    int bar = 0;                            // pixels kept free under the picture (the time bar)
    bool resized = false, closed = false, fullscreen = false;
    bool lost = false;                      // the graphics device is gone: nothing can be shown any more
    bool displays = false;                  // Windows says its displays changed (one plugged in, taken away, another size)
    WINDOWPLACEMENT placement = { sizeof(WINDOWPLACEMENT) };
    std::vector<WPARAM> keys;               // keys pressed since the last pump
    std::wstring chars;                     // ... and what they typed (the comment line)
    int mx = 0, my = 0;
    bool down = false, pressed = false, released = false;
    // Windows runs a loop of its own while the window is dragged or resized, or while its menu is open:
    // the main loop does not get its turn. `tick` is called from a timer during that time, so that the
    // picture and the Spout output go on.
    std::function<void()> tick;
    bool modal = false;

    static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM wp, LPARAM lp)
    {
        Window* s = (Window*)GetWindowLongPtrW(h, GWLP_USERDATA);
        if (s) switch (m) {
            case WM_CLOSE: s->closed = true; return 0;
            case WM_DISPLAYCHANGE: s->displays = true; return 0;
            case WM_SIZE:
                if (wp != SIZE_MINIMIZED) { s->w = LOWORD(lp); s->h = HIWORD(lp); s->resized = true; }
                return 0;
            case WM_KEYDOWN:
                if (wp == 'S' && (lp & (1 << 30))) return 0;       // (a key held down repeats: one snapshot per press)
                s->keys.push_back(wp);
                return 0;
            case WM_CHAR: if (wp >= 32 && wp != 127) s->chars.push_back((wchar_t)wp); return 0;
            case WM_MOUSEMOVE: s->mx = GET_X_LPARAM(lp); s->my = GET_Y_LPARAM(lp); return 0;
            case WM_LBUTTONDOWN: s->mx = GET_X_LPARAM(lp); s->my = GET_Y_LPARAM(lp); s->down = s->pressed = true; SetCapture(h); return 0;
            case WM_LBUTTONUP: s->down = false; s->released = true; ReleaseCapture(); return 0;
            case WM_SYSCOMMAND:
                // Alt or F10 alone would open the (empty) menu of the window and stop the picture until
                // another key is pressed: there is no menu to open
                if ((wp & 0xFFF0) == SC_KEYMENU) return 0;
                break;
            case WM_ENTERSIZEMOVE:
            case WM_ENTERMENULOOP:
                s->modal = true;
                SetTimer(h, 1, 1, nullptr);
                break;
            case WM_EXITSIZEMOVE:
            case WM_EXITMENULOOP:
                s->modal = false;
                KillTimer(h, 1);
                break;
            case WM_TIMER:
                if (wp == 1 && s->modal && s->tick) { s->tick(); return 0; }
                break;
        }
        return DefWindowProcW(h, m, wp, lp);
    }

    bool create(Gpu& g, int cw, int ch, std::string& err)
    {
        WNDCLASSW wc = {};
        wc.lpfnWndProc = proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.lpszClassName = L"MuonBloomEngine";
        RegisterClassW(&wc);
        RECT r = { 0, 0, cw, ch };
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        hwnd = CreateWindowExW(0, wc.lpszClassName, L"MUON : BLOOM", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                               r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd) { err = "cannot create the window"; return false; }
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)this);
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        if (FAILED(g.factory->CreateSwapChainForHwnd(g.dev.Get(), hwnd, &sd, nullptr, nullptr, &swap))) { err = "cannot create the swap chain"; return false; }
        g.factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        w = cw;
        h = ch;
        resized = true;
        ShowWindow(hwnd, SW_SHOW);
        return true;
    }

    void pump()
    {
        keys.clear();
        chars.clear();
        pressed = released = false;
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&m);
            DispatchMessageW(&m);
        }
    }

    void toggleFullscreen()
    {
        DWORD style = GetWindowLongW(hwnd, GWL_STYLE);
        if (!fullscreen) {
            MONITORINFO mi = { sizeof mi };
            GetWindowPlacement(hwnd, &placement);
            GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
            SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED);
        } else {
            SetWindowLongW(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(hwnd, &placement);
            SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
        fullscreen = !fullscreen;
    }

    // The picture, and under it what the window alone shows (`extra`, in window pixels).
    void present(Gpu& g, Renderer& r, const std::vector<Renderer::Over>& extra)
    {
        if (w <= 0 || h <= 0 || lost) return;
        if (resized || !rtv) {
            rtv.Reset();
            swap->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
            ComPtr<ID3D11Texture2D> back;
            if (FAILED(swap->GetBuffer(0, IID_PPV_ARGS(&back)))) { lost = g.removed(); return; }
            g.dev->CreateRenderTargetView(back.Get(), nullptr, &rtv);
            resized = false;
        }
        r.blit(rtv.Get(), w, h, bar);
        r.windowRects(rtv.Get(), w, h, extra);
        HRESULT hr = swap->Present(0, 0);
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) lost = true;
    }

    // window pixel -> pixel of the picture
    void toPicture(float& x, float& y) const
    {
        const int hp = std::max(1, h - bar);
        float k = std::min((float)w / PIC_W, (float)hp / PIC_H);
        x = (mx - 0.5f * (w - PIC_W * k)) / k;
        y = (my - 0.5f * (hp - PIC_H * k)) / k;
    }
    bool overBar() const { return bar > 0 && my >= h - bar && my < h && mx >= 0 && mx < w; }
    bool overButton() const { return overBar() && mx < BTN_W; }         // the play / pause button
    bool overTowers() const { return overBar() && mx >= w - RIGHT_W && mx < w - RIGHT_W + TWR_W; }
    bool overOutput() const { return overBar() && mx >= w - RIGHT_W + TWR_W && mx < w - RIGHT_W + TWR_W + OUT_W; }
    bool overSound() const { return overBar() && mx >= w - RIGHT_W + TWR_W + OUT_W && mx < w - SNP_W - CMT_W - ASK_W; }
    bool overSnapshot() const { return overBar() && mx >= w - SNP_W - CMT_W - ASK_W && mx < w - CMT_W - ASK_W; }
    bool overComment() const { return overBar() && mx >= w - CMT_W - ASK_W && mx < w - ASK_W; }
    bool overAsk() const { return overBar() && mx >= w - ASK_W; }
    float lineEnd() const { return std::max(BAR_X0 + 1.0f, (float)w - RIGHT_W - BAR_PAD); }    // where the time line ends
    bool overLine() const { return overBar() && mx >= BAR_X0 - 6.0f && mx <= lineEnd() + 6.0f; }
    // mouse position -> 0..1 along the time line
    double barFraction() const
    {
        return std::clamp(((double)mx - BAR_X0) / std::max(1.0, (double)lineEnd() - BAR_X0), 0.0, 1.0);
    }
};

// ------------------------------------------------------------------------------------------------
// towers: data/towers.json, read and written as muonbloom/layout.py does
// ------------------------------------------------------------------------------------------------

struct Tower { float x0, x1, top, bot, det_h; };

struct Towers {
    Tower t[3];                             // L, C, R
    fs::path path;

    bool load()
    {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string s = ss.str();
        const char* keys[3] = { "\"L\"", "\"C\"", "\"R\"" };
        const char* names[5] = { "\"x0\"", "\"x1\"", "\"top\"", "\"bot\"", "\"det_h\"" };
        for (int k = 0; k < 3; k++) {
            size_t a = s.find(keys[k]);
            if (a == std::string::npos) return false;
            size_t b = s.find('}', a);
            float* v = &t[k].x0;
            for (int j = 0; j < 5; j++) {
                size_t p = s.find(names[j], a);
                if (p == std::string::npos || p > b) return false;
                v[j] = (float)atof(s.c_str() + s.find(':', p) + 1);
            }
        }
        return true;
    }

    // Written beside the file and moved over it: the scene workers never read half a file.
    bool save() const
    {
        fs::path tmp = path;
        tmp += L".tmp";
        {
            std::ofstream f(tmp);
            if (!f) return false;
            const char* keys[3] = { "L", "C", "R" };
            f << "{\n";
            for (int k = 0; k < 3; k++) {
                char b[256];
                snprintf(b, sizeof b, "  \"%s\": {\n    \"x0\": %.1f,\n    \"x1\": %.1f,\n    \"top\": %.1f,\n    \"bot\": %.1f,\n    \"det_h\": %.1f\n  }%s\n",
                         keys[k], t[k].x0, t[k].x1, t[k].top, t[k].bot, t[k].det_h, k < 2 ? "," : "");
                f << b;
            }
            f << "}";
            if (!f) return false;
        }
        return MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
    }

    // Width of the widest free stretch of wall between the towers and the frame.
    float widestBay() const
    {
        float best = 0.0f, x = FRAME_X0;
        for (int k = 0; k < 3; k++) {
            best = std::max(best, t[k].x0 - x);
            x = t[k].x1;
        }
        return std::max(best, FRAME_X1 - x);
    }
};

// The placement tool: the three towers drawn over the picture (so they are seen on the wall, through
// the output), moved with the mouse, the arrow keys or the numbers of the TOWERS panel, written to
// data/towers.json on Enter (or SAVE in the panel).
struct TowerTool {
    static constexpr float MIN_BAY = 300.0f;        // the scenes need one free stretch of wall at least this wide
    Towers tw;
    bool on = false, dirty = false;
    bool edited = false;                    // moved since it was read or saved
    bool calib = false;                     // the calibration picture instead of the show (CALIBRATION in the TOWERS panel)
    int sel = 1;                            // tower being edited
    int handle = 0;                         // 0 whole, 1 left edge, 2 right edge, 3 top, 4 detector height
    bool dragging = false;
    float gx = 0, gy = 0;                   // mouse position of the last drag step

    void clamp()
    {
        for (int k = 0; k < 3; k++) {
            Tower& a = tw.t[k];
            float lo = k ? tw.t[k - 1].x1 + 30.0f : FRAME_X0, hi = k < 2 ? tw.t[k + 1].x0 - 30.0f : FRAME_X1;
            a.x0 = std::clamp(a.x0, lo, hi - 20.0f);
            a.x1 = std::clamp(a.x1, a.x0 + 20.0f, hi);
            a.top = std::clamp(a.top, HEAD_Y, a.bot - 60.0f);
            a.det_h = std::clamp(a.det_h, 20.0f, a.bot - a.top - 20.0f);
        }
    }

    void move(float dx, float dy)
    {
        Tower& a = tw.t[sel];
        float w = a.x1 - a.x0;
        switch (handle) {
        case 0: a.x0 += dx; a.x1 = a.x0 + w; a.top += dy; break;
        case 1: a.x0 += dx; break;
        case 2: a.x1 += dx; break;
        case 3: a.top += dy; break;
        case 4: a.det_h += dy; break;
        }
        if (handle == 0) {                  // keep the width when the tower is pushed against a neighbour
            clamp();
            Tower& c = tw.t[sel];
            if (c.x1 - c.x0 < w) { if (dx > 0) c.x0 = c.x1 - w; else c.x1 = c.x0 + w; }
        }
        clamp();
        dirty = edited = true;
    }

    // The placement as it is saved, shown.
    bool start()
    {
        if (!tw.load()) return false;
        on = dirty = true;
        edited = false;
        return true;
    }

    // The numbers of the TOWERS panel: 0 the centre, 1 the width (about the centre), 2 the height (the foot
    // stays on the ground), 3 the height of the detector.
    void nudge(int what, float d)
    {
        Tower& a = tw.t[sel];
        if (what == 0) { handle = 0; move(d, 0); return; }
        handle = what == 1 ? 0 : what == 2 ? 3 : 4;
        if (what == 1) { a.x0 -= 0.5f * d; a.x1 += 0.5f * d; }
        else if (what == 2) a.top -= d;
        else a.det_h += d;
        clamp();
        dirty = edited = true;
    }

    void press(float x, float y)
    {
        float best = 1e9f;
        for (int k = 0; k < 3; k++) {
            const Tower& a = tw.t[k];
            if (x < a.x0 - 16 || x > a.x1 + 16 || y < a.top - 16 || y > a.bot + 16) continue;
            struct { int h; float d; } c[4] = { { 1, std::abs(x - a.x0) }, { 2, std::abs(x - a.x1) }, { 3, std::abs(y - a.top) },
                                                { 4, std::abs(y - (a.top + a.det_h)) } };
            int hh = 0;
            float d = 14.0f;
            for (auto& e : c) if (e.d < d) { d = e.d; hh = e.h; }
            if (d < best) { best = d; sel = k; handle = hh; }
        }
        dragging = best < 1e8f;
        gx = x;
        gy = y;
    }

    // The calibration picture, sent out instead of the show: black, the grid of the canvas with its numbers
    // (a line every 100 pixels, a stronger one every 500), and the three towers as blocks of light with
    // full-level edges, to be moved until each block sits on its tower and no light falls on the wall
    // beside it. Above a tower: its name, centre, width and height. Yellow: what the next move changes.
    void pattern(std::vector<Renderer::Over>& o) const
    {
        const float W = (float)PIC_W, H = (float)PIC_H;
        char b[48];
        o.push_back({ 0.0f, 0.0f, W, H, 0.0f, 0.0f, 0.0f, 1.0f });
        for (int x = 100; x < PIC_W; x += 100) {
            const bool big = x % 500 == 0;
            const float g = big ? 0.6f : 0.25f, t = big ? 1.0f : 0.5f;
            o.push_back({ x - t, 0.0f, x + t, H, g, g, g, 1.0f });
            snprintf(b, sizeof b, "%d", x);
            if (big) gui::text(o, x + 9.0f, 16.0f, b, 3.0f, 1.0f, 1.0f, 1.0f);
        }
        for (int y = 100; y < PIC_H; y += 100) {
            const bool big = y % 500 == 0;
            const float g = big ? 0.6f : 0.25f, t = big ? 1.0f : 0.5f;
            o.push_back({ 0.0f, y - t, W, y + t, g, g, g, 1.0f });
            snprintf(b, sizeof b, "%d", y);
            if (big) gui::text(o, 16.0f, y + 9.0f, b, 3.0f, 1.0f, 1.0f, 1.0f);
        }
        const float e = 3.0f;                       // the edge of the canvas
        o.push_back({ 0.0f, 0.0f, W, e, 1.0f, 1.0f, 1.0f, 1.0f });
        o.push_back({ 0.0f, H - e, W, H, 1.0f, 1.0f, 1.0f, 1.0f });
        o.push_back({ 0.0f, 0.0f, e, H, 1.0f, 1.0f, 1.0f, 1.0f });
        o.push_back({ W - e, 0.0f, W, H, 1.0f, 1.0f, 1.0f, 1.0f });
        const char* names[3] = { "LEFT", "CENTRE", "RIGHT" };
        for (int k = 0; k < 3; k++) {
            const Tower& a = tw.t[k];
            const bool s = k == sel;
            const float cx = 0.5f * (a.x0 + a.x1);
            auto line = [&](float x0, float y0, float x1, float y1, bool hot) {
                o.push_back({ x0, y0, x1, y1, 1.0f, hot ? 0.85f : 1.0f, hot ? 0.0f : 1.0f, 1.0f });
            };
            o.push_back({ a.x0 - 12.0f, a.top - 12.0f, a.x1 + 12.0f, a.bot + 12.0f, 0.0f, 0.0f, 0.0f, 1.0f });     // no grid against it
            o.push_back({ a.x0, a.top, a.x1, a.bot, 0.4f, 0.4f, 0.4f, 1.0f });
            line(a.x0, a.top, a.x0 + e, a.bot, s && (handle == 1 || handle == 0));
            line(a.x1 - e, a.top, a.x1, a.bot, s && (handle == 2 || handle == 0));
            line(a.x0, a.top, a.x1, a.top + e, s && (handle == 3 || handle == 0));
            line(a.x0, a.bot - e, a.x1, a.bot, false);
            line(a.x0, a.top + a.det_h - 1.5f, a.x1, a.top + a.det_h + 1.5f, s && handle == 4);
            line(a.x0 - 150.0f, a.bot, a.x0 - 20.0f, a.bot + 2.0f, false);          // the ground, either side
            line(a.x1 + 20.0f, a.bot, a.x1 + 150.0f, a.bot + 2.0f, false);
            line(cx - 1.0f, a.top - 150.0f, cx + 1.0f, a.top - 20.0f, false);        // its axis, above the head
            const float tx = std::clamp(cx + 16.0f, 12.0f, W - 190.0f), ty = a.top - 150.0f, g = s ? 0.85f : 1.0f, bl = s ? 0.0f : 1.0f;
            o.push_back({ tx - 8.0f, ty - 8.0f, tx + 180.0f, ty + 120.0f, 0.0f, 0.0f, 0.0f, 1.0f });
            gui::text(o, tx, ty, names[k], 3.0f, 1.0f, g, bl);
            snprintf(b, sizeof b, "X %.1f", cx);
            gui::text(o, tx, ty + 30.0f, b, 3.0f, 1.0f, g, bl);
            snprintf(b, sizeof b, "W %.1f", a.x1 - a.x0);
            gui::text(o, tx, ty + 60.0f, b, 3.0f, 1.0f, g, bl);
            snprintf(b, sizeof b, "H %.1f", a.bot - a.top);
            gui::text(o, tx, ty + 90.0f, b, 3.0f, 1.0f, g, bl);
        }
    }

    void rects(std::vector<Renderer::Over>& o) const
    {
        if (calib) { pattern(o); return; }
        for (int k = 0; k < 3; k++) {
            const Tower& a = tw.t[k];
            bool s = k == sel;
            auto line = [&](float x0, float y0, float x1, float y1, bool hot) {
                o.push_back({ x0, y0, x1, y1, 1.0f, hot ? 0.85f : 1.0f, hot ? 0.0f : 1.0f, 1.0f });
            };
            o.push_back({ a.x0, a.top, a.x1, a.bot, s ? 1.0f : 0.0f, s ? 0.15f : 0.55f, s ? 0.1f : 1.0f, 0.35f });
            float t = 3.0f;
            line(a.x0, a.top, a.x0 + t, a.bot, s && (handle == 1 || handle == 0));
            line(a.x1 - t, a.top, a.x1, a.bot, s && (handle == 2 || handle == 0));
            line(a.x0, a.top, a.x1, a.top + t, s && (handle == 3 || handle == 0));
            line(a.x0, a.bot - t, a.x1, a.bot, false);
            line(a.x0, a.top + a.det_h - 1.0f, a.x1, a.top + a.det_h + 1.0f, s && handle == 4);
            line(0.5f * (a.x0 + a.x1) - 1.0f, a.top - 40.0f, 0.5f * (a.x0 + a.x1) + 1.0f, a.top, false);     // the detector
        }
    }
};

// ------------------------------------------------------------------------------------------------
// files of the show: a change restarts the workers
// ------------------------------------------------------------------------------------------------

struct Watcher {
    fs::path root;
    uint64_t last = 0;
    double checked = 0, changed = 0;
    bool pending = false;

    uint64_t scan() const
    {
        uint64_t h = 1469598103934665603ull;
        std::error_code ec;
        auto add = [&](const fs::path& p) {
            auto t = fs::last_write_time(p, ec);
            if (!ec) h = (h ^ (uint64_t)t.time_since_epoch().count()) * 1099511628211ull;
        };
        for (const wchar_t* d : { L"muonbloom", L"muonbloom/scenes" })
            for (auto& e : fs::directory_iterator(root / d, ec))
                if (e.path().extension() == L".py") add(e.path());
        for (const wchar_t* f : { L"data/towers.json", L"data/cues.npz", L"subtitletimecode.txt", L"engine/worker.py", L"engine/detectors.py" })
            add(root / f);
        return h;
    }

    // True once, when the files have changed and then stayed still for a moment.
    bool poll()
    {
        double t = now();
        if (t - checked < 0.5) return false;
        checked = t;
        uint64_t h = scan();
        if (!last) last = h;
        if (h != last) {
            last = h;
            pending = true;
            changed = t;
            return false;
        }
        if (pending && t - changed > 0.4) {
            pending = false;
            return true;
        }
        return false;
    }
};

std::wstring timecode(double t)
{
    wchar_t b[32];
    int m = (int)(t / 60);
    swprintf(b, 32, L"%02d:%06.3f", m, t - 60 * m);
    return b;
}

float percentile(std::vector<float> v, double q)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min<double>((double)v.size() - 1, q * v.size())];
}

// ------------------------------------------------------------------------------------------------
// the prompt box: what is typed in the window goes to Claude Code, its answer comes back in a panel
// ------------------------------------------------------------------------------------------------

// Claude Code without its terminal: `claude -p`, started in the repository folder, the prompt on its standard
// input, its answer read from a pipe by a thread. Every prompt of one run of the engine goes on in the same
// conversation (--session-id for the first, --resume for the next ones). Nobody can answer a permission
// request in that mode: what it may do is given on the command line (ARGS, or the one line of
// engine/claude_args.txt when that file exists). Never in the Spout output: it is drawn with the bar.
struct Claude {
    static constexpr const wchar_t* ARGS = L"--permission-mode acceptEdits --allowedTools \"Read,Edit,Write,Glob,Grep,Bash(python:*),PowerShell(python:*)\"";
    static constexpr const wchar_t* SYSTEM =
        L"You are called from the prompt box of the Muon Bloom engine window, not from a terminal. Your answer is shown in a "
        L"small panel with a 5 x 7 pixel ASCII font: answer in plain text, at most 8 short lines, no markdown, no tables, no "
        L"accents. Each message starts with the show time and the scene that are on screen. The engine reloads the scenes "
        L"as soon as a .py file under muonbloom is saved: save such a file only in a state that imports and draws. "
        L"After a change of the picture, look at it before you answer: run  python tools/engine_frame.py MM:SS:FF  (several "
        L"times, and --crop x0,y0,x1,y1 in wall pixels for a part at full size) - it draws those frames with the renderer of "
        L"the engine and the lift the user has, and prints the paths of the pictures: read them, and say what you saw. "
        L"Nobody can approve a permission request here: if something you need is refused, say what. "
        L"Do not commit or push unless the message asks for it.";

    fs::path root;
    std::wstring exe, args = ARGS;
    std::string session;                    // the conversation of this run
    bool begun = false;                     // it exists: the next prompt resumes it
    bool running = false, show = false;
    double startedAt = 0;
    std::vector<std::string> lines;         // what the panel shows: the questions ("> ...") and the answers
    HANDLE proc = nullptr, rd = nullptr;
    std::thread reader;
    std::mutex mu;
    std::string buf;                        // what the process has written so far
    std::atomic<bool> eof{ false };

    static std::string uuid()
    {
        GUID g;
        if (CoCreateGuid(&g) != S_OK) return "00000000-0000-4000-8000-000000000000";
        char b[40];
        snprintf(b, sizeof b, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g.Data1, g.Data2, g.Data3, g.Data4[0], g.Data4[1],
                 g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
        return b;
    }

    void init(const fs::path& r)
    {
        root = r;
        session = uuid();
        wchar_t found[MAX_PATH];
        if (SearchPathW(nullptr, L"claude.exe", nullptr, MAX_PATH, found, nullptr)) exe = found;
        else {
            wchar_t home[MAX_PATH];
            std::error_code ec;
            if (GetEnvironmentVariableW(L"USERPROFILE", home, MAX_PATH) && fs::exists(fs::path(home) / L".local/bin/claude.exe", ec))
                exe = (fs::path(home) / L".local/bin/claude.exe").wstring();
        }
        std::ifstream f(root / L"engine" / L"claude_args.txt");
        std::string line;
        if (f && std::getline(f, line) && !line.empty()) args = std::wstring(line.begin(), line.end());
    }

    // Sends a prompt (UTF-8). False when it cannot be started: `why` says it.
    bool ask(const std::string& prompt, const std::string& shown, std::string& why)
    {
        if (running) { why = "CLAUDE IS STILL WORKING"; return false; }
        if (exe.empty()) { why = "claude.exe NOT FOUND (is Claude Code installed?)"; return false; }
        SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
        HANDLE inR = nullptr, inW = nullptr, outR = nullptr, outW = nullptr;
        if (!CreatePipe(&inR, &inW, &sa, 0) || !CreatePipe(&outR, &outW, &sa, 0)) { why = "CANNOT OPEN A PIPE"; return false; }
        SetHandleInformation(inW, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(outR, HANDLE_FLAG_INHERIT, 0);
        std::wstring cmd = L"\"" + exe + L"\" -p " + (begun ? L"--resume " : L"--session-id ") + std::wstring(session.begin(), session.end())
                           + L" " + args + L" --append-system-prompt \"" + SYSTEM + L"\"";
        STARTUPINFOW si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = inR;
        si.hStdOutput = outW;
        si.hStdError = outW;
        PROCESS_INFORMATION pi{};
        std::vector<wchar_t> line(cmd.begin(), cmd.end());
        line.push_back(0);
        const BOOL ok = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, root.wstring().c_str(), &si, &pi);
        CloseHandle(inR);
        CloseHandle(outW);
        if (!ok) {
            CloseHandle(inW);
            CloseHandle(outR);
            why = "CLAUDE CANNOT BE STARTED";
            return false;
        }
        CloseHandle(pi.hThread);
        DWORD n = 0;
        WriteFile(inW, prompt.data(), (DWORD)prompt.size(), &n, nullptr);
        CloseHandle(inW);                   // the end of the prompt
        proc = pi.hProcess;
        rd = outR;
        buf.clear();
        eof = false;
        reader = std::thread([this] {
            char b[4096];
            DWORD got = 0;
            while (ReadFile(rd, b, sizeof b, &got, nullptr) && got) {
                std::lock_guard<std::mutex> g(mu);
                buf.append(b, got);
            }
            eof = true;
        });
        running = show = true;
        startedAt = now();
        lines.push_back("> " + shown);
        return true;
    }

    // True once, when the answer is there (it is then in `lines`); `code` = how the process ended.
    bool poll(DWORD& code)
    {
        if (!running || !eof) return false;
        reader.join();
        WaitForSingleObject(proc, 2000);
        code = 1;
        GetExitCodeProcess(proc, &code);
        CloseHandle(proc);
        CloseHandle(rd);
        proc = rd = nullptr;
        running = false;
        if (code == 0) begun = true;
        else if (!begun) session = uuid();  // (a conversation that did not start cannot be given the same name again)
        std::string cur;
        size_t added = 0;
        for (unsigned char c : buf) {       // the 5 x 7 font is ASCII: one '?' for a character it does not have
            if (c == '\n') { lines.push_back(cur); cur.clear(); added++; }
            else if (c == '\r' || (c >= 0x80 && c < 0xC0)) continue;
            else cur += (char)(c >= 32 && c < 127 ? c : c == '\t' ? ' ' : '?');
        }
        if (!cur.empty() || !added) lines.push_back(cur.empty() ? (code ? "(no answer: it ended with an error)" : "(done)") : cur);
        lines.push_back("");
        if (lines.size() > 400) lines.erase(lines.begin(), lines.end() - 400);
        return true;
    }

    void close()
    {
        if (!running) return;
        TerminateProcess(proc, 1);
        reader.join();
        CloseHandle(proc);
        CloseHandle(rd);
        running = false;
    }

    // The panel above the bar (above the line being typed): the end of the conversation, wrapped to the window.
    void rects(std::vector<Renderer::Over>& o, int w, int h, int bar, bool typing) const
    {
        const float px = 2.0f, lh = 20.0f, bottom = (float)(h - bar) - (typing ? BOX_H : 0.0f);
        const size_t room = (size_t)std::max(20.0f, ((float)w - 28.0f) / (6.0f * px));
        std::vector<std::pair<std::string, bool>> out;      // (text, it is a question)
        for (const std::string& l : lines) {
            const bool q = l.rfind("> ", 0) == 0;
            size_t i = 0;
            do {
                size_t n = std::min(room, l.size() - i);
                if (i + n < l.size()) {                     // break at a space when there is one
                    size_t sp = l.rfind(' ', i + n);
                    if (sp != std::string::npos && sp > i + room / 2) n = sp - i;
                }
                out.push_back({ l.substr(i, n), q });
                i += n;
                while (i < l.size() && l[i] == ' ') i++;
            } while (i < l.size());
        }
        while (!out.empty() && out.back().first.empty()) out.pop_back();
        const size_t maxLines = (size_t)std::max(3.0f, std::min(16.0f, (bottom - 60.0f) / lh));
        const size_t first = out.size() > maxLines ? out.size() - maxLines : 0, n = out.size() - first;
        const float y0 = bottom - 34.0f - (float)n * lh;
        o.push_back({ 0.0f, y0, (float)w, bottom, 0.0f, 0.0f, 0.0f, 0.88f });
        o.push_back({ 0.0f, y0, (float)w, y0 + 1.0f, 0.35f, 0.8f, 1.0f, 1.0f });
        char head[96];
        if (running) snprintf(head, sizeof head, "CLAUDE IS WORKING   %d S", (int)(now() - startedAt));
        else snprintf(head, sizeof head, "CLAUDE   A: ASK   SHIFT+A: HIDE");
        gui::text(o, 14.0f, y0 + 9.0f, head, px, 0.35f, 0.8f, 1.0f);
        for (size_t k = 0; k < n; k++) {
            const auto& l = out[first + k];
            const float c = l.second ? 0.6f : 1.0f;
            gui::text(o, 14.0f, y0 + 30.0f + (float)k * lh, l.first, px, c, c, c);
        }
    }
};

// What is drawn on the frame (drag on the picture, at any time while the tower tool is away): strokes, in pixels
// of the picture. They are shown over the preview only - never in the output - and painted into the next
// snapshot, or into the picture that goes with the next comment or prompt; then they are gone. Backspace takes
// the last stroke away, Delete all of them.
struct Sketch {
    static constexpr float R = 0.2f, G = 0.9f, B = 1.0f;       // their colour: the blue of the prompt box (not a colour of the show)
    std::vector<std::vector<std::pair<float, float>>> strokes;
    bool drawing = false;

    bool empty() const { return strokes.empty(); }
    void clear() { strokes.clear(); drawing = false; }

    // Over the preview: window pixels (the picture is fitted and centred above the bar, as Window::toPicture has it).
    void rects(std::vector<Renderer::Over>& o, int w, int h, int bar) const
    {
        const int hp = std::max(1, h - bar);
        const float k = std::min((float)w / PIC_W, (float)hp / PIC_H), ox = 0.5f * (w - PIC_W * k), oy = 0.5f * (hp - PIC_H * k);
        for (auto& st : strokes)
            for (size_t i = 0; i < st.size(); i++) {
                const float x1 = st[i].first * k + ox, y1 = st[i].second * k + oy;
                const float x0 = i ? st[i - 1].first * k + ox : x1, y0 = i ? st[i - 1].second * k + oy : y1;
                const int n = std::max(1, (int)(std::hypot(x1 - x0, y1 - y0) / 1.5f));
                for (int j = 0; j <= n; j++) {
                    const float x = x0 + (x1 - x0) * j / n, y = y0 + (y1 - y0) * j / n;
                    o.push_back({ x - 1.5f, y - 1.5f, x + 1.5f, y + 1.5f, R, G, B, 1.0f });
                }
            }
    }

    // Into a picture read back from the renderer (BGRA, w x h): lines 7 pixels thick at the size of the show.
    void paint(std::vector<uint8_t>& px, int w, int h) const
    {
        const float k = (float)w / PIC_W, rad = std::max(1.5f, 3.5f * k);
        auto disc = [&](float cx, float cy) {
            for (int y = std::max(0, (int)(cy - rad)); y <= std::min(h - 1, (int)(cy + rad)); y++)
                for (int x = std::max(0, (int)(cx - rad)); x <= std::min(w - 1, (int)(cx + rad)); x++)
                    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= rad * rad) {
                        uint8_t* q = &px[4 * ((size_t)y * w + x)];
                        q[0] = (uint8_t)(255 * B); q[1] = (uint8_t)(255 * G); q[2] = (uint8_t)(255 * R); q[3] = 255;
                    }
        };
        for (auto& st : strokes)
            for (size_t i = 0; i < st.size(); i++) {
                const float x1 = st[i].first * k, y1 = st[i].second * k;
                const float x0 = i ? st[i - 1].first * k : x1, y0 = i ? st[i - 1].second * k : y1;
                const int n = std::max(1, (int)std::hypot(x1 - x0, y1 - y0));
                for (int j = 0; j <= n; j++) disc(x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * j / n);
            }
    }
};

// What the bar shows beside the time line.
struct BarState {
    std::string tc;                         // the time code, MM:SS:FF
    bool sound = false;                     // the engine plays sound itself
    bool commenting = false;                // a comment is being typed
    int output = 0;                         // 0 no output window, 1 it is on a display, 2 its display is gone
    bool card = false;                      // the test card is shown instead of the show
    bool panel = false;                     // the OUTPUT panel is open
    bool towers = false;                    // the TOWERS panel is open
    bool snapped = false;                   // a snapshot has just been taken
    int claude = 0;                         // the prompt box: 1 a prompt is being typed, 2 Claude is working
    int hot = 0;                            // button under the mouse: 1 play / pause, 2 SOUND, 3 COMMENT, 4 OUTPUT, 5 SNAPSHOT, 6 CLAUDE, 7 TOWERS
    const std::vector<gui::Comments::Mark>* marks = nullptr;
};

// The time bar of the preview window: a play / pause button, the time code, then the scenes of the show as
// blocks, what has been played, where the clock is, where the mouse points, where the comments are; at the
// right the TOWERS, OUTPUT, SOUND, SNAPSHOT, COMMENT and CLAUDE buttons. Window pixels.
void timeBar(std::vector<Renderer::Over>& o, int w, int h, const std::vector<Pool::Look>& looks, double end, double t, double hover,
             bool playing, const BarState& st)
{
    const bool hot = st.hot == 1;
    const float y0 = (float)(h - BAR_H), y1 = (float)h, x0 = BAR_X0, x1 = std::max(BAR_X0 + 1.0f, (float)w - RIGHT_W - BAR_PAD);
    auto X = [&](double v) { return x0 + (float)(std::clamp(v / std::max(end, 1e-9), 0.0, 1.0) * (x1 - x0)); };
    o.push_back({ 0.0f, y0, (float)w, y1, 0.0f, 0.0f, 0.0f, 1.0f });
    o.push_back({ 0.0f, y0, (float)w, y0 + 1.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    // the button: what a click will do - two bars while it plays (pause), a triangle while it is paused (play)
    {
        const float g = hot ? 1.0f : 0.8f, cx = 0.5f * BTN_W, cy = 0.5f * (y0 + y1), s = 8.0f;
        if (hot) o.push_back({ 4.0f, y0 + 4.0f, BTN_W - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
        if (playing) {
            o.push_back({ cx - s, cy - s, cx - 2.5f, cy + s, g, g, g, 1.0f });
            o.push_back({ cx + 2.5f, cy - s, cx + s, cy + s, g, g, g, 1.0f });
        } else {
            for (int k = 0; k < 16; k++) {          // a triangle pointing right, in one-pixel columns
                float half = s * (1.0f - (k + 0.5f) / 16.0f);
                o.push_back({ cx - 6.0f + k, cy - half, cx - 5.0f + k, cy + half, g, g, g, 1.0f });
            }
        }
        o.push_back({ BTN_W, y0 + 6.0f, BTN_W + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    }
    const float ty0 = y0 + 12.0f, ty1 = y1 - 12.0f;
    int k = 0;
    for (auto& l : looks) {                         // one block per scene, two greys in turn
        float g = (k++ & 1) ? 0.30f : 0.20f;
        o.push_back({ X(l.t0), ty0, X(l.t1), ty1, g, g, g, 1.0f });
    }
    o.push_back({ x0, ty0, X(t), ty1, 1.0f, 1.0f, 1.0f, 0.55f });                   // played
    for (auto& l : looks)                                                           // where a scene starts
        o.push_back({ X(l.t0), y0 + 6.0f, X(l.t0) + 1.0f, y1 - 6.0f, 0.75f, 0.75f, 0.75f, 1.0f });
    if (hover >= 0.0) o.push_back({ X(hover) - 0.5f, y0 + 3.0f, X(hover) + 0.5f, y1 - 3.0f, 1.0f, 1.0f, 1.0f, 0.8f });
    if (st.marks)                                                                   // the comments: yellow, grey once done
        for (auto& m : *st.marks) {
            float g = m.done ? 0.5f : 1.0f;
            o.push_back({ X(m.t) - 1.5f, y0 + 2.0f, X(m.t) + 1.5f, y0 + 10.0f, g, m.done ? 0.5f : 0.85f, m.done ? 0.5f : 0.0f, 1.0f });
        }
    o.push_back({ X(t) - 1.5f, y0 + 2.0f, X(t) + 1.5f, y1 - 2.0f, 1.0f, 0.1f, 0.06f, 1.0f });     // the clock
    // the time code
    gui::text(o, BTN_W + 11.0f, y0 + 11.0f, st.tc, 2.0f, 1.0f, 1.0f, 1.0f);
    // the buttons at the right: OUTPUT (white while a display takes the raster, yellow when that display is
    // gone, red while the test card is on), SOUND (white while the engine plays sound), SNAPSHOT (lit for a
    // moment when one is taken), COMMENT (yellow while one is typed)
    const float xt = (float)w - RIGHT_W, xo = xt + TWR_W, xs = xo + OUT_W, xa = (float)w - ASK_W, xc = xa - CMT_W, xn = xc - SNP_W;
    o.push_back({ xt, y0 + 6.0f, xt + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    if (st.hot == 7) o.push_back({ xt + 4.0f, y0 + 4.0f, xo - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
    {                                               // TOWERS: white and underlined while its panel is open
        const float g = st.towers ? 1.0f : 0.8f, tx = xt + 0.5f * (TWR_W - gui::textWidth(6, 2.0f));
        gui::text(o, tx, y0 + 11.0f, "TOWERS", 2.0f, g, g, g);
        if (st.towers) o.push_back({ tx, y1 - 8.0f, tx + gui::textWidth(6, 2.0f), y1 - 6.0f, g, g, g, 1.0f });
    }
    o.push_back({ xo, y0 + 6.0f, xo + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    if (st.hot == 4) o.push_back({ xo + 4.0f, y0 + 4.0f, xs - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
    {
        const float r = st.card || st.output ? 1.0f : 0.55f, g = st.card ? 0.12f : st.output == 2 ? 0.85f : r, b = st.card ? 0.08f : st.output == 2 ? 0.0f : r;
        const float tx = xo + 0.5f * (OUT_W - gui::textWidth(6, 2.0f));
        gui::text(o, tx, y0 + 11.0f, "OUTPUT", 2.0f, r, g, b);
        if (st.panel) o.push_back({ tx, y1 - 8.0f, tx + gui::textWidth(6, 2.0f), y1 - 6.0f, r, g, b, 1.0f });
    }
    o.push_back({ xs, y0 + 6.0f, xs + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    o.push_back({ xc, y0 + 6.0f, xc + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    o.push_back({ xn, y0 + 6.0f, xn + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    if (st.hot == 2) o.push_back({ xs + 4.0f, y0 + 4.0f, xn - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
    if (st.hot == 5 || st.snapped) o.push_back({ xn + 4.0f, y0 + 4.0f, xc - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, st.snapped ? 0.32f : 0.14f });
    if (st.hot == 3) o.push_back({ xc + 4.0f, y0 + 4.0f, xa - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
    o.push_back({ xa, y0 + 6.0f, xa + 1.0f, y1 - 6.0f, 0.35f, 0.35f, 0.35f, 1.0f });
    if (st.hot == 6) o.push_back({ xa + 4.0f, y0 + 4.0f, (float)w - 4.0f, y1 - 4.0f, 1.0f, 1.0f, 1.0f, 0.14f });
    gui::text(o, xa + 0.5f * (ASK_W - gui::textWidth(6, 2.0f)), y0 + 11.0f, "CLAUDE", 2.0f, st.claude ? 0.35f : 0.8f, 0.8f, st.claude ? 1.0f : 0.8f);
    if (st.claude == 2 && (int)(now() * 2.0) % 2)           // working: its name is underlined, on and off
        o.push_back({ xa + 0.5f * (ASK_W - gui::textWidth(6, 2.0f)), y1 - 8.0f, xa + 0.5f * (ASK_W + gui::textWidth(6, 2.0f)), y1 - 6.0f, 0.35f, 0.8f, 1.0f, 1.0f });
    const float gs = st.sound ? 1.0f : 0.55f, gn = st.snapped ? 1.0f : 0.8f;
    gui::text(o, xs + 0.5f * (SND_W - gui::textWidth(5, 2.0f)), y0 + 11.0f, "SOUND", 2.0f, gs, gs, gs);
    gui::text(o, xn + 0.5f * (SNP_W - gui::textWidth(8, 2.0f)), y0 + 11.0f, "SNAPSHOT", 2.0f, gn, gn, gn);
    gui::text(o, xc + 0.5f * (CMT_W - gui::textWidth(7, 2.0f)), y0 + 11.0f, "COMMENT", 2.0f, st.commenting ? 1.0f : 0.8f,
              st.commenting ? 0.85f : 0.8f, st.commenting ? 0.0f : 0.8f);
}

// The line a comment is typed in, above the bar: its time code, then the text (its end when it is too long).
// ask: the line is a prompt for Claude (blue) instead of a comment (yellow).
void commentBox(std::vector<Renderer::Over>& o, int w, int h, int bar, const std::string& tc, const std::wstring& typed, bool ask = false)
{
    const float y1 = (float)(h - bar), y0 = y1 - BOX_H, px = 2.0f;
    const float cr = ask ? 0.35f : 1.0f, cg = ask ? 0.8f : 0.85f, cb = ask ? 1.0f : 0.0f;
    o.push_back({ 0.0f, y0, (float)w, y1, 0.0f, 0.0f, 0.0f, 0.92f });
    o.push_back({ 0.0f, y0, (float)w, y0 + 1.0f, cr, cg, cb, 1.0f });
    const std::string head = (ask ? "CLAUDE " : "COMMENT ") + tc + " > ";
    gui::text(o, 14.0f, y0 + 13.0f, head, px, cr, cg, cb);
    const float x = 14.0f + gui::textWidth(head.size(), px) + 6.0f * px;
    const size_t room = (size_t)std::max(1.0f, ((float)w - 30.0f - x) / (6.0f * px));
    std::string shown;
    for (wchar_t c : typed) shown += (char)(c >= 32 && c < 127 ? c : '?');
    if (shown.size() > room) shown = shown.substr(shown.size() - room);
    gui::text(o, x, y0 + 13.0f, shown, px, 1.0f, 1.0f, 1.0f);
    const float xe = x + (shown.empty() ? 0.0f : gui::textWidth(shown.size(), px) + px);
    o.push_back({ xe, y0 + 11.0f, xe + 5.0f * px, y0 + 29.0f, 1.0f, 1.0f, 1.0f, 0.9f });          // the cursor
    const char* help = ask ? "ENTER SEND   ESC CANCEL   DRAG ON THE PICTURE TO DRAW   BACKSPACE: LAST STROKE AWAY"
                           : "ENTER SAVE   ESC CANCEL   DRAG ON THE PICTURE TO DRAW";
    if (typed.empty()) gui::text(o, xe + 24.0f, y0 + 13.0f, help, px, 0.5f, 0.5f, 0.5f);
}

// ------------------------------------------------------------------------------------------------
// the output: a window that fills one display (the HDMI output that feeds the media server)
// ------------------------------------------------------------------------------------------------

struct Display {
    std::wstring device;                    // \\.\DISPLAY2
    RECT rc = {};                           // where it is on the desktop, in its own pixels
    int hz = 0;
    int w() const { return rc.right - rc.left; }
    int h() const { return rc.bottom - rc.top; }
};

// The displays of the machine, in the order of their names: 1, 2 ... in the OUTPUT panel. Asked as a thread
// that knows every display has its own scaling: the preview window does not, and is told sizes that are
// not pixels.
std::vector<Display> displays()
{
    std::vector<Display> out;
    DPI_AWARENESS_CONTEXT old = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR m, HDC, LPRECT, LPARAM p) -> BOOL {
        MONITORINFOEXW mi = {};
        mi.cbSize = sizeof mi;
        if (!GetMonitorInfoW(m, &mi)) return TRUE;
        DEVMODEW dm = {};
        dm.dmSize = sizeof dm;
        EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm);
        Display d;
        d.device = mi.szDevice;
        d.rc = mi.rcMonitor;
        d.hz = (int)dm.dmDisplayFrequency;
        ((std::vector<Display>*)p)->push_back(d);
        return TRUE;
    }, (LPARAM)&out);
    if (old) SetThreadDpiAwarenessContext(old);
    auto number = [](const std::wstring& s) {
        size_t k = s.find_last_not_of(L"0123456789");
        return k + 1 < s.size() ? _wtoi(s.c_str() + k + 1) : 0;
    };
    std::sort(out.begin(), out.end(), [&](const Display& a, const Display& b) { return number(a.device) < number(b.device); });
    return out;
}

// The display a window is on.
std::wstring displayOf(HWND h)
{
    MONITORINFOEXW mi = {};
    mi.cbSize = sizeof mi;
    return GetMonitorInfoW(MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST), &mi) ? mi.szDevice : L"";
}

// The window of the output: no border, no mouse pointer, above everything on its display, never the window
// the keyboard goes to. What it shows is Renderer::output: the raster, pixel for pixel.
struct OutWindow {
    HWND hwnd = nullptr;
    ComPtr<IDXGISwapChain1> swap;
    ComPtr<ID3D11Texture2D> back;
    ComPtr<ID3D11RenderTargetView> rtv;
    Display on;                             // the display it fills
    bool exact = false;                     // the raster fits in it: pixel for pixel
    bool lost = false;

    static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM wp, LPARAM lp)
    {
        switch (m) {
        case WM_SETCURSOR: SetCursor(nullptr); return TRUE;
        case WM_MOUSEACTIVATE: return MA_NOACTIVATE;
        case WM_CLOSE: return 0;            // only the engine closes it
        case WM_ERASEBKGND: return 1;
        }
        return DefWindowProcW(h, m, wp, lp);
    }

    bool open(Gpu& g, const Display& d, std::string& err)
    {
        close();
        WNDCLASSW wc = {};
        wc.lpfnWndProc = proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"MuonBloomOutput";
        RegisterClassW(&wc);
        // made as a window that counts in pixels, whatever the scaling of its display
        DPI_AWARENESS_CONTEXT old = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, wc.lpszClassName, L"MUON : BLOOM output", WS_POPUP,
                               d.rc.left, d.rc.top, d.w(), d.h(), nullptr, nullptr, wc.hInstance, nullptr);
        if (old) SetThreadDpiAwarenessContext(old);
        if (!hwnd) { err = "cannot create the window of the output"; return false; }
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = (UINT)d.w();
        sd.Height = (UINT)d.h();
        sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Scaling = DXGI_SCALING_NONE;
        if (FAILED(g.factory->CreateSwapChainForHwnd(g.dev.Get(), hwnd, &sd, nullptr, nullptr, &swap))
            || FAILED(swap->GetBuffer(0, IID_PPV_ARGS(&back))) || FAILED(g.dev->CreateRenderTargetView(back.Get(), nullptr, &rtv))) {
            close();
            err = "cannot create the swap chain of the output";
            return false;
        }
        g.factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_WINDOW_CHANGES);
        on = d;
        lost = false;
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        return true;
    }

    void close()
    {
        rtv.Reset();
        back.Reset();
        swap.Reset();
        if (hwnd) DestroyWindow(hwnd);
        hwnd = nullptr;
    }

    int moveX = 0, moveY = 0;               // the MOVE of the OUTPUT panel: what is sent, shifted by that many pixels

    void present(Renderer& r, const LiveOptions& lo)
    {
        if (!hwnd || lost) return;
        exact = r.output(back.Get(), rtv.Get(), on.w(), on.h(), lo.rasterW, lo.rasterH, lo.rasterX, lo.rasterY, lo.picX, lo.picY, moveX, moveY);
        HRESULT hr = swap->Present(0, 0);
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) lost = true;
    }
};

// The OUTPUT panel, above the bar (in the preview window only): what is sent out, and how.
//     LIFT       the mid levels of the picture raised (Renderer::lift): a slider, 0 at its left; a click on the name: 0
//     TEST CARD  the test card instead of the show
//     GLOW W     the glow of the white: how much of the glow of the scenes is kept (1: all of it, 0: none, 2: twice)
//     GLOW R     the same for the red, on its own
//     RED        gain of the red, before the tonemap (1: as the scenes give it): thin red lines come up, full red stays
//     WEIGHT     pixels added to the width of every line (0: none; a hairline starts to gain from 0.3)
//                a click on a name: back to what the scenes give
//     MOVE       what is sent to the display, shifted a pixel at a time (Shift: ten), to sit the picture on the wall;
//                0 puts it back. Only the output moves: not the preview, not Spout
//     OUTPUT     OFF, AUTO, or the display that takes the raster (the one this window is on cannot be chosen).
//                AUTO: the other display of the machine, whichever it is - taken as soon as it is plugged in,
//                let go when it is taken away, on any machine (nothing of this one is kept in output.json)
struct OutputPanel {
    struct Hit { float x0, y0, x1, y1; int id; };
    bool open = false;
    int drag = 0;                           // the slider being dragged: 2 LIFT, 31 GLOW W, 33 GLOW R, 35 RED, 37 WEIGHT
    std::vector<Hit> hits;                  // 1 the name LIFT, 2 its slider, 3 TEST CARD, 4 .. 8 MOVE left right up down 0, 9 AUTO, 10 OFF, 11 .. the displays
    std::vector<Display> list;              // the displays, as shown

    int at(int mx, int my) const
    {
        for (auto& h : hits) if (mx >= h.x0 && mx < h.x1 && my >= h.y0 && my < h.y1) return h.id;
        return 0;
    }

    void rects(std::vector<Renderer::Over>& o, int w, int h, int bar, float lift, bool card, const std::string& cardNote, const std::wstring& active,
               bool gone, bool exact, const std::wstring& own, const LiveOptions& lo, int mx, int my, int moveX, int moveY, float glow, float glowRed, float red, float weight,
               bool autoOn)
    {
        const float y1 = (float)(h - bar), y0 = y1 - PANEL_H, px = 2.0f;
        hits.clear();
        o.push_back({ 0.0f, y0, (float)w, y1, 0.0f, 0.0f, 0.0f, 0.92f });
        o.push_back({ 0.0f, y0, (float)w, y0 + 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        auto button = [&](float x, float y, const std::string& s, int id, bool lit, bool dead = false) {
            const float bw = gui::textWidth(s.size(), px) + 20.0f, bh = 28.0f;
            const bool hot = !dead && mx >= x && mx < x + bw && my >= y && my < y + bh;
            const float g = lit ? 1.0f : hot ? 0.75f : 0.3f, in = lit ? 1.0f : hot ? 0.14f : 0.0f, t = lit ? 0.0f : dead ? 0.4f : 0.85f;
            o.push_back({ x, y, x + bw, y + bh, g, g, g, 1.0f });
            o.push_back({ x + 1.0f, y + 1.0f, x + bw - 1.0f, y + bh - 1.0f, in, in, in, 1.0f });
            gui::text(o, x + 10.0f, y + 7.0f, s, px, t, t, t);
            if (!dead) hits.push_back({ x, y, x + bw, y + bh, id });
            return x + bw + 10.0f;
        };
        // the lift and the test card
        float y = y0 + 10.0f;
        char b[96];
        snprintf(b, sizeof b, "LIFT %.2f", lift);
        gui::text(o, 14.0f, y + 7.0f, b, px, 1.0f, 1.0f, 1.0f);
        hits.push_back({ 8.0f, y, SLIDER_X0 - 16.0f, y + 28.0f, 1 });
        const float xk = SLIDER_X0 + SLIDER_W * std::clamp(lift / LIFT_MAX, 0.0f, 1.0f);
        o.push_back({ SLIDER_X0, y + 13.0f, SLIDER_X0 + SLIDER_W, y + 15.0f, 0.35f, 0.35f, 0.35f, 1.0f });
        o.push_back({ SLIDER_X0, y + 13.0f, xk, y + 15.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        o.push_back({ xk - 3.0f, y + 4.0f, xk + 3.0f, y + 24.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        hits.push_back({ SLIDER_X0 - 10.0f, y, SLIDER_X0 + SLIDER_W + 10.0f, y + 28.0f, 2 });
        float x = button(SLIDER_X0 + SLIDER_W + 40.0f, y, "TEST CARD", 3, card);
        snprintf(b, sizeof b, "MOVE %d, %d", moveX, moveY);
        gui::text(o, x + 30.0f, y + 7.0f, b, px, 1.0f, 1.0f, 1.0f);
        x = button(x + 30.0f + gui::textWidth(16, px), y, "<", 4, false);
        x = button(x, y, ">", 5, false);
        x = button(x, y, "UP", 6, false);
        x = button(x, y, "DOWN", 7, false);
        x = button(x, y, "0", 8, false, !moveX && !moveY);
        const float xm = x;                 // (where the first row ends)
        if (!cardNote.empty()) gui::text(o, x + 6.0f, y + 7.0f, cardNote, px, 1.0f, 0.85f, 0.0f);
        // glow of the white, glow of the red, red, line weight: four sliders
        y = y0 + 94.0f;
        const struct { const char* name; float v, vmax; int id; } tune[4] = { { "GLOW W", glow, GLOW_MAX, 30 }, { "GLOW R", glowRed, GLOW_MAX, 32 },
                                                                             { "RED", red, RED_MAX, 34 }, { "WEIGHT", weight, WEIGHT_MAX, 36 } };
        for (int k = 0; k < 4; k++) {
            const float xl = 14.0f + k * TUNE_PITCH, xs = TUNE_X0 + k * TUNE_PITCH, xk = xs + TUNE_W * std::clamp(tune[k].v / tune[k].vmax, 0.0f, 1.0f);
            snprintf(b, sizeof b, "%s %.2f", tune[k].name, tune[k].v);
            gui::text(o, xl, y + 7.0f, b, px, 1.0f, 1.0f, 1.0f);
            hits.push_back({ xl - 6.0f, y, xs - 16.0f, y + 28.0f, tune[k].id });
            o.push_back({ xs, y + 13.0f, xs + TUNE_W, y + 15.0f, 0.35f, 0.35f, 0.35f, 1.0f });
            o.push_back({ xs, y + 13.0f, xk, y + 15.0f, 1.0f, 1.0f, 1.0f, 1.0f });
            o.push_back({ xk - 3.0f, y + 4.0f, xk + 3.0f, y + 24.0f, 1.0f, 1.0f, 1.0f, 1.0f });
            hits.push_back({ xs - 10.0f, y, xs + TUNE_W + 10.0f, y + 28.0f, tune[k].id + 1 });
        }
        // the display of the output
        y = y0 + 52.0f;
        gui::text(o, 14.0f, y + 7.0f, "OUTPUT", px, 1.0f, 1.0f, 1.0f);
        x = button(110.0f, y, "OFF", 10, active.empty() && !autoOn);
        x = button(x, y, "AUTO", 9, autoOn);
        for (size_t k = 0; k < list.size(); k++) {
            const bool self = list[k].device == own;
            snprintf(b, sizeof b, "%d: %d x %d %d HZ%s", (int)k + 1, list[k].w(), list[k].h(), list[k].hz, self ? " (THIS SCREEN)" : "");
            x = button(x, y, b, 11 + (int)k, list[k].device == active && !gone, self);
        }
        int hz = 60;
        for (auto& d : list) if (d.device == active && d.hz) hz = d.hz;
        if (autoOn && active.empty()) gui::text(o, x + 6.0f, y + 7.0f, "AUTO: WAITING FOR ANOTHER DISPLAY", px, 1.0f, 0.85f, 0.0f);
        else if (gone) gui::text(o, x + 6.0f, y + 7.0f, "THE DISPLAY OF THE OUTPUT IS GONE", px, 1.0f, 0.85f, 0.0f);
        else if (!active.empty() && !exact) gui::text(o, x + 6.0f, y + 7.0f, "SMALLER THAN THE RASTER: SCALED TO FIT", px, 1.0f, 0.85f, 0.0f);
        else if (!active.empty() && hz < 59) gui::text(o, x + 6.0f, y + 7.0f, "UNDER 60 HZ: FRAMES OF THE SHOW ARE LOST", px, 1.0f, 0.85f, 0.0f);
        else if (!active.empty()) gui::text(o, x + 6.0f, y + 7.0f, "PIXEL FOR PIXEL", px, 0.8f, 0.8f, 0.8f);
        snprintf(b, sizeof b, "RASTER %d x %d, PICTURE AT %d, %d", lo.rasterW, lo.rasterH, lo.picX, lo.picY);
        const float tw = gui::textWidth(strlen(b), px);
        if ((float)w - 14.0f - tw > xm + 40.0f) gui::text(o, (float)w - 14.0f - tw, y0 + 17.0f, b, px, 0.5f, 0.5f, 0.5f);
    }
};

// The TOWERS panel, above the bar (in the preview window only): the placement tool (key T) with its numbers.
// The three towers are drawn over the picture - on the wall too - and can be dragged there. Here, for the
// tower that is chosen, in pixels of the 2978 x 1400 canvas:
//     X          where its centre is
//     WIDTH      its width, about its centre
//     HEIGHT     its height: the foot stays on the ground
//     DETECTOR   the height of the detector at its head
//                a click: one pixel, ten with Shift; a button kept down goes on
//     SAVE       writes data/towers.json: the scenes are built again around that placement
//     CANCEL     back to the placement as it is saved, and the tool away
//     CALIBRATION  the calibration picture instead of the show (TowerTool::pattern), in the output too: the
//                towers as blocks of light on black, to be moved until they sit on the real ones
struct TowerPanel {
    struct Hit { float x0, y0, x1, y1; int id; };
    std::vector<Hit> hits;                  // 1 .. 3 the towers, 10 11 X, 12 13 WIDTH, 14 15 HEIGHT, 16 17 DETECTOR, 20 SAVE, 21 CANCEL, 22 CALIBRATION
    int held = 0;                           // the button kept down
    double heldAt = 0, stepAt = 0;

    int at(int mx, int my) const
    {
        for (auto& h : hits) if (mx >= h.x0 && mx < h.x1 && my >= h.y0 && my < h.y1) return h.id;
        return 0;
    }

    void rects(std::vector<Renderer::Over>& o, int w, int h, int bar, const TowerTool& tool, int mx, int my, bool outputOn)
    {
        const float y1 = (float)(h - bar), y0 = y1 - PANEL_H, px = 2.0f;
        hits.clear();
        o.push_back({ 0.0f, y0, (float)w, y1, 0.0f, 0.0f, 0.0f, 0.92f });
        o.push_back({ 0.0f, y0, (float)w, y0 + 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        auto button = [&](float x, float y, const std::string& s, int id, bool lit, bool dead = false) {
            const float bw = gui::textWidth(s.size(), px) + 20.0f, bh = 28.0f;
            const bool hot = !dead && mx >= x && mx < x + bw && my >= y && my < y + bh;
            const float g = lit ? 1.0f : hot ? 0.75f : 0.3f, in = lit ? 1.0f : hot ? 0.14f : 0.0f, t = lit ? 0.0f : dead ? 0.4f : 0.85f;
            o.push_back({ x, y, x + bw, y + bh, g, g, g, 1.0f });
            o.push_back({ x + 1.0f, y + 1.0f, x + bw - 1.0f, y + bh - 1.0f, in, in, in, 1.0f });
            gui::text(o, x + 10.0f, y + 7.0f, s, px, t, t, t);
            if (!dead) hits.push_back({ x, y, x + bw, y + bh, id });
            return x + bw + 10.0f;
        };
        // which tower, and what becomes of the changes
        float y = y0 + 10.0f;
        gui::text(o, 14.0f, y + 7.0f, "TOWER", px, 1.0f, 1.0f, 1.0f);
        const char* names[3] = { "LEFT", "CENTRE", "RIGHT" };
        float x = 110.0f;
        for (int k = 0; k < 3; k++) x = button(x, y, names[k], 1 + k, k == tool.sel);
        x = button(x + 30.0f, y, "SAVE", 20, false, !tool.edited);
        x = button(x, y, "CANCEL", 21, false);
        x = button(x + 30.0f, y, "CALIBRATION", 22, tool.calib);
        if (tool.calib && !outputOn)
            gui::text(o, x + 6.0f, y + 7.0f, "NO OUTPUT: CHOOSE ITS DISPLAY IN THE OUTPUT PANEL", px, 1.0f, 0.85f, 0.0f);
        else if (tool.tw.widestBay() < TowerTool::MIN_BAY)
            gui::text(o, x + 6.0f, y + 7.0f, "NO 3 M OF FREE WALL LEFT: CANNOT BE SAVED", px, 1.0f, 0.85f, 0.0f);
        else if (tool.edited)
            gui::text(o, x + 6.0f, y + 7.0f, "CHANGED: SAVE BUILDS THE SCENES AROUND IT", px, 1.0f, 0.85f, 0.0f);
        // its numbers, each with its two buttons
        const Tower& a = tool.tw.t[tool.sel];
        auto value = [&](float x, float y, const char* name, float v, int id) {
            char b[64];
            snprintf(b, sizeof b, "%s %.1f", name, v);
            gui::text(o, x, y + 7.0f, b, px, 1.0f, 1.0f, 1.0f);
            x = button(x + gui::textWidth(16, px), y, id == 10 ? "<" : "-", id, false);
            button(x, y, id == 10 ? ">" : "+", id + 1, false);
        };
        value(14.0f, y0 + 52.0f, "X", 0.5f * (a.x0 + a.x1), 10);
        value(14.0f + COL, y0 + 52.0f, "WIDTH", a.x1 - a.x0, 12);
        value(14.0f, y0 + 94.0f, "HEIGHT", a.bot - a.top, 14);
        value(14.0f + COL, y0 + 94.0f, "DETECTOR", a.det_h, 16);
        const char* help = "PIXELS OF THE 2978 x 1400 CANVAS   SHIFT: TEN   OR DRAG A TOWER ON THE PICTURE";
        const float tw = gui::textWidth(strlen(help), px), xh = 14.0f + 2.0f * COL;
        if (xh + tw < (float)w - 14.0f) gui::text(o, xh, y0 + 59.0f, help, px, 0.5f, 0.5f, 0.5f);
    }
    static constexpr float COL = 340.0f;    // from one number to the next
};

// What was set in the OUTPUT panel, kept from one run to the next: engine/output.json
//     { "lift": 0.60, "display": "\\\\.\\DISPLAY2", "move": [0, 0], "glow": 1.00, "glow_red": 1.00, "red": 1.00, "weight": 0.00 }
// "glow" is the glow of the white; a file written before "glow_red" gives its "glow" to both.
// "display": the device name of the display that takes the raster, "" for none, or "auto" (the OUTPUT panel).
bool loadOutput(const fs::path& path, float& lift, std::wstring& display, int& moveX, int& moveY, float& glow, float& glowRed, float& red, float& weight)
{
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    size_t a = s.find("\"lift\""), c = a == std::string::npos ? a : s.find(':', a);
    if (c != std::string::npos) {
        const double v = atof(s.c_str() + c + 1);
        if (v >= 0.0 && v <= LIFT_MAX) lift = (float)v;
    }
    struct { const char* key; float* v; float vmax; } num[4] = { { "\"glow\"", &glow, GLOW_MAX }, { "\"glow_red\"", &glowRed, GLOW_MAX },
                                                                 { "\"red\"", &red, RED_MAX }, { "\"weight\"", &weight, WEIGHT_MAX } };
    if (s.find("\"glow_red\"") == std::string::npos) num[1].key = "\"glow\"";      // (an older file)
    for (auto& n : num) {
        a = s.find(n.key);
        c = a == std::string::npos ? a : s.find(':', a);
        if (c == std::string::npos) continue;
        const double v = atof(s.c_str() + c + 1);
        if (v >= 0.0 && v <= n.vmax) *n.v = (float)v;
    }
    a = s.find("\"move\"");
    c = a == std::string::npos ? a : s.find('[', a);
    if (c != std::string::npos) {
        int mx = 0, my = 0;
        if (sscanf(s.c_str() + c, "[ %d , %d", &mx, &my) == 2) {
            moveX = std::clamp(mx, -MOVE_MAX, MOVE_MAX);
            moveY = std::clamp(my, -MOVE_MAX, MOVE_MAX);
        }
    }
    a = s.find("\"display\"");
    c = a == std::string::npos ? a : s.find(':', a);
    const size_t q0 = c == std::string::npos ? c : s.find('"', c), q1 = q0 == std::string::npos ? q0 : s.find('"', q0 + 1);
    if (q1 != std::string::npos) {
        display.clear();
        for (size_t k = q0 + 1; k < q1; k++) {
            if (s[k] == '\\' && k + 1 < q1) k++;
            display += (wchar_t)(unsigned char)s[k];
        }
    }
    return true;
}

bool saveOutput(const fs::path& path, float lift, const std::wstring& display, int moveX, int moveY, float glow, float glowRed, float red, float weight)
{
    fs::path tmp = path;
    tmp += L".tmp";
    {
        std::ofstream f(tmp);
        if (!f) return false;
        std::string d;
        for (wchar_t ch : display) {
            if (ch == L'\\') d += '\\';
            d += (char)(ch < 128 ? ch : '?');
        }
        char b[128];
        snprintf(b, sizeof b, "{\n  \"lift\": %.2f,\n  \"display\": \"", lift);
        f << b << d << "\",\n  \"move\": [" << moveX << ", " << moveY << "]";
        snprintf(b, sizeof b, ",\n  \"glow\": %.2f,\n  \"glow_red\": %.2f,\n  \"red\": %.2f,\n  \"weight\": %.2f\n}\n", glow, glowRed, red, weight);
        f << b;
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    return !ec;
}

}  // namespace

// The time of the show, given by whoever plays the sound (Ableton, through TouchDesigner):
//     /muonbloom/time <seconds>      sent all the time, many times a second
// That time is not steady: Ableton counts in beats, and what it gives in seconds comes in uneven steps and
// wobbles when the tempo changes. So the show does not take it message by message. It runs on the machine's
// own timer, which is smooth; once a second it looks how far it is from the times received, and runs a
// little faster or slower (never more than SLEW) until it is there again. The picture is never stepped
// while it plays, except for a locate.
//     the time moves          the show plays, from that time
//     the time stands still   the sender has stopped: the show pauses there
//     the time is far away    and stays there: a locate, the show is moved (one odd value is not followed)
//     nothing arrives         the sender is gone: the show goes on by the machine's timer (a picture that
//                             freezes because a cable fell out is worse than one that drifts)
struct ExtClock {
    static constexpr double JUMP = 0.08;        // paused, and further than this from the time received: moved there
    static constexpr double STALL = 0.4;        // the time has not moved for this long, and still arrives: paused
    static constexpr double SILENT = 1.5;       // nothing has arrived for this long: the sender is gone
    static constexpr double SYNC = 1.0;         // the clock is compared with the times received this often
    static constexpr double CATCH = 2.0;        // ... and set to be back on them after this long
    static constexpr double SLEW = 0.05;        // ... but never more than this faster or slower than the machine
    static constexpr double CLOSE = 0.003;      // nearer than this: left alone
    static constexpr double LOCATE = 0.5;       // further than this from the time received ...
    static constexpr double SAME = 0.25;        // ... by the same amount ...
    static constexpr double HOLD = 0.3;         // ... for this long: the sender is elsewhere in the show
    bool heard = false;                         // a time has arrived at least once
    bool following = false;                     // the show is running on the times received
    bool stalled = false;                       // ... and they stand still: paused there
    double last = -1e18, movedAt = 0, heardAt = 0;
    uint64_t n = 0, jumps = 0;                  // times followed, and times the show was moved
    double errSum = 0, errWorst = 0;            // received - clock at arrival, once locked
    double lockedAt = 0;
    std::vector<double> errs;                   // received - clock, since the last comparison
    double syncAt = 0;
    double farSince = 0, farErr = 0;            // far away since then, by that much (0: not far)
    bool catching = false;                      // the clock is being brought back from more than a frame away

    void fresh(double w)                        // the clock was just put on a time received
    {
        errs.clear();
        syncAt = lockedAt = w;
        farSince = 0;
        catching = false;
    }
    // How far the clock is from the times received since the last comparison. The messages arrive late,
    // never early: the upper quarter is taken, and a few odd values on either side do not count.
    bool error(double& e)
    {
        if (errs.empty()) return false;
        size_t k = (errs.size() * 3) / 4;
        std::nth_element(errs.begin(), errs.begin() + k, errs.end());
        e = errs[k];
        errs.clear();
        return true;
    }
};

// The trigger levels of the detectors, kept from one run to the next: engine/detectors.json
//     { "level": { "L": 0.10, "C": 0.10, "R": 0.10 } }
static bool loadLevels(const fs::path& path, float lv[3])
{
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    size_t a = s.find("\"level\"");
    if (a == std::string::npos) return false;
    const char* keys[3] = { "\"L\"", "\"C\"", "\"R\"" };
    for (int k = 0; k < 3; k++) {
        size_t p = s.find(keys[k], a);
        if (p == std::string::npos) return false;
        double v = atof(s.c_str() + s.find(':', p) + 1);
        if (v > 0.0 && v <= 1.0) lv[k] = (float)v;
    }
    return true;
}

static bool saveLevels(const fs::path& path, const float lv[3])
{
    fs::path tmp = path;
    tmp += L".tmp";
    {
        std::ofstream f(tmp);
        if (!f) return false;
        char b[160];
        snprintf(b, sizeof b, "{\n  \"level\": { \"L\": %.3f, \"C\": %.3f, \"R\": %.3f }\n}\n", lv[0], lv[1], lv[2]);
        f << b;
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    return !ec;
}

// What the window shows of the detectors (never the Spout output): the value of each one against its
// trigger level, to set that level by eye. The hits it lights are found the way engine/detectors.py finds them.
struct DetMeter {
    float last[3] = { 0, 0, 0 }, peak[3] = { 0, 0, 0 };
    double peakAt[3] = { 0, 0, 0 }, hitAt[3] = { -9, -9, -9 }, heardAt[3] = { -9, -9, -9 };
    bool armed[3] = { true, true, true };
    uint64_t hits[3] = { 0, 0, 0 };

    void feed(int k, float v, double w, float level)
    {
        if (w - heardAt[k] > 0.25) armed[k] = true;            // (detectors.py: REARM)
        if (armed[k] && v >= level) { hitAt[k] = w; hits[k]++; armed[k] = false; }
        else if (!armed[k] && v < level) armed[k] = true;              // (detectors.py: RELEASE)
        if (v >= peak[k] || w - peakAt[k] > 1.0) { peak[k] = v; peakAt[k] = w; }
        last[k] = v;
        heardAt[k] = w;
    }

    // Three rows at the top left of the window, L C R from the top: the value (red above the trigger level),
    // its peak of the last second, the trigger level (yellow), a lamp at each hit; a mark on the selected row.
    void rects(std::vector<Renderer::Over>& o, const float level[3], int sel, double w) const
    {
        const float x0 = 28.0f, W = 260.0f, x1 = x0 + W, H = 12.0f;
        o.push_back({ x0 - 21.0f, 7.0f, x1 + 35.0f, 8.0f + 3 * 20.0f + 7.0f, 0.45f, 0.45f, 0.45f, 1.0f });
        o.push_back({ x0 - 20.0f, 8.0f, x1 + 34.0f, 8.0f + 3 * 20.0f + 6.0f, 0.0f, 0.0f, 0.0f, 1.0f });
        for (int k = 0; k < 3; k++) {
            float y = 16.0f + 20.0f * k;
            float v = w - heardAt[k] > 1.5 ? 0.0f : std::clamp(last[k], 0.0f, 1.0f);       // silent: nothing to show
            float pk = w - peakAt[k] > 1.5 ? v : std::clamp(peak[k], 0.0f, 1.0f);
            bool over = v >= level[k];
            o.push_back({ x0, y, x1, y + H, 0.22f, 0.22f, 0.22f, 1.0f });
            o.push_back({ x0, y, x0 + v * W, y + H, 1.0f, over ? 0.12f : 1.0f, over ? 0.08f : 1.0f, 1.0f });
            o.push_back({ x0 + pk * W - 1.0f, y, x0 + pk * W + 1.0f, y + H, 0.75f, 0.75f, 0.75f, 1.0f });
            o.push_back({ x0 + level[k] * W - 1.0f, y - 3.0f, x0 + level[k] * W + 1.0f, y + H + 3.0f, 1.0f, 0.85f, 0.0f, 1.0f });
            bool lamp = w - hitAt[k] < 0.15;
            o.push_back({ x1 + 10.0f, y, x1 + 10.0f + H, y + H, lamp ? 1.0f : 0.22f, lamp ? 0.12f : 0.22f, lamp ? 0.08f : 0.22f, 1.0f });
            if (sel == k || sel < 0) o.push_back({ x0 - 12.0f, y + 3.0f, x0 - 6.0f, y + H - 3.0f, 1.0f, 0.85f, 0.0f, 1.0f });
        }
    }
};

// Which detector the last part of an OSC address is about: L, C, R (or left / centre / right, or 1 2 3).
static int detectorOf(const std::string& last)
{
    static const char* names[3][5] = { { "L", "l", "left", "1", "DET_L" }, { "C", "c", "centre", "2", "DET_C" }, { "R", "r", "right", "3", "DET_R" } };
    for (int k = 0; k < 3; k++)
        for (const char* n : names[k]) if (last == n) return k;
    return last == "center" ? 1 : -1;
}

int runLive(const PlayerOptions& options, const LiveOptions& lo)
{
    PlayerOptions o = options;
    if (lo.picX < 0 || lo.picY < 0 || lo.picX + PIC_W > lo.rasterW || lo.picY + PIC_H > lo.rasterH) {
        fprintf(stderr, "the picture of the show (%d x %d) does not fit in a %d x %d raster at %d, %d (--raster, --picture-at)\n", PIC_W, PIC_H,
                lo.rasterW, lo.rasterH, lo.picX, lo.picY);
        return 2;
    }
    if (!lo.log.empty()) {
        std::error_code ec;
        fs::create_directories(fs::path(lo.log).parent_path(), ec);
        gLog = _wfopen(lo.log.c_str(), L"a");
    }
    say("muonengine live: root %s, shaders %s", narrow(o.root).c_str(), narrow(o.shaders).c_str());

    // nobody is at the keyboard for hours: the machine must not go to sleep or switch its screens off, and
    // a click in the console (text selection) must not suspend the engine
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    {
        HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode;
        if (GetConsoleMode(in, &mode)) SetConsoleMode(in, (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS);
    }

    DetectorRing ring;
    bool liveDet = lo.detectors == "live" || lo.detectors == "both";
    if (liveDet) {
        wchar_t name[64];
        swprintf(name, 64, L"muonengine-%lu-det", GetCurrentProcessId());
        if (!ring.create(name)) { warn("cannot create the shared memory of the detectors"); return 2; }
        o.workerArgs = std::wstring(L" --det ") + name + L" --det-mode " + std::wstring(lo.detectors.begin(), lo.detectors.end());
    }
    // the trigger levels: the command line, else what was set the last time, else 0.10
    const fs::path levelFile = fs::path(o.root) / L"engine" / L"detectors.json";
    float level[3] = { 0.10f, 0.10f, 0.10f };
    const bool levelsFromFile = loadLevels(levelFile, level);
    for (int k = 0; k < 3; k++) if (lo.detLevel[k] > 0.0f) level[k] = std::min(lo.detLevel[k], 1.0f);
    if (liveDet) {
        ring.levels(level);
        say("detectors: trigger level L %.3f  C %.3f  R %.3f%s", level[0], level[1], level[2],
            lo.detLevel[0] > 0.0f ? " (--det-level)" : levelsFromFile ? " (engine/detectors.json)" : "");
    }
    DetMeter meter;
    int levelSel = -1;                              // detector the Up / Down keys act on (-1: the three)
    bool panel = liveDet;                           // the detector meters over the preview
    double panelAt = 0;
    std::string note;
    auto setLevels = [&](int which, float v, bool relative) {
        if (!liveDet) { note = "the detectors are scripted: start with --detectors live to set a trigger level"; return; }
        if (!std::isfinite(v)) return;
        for (int k = 0; k < 3; k++)
            if (which < 0 || which == k) level[k] = std::clamp(relative ? level[k] + v : v, 0.005f, 1.0f);
        ring.levels(level);
        if (!saveLevels(levelFile, level)) warn("cannot write %s", narrow(levelFile.wstring()).c_str());
        say("detectors: trigger level L %.3f  C %.3f  R %.3f", level[0], level[1], level[2]);
    };
    OscIn osc;
    if (lo.oscPort > 0) {
        if (osc.open(lo.oscPort))
            say("OSC: listening on UDP port %d (detectors: %s, at %s/L /C /R; transport orders from %s)", lo.oscPort, lo.detectors.c_str(),
                lo.detPrefix.c_str(), lo.oscAllow.empty() ? "anybody" : lo.oscAllow.size() == 1 && lo.oscAllow[0] == 0x7F000001u ? "this machine only" : "the allowed addresses");
        else warn("cannot listen on UDP port %d: no OSC", lo.oscPort);
    }
    std::vector<OscMsg> msgs;

    // The sound: files that cannot be read are a mistake in the set-up (stop); an output that cannot be
    // opened may come back (go on without sound, and keep trying).
    Audio audio;
    bool wantAudio = !lo.audio.empty();             // (the SOUND button changes it while the show runs)
    std::string err;
    if (wantAudio) {
        for (auto& f : lo.audio) say("sound: %s", narrow(f).c_str());
        if (!audio.load(lo.audio, err)) { warn("%s", err.c_str()); return 2; }
        if (!audio.openDevice()) warn("NO SOUND OUTPUT: playing without sound, trying again every few seconds");
    } else {
        say("no sound from the engine (Ableton plays it): the clock is /muonbloom/time if it arrives, else the machine's timer");
    }
    audio.volume(lo.volume);

    Player p;
    if (!p.start(o, err)) { warn("%s", err.c_str()); return 2; }
    say("gpu: %s", narrow(p.gpu.name).c_str());

    Window win;
    if (!win.create(p.gpu, 1489, 700 + (lo.bar ? BAR_H : 0), err)) { warn("%s", err.c_str()); return 2; }
    win.bar = lo.bar ? BAR_H : 0;

    spoutDX spout;
    bool spoutOn = false;
    if (!lo.spout.empty()) {
        spoutOn = spout.OpenDirectX11(p.gpu.dev.Get()) && spout.SetSenderName(lo.spout.c_str());
        if (spoutOn) say("Spout sender: %s", lo.spout.c_str());
        else warn("cannot open the Spout sender %s", lo.spout.c_str());
    }

    TowerTool tool;
    tool.tw.path = fs::path(o.root) / L"data" / L"towers.json";
    if (lo.towers) tool.start();
    TowerPanel towerPanel;

    // ---- the output: the lift, the test card, the display that takes the raster (the OUTPUT panel) ----
    const fs::path outputFile = fs::path(o.root) / L"engine" / L"output.json";
    float lift = 0.0f;
    std::wstring outDevice;                         // the display of the output ("" = none)
    OutWindow out;
    float glow = 1.0f, glowRed = 1.0f, red = 1.0f, weight = 0.0f;   // GLOW W, GLOW R, RED and WEIGHT of the panel
    loadOutput(outputFile, lift, outDevice, out.moveX, out.moveY, glow, glowRed, red, weight);
    bool outAuto = outDevice == L"auto";            // AUTO: the display is whichever other one the machine has
    if (outAuto) outDevice.clear();
    std::wstring autoFailed;                        // (a display AUTO could not open: not tried again until the displays change)
    p.renderer.tune(glow, glowRed, red, weight);
    if (lo.lift >= 0.0f) lift = std::min(lo.lift, LIFT_MAX);
    p.renderer.lift(lift);
    OutputPanel outPanel;
    bool outGone = false;                           // that display is not there at the moment
    bool outDirty = false;                          // what is sent out has to be composed again
    bool settingsDirty = false;
    int panelMx = -1, panelMy = -1;
    // the test card: engine/out/testcard_<W>x<H>.bgra, made by tools/test_card.py - run from here when the card
    // is missing, or older than the tower placement it shows
    wchar_t cardName[64];
    swprintf(cardName, 64, L"testcard_%dx%d.bgra", lo.rasterW, lo.rasterH);
    const fs::path cardFile = fs::path(o.root) / L"engine" / L"out" / cardName;
    HANDLE cardMaker = nullptr;                     // tools/test_card.py at work
    bool cardWanted = lo.card;
    bool cardBroken = false;                        // it could not be made: not tried again by itself
    std::string cardNote;
    auto cardStale = [&] {
        std::error_code ec;
        const auto made = fs::last_write_time(cardFile, ec);
        if (ec) return true;
        for (const wchar_t* dep : { L"data/towers.json", L"tools/test_card.py" }) {
            const auto t = fs::last_write_time(fs::path(o.root) / dep, ec);
            if (!ec && t > made) return true;
        }
        return false;
    };
    auto makeCard = [&] {
        wchar_t cmd[2048];
        swprintf(cmd, 2048, L"\"%ls\" \"%ls\" --raster %dx%d --at %d,%d", o.python.c_str(), (fs::path(o.root) / L"tools" / L"test_card.py").c_str(),
                 lo.rasterW, lo.rasterH, lo.picX, lo.picY);
        STARTUPINFOW si = { sizeof si };
        PROCESS_INFORMATION pi = {};
        if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, FALSE, 0, nullptr, o.root.c_str(), &si, &pi)) return false;
        CloseHandle(pi.hThread);
        cardMaker = pi.hProcess;
        return true;
    };
    std::function<void(bool)> showCard = [&](bool on) {
        cardWanted = on;
        cardBroken = false;
        outDirty = true;
        if (!on) {
            if (p.renderer.cardOn()) say("test card off");
            p.renderer.card(false, 0, 0);
            cardNote.clear();
            return;
        }
        if (cardMaker || p.renderer.cardOn()) return;
        if (cardStale()) {
            if (makeCard()) cardNote = "MAKING IT ...";
            else { cardNote = "CANNOT RUN tools/test_card.py"; cardWanted = false; }
            return;
        }
        if (p.shown() < 0) { cardNote = "WITH THE FIRST PICTURE"; return; }        // (tried again then)
        if (!p.renderer.loadCard(cardFile.wstring(), lo.rasterW, lo.rasterH) || !p.renderer.card(true, lo.picX, lo.picY)) {
            cardNote = "THE CARD CANNOT BE READ";
            cardWanted = false;
        } else {
            cardNote.clear();
            say("test card on");
        }
    };
    auto setLift = [&](float v) {
        v = std::round(std::clamp(v, 0.0f, LIFT_MAX) * 100.0f) / 100.0f;
        if (v == lift) return;
        lift = v;
        p.renderer.lift(lift);
        outDirty = settingsDirty = true;
    };
    auto setOutput = [&](const std::wstring& device, bool byAuto = false) {
        out.close();
        outDevice = device;
        outAuto = byAuto;
        outGone = false;
        outDirty = true;
        if (device.empty()) { say("output: off"); return; }
        const std::vector<Display> list = displays();
        const auto it = std::find_if(list.begin(), list.end(), [&](const Display& d) { return d.device == device; });
        std::string e;
        if (it == list.end()) {
            outGone = true;
            warn("OUTPUT: the display %s is not there; it is taken as soon as it is", narrow(device).c_str());
        } else if (it->device == displayOf(win.hwnd)) {
            outDevice.clear();
            note = "THE OUTPUT CANNOT BE THE SCREEN THIS WINDOW IS ON";
            warn("output: %s is the screen of the preview window: not taken", narrow(device).c_str());
        } else if (!out.open(p.gpu, *it, e)) {
            if (byAuto) autoFailed = device;
            outDevice.clear();
            note = e;
            warn("output: %s", e.c_str());
        } else {
            const bool exact = lo.rasterX >= 0 && lo.rasterY >= 0 && lo.rasterX + lo.rasterW <= it->w() && lo.rasterY + lo.rasterH <= it->h();
            say("output: display %d (%s), %d x %d at %d Hz; the %d x %d raster at %d, %d of it, the picture at %d, %d of the raster%s", (int)(it - list.begin()) + 1,
                narrow(device).c_str(), it->w(), it->h(), it->hz, lo.rasterW, lo.rasterH, lo.rasterX, lo.rasterY, lo.picX, lo.picY,
                exact ? ": pixel for pixel" : ": THE DISPLAY IS SMALLER THAN THE RASTER, which is scaled to fit (not pixel for pixel)");
            if (it->hz && it->hz < 59) warn("OUTPUT: that display runs at %d Hz: the show is 60 frames a second, frames are lost on the way", it->hz);
        }
    };
    auto describe = [&](const std::vector<Display>& list) {
        const std::wstring own = displayOf(win.hwnd);
        std::string all;
        for (size_t k = 0; k < list.size(); k++) {
            char b[160];
            snprintf(b, sizeof b, "%s%d: %s %d x %d at %d Hz%s", k ? ", " : "", (int)k + 1, narrow(list[k].device).c_str(), list[k].w(), list[k].h(), list[k].hz,
                     list[k].device == own ? " (this window)" : "");
            all += b;
        }
        return all;
    };
    // AUTO: the display this window is not on; of several, one the raster fits in, then the largest
    auto autoDisplay = [&](const std::vector<Display>& list) {
        const std::wstring own = displayOf(win.hwnd);
        auto fits = [&](const Display& d) { return lo.rasterX >= 0 && lo.rasterY >= 0 && lo.rasterX + lo.rasterW <= d.w() && lo.rasterY + lo.rasterH <= d.h(); };
        const Display* best = nullptr;
        for (auto& d : list) {
            if (d.device == own || d.device == autoFailed) continue;
            if (!best || fits(d) > fits(*best) || (fits(d) == fits(*best) && (long long)d.w() * d.h() > (long long)best->w() * best->h())) best = &d;
        }
        return best ? best->device : std::wstring();
    };
    std::vector<Display> known = displays();        // the displays of the machine, as last seen
    {
        const std::vector<Display>& list = known;
        say("displays: %s", describe(list).c_str());
        if (lo.output == "off") { outDevice.clear(); outAuto = false; }
        else if (lo.output == "auto") { outDevice.clear(); outAuto = true; }
        else if (!lo.output.empty()) {              // a number of that list, or a device name
            const int n = atoi(lo.output.c_str());
            if (lo.output.find_first_not_of("0123456789") == std::string::npos && n >= 1 && n <= (int)list.size()) outDevice = list[n - 1].device;
            else outDevice.assign(lo.output.begin(), lo.output.end());
            outAuto = false;
        }
        if (!outDevice.empty()) setOutput(outDevice);
        else if (outAuto) {
            const std::wstring d = autoDisplay(list);
            if (!d.empty()) setOutput(d, true);
            else say("output: AUTO, and no other display yet: it is taken as soon as there is one");
        }
        if (lift > 0.0f) say("lift of the output: %.2f", lift);
    }
    Watcher watch;
    watch.root = o.root;

    // ---- the towers: SAVE (or Enter), and the buttons of the TOWERS panel ----
    auto saveTowers = [&](bool close) {
        // a placement with no room left for the picture would stop the scenes from starting at all
        if (tool.tw.widestBay() < TowerTool::MIN_BAY) note = "NOT SAVED: the towers must leave 3 m of wall free somewhere";
        else if (tool.tw.save()) {                  // the watcher restarts the workers
            tool.edited = false;
            if (close) tool.on = false;
            note = "towers saved";
            say("towers saved: data/towers.json");
        } else note = "cannot write data/towers.json";
    };
    auto towerButton = [&](int id) {
        const float step = GetKeyState(VK_SHIFT) < 0 ? 10.0f : 1.0f;
        if (id >= 1 && id <= 3) tool.sel = id - 1;
        else if (id >= 10 && id <= 17) tool.nudge((id - 10) / 2, (id & 1) ? step : -step);
        else if (id == 20) saveTowers(false);
        else if (id == 21) tool.on = false;
        else if (id == 22) {                        // the calibration picture, in the output too
            tool.calib = !tool.calib;
            say(tool.calib ? "towers: the calibration picture is sent out instead of the show" : "towers: the show is sent out again");
        }
        tool.dirty = true;
    };

    Clock clock;
    clock.set(lo.from);
    ExtClock ext;
    bool extRefused = false;
    bool startHeld = false;                     // the last value of /muonbloom/start was 1
    SoundWatch sound;
    bool playing = false, warmed = false, warmSent = false, wantPlay = !lo.paused;
    double showEnd = 0;
    double offset = lo.offset;                      // the picture is drawn this much ahead of the sound
    double soundRetryAt = 0;
    int rc = 0;

    auto play = [&] {
        if (audio.ok()) {
            int failed = audio.play(clock.now());
            if (failed) warn("%d of the sound stems could not be started", failed);
        }
        sound.reset();
        clock.start();
        playing = true;
    };
    auto pause = [&] {
        audio.stop();
        clock.stop();
        playing = false;
    };
    auto seek = [&](double t) {
        if (!std::isfinite(t)) return;              // (a NaN would freeze the clock for good)
        t = std::clamp(t, 0.0, std::max(0.0, showEnd - 0.05));
        clock.set(t);
        if (playing && audio.ok()) audio.play(t);
        sound.reset();
        ring.clear();
    };
    auto soundDown = [&](const char* why) {
        warn("SOUND LOST (%s): the show goes on without it, on the machine's timer; trying to open it again", why);
        audio.closeDevice();
        soundRetryAt = now() + 3.0;
    };

    timeBeginPeriod(1);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    double titleAt = 0, statAt = now(), positionAt = 0, scrubAt = 0;
    uint64_t drawnCount = 0, lastDrawn = 0;
    std::vector<Renderer::Over> over, barRects;
    double playedSince = 0, respawnAt = 0, barDrawnT = -1.0, barDrawnHover = -2.0;
    uint64_t sent = 0;
    int warmFailedSeen = 0, hungSeen = 0;
    bool scrubbing = false, strangerTold = false, barDirty = true, barDrawnPlaying = false;
    int barDrawnHot = 0;

    // comments typed in the window (key C, or the COMMENT button): written to <root>/comments.txt with the
    // time code they were started at
    gui::Comments comments;
    comments.path = fs::path(o.root) / L"comments.txt";
    comments.load(o.fps);
    bool commenting = false, commentResume = false;
    double commentT = 0;
    std::wstring commentText;
    // the prompt box (key A, or the CLAUDE button): the same line, sent to Claude Code instead of written to the
    // file; the answer comes in a panel above the bar (Shift+A hides and shows it)
    bool asking = false;                            // the line being typed is a prompt
    Claude claude;
    claude.init(o.root);
    double claudeTick = 0;
    Sketch sketch;                                  // what is drawn on the frame, until a snapshot, a comment or a prompt takes it
    bool askSound = false;                          // the SOUND button: the dialog is opened by the main loop

    // snapshots (key S, or the SNAPSHOT button): the picture that is on screen, written to
    // <root>/snapshots/MM-SS-FF_scene.png to be drawn and written on, with a copy nobody touches in
    // snapshots/untouched (what was drawn on a snapshot is what differs from that copy). When something is
    // drawn on the frame, the snapshot is MM-SS-FF_scene_drawn.png with the drawing in it (the copy without),
    // and the drawing is gone from the window. A line of comments.txt
    // names the picture, so that it has its mark on the time line and is read with the comments. The PNG is
    // packed by a thread of its own: the show does not wait for it.
    const fs::path snapDir = fs::path(o.root) / L"snapshots";
    std::atomic<int> snapState{ 0 };                // 1 being written, 2 written, 3 could not be written
    std::jthread snapThread;                        // (declared after snapState: it is joined before that goes)
    fs::path snapFile;
    std::string snapScene;
    double snapT = 0, snapAt = 0;                   // show time of the picture; when it was taken
    bool snapLit = false;                           // the button is lit for a moment
    auto snapshot = [&] {
        Target& pic = p.renderer.shown();
        if (snapState != 0 || !warmed || !pic.tex) return;          // one at a time; and there has to be a picture
        std::vector<uint8_t> px;
        if (!p.gpu.readback(pic.tex.Get(), 4, px)) {
            note = "THE SNAPSHOT CANNOT BE TAKEN";
            warn("snapshot: the picture cannot be read back");
            return;
        }
        snapT = p.shown() >= 0 ? p.shown() / o.fps : clock.now();   // the frame that is on screen: its own time code
        snapScene.clear();
        if (p.renderer.cardOn()) snapScene = "testcard";
        else for (auto& l : p.pool->looks) if (snapT >= l.t0 && snapT < l.t1) snapScene = l.name;
        std::string name = gui::timecode(snapT, o.fps) + "_" + (snapScene.empty() ? "show" : snapScene) + (sketch.empty() ? "" : "_drawn");
        for (char& c : name) if (!isalnum((unsigned char)c) && c != '_') c = '-';      // (MM:SS:FF: no colon in a file name)
        std::error_code ec;
        fs::create_directories(snapDir / L"untouched", ec);
        for (int n = 1;; n++) {                     // never over a picture that is there: it may have been drawn on
            snapFile = snapDir / (name + (n > 1 ? "_" + std::to_string(n) : "") + ".png");
            if (!fs::exists(snapFile, ec)) break;
        }
        snapState = 1;
        snapAt = now();
        snapLit = barDirty = true;
        snapThread = std::jthread([&snapState, file = snapFile, px = std::move(px), w = pic.w, h = pic.h, drawn = sketch]() mutable {
            const fs::path clean = file.parent_path() / L"untouched" / file.filename();
            bool ok;
            if (drawn.empty()) {
                ok = gui::savePng(file.wstring(), px.data(), (unsigned)w, (unsigned)h);
                std::error_code e;
                if (ok) fs::copy_file(file, clean, fs::copy_options::overwrite_existing, e);
            } else {                                // the frame as it was, then the frame with what was drawn on it
                ok = gui::savePng(clean.wstring(), px.data(), (unsigned)w, (unsigned)h);
                drawn.paint(px, w, h);
                ok = gui::savePng(file.wstring(), px.data(), (unsigned)w, (unsigned)h) && ok;
            }
            snapState = ok ? 2 : 3;
        });
        sketch.clear();                             // (it is in the picture now)
    };
    auto snapDone = [&] {                           // the picture is written, or could not be
        if (snapState < 2) return;
        const std::string rel = "snapshots/" + snapFile.filename().string();
        if (snapState == 2) {
            char lf[32] = "";
            if (lift > 0.0f && snapScene != "testcard") snprintf(lf, sizeof lf, " (lift %.2f)", lift);
            note = "snapshot: " + rel;
            say("snapshot at %s: %s%s", gui::timecode(snapT, o.fps).c_str(), rel.c_str(), lf);
            if (!comments.add(snapT, o.fps, snapScene, L"snapshot " + std::wstring(rel.begin(), rel.end()))) {
                note = "CANNOT WRITE comments.txt";
                warn("cannot write %s", narrow(comments.path.wstring()).c_str());
            }
        } else {
            note = "THE SNAPSHOT COULD NOT BE WRITTEN";
            warn("cannot write %s", narrow(snapFile.wstring()).c_str());
        }
        snapState = 0;
        barDirty = true;
    };
    // The frame with what was drawn on it, for a prompt or a comment: snapshots/MM-SS-FF_scene_drawn.png, and the same frame
    // without the drawing in snapshots/untouched. Written at once (the prompt names the file). Returns its
    // path from the repository folder, or nothing when it cannot be made.
    auto drawnSnapshot = [&](double tAt, const std::string& scene) -> std::string {
        Target& pic = p.renderer.shown();
        std::vector<uint8_t> px;
        if (!pic.tex || !p.gpu.readback(pic.tex.Get(), 4, px)) return {};
        std::string name = gui::timecode(tAt, o.fps) + "_" + (scene.empty() ? "show" : scene) + "_drawn";
        for (char& c : name) if (!isalnum((unsigned char)c) && c != '_') c = '-';
        std::error_code ec;
        fs::create_directories(snapDir / L"untouched", ec);
        fs::path file;
        for (int n = 1;; n++) {
            file = snapDir / (name + (n > 1 ? "_" + std::to_string(n) : "") + ".png");
            if (!fs::exists(file, ec)) break;
        }
        if (!gui::savePng((snapDir / L"untouched" / file.filename()).wstring(), px.data(), (unsigned)pic.w, (unsigned)pic.h)) return {};
        sketch.paint(px, pic.w, pic.h);
        if (!gui::savePng(file.wstring(), px.data(), (unsigned)pic.w, (unsigned)pic.h)) return {};
        return "snapshots/" + file.filename().string();
    };
    auto openSnapshots = [&] {                      // the folder the snapshots are in, in the Explorer
        std::error_code ec;
        fs::create_directories(snapDir, ec);
        ShellExecuteW(nullptr, L"open", snapDir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    };

    // One turn of the loop. Also called from a timer while Windows holds the window thread (drag, resize).
    auto step = [&] {
        double t = clock.now();

        // ---- start-up: wait for the workers, build every scene, then play ----
        if (!warmed) {
            p.pool->pump();
            if (p.pool->dead()) {
                warn("a Python worker stopped at start-up (see above): the show cannot start");
                rc = 2;
                win.closed = true;
                return;
            }
            if (p.pool->ready() == p.pool->workers()) {
                if (!warmSent) {
                    for (auto& l : p.pool->looks) showEnd = std::max(showEnd, l.t1);
                    p.warmAll();
                    warmSent = true;
                }
                if (!p.warmStep(p.pool->workers())) {
                    warmed = true;
                    say("scenes built, %s", wantPlay ? "playing" : "paused");
                    playedSince = now();
                    if (wantPlay) play();
                }
            } else if (p.pool->age() > Player::START_LIMIT) {
                warn("the Python workers did not start in %.0f s: the show cannot start", Player::START_LIMIT);
                rc = 2;
                win.closed = true;
                return;
            }
        }

        // ---- keys ----
        auto startComment = [&](bool ask = false) {
            if (commenting || !warmed) return;
            if (ask && claude.running) { note = "CLAUDE IS STILL WORKING"; claude.show = true; barDirty = true; return; }
            asking = ask;
            commenting = true;
            commentT = clock.now();
            commentText.clear();
            commentResume = playing && !ext.following;      // (a show that follows Ableton is not ours to pause)
            if (commentResume) pause();
            barDirty = true;
        };
        auto endComment = [&](bool save) {
            if (save && asking && (!commentText.empty() || !sketch.empty())) {      // a prompt: to Claude, with where the show is
                std::string scene, why, drawn;
                for (auto& l : p.pool->looks) if (commentT >= l.t0 && commentT < l.t1) scene = l.name;
                const std::string text = gui::utf8(commentText), tc = gui::timecode(commentT, o.fps);
                if (!sketch.empty()) {              // ... and with the frame, and what was drawn on it
                    const std::string file = drawnSnapshot(commentT, scene);
                    if (file.empty()) warn("the drawing could not be written: the prompt goes without it");
                    else drawn = " [The user drew on the frame, in light blue: read the picture " + file + " (2978 x 1400, the frame that was "
                                 "on screen; the same frame without the drawing is snapshots/untouched/" + file.substr(10) + "). What is drawn "
                                 "points at what the message is about.]";
                }
                if (claude.ask("[engine window: show time " + tc + " (MM:SS:FF), scene " + scene + "] " + text + drawn,
                               tc + "  " + text + (drawn.empty() ? "" : "  [+ drawing]"), why)) {
                    note = "sent to Claude";
                    say("to Claude at %s: %s", tc.c_str(), text.c_str());
                } else {
                    note = why;
                    warn("the prompt was not sent: %s", why.c_str());
                }
            } else if (save && (!commentText.empty() || !sketch.empty())) {
                std::string scene;
                for (auto& l : p.pool->looks) if (commentT >= l.t0 && commentT < l.t1) scene = l.name;
                if (!sketch.empty()) {              // a comment with a drawing: the frame with the drawing goes with it
                    const std::string file = drawnSnapshot(commentT, scene);
                    if (file.empty()) warn("the drawing could not be written: the comment goes without it");
                    else commentText += (commentText.empty() ? L"drawing " : L" [drawing: ") + std::wstring(file.begin(), file.end()) + (commentText.empty() ? L"" : L"]");
                }
                if (comments.add(commentT, o.fps, scene, commentText)) {
                    note = "comment saved in comments.txt";
                    say("comment at %s: %s", gui::timecode(commentT, o.fps).c_str(), gui::utf8(commentText).c_str());
                } else {
                    note = "CANNOT WRITE comments.txt";
                    warn("cannot write %s", narrow(comments.path.wstring()).c_str());
                }
            }
            commenting = false;
            if (save) sketch.clear();               // (cancelled: what was drawn stays, for a snapshot or another line)
            if (commentResume && !playing) play();
            barDirty = true;
        };
        const bool typing = commenting;             // (a comment started by a key in this turn: that key is not text)
        for (WPARAM k : win.keys) {
            bool shift = GetKeyState(VK_SHIFT) < 0, ctrl = GetKeyState(VK_CONTROL) < 0;
            if (commenting) {                       // every key belongs to the line being typed
                if (k == VK_RETURN) endComment(true);
                else if (k == VK_ESCAPE) endComment(false);
                else if (k == VK_BACK && !commentText.empty()) commentText.pop_back();
                else if (k == VK_BACK && !sketch.empty()) sketch.strokes.pop_back();                // an empty line: the last stroke goes
                barDirty = true;
                continue;
            }
            if (tool.on) {
                float stepPx = shift ? 10.0f : 1.0f;
                if (k == '1' || k == '2' || k == '3') tool.sel = (int)(k - '1');
                else if (k == VK_TAB) tool.handle = (tool.handle + (shift ? 4 : 1)) % 5;
                else if (k == VK_LEFT) tool.move(-stepPx, 0);
                else if (k == VK_RIGHT) tool.move(stepPx, 0);
                else if (k == VK_UP) tool.move(0, -stepPx);
                else if (k == VK_DOWN) tool.move(0, stepPx);
                else if (k == VK_RETURN) saveTowers(true);
                else if (k == VK_ESCAPE || k == 'T') tool.on = false;
                tool.dirty = barDirty = true;
                if (k != VK_SPACE && k != 'S') continue;
            }
            switch (k) {
            case VK_ESCAPE: win.closed = true; break;
            case VK_SPACE: if (warmed) { if (playing) pause(); else play(); } break;
            case VK_LEFT: seek(t - (ctrl ? 1.0 / o.fps : shift ? 30.0 : 5.0)); break;
            case VK_RIGHT: seek(t + (ctrl ? 1.0 / o.fps : shift ? 30.0 : 5.0)); break;
            case VK_HOME: seek(0.0); break;
            case VK_NEXT:
                for (auto& l : p.pool->looks) if (l.t0 > t + 0.01) { seek(l.t0); break; }
                break;
            case VK_PRIOR: {
                double best = 0.0;
                for (auto& l : p.pool->looks) if (l.t0 < t - 1.0) best = l.t0;
                seek(best);
                break;
            }
            case 'F': case VK_F11: win.toggleFullscreen(); break;
            case VK_BACK: if (!sketch.empty()) { sketch.strokes.pop_back(); barDirty = true; } break;      // the drawing: the last stroke away
            case VK_DELETE: if (!sketch.empty()) { sketch.clear(); note.clear(); barDirty = true; } break;  // ... all of it
            case 'B': win.bar = win.bar ? 0 : BAR_H; win.resized = true; break;
            case 'D': panel = !panel; barDirty = true; break;       // the detector meters
            case '1': case '2': case '3': levelSel = (int)(k - '1'); panel = true; break;
            case '0': levelSel = -1; panel = true; break;
            case VK_UP: setLevels(levelSel, shift ? 0.05f : 0.01f, true); panel = true; break;
            case VK_DOWN: setLevels(levelSel, shift ? -0.05f : -0.01f, true); panel = true; break;
            case 'R': p.reload(); note = "reloading"; break;
            case 'C': startComment(); break;
            case 'A':                               // the prompt box; Shift: its panel away / back
                if (shift) { claude.show = !claude.show; barDirty = true; }
                else startComment(true);
                break;
            case 'P':                               // the OUTPUT panel
                outPanel.open = !outPanel.open;
                outPanel.list = displays();
                barDirty = true;
                break;
            case 'K': showCard(!cardWanted); barDirty = true; break;      // the test card
            case 'S': if (shift) openSnapshots(); else snapshot(); break;       // a snapshot; Shift: the folder they are in
            case 'O':                               // the sound: choose the files; Shift: no sound from the engine
                if (!shift) askSound = true;
                else if (wantAudio) {
                    audio.close();
                    wantAudio = false;
                    note = "sound off";
                    say("sound off: the clock is /muonbloom/time if it arrives, else the machine's timer");
                    barDirty = true;
                }
                break;
            case 'T':                               // the towers: the placement tool and its panel
                if (tool.start()) { outPanel.open = false; note.clear(); }
                else note = "cannot read data/towers.json";
                barDirty = true;
                break;
            case VK_OEM_4: offset -= 0.005; break;          // [
            case VK_OEM_6: offset += 0.005; break;          // ]
            }
        }
        win.keys.clear();                           // (taken: a turn called from the timer has no pump to clear them)
        if (commenting && typing && !win.chars.empty()) {
            for (wchar_t c : win.chars) if (commentText.size() < 400) commentText.push_back(c);
            barDirty = true;
        }
        win.chars.clear();

        // ---- mouse: the time bar first, then the tower tool ----
        double hover = -1.0;
        const int hotButton = !win.bar || scrubbing ? 0 : win.overButton() ? 1 : win.overSound() ? 2 : win.overComment() ? 3 : win.overOutput() ? 4
                              : win.overSnapshot() ? 5 : win.overAsk() ? 6 : win.overTowers() ? 7 : 0;
        if (win.bar && win.pressed && win.overTowers()) {       // the TOWERS button: the placement tool and its panel
            if (tool.on && !outPanel.open) tool.on = false;
            else if (tool.on || tool.start()) outPanel.open = false;
            else note = "cannot read data/towers.json";
            tool.dirty = true;
            win.pressed = false;
            barDirty = true;
        }
        if (win.bar && win.pressed && win.overOutput()) {       // the OUTPUT button: its panel
            outPanel.open = !outPanel.open;
            outPanel.list = displays();
            win.pressed = false;
            barDirty = true;
        }
        if (outPanel.open && !commenting) {
            const float py1 = (float)(win.h - win.bar), py0 = py1 - PANEL_H;
            const bool inside = win.my >= py0 && win.my < py1;
            if (inside && (win.mx != panelMx || win.my != panelMy)) barDirty = true;        // a button lights under the mouse
            panelMx = win.mx;
            panelMy = win.my;
            if (win.pressed && inside) {
                const int id = outPanel.at(win.mx, win.my);
                if (id == 1) setLift(0.0f);
                else if (id == 2) outPanel.drag = 2;
                else if (id == 3) showCard(!cardWanted);
                else if (id == 31 || id == 33 || id == 35 || id == 37) outPanel.drag = id;
                else if (id == 30 || id == 32 || id == 34 || id == 36) {            // a click on a name: as the scenes give it
                    (id == 30 ? glow : id == 32 ? glowRed : id == 34 ? red : weight) = id == 36 ? 0.0f : 1.0f;
                    p.renderer.tune(glow, glowRed, red, weight);
                    if (id == 36) p.invalidate();           // (the lines are drawn again)
                    outDirty = settingsDirty = true;
                }
                else if (id >= 4 && id <= 8) {              // MOVE: a pixel, ten with Shift; 0 puts it back
                    const int step = GetKeyState(VK_SHIFT) < 0 ? 10 : 1;
                    if (id == 8) out.moveX = out.moveY = 0;
                    else if (id <= 5) out.moveX = std::clamp(out.moveX + (id == 4 ? -step : step), -MOVE_MAX, MOVE_MAX);
                    else out.moveY = std::clamp(out.moveY + (id == 6 ? -step : step), -MOVE_MAX, MOVE_MAX);
                    outDirty = settingsDirty = true;
                }
                else if (id == 9) {                         // AUTO: the display it has, else the other one, else the next to come
                    if (outDevice.empty() || outGone) {
                        autoFailed.clear();
                        const std::wstring d = autoDisplay(displays());
                        setOutput(d, true);
                    }
                    outAuto = true;
                    settingsDirty = true;
                }
                else if (id == 10) { setOutput(L""); settingsDirty = true; }
                else if (id >= 11 && id - 11 < (int)outPanel.list.size()) { setOutput(outPanel.list[id - 11].device); settingsDirty = true; }
                win.pressed = false;                // (the click is taken: not for the tower tool under the panel)
                barDirty = true;
            }
            if (outPanel.drag == 2) {
                if (win.down) setLift(((float)win.mx - SLIDER_X0) / SLIDER_W * LIFT_MAX);
                else outPanel.drag = 0;
                barDirty = true;
            }
            if (outPanel.drag >= 31) {              // GLOW W, GLOW R, RED, WEIGHT
                const int k = (outPanel.drag - 31) / 2;
                float& v = k == 0 ? glow : k == 1 ? glowRed : k == 2 ? red : weight;
                const float vmax = k <= 1 ? GLOW_MAX : k == 2 ? RED_MAX : WEIGHT_MAX;
                if (win.down) {
                    const float nv = std::round(std::clamp(((float)win.mx - TUNE_X0 - k * TUNE_PITCH) / TUNE_W, 0.0f, 1.0f) * vmax * 20.0f) / 20.0f;
                    if (nv != v) {
                        v = nv;
                        p.renderer.tune(glow, glowRed, red, weight);
                        if (k == 3) p.invalidate();
                        outDirty = settingsDirty = true;
                    }
                } else outPanel.drag = 0;
                barDirty = true;
            }
        } else {
            outPanel.drag = 0;
        }
        if (tool.on && !outPanel.open && !commenting) {         // the TOWERS panel
            const float py1 = (float)(win.h - win.bar), py0 = py1 - PANEL_H;
            const bool inside = win.my >= py0 && win.my < py1;
            if (inside && (win.mx != panelMx || win.my != panelMy)) barDirty = true;
            panelMx = win.mx;
            panelMy = win.my;
            if (win.pressed && inside) {
                const int id = towerPanel.at(win.mx, win.my);
                towerButton(id);
                towerPanel.held = id >= 10 && id <= 17 ? id : 0;
                towerPanel.heldAt = towerPanel.stepAt = now();
                win.pressed = false;                // (the click is taken: not for the tower under the panel)
                barDirty = true;
            }
            if (towerPanel.held) {                  // a button kept down goes on
                if (!win.down || towerPanel.at(win.mx, win.my) != towerPanel.held) towerPanel.held = 0;
                else if (now() - towerPanel.heldAt > 0.35 && now() - towerPanel.stepAt > 0.03) {
                    towerButton(towerPanel.held);
                    towerPanel.stepAt = now();
                    barDirty = true;
                }
            }
        } else {
            towerPanel.held = 0;
        }
        if (win.bar && win.pressed && win.overButton()) {       // the play / pause button
            if (warmed && !commenting) { if (playing) pause(); else play(); }
            win.pressed = false;
        }
        if (win.bar && win.pressed && win.overSound()) {
            askSound = true;
            win.pressed = false;
        }
        if (win.bar && win.pressed && win.overComment()) {
            if (commenting) endComment(true); else startComment();
            win.pressed = false;
        }
        if (win.bar && win.pressed && win.overAsk()) {          // the CLAUDE button; with Shift: its panel away / back
            if (commenting) endComment(true);
            else if (GetKeyState(VK_SHIFT) < 0) { claude.show = !claude.show; barDirty = true; }
            else startComment(true);
            win.pressed = false;
        }
        {
            DWORD code = 0;
            if (claude.poll(code)) {                            // the answer is there
                note = code ? "CLAUDE ENDED WITH AN ERROR" : "Claude has answered";
                say("Claude answered (exit code %lu)", code);
                claude.show = true;
                barDirty = true;
            } else if (claude.running && now() - claudeTick > 0.5) {    // its seconds, its blinking name
                claudeTick = now();
                barDirty = true;
            }
        }
        if (win.bar && win.pressed && win.overSnapshot()) {     // the SNAPSHOT button; with Shift: the folder they are in
            if (GetKeyState(VK_SHIFT) < 0) openSnapshots(); else snapshot();
            win.pressed = false;
        }
        snapDone();
        if (snapLit && now() - snapAt > 0.6) {      // the button goes back to grey
            snapLit = false;
            barDirty = true;
        }
        if (win.bar && showEnd > 0) {
            if (win.pressed && win.overLine()) scrubbing = true;
            if (win.overLine() || scrubbing) hover = win.barFraction() * showEnd;
            if (scrubbing && win.down && now() - scrubAt > 0.03 && std::abs(hover - clock.now()) > 0.5 / o.fps) {
                scrubAt = now();
                seek(hover);
            }
            if (win.released || !win.down) scrubbing = false;
        }
        if (!tool.on && !scrubbing) {                           // the mouse draws on the picture (the tower tool has it otherwise)
            float x, y;
            win.toPicture(x, y);
            const int above = win.bar + (commenting ? (int)BOX_H : outPanel.open ? (int)PANEL_H : 0);       // what lies over the picture
            const bool onPic = win.my < win.h - above && x >= 0.0f && x < (float)PIC_W && y >= 0.0f && y < (float)PIC_H;
            if (win.pressed && onPic) {
                sketch.strokes.push_back({ { x, y } });
                sketch.drawing = barDirty = true;
                if (!commenting) note = "drawing: S snapshot with it, C comment with it, Backspace last stroke away, Delete all";
            } else if (sketch.drawing && win.down) {
                auto& st = sketch.strokes.back();
                x = std::clamp(x, 0.0f, (float)PIC_W - 1.0f);
                y = std::clamp(y, 0.0f, (float)PIC_H - 1.0f);
                if (std::hypot(x - st.back().first, y - st.back().second) > 3.0f) { st.push_back({ x, y }); barDirty = true; }
            }
            if (sketch.drawing && (win.released || !win.down)) {
                sketch.drawing = false;
                if (sketch.strokes.back().size() < 2) {         // a click is not a stroke (the window was only given the focus)
                    sketch.strokes.pop_back();
                    barDirty = true;
                }
            }
        } else if (tool.on && !scrubbing) {
            float x, y;
            win.toPicture(x, y);
            if (win.pressed && !win.overBar()) tool.press(x, y);
            if (tool.dragging && win.down) {
                if (x != tool.gx || y != tool.gy) tool.move(std::round(x - tool.gx), std::round(y - tool.gy));
                tool.gx += std::round(x - tool.gx);
                tool.gy += std::round(y - tool.gy);
            }
            if (win.released) tool.dragging = false;
        }
        win.pressed = win.released = false;         // (taken: a turn called from the timer has no pump to clear them)

        // ---- OSC: the detectors, and orders for the transport ----
        msgs.clear();
        osc.poll(msgs);
        for (auto& m : msgs) {
            if (m.addr.rfind("/muonbloom/", 0) == 0) {
                bool allowed = lo.oscAllow.empty() || std::find(lo.oscAllow.begin(), lo.oscAllow.end(), m.from) != lo.oscAllow.end();
                if (!allowed) {
                    if (!strangerTold)
                        warn("OSC order %s from %u.%u.%u.%u ignored: only the addresses given with --osc-allow may drive the show",
                             m.addr.c_str(), m.from >> 24, (m.from >> 16) & 255, (m.from >> 8) & 255, m.from & 255);
                    strangerTold = true;
                    continue;
                }
                if (m.addr == "/muonbloom/time") {
                    if (m.args.empty()) continue;
                    if (wantAudio) {                // the engine plays the sound itself: that sound is the clock
                        if (!extRefused) warn("/muonbloom/time ignored: the engine was started with its own sound (--sound, --audio)");
                        extRefused = true;
                        continue;
                    }
                    const double T = m.args[0], w = now();
                    if (!ext.heard) say("clock: a time is arriving by OSC (%s): the show follows it", narrow(timecode(T)).c_str());
                    ext.heard = true;
                    ext.heardAt = w;
                    if (T == ext.last) continue;                    // the same time again: nothing new
                    ext.last = T;
                    ext.movedAt = w;
                    if (!warmed) { clock.set(std::max(0.0, T)); continue; }
                    if (showEnd > 0 && T >= showEnd - 1.0 / o.fps) {        // past the end: the last picture stays
                        if (playing) pause();
                        clock.set(showEnd - 1.0 / o.fps);
                        continue;
                    }
                    if (!playing) {
                        if (std::abs(T - clock.now()) > ExtClock::JUMP) seek(T);    // elsewhere in the show
                        else clock.set(T);                                          // going on from where it stopped
                        play();
                        ext.fresh(w);
                        if (!ext.following || ext.stalled) say("clock: running on the time received, from %s", narrow(timecode(T)).c_str());
                        ext.stalled = false;
                    } else if (!ext.following) {            // the show was on its own timer: taken up from where it is
                        ext.fresh(w);
                        say("clock: following the time received (%s, the show is at %s)", narrow(timecode(T)).c_str(), narrow(timecode(clock.now())).c_str());
                    } else {
                        const double e = T - clock.now();
                        if (std::abs(e) > ExtClock::LOCATE) {       // far away: a locate if it stays there, else an odd value
                            if (ext.farSince == 0 || std::abs(e - ext.farErr) > ExtClock::SAME) {
                                ext.farSince = w;
                                ext.farErr = e;
                            } else if (w - ext.farSince > ExtClock::HOLD) {
                                say("clock: the time received is elsewhere in the show (%+.3f s): moved to %s", e, narrow(timecode(T)).c_str());
                                seek(T);
                                ext.jumps++;
                                ext.fresh(w);
                            }
                        } else {
                            ext.farSince = 0;
                            ext.errs.push_back(e);
                            if (w - ext.lockedAt > 2.0) {           // (the first seconds are the clock settling)
                                ext.n++;
                                ext.errSum += e;
                                ext.errWorst = std::max(ext.errWorst, std::abs(e));
                            }
                        }
                    }
                    ext.following = true;
                }
                else if (m.addr.rfind("/muonbloom/level", 0) == 0) {       // the trigger level: one for all, three, or /level/L
                    int k = m.addr.size() > 17 ? detectorOf(m.addr.substr(17)) : -1;
                    if (m.addr.size() == 16 && m.args.size() >= 3) {
                        for (int j = 0; j < 3; j++) level[j] = (float)m.args[j];
                        setLevels(-1, 0.0f, true);
                    } else if (!m.args.empty() && (m.addr.size() == 16 || k >= 0)) setLevels(k, (float)m.args[0], false);
                }
                else if (m.addr == "/muonbloom/play") { if (warmed && !playing) play(); }
                else if (m.addr == "/muonbloom/start") {                   // the show from its beginning
                    // a button: 0, then 1 when it is pressed. The order is the passage from 0 to 1, not the
                    // value: a 1 that is sent again and again, and the 0 of the release, do nothing.
                    // (TouchDesigner puts two values in one message when it has a frame to make up: all are read)
                    bool pressed = m.args.empty();
                    for (double v : m.args) {
                        const bool on = v >= 0.5;
                        if (on && !startHeld) pressed = true;
                        startHeld = on;
                    }
                    if (pressed && warmed) {
                        seek(0.0);
                        if (!playing) play();
                        ext.fresh(now());
                        say("playing from the beginning (OSC)");
                    }
                }
                else if (m.addr == "/muonbloom/pause") { if (playing) pause(); }
                else if (m.addr == "/muonbloom/seek") { if (!m.args.empty()) seek(m.args[0]); }
                else if (m.addr == "/muonbloom/reload") { if (warmed) { p.reload(); note = "reloading"; say("reloading the scenes (OSC)"); } }
            } else if (liveDet && m.addr.rfind(lo.detPrefix, 0) == 0) {
                // stamped with the first frame that can still show it: the frames up to there are already asked
                double stamp = std::max(0.0, clock.now() + offset) + (o.lead + 1) / o.fps;
                const size_t n = lo.detPrefix.size();
                if (m.addr.size() == n) {                               // <prefix> L C R
                    if (m.args.size() >= 3)
                        for (int j = 0; j < 3; j++) {
                            ring.set(j, (float)m.args[j], stamp);
                            meter.feed(j, (float)m.args[j], now(), level[j]);
                        }
                } else if (m.addr[n] == '/' && m.addr.find('/', n + 1) == std::string::npos) {     // <prefix>/L
                    int k = detectorOf(m.addr.substr(n + 1));
                    if (k >= 0 && !m.args.empty()) {
                        ring.set(k, (float)m.args[0], stamp);
                        meter.feed(k, (float)m.args[0], now(), level[k]);
                    }
                }
            }
        }
        ring.commit();

        // ---- the test card: made, wanted before there was a picture, older than the towers it shows ----
        if (cardMaker && WaitForSingleObject(cardMaker, 0) == WAIT_OBJECT_0) {
            DWORD code = 1;
            GetExitCodeProcess(cardMaker, &code);
            CloseHandle(cardMaker);
            cardMaker = nullptr;
            if (code == 0 && !cardStale()) {
                cardNote.clear();
                if (cardWanted) {
                    p.renderer.card(false, 0, 0);       // (a card that was on is the old one)
                    showCard(true);
                }
            } else {
                cardNote = "IT COULD NOT BE MADE (python tools/test_card.py)";
                cardBroken = true;
                cardWanted = p.renderer.cardOn();   // (a card that is on stays: it is the one of before)
                warn("the test card could not be made: python tools/test_card.py ended with code %lu", code);
            }
            barDirty = true;
        }
        if (cardWanted && !cardMaker && !p.renderer.cardOn() && p.shown() >= 0) {
            showCard(true);
            barDirty = true;
        }

        // ---- the files of the show ----
        if (warmed && watch.poll()) {               // (during a reload too: what was being started is already old)
            say(p.reloading() ? "a file of the show changed again: starting the reload again" : "a file of the show changed: reloading the scenes");
            p.reload();
            note = "reloading";
        }
        if (p.reloadStep(t)) {
            note = "reloaded";
            say("scenes reloaded at %s", narrow(timecode(t)).c_str());
            warmFailedSeen = hungSeen = 0;
            p.warmAll();
        }
        if (!p.reloadError.empty()) {
            note = p.reloadError;
            warn("%s", note.c_str());
            p.reloadError.clear();
        }
        if (warmed && !p.reloading()) p.warmStep(1);
        if (p.pool->warmFailed != warmFailedSeen) {     // a scene that cannot be built will show as a frozen picture
            warmFailedSeen = p.pool->warmFailed;
            warn("A SCENE CANNOT BE BUILT: %s", p.pool->warmError.c_str());
            note = "A SCENE CANNOT BE BUILT (see the console)";
        }
        if (p.pool->hung() != hungSeen) {
            hungSeen = p.pool->hung();
            warn("a scene worker did not answer for too long and was stopped");
        }
        if (warmed && p.pool->dead() && !p.reloading() && now() - respawnAt > 5.0) {     // a worker died: start a new set
            respawnAt = now();
            warn("a scene worker stopped: starting the workers again");
            p.reload();
        }

        // ---- clock ----
        if (ext.following) {                        // the time comes from outside: has it stopped, or gone?
            double w = now();
            if (w - ext.heardAt > ExtClock::SILENT) {
                ext.following = false;
                clock.speed(1.0);
                warn("CLOCK: no time has arrived for %.1f s (at %s): the show goes on, on the machine's timer", ExtClock::SILENT,
                     narrow(timecode(clock.now())).c_str());
            } else if (playing && ext.heardAt - ext.movedAt > ExtClock::STALL && w - ext.heardAt < ExtClock::STALL) {
                pause();                            // (the same time, still received that long after it last moved)
                clock.set(std::clamp(ext.last, 0.0, std::max(0.0, showEnd - 1.0 / o.fps)));
                ext.stalled = true;
                say("clock: the time received stands still at %s: paused", narrow(timecode(clock.now())).c_str());
            } else if (playing && w - ext.syncAt >= ExtClock::SYNC) {   // how far from the times received? no step: a speed
                ext.syncAt = w;
                double e = 0;
                if (!ext.error(e) || std::abs(e) < ExtClock::CLOSE) {
                    clock.speed(1.0);
                    ext.catching = false;
                } else {
                    clock.speed(1.0 + std::clamp(e / ExtClock::CATCH, -ExtClock::SLEW, ExtClock::SLEW));
                    if (std::abs(e) > 1.5 / o.fps && !ext.catching && w - ext.lockedAt > 2.0)
                        say("clock: %+.0f ms from the time received at %s: catching up without a step", 1e3 * e, narrow(timecode(clock.now())).c_str());
                    ext.catching = std::abs(e) > 1.0 / o.fps;
                }
            }
        }
        if (playing) {
            double a;
            if (audio.ok()) {
                if (audio.lost()) soundDown("the sound output reported an error");
                else if (audio.position(a)) {
                    int s = sound.check(a);         // a position that stands still is not followed
                    if (s == 0) {
                        double before = clock.now();
                        clock.follow(a);
                        if (std::abs(clock.now() - before) > 0.05)
                            warn("the picture was moved by %+.3f s onto the sound (at %s)", clock.now() - before, narrow(timecode(a)).c_str());
                    } else if (s == 2) soundDown("the position of the sound stopped moving");
                }
            }
            t = clock.now();
            if (showEnd > 0 && t >= showEnd) {
                if (lo.loop && !ext.following) { seek(0.0); if (!audio.ok()) clock.start(); }   // (followed: the sender decides)
                else { pause(); clock.set(showEnd - 1.0 / o.fps); }
                t = clock.now();
            }
        }
        if (wantAudio && !audio.ok() && now() > soundRetryAt) {
            soundRetryAt = now() + 5.0;
            if (audio.openDevice()) {
                say("sound output opened again");
                if (playing) audio.play(clock.now());
                sound.reset();
            }
        }

        // ---- picture ----
        const double ts = std::max(0.0, t + offset);
        bool drew = p.tick(ts);
        if (!p.error.empty()) {
            warn("renderer: %s", p.error.c_str());
            note = "RENDERER ERROR (see the console)";
            p.error.clear();
        }
        if (win.bar && showEnd > 0) {               // the bar moves by itself: redraw the window when it has moved a pixel
            double px = showEnd / std::max(1, win.w);
            if (std::abs(t - barDrawnT) > px || hover != barDrawnHover) barDirty = true;
        }
        if (win.bar && (playing != barDrawnPlaying || hotButton != barDrawnHot)) barDirty = true;
        if (win.bar && !playing && std::abs(t - barDrawnT) > 0.25 / o.fps) barDirty = true;     // the time code
        if (panel && liveDet && now() - panelAt > 0.033) {      // the meters move by themselves
            panelAt = now();
            barDirty = true;
        }
        bool overNow = tool.on || !over.empty();
        if (drew || (overNow && tool.dirty) || win.resized || barDirty || outDirty) {
            if (drew || tool.dirty || outDirty) {
                over.clear();
                if (tool.on) tool.rects(over);
                p.renderer.overlay(over);
                tool.dirty = outDirty = false;
                if (spoutOn && p.shown() >= 0) sent += spout.SendTexture(p.renderer.shown().tex.Get());
                out.present(p.renderer, lo);        // before the preview: this is the picture on the wall
            }
            barRects.clear();
            if (!sketch.empty()) sketch.rects(barRects, win.w, win.h, win.bar);
            if (win.bar) {
                BarState st;
                st.tc = gui::timecode(t, o.fps);
                st.sound = wantAudio;
                st.commenting = commenting && !asking;
                st.claude = commenting && asking ? 1 : claude.running ? 2 : 0;
                st.hot = hotButton;
                st.output = outDevice.empty() ? (outAuto ? 2 : 0) : outGone ? 2 : 1;
                st.towers = tool.on && !outPanel.open;
                st.card = p.renderer.cardOn();
                st.panel = outPanel.open;
                st.snapped = snapLit;
                st.marks = &comments.marks;
                timeBar(barRects, win.w, win.h, p.pool->looks, std::max(showEnd, 1e-9), t, hover, playing, st);
            }
            if (claude.show && !claude.lines.empty() && !((outPanel.open || tool.on) && !commenting) && !(commenting && asking))
                claude.rects(barRects, win.w, win.h, win.bar, commenting);     // (not while a prompt is typed: one draws on the picture)
            if (commenting) commentBox(barRects, win.w, win.h, win.bar, gui::timecode(commentT, o.fps), commentText, asking);
            else if (outPanel.open)
                outPanel.rects(barRects, win.w, win.h, win.bar, lift, cardWanted, cardNote, outDevice, outGone, out.exact, displayOf(win.hwnd), lo, win.mx,
                               win.my, out.moveX, out.moveY, glow, glowRed, red, weight, outAuto);
            else if (tool.on)
                towerPanel.rects(barRects, win.w, win.h, win.bar, tool, win.mx, win.my, !outDevice.empty() && !outGone);
            if (panel && liveDet) meter.rects(barRects, level, levelSel, now());
            barDrawnT = t;
            barDrawnHover = hover;
            barDrawnPlaying = playing;
            barDrawnHot = hotButton;
            barDirty = false;
            win.present(p.gpu, p.renderer, barRects);
            drawnCount += drew;
        } else if (!playing) {
            Sleep(2);
        } else if (!win.modal) {
            double boundary = (p.frameAt(ts) + 1) / o.fps;         // show time of the next frame period
            if (p.shown() < p.frameAt(ts) || boundary - ts > 0.0015) Sleep(1);
            else while (clock.now() + offset < boundary) YieldProcessor();
        }

        if (lo.quitAfter > 0 && warmed && now() - playedSince > lo.quitAfter) win.closed = true;

        // ---- once a second or so: the title, what went wrong, where we are ----
        double w = now();
        if (win.displays) {                         // (looked at now, and again half a second later: Windows may not be done)
            win.displays = false;
            titleAt = 0;
        }
        if (w - titleAt > 0.5) {
            if (win.lost || p.gpu.removed()) {
                warn("THE GRAPHICS DEVICE IS GONE (driver reset, card removed): the engine has to be started again");
                rc = 3;
                win.closed = true;
            }
            // the displays: one plugged in or taken away; the one of the output gone, back, or of another size
            const std::vector<Display> list = displays();
            if (list.size() != known.size() || !std::equal(list.begin(), list.end(), known.begin(), [](const Display& a, const Display& b) {
                    return a.device == b.device && EqualRect(&a.rc, &b.rc) && a.hz == b.hz; })) {
                known = list;
                autoFailed.clear();
                say("displays: %s", describe(list).c_str());
                barDirty = true;
            }
            outPanel.list = list;
            if (!outDevice.empty()) {
                const auto it = std::find_if(list.begin(), list.end(), [&](const Display& d) { return d.device == outDevice; });
                if (it == list.end()) {
                    if (outAuto) {                  // AUTO lets it go, and takes the next one
                        out.close();
                        warn("OUTPUT LOST: the display %s is gone (AUTO: the next one is taken)", narrow(outDevice).c_str());
                        outDevice.clear();
                        outGone = false;
                        outDirty = barDirty = true;
                    } else if (!outGone) {
                        out.close();
                        outGone = true;
                        warn("OUTPUT LOST: the display %s is gone; it is taken again as soon as it is back", narrow(outDevice).c_str());
                        barDirty = true;
                    }
                } else if (outGone || out.lost || !out.hwnd || !EqualRect(&it->rc, &out.on.rc)) {
                    std::string e;
                    if (out.open(p.gpu, *it, e)) {
                        say("output: the display %s is taken (%d x %d at %d Hz)", narrow(outDevice).c_str(), it->w(), it->h(), it->hz);
                        outGone = false;
                        outDirty = barDirty = true;
                    }
                }
            }
            if (outAuto && outDevice.empty()) {
                const std::wstring d = autoDisplay(list);
                if (!d.empty()) {
                    setOutput(d, true);
                    outAuto = true;                 // (also when it could not be opened: the next one is tried)
                    barDirty = true;
                }
            }
            if (p.renderer.cardOn() && !cardMaker && !cardBroken && cardStale() && makeCard()) cardNote = "MAKING IT AGAIN ...";   // the towers moved
            if (settingsDirty && !outPanel.drag) {
                settingsDirty = false;
                if (!saveOutput(outputFile, lift, outAuto ? std::wstring(L"auto") : outDevice, out.moveX, out.moveY, glow, glowRed, red, weight)) warn("cannot write %s", narrow(outputFile.wstring()).c_str());
            }
            double dt = w - statAt;
            std::wstring look, hov;
            for (auto& l : p.pool->looks) if (t >= l.t0 && t < l.t1) look.assign(l.name.begin(), l.name.end());
            if (hover >= 0.0) {
                std::wstring hl;
                for (auto& l : p.pool->looks) if (hover >= l.t0 && hover < l.t1) hl.assign(l.name.begin(), l.name.end());
                hov = L"   -> " + timecode(hover) + L" " + hl;
            }
            auto& ms = p.pool->stats.ms;
            wchar_t off[48] = L"";
            if (std::abs(offset) > 1e-6) swprintf(off, 48, L"   picture %+.0f ms (--offset)", offset * 1e3);
            wchar_t trig[160] = L"";
            if (liveDet)
                swprintf(trig, 160, L"   trigger L %.2f  C %.2f  R %.2f  [%ls: Up / Down]", level[0], level[1], level[2],
                         levelSel < 0 ? L"all" : levelSel == 0 ? L"L" : levelSel == 1 ? L"C" : L"R");
            wchar_t outs[200] = L"";
            if (!outDevice.empty() || lift > 0.0f || p.renderer.cardOn()) {     // what is sent out, when it is not just the show
                wchar_t lf[24] = L"";
                if (lift > 0.0f) swprintf(lf, 24, L"LIFT %.2f", lift);
                swprintf(outs, 200, L"   %ls%ls%ls%ls%ls", p.renderer.cardOn() ? L"TEST CARD  " : L"",
                         outDevice.empty() ? L"" : outGone ? L"OUTPUT LOST  " : !out.exact ? L"OUTPUT SCALED  " : L"OUTPUT ON  ",
                         outDevice.size() > 4 ? outDevice.c_str() + 4 : L"", outDevice.size() > 4 ? L"  " : L"", lf);
            }
            wchar_t title[1000];
            std::wstring wnote(note.begin(), note.end());
            swprintf(title, 1000, L"MUON : BLOOM   %ls  %ls   %ls%ls   %.1f fps   dropped %llu   scenes %.0f / %.0f ms (median / max)%ls%ls%ls%ls%ls%ls%ls%ls%ls",
                     timecode(t).c_str(), look.c_str(), !warmed ? L"BUILDING THE SCENES" : playing ? L"PLAYING" : L"PAUSED",
                     ext.following ? L" (time by OSC)" : ext.heard ? L" (TIME BY OSC LOST: own timer)" : wantAudio ? L" (own sound)" : L" (own timer)",
                     (drawnCount - lastDrawn) / dt, (unsigned long long)p.stats.stale, percentile(ms, 0.5), percentile(ms, 1.0), hov.c_str(), off,
                     outs, trig, wantAudio && !audio.ok() ? L"   NO SOUND" : L"",
                     tool.on ? L"   TOWERS: 1 2 3 select, Tab handle, arrows / mouse / panel move, Enter save, Esc cancel" : L"",
                     p.pool->lastError.empty() ? L"" : L"   SCENE ERROR (see the console)", wnote.empty() ? L"" : L"   ", wnote.c_str());
            SetWindowTextW(win.hwnd, title);
            if (!p.pool->lastError.empty()) {
                warn("scene error at frame %lld (%s): %s", (long long)p.pool->lastErrorFrame,
                     narrow(timecode(p.pool->lastErrorFrame / o.fps)).c_str(), p.pool->lastError.c_str());
                p.pool->lastError.clear();
            }
            if (p.unknownPost) warn("%llu frames drawn WITHOUT their post-process: the recorder does not know it (muonbloom/drawlist.py: post_ops)",
                                    (unsigned long long)p.unknownPost);
            if (p.missingGlyphs) warn("%llu letters could not be drawn (glyphs the renderer never received): reload the scenes (R)",
                                      (unsigned long long)p.missingGlyphs);
            p.unknownPost = p.missingGlyphs = 0;
            lastDrawn = drawnCount;
            p.pool->stats.reset();
            statAt = titleAt = w;
        }
        if (!lo.position.empty() && playing && w - positionAt > 1.0) {      // for run.bat: where to start again after a crash
            positionAt = w;
            if (FILE* f = _wfopen(lo.position.c_str(), L"w")) {
                fprintf(f, "%.2f\n", t);
                fclose(f);
            }
        }
    };

    // The SOUND button (or the key O): choose the WAV files the engine plays, and follows, from now on.
    // The dialog runs a loop of its own: the picture goes on behind it, from the timer of the window.
    auto chooseSound = [&] {
        std::error_code ec;
        fs::path dir = fs::path(o.root) / L".." / L"audio";
        std::vector<std::wstring> files;
        win.modal = true;
        SetTimer(win.hwnd, 1, 1, nullptr);
        const bool chosen = gui::chooseWav(win.hwnd, fs::exists(dir, ec) ? fs::weakly_canonical(dir, ec).wstring() : std::wstring(), files);
        KillTimer(win.hwnd, 1);
        win.modal = false;
        win.down = win.pressed = win.released = false;
        barDirty = true;
        if (!chosen) return;
        audio.stop();                               // (the voices play out of the memory that is replaced)
        std::string e;
        if (!audio.load(files, e)) {
            warn("%s", e.c_str());
            note = e + " (PCM WAV files only): no sound";
            audio.close();
            wantAudio = false;
            return;
        }
        wantAudio = true;
        ext.following = false;                      // the sound of the engine is the clock from here on
        clock.speed(1.0);
        for (auto& f : files) say("sound: %s", narrow(f).c_str());
        note = "sound: " + narrow(fs::path(files[0]).filename().wstring()) + (files.size() > 1 ? " + " + std::to_string(files.size() - 1) + " more" : "");
        if (!audio.ok() && !audio.openDevice()) warn("NO SOUND OUTPUT: playing without sound, trying again every few seconds");
        audio.volume(lo.volume);
        if (playing && audio.ok()) audio.play(clock.now());
        sound.reset();
    };

    win.tick = step;
    while (!win.closed) {
        win.pump();
        step();
        if (askSound) {
            askSound = false;
            chooseSound();
        }
    }
    win.tick = nullptr;
    timeEndPeriod(1);
    say("stopped at %s: %llu frames drawn, %llu dropped, %llu late, %llu sent to Spout", narrow(timecode(clock.now())).c_str(),
        (unsigned long long)drawnCount, (unsigned long long)p.stats.stale, (unsigned long long)p.stats.late, (unsigned long long)sent);
    if (liveDet)
        say("detectors: %llu, %llu, %llu hits (L, C, R) with the trigger level at %.3f, %.3f, %.3f at the end", (unsigned long long)meter.hits[0],
            (unsigned long long)meter.hits[1], (unsigned long long)meter.hits[2], level[0], level[1], level[2]);
    if (ext.heard)
        say("clock: %llu times followed, on average %+.1f ms from the clock when they arrived, at worst %.1f ms; the show was moved %llu times",
            (unsigned long long)ext.n, ext.n ? 1e3 * ext.errSum / ext.n : 0.0, 1e3 * ext.errWorst, (unsigned long long)ext.jumps);
    if (rc == 0 && !lo.position.empty()) {          // stopped on purpose: nothing to resume
        std::error_code ec;
        fs::remove(lo.position, ec);
    }
    claude.close();                                 // (a prompt still at work is stopped with the engine)
    if (snapThread.joinable()) snapThread.join();   // (a snapshot that is being written is finished, and gets its line)
    snapDone();
    audio.close();
    out.close();
    if (cardMaker) CloseHandle(cardMaker);
    if (settingsDirty) saveOutput(outputFile, lift, outDevice, out.moveX, out.moveY, glow, glowRed, red, weight);
    if (spoutOn) spout.ReleaseSender();
    SetThreadExecutionState(ES_CONTINUOUS);
    if (gLog) fclose(gLog);
    gLog = nullptr;
    return rc;
}

int runSpoutCheck(const char* name, const wchar_t* dump)
{
    Gpu g;                                          // the same graphics card as the engine: a texture is shared on one card only
    std::string err;
    if (!g.create(false, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    spoutDX rx;
    if (!rx.OpenDirectX11(g.dev.Get())) { fprintf(stderr, "cannot open Direct3D 11\n"); return 1; }
    if (name && *name) rx.SetReceiverName(name);
    std::vector<unsigned char> px;
    double start = now();
    long first = -1, last = -1;
    int got = 0;
    unsigned w = 0, h = 0;
    while (now() - start < 3.0) {
        if (rx.ReceiveTexture() && rx.IsConnected()) {
            w = rx.GetSenderWidth();
            h = rx.GetSenderHeight();
            if (w && rx.IsFrameNew()) {
                got++;
                last = rx.GetSenderFrame();
                if (first < 0) first = last;
            }
        }
        Sleep(2);
    }
    if (!w) { printf("no Spout sender%s%s\n", name && *name ? " named " : "", name ? name : ""); return 1; }
    double lit = 0;
    rx.ReceiveTexture();
    if (rx.GetSenderTexture() && g.readback(rx.GetSenderTexture(), 4, px)) {
        for (size_t k = 0; k < px.size(); k += 4) lit += px[k] > 8 || px[k + 1] > 8 || px[k + 2] > 8;
        if (dump && *dump) {
            std::ofstream f(dump, std::ios::binary);
            f.write((const char*)px.data(), (std::streamsize)px.size());
        }
    }
    printf("Spout sender %s: %u x %u, format %d, %d new frames in 3 s (sender frame %ld -> %ld, %.1f fps), %.1f %% of the picture lit\n",
           rx.GetSenderName(), w, h, (int)rx.GetSenderFormat(), got, first, last, rx.GetSenderFps(), 100.0 * lit / ((double)w * h));
    rx.ReleaseReceiver();
    return 0;
}
