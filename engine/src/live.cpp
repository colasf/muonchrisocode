// muonengine live: the show in real time, with sound, in a preview window and as a Spout sender.
#include "osc.h"                            // first: winsock2.h has to come before windows.h

#include <windows.h>
#include <timeapi.h>
#include <windowsx.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "SpoutDX.h"
#include "audio.h"
#include "live.h"
#include "player.h"

namespace fs = std::filesystem;

namespace {

const int PIC_W = 2978, PIC_H = 1400;       // the picture of the show (layout.W, layout.H)
const float FRAME_X0 = 28.0f, FRAME_X1 = 2950.0f, FRAME_Y1 = 1354.0f, HEAD_Y = 220.0f;
const int BAR_H = 36;                       // height of the time bar under the picture, window pixels
const float BAR_PAD = 12.0f;                // its margin left and right
const float BTN_W = 44.0f;                  // the play / pause button, at its left
const float BAR_X0 = BTN_W + BAR_PAD;       // where the time line starts

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
    WINDOWPLACEMENT placement = { sizeof(WINDOWPLACEMENT) };
    std::vector<WPARAM> keys;               // keys pressed since the last pump
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
            case WM_SIZE:
                if (wp != SIZE_MINIMIZED) { s->w = LOWORD(lp); s->h = HIWORD(lp); s->resized = true; }
                return 0;
            case WM_KEYDOWN: s->keys.push_back(wp); return 0;
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
    // mouse position -> 0..1 along the time line
    double barFraction() const
    {
        return std::clamp(((double)mx - BAR_X0) / std::max(1.0, (double)w - BAR_PAD - BAR_X0), 0.0, 1.0);
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
// MadMapper), moved with the mouse or the arrow keys, written to data/towers.json on Enter.
struct TowerTool {
    static constexpr float MIN_BAY = 300.0f;        // the scenes need one free stretch of wall at least this wide
    Towers tw;
    bool on = false, dirty = false;
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
        dirty = true;
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

    void rects(std::vector<Renderer::Over>& o) const
    {
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

// The time bar of the preview window: a play / pause button, then the scenes of the show as blocks, what
// has been played, where the clock is, where the mouse points. Window pixels.
void timeBar(std::vector<Renderer::Over>& o, int w, int h, const std::vector<Pool::Look>& looks, double end, double t, double hover,
             bool playing, bool hot)
{
    const float y0 = (float)(h - BAR_H), y1 = (float)h, x0 = BAR_X0, x1 = std::max(BAR_X0 + 1.0f, (float)w - BAR_PAD);
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
    o.push_back({ X(t) - 1.5f, y0 + 2.0f, X(t) + 1.5f, y1 - 2.0f, 1.0f, 0.1f, 0.06f, 1.0f });     // the clock
}

}  // namespace

// The time of the show, given by whoever plays the sound (Ableton, through TouchDesigner):
//     /muonbloom/time <seconds>      sent all the time, many times a second
// The engine keeps running on the machine's own timer and is pulled onto that time (Clock::follow: the
// freshest of the times received over half a second is the true one, the others arrived late).
//     the time moves          the show plays, at that time
//     the time stands still   the sender has stopped: the show pauses there
//     nothing arrives         the sender is gone: the show goes on by the machine's timer (a picture that
//                             freezes because a cable fell out is worse than one that drifts)
struct ExtClock {
    static constexpr double JUMP = 0.08;        // further than this from the time received: the show is moved there
    static constexpr double STALL = 0.4;        // the time has not moved for this long, and still arrives: paused
    static constexpr double SILENT = 1.5;       // nothing has arrived for this long: the sender is gone
    bool heard = false;                         // a time has arrived at least once
    bool following = false;                     // the show is running on the times received
    bool stalled = false;                       // ... and they stand still: paused there
    double last = -1e18, movedAt = 0, heardAt = 0;
    uint64_t n = 0, jumps = 0;                  // times followed, and times the show was moved
    double errSum = 0, errWorst = 0;            // received - clock at arrival, once locked
    double lockedAt = 0;
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
        else if (!armed[k] && v < 0.6f * level) armed[k] = true;       // (detectors.py: RELEASE)
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
    const bool wantAudio = !lo.audio.empty();
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
    if (lo.towers && tool.tw.load()) tool.on = tool.dirty = true;
    Watcher watch;
    watch.root = o.root;

    Clock clock;
    clock.set(lo.from);
    ExtClock ext;
    bool extRefused = false;
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
    bool scrubbing = false, strangerTold = false, barDirty = true, barDrawnPlaying = false, barDrawnHot = false;

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
        for (WPARAM k : win.keys) {
            bool shift = GetKeyState(VK_SHIFT) < 0, ctrl = GetKeyState(VK_CONTROL) < 0;
            if (tool.on) {
                float stepPx = shift ? 10.0f : 1.0f;
                if (k == '1' || k == '2' || k == '3') tool.sel = (int)(k - '1');
                else if (k == VK_TAB) tool.handle = (tool.handle + (shift ? 4 : 1)) % 5;
                else if (k == VK_LEFT) tool.move(-stepPx, 0);
                else if (k == VK_RIGHT) tool.move(stepPx, 0);
                else if (k == VK_UP) tool.move(0, -stepPx);
                else if (k == VK_DOWN) tool.move(0, stepPx);
                else if (k == VK_RETURN) {
                    // a placement with no room left for the picture would stop the scenes from starting at all
                    if (tool.tw.widestBay() < TowerTool::MIN_BAY) note = "NOT SAVED: the towers must leave 3 m of wall free somewhere";
                    else if (tool.tw.save()) { tool.on = false; note = "towers saved"; }   // the watcher restarts the workers
                    else note = "cannot write data/towers.json";
                } else if (k == VK_ESCAPE || k == 'T') tool.on = false;
                tool.dirty = true;
                if (k != VK_SPACE) continue;
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
            case 'B': win.bar = win.bar ? 0 : BAR_H; win.resized = true; break;
            case 'D': panel = !panel; barDirty = true; break;       // the detector meters
            case '1': case '2': case '3': levelSel = (int)(k - '1'); panel = true; break;
            case '0': levelSel = -1; panel = true; break;
            case VK_UP: setLevels(levelSel, shift ? 0.05f : 0.01f, true); panel = true; break;
            case VK_DOWN: setLevels(levelSel, shift ? -0.05f : -0.01f, true); panel = true; break;
            case 'R': p.reload(); note = "reloading"; break;
            case 'T':
                if (tool.tw.load()) { tool.on = true; tool.dirty = true; note.clear(); }
                else note = "cannot read data/towers.json";
                break;
            case VK_OEM_4: offset -= 0.005; break;          // [
            case VK_OEM_6: offset += 0.005; break;          // ]
            }
        }
        win.keys.clear();                           // (taken: a turn called from the timer has no pump to clear them)

        // ---- mouse: the time bar first, then the tower tool ----
        double hover = -1.0;
        const bool hotButton = win.overButton() && !scrubbing;
        if (win.bar && win.pressed && win.overButton()) {       // the play / pause button
            if (warmed) { if (playing) pause(); else play(); }
            win.pressed = false;
        }
        if (win.bar && showEnd > 0) {
            if (win.pressed && win.overBar()) scrubbing = true;
            if ((win.overBar() && !win.overButton()) || scrubbing) hover = win.barFraction() * showEnd;
            if (scrubbing && win.down && now() - scrubAt > 0.03 && std::abs(hover - clock.now()) > 0.5 / o.fps) {
                scrubAt = now();
                seek(hover);
            }
            if (win.released || !win.down) scrubbing = false;
        }
        if (tool.on && !scrubbing) {
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
                        ext.lockedAt = w;
                        if (!ext.following || ext.stalled) say("clock: running on the time received, from %s", narrow(timecode(T)).c_str());
                        ext.stalled = false;
                    } else {
                        double e = T - clock.now();
                        if (std::abs(e) > ExtClock::JUMP) {
                            if (ext.following) say("clock: the time received jumped by %+.3f s, to %s", e, narrow(timecode(T)).c_str());
                            seek(T);
                            ext.jumps++;
                            ext.lockedAt = w;
                        } else {
                            clock.follow(T);
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
                warn("CLOCK: no time has arrived for %.1f s (at %s): the show goes on, on the machine's timer", ExtClock::SILENT,
                     narrow(timecode(clock.now())).c_str());
            } else if (playing && w - ext.movedAt > ExtClock::STALL && w - ext.heardAt < ExtClock::STALL) {
                pause();
                clock.set(std::clamp(ext.last, 0.0, std::max(0.0, showEnd - 1.0 / o.fps)));
                ext.stalled = true;
                say("clock: the time received stands still at %s: paused", narrow(timecode(clock.now())).c_str());
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
        if (panel && liveDet && now() - panelAt > 0.033) {      // the meters move by themselves
            panelAt = now();
            barDirty = true;
        }
        bool overNow = tool.on || !over.empty();
        if (drew || (overNow && tool.dirty) || win.resized || barDirty) {
            if (drew || tool.dirty) {
                over.clear();
                if (tool.on) tool.rects(over);
                p.renderer.overlay(over);
                tool.dirty = false;
                if (spoutOn && p.shown() >= 0) sent += spout.SendTexture(p.renderer.shown().tex.Get());
            }
            barRects.clear();
            if (win.bar) timeBar(barRects, win.w, win.h, p.pool->looks, std::max(showEnd, 1e-9), t, hover, playing, hotButton);
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
        if (w - titleAt > 0.5) {
            if (win.lost || p.gpu.removed()) {
                warn("THE GRAPHICS DEVICE IS GONE (driver reset, card removed): the engine has to be started again");
                rc = 3;
                win.closed = true;
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
            wchar_t title[800];
            std::wstring wnote(note.begin(), note.end());
            swprintf(title, 800, L"MUON : BLOOM   %ls  %ls   %ls%ls   %.1f fps   dropped %llu   scenes %.0f / %.0f ms (median / max)%ls%ls%ls%ls%ls%ls%ls%ls",
                     timecode(t).c_str(), look.c_str(), !warmed ? L"BUILDING THE SCENES" : playing ? L"PLAYING" : L"PAUSED",
                     ext.following ? L" (time by OSC)" : ext.heard ? L" (TIME BY OSC LOST: own timer)" : wantAudio ? L" (own sound)" : L" (own timer)",
                     (drawnCount - lastDrawn) / dt, (unsigned long long)p.stats.stale, percentile(ms, 0.5), percentile(ms, 1.0), hov.c_str(), off,
                     trig, wantAudio && !audio.ok() ? L"   NO SOUND" : L"",
                     tool.on ? L"   TOWERS: 1 2 3 select, Tab handle, arrows / mouse move, Enter save, Esc cancel" : L"",
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

    win.tick = step;
    while (!win.closed) {
        win.pump();
        step();
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
    audio.close();
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
