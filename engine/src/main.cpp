// muonengine - realtime engine of the Muon Bloom show.
//
//   muonengine render <frames.mbdl> <out prefix> [--float] [--repeat N] [--lift V]
//       draw every frame of a draw-list file (muonbloom/drawlist.py) without a window and write
//       <prefix>NNNN.rgba (W x H x 4 bytes), or <prefix>NNNN.rgbaf (float32, no dither) with --float.
//       --repeat N draws each frame N times and prints the time of one. --lift V: with the lift of the
//       output (the OUTPUT panel of `live`) at V. --display WxH: also write <prefix>NNNN.display, what the
//       output of `live` sends to a display of that size (--raster, --raster-at, --picture-at as there;
//       --card-file F: with that test card on): the check that the raster leaves pixel for pixel.
//
//   muonengine bench [--from S] [--to S] [--workers N] [--lead N] [--fps F]
//       play the show (or a part of it) in real time without a window or sound, and report per scene how
//       many frames were on screen in time. This is the measure of "does it hold 60 fps".
//
//   muonengine live [--from S] [--paused] [--loop] [--sound | --audio a.wav --audio b.wav]
//                   [--spout NAME | --no-spout] [--offset MS] [--workers N] [--lead N]
//                   [--osc PORT] [--detectors scripted|live|both]
//       the show in real time: preview window, Spout sender (default name: MuonBloom).
//       Keys: Space play / pause, Left / Right 5 s (Shift 30 s, Ctrl one frame), Page Up / Down scene,
//       Home start, F full screen, T tower placement, R reload the scenes, [ ] picture earlier / later, Esc quit.
//       The bar under the picture shows the time code (MM:SS:FF, as in the timings sheet). O (or the SOUND
//       button): choose the WAV file(s) the engine plays and follows from then on; Shift+O: no sound again.
//       C (or the COMMENT button): type a comment, Enter writes it with its time code to comments.txt in
//       the repository folder (Esc cancels); the show is paused while it is typed.
//       A (or the CLAUDE button): the same line, sent to Claude Code; its answer comes in a panel above the
//       bar (Shift+A hides and shows it).
//       S (or the SNAPSHOT button): the picture on screen is written to snapshots/MM-SS-FF_scene.png in the
//       repository folder, to be drawn and written on, and a line of comments.txt names it. Shift+S (or
//       Shift and the button) opens that folder.
//       Drawing: a drag on the picture draws on it, at any time (light blue, in this window only; not while
//       the towers are placed). The next snapshot (..._drawn.png), comment or prompt takes the drawing with
//       the frame, and it is gone. Backspace: the last stroke away; Delete: all of it.
//       P (or the OUTPUT button): the output panel. LIFT raises the mid levels of the picture (0 = as
//       rendered; drag the slider, click its name for 0). TEST CARD (or K) shows the test card instead of
//       the show (tools/test_card.py makes it). OUTPUT chooses the display the raster is sent to: a window
//       that fills it, no border, no pointer - the HDMI output to the media server. The raster (--raster
//       3000x1688) sits at the top left of that display (--raster-at X,Y) with the picture of the show at
//       --picture-at 11,272 of it, pixel for pixel; a display smaller than the raster gets it scaled to fit.
//       AUTO: the display this window is not on, whichever it is, taken as soon as it is plugged in.
//       At start: --output 2 (a display as the panel numbers them, or its device name; auto; off = none;
//       default: as in the last run), --lift 0.6, --card. What is set in the panel is kept in engine/output.json.
//       T (or the TOWERS button): the three towers over the picture and their panel - choose one, set its
//       centre, width, height and detector height with the buttons (a pixel, ten with Shift) or drag it on
//       the picture; SAVE writes data/towers.json and the scenes are built again around it.
//       The clock: the sound is played by Ableton, and the time of the show comes by OSC:
//       /muonbloom/time <seconds>, sent all the time. The show follows it (plays when it moves, pauses when
//       it stands still, goes on by the machine's timer if nothing arrives any more). Without it the engine
//       runs on its own timer. --sound / --audio: the engine plays sound itself and follows that instead
//       (work at the desk). --offset: milliseconds the picture is drawn ahead of the clock.
//       OSC (UDP, port 9000): /muonbloom/time <seconds>, /muonbloom/start <0 or 1> (from the beginning, on 0 to 1), /muonbloom/play,
//       /muonbloom/pause, /muonbloom/seek
//       <seconds>; with --detectors live or both, the detector values: /muon/L, /muon/C, /muon/R <float 0..1>
//       (or /muon <L> <C> <R>).
//       Trigger level of the detectors (a value that rises above it is a hit): --det-level 0.25 or
//       --det-level L,C,R; while it runs: D shows the meters, 1 2 3 choose a detector (0: the three),
//       Up / Down move its level by 0.01 (Shift 0.05); OSC /muonbloom/level <v> (or <L> <C> <R>, or
//       /muonbloom/level/L <v>). What is set while it runs is kept in engine/detectors.json.
//
//       --osc-allow IP[,IP] | any: who may send those orders (default: this machine only);
//       --det-prefix /muon: OSC address of the detectors; --no-bar: no time bar in the preview window (key B);
//       --log FILE | --no-log: where the messages are also written (default engine/out/engine.log).
//       Exit code: 0 stopped by somebody, 2 cannot start as set up, 3 failed while running (start it again).
//
// Common options: --python <interpreter>, --root <repository folder>, --debug (Direct3D debug layer).
#include "osc.h"                            // first: winsock2.h has to come before windows.h

#include <windows.h>
#include <timeapi.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu.h"
#include "live.h"
#include "player.h"
#include "renderer.h"

#ifndef MUON_SHADER_DIR
#define MUON_SHADER_DIR L""
#endif
#ifndef MUON_ROOT_DIR
#define MUON_ROOT_DIR L""
#endif

namespace fs = std::filesystem;

static fs::path exeDir()
{
    wchar_t exe[1024];
    GetModuleFileNameW(nullptr, exe, 1024);
    return fs::path(exe).parent_path();
}

// The shaders: beside the exe, else in engine/shaders when the exe sits in engine/build, else where they
// were when the engine was built.
static std::wstring shaderDir()
{
    wchar_t env[1024];
    if (GetEnvironmentVariableW(L"MUONENGINE_SHADERS", env, 1024)) return env;
    for (const fs::path& p : { exeDir() / L"shaders", exeDir().parent_path() / L"shaders" })
        if (fs::exists(p / L"segments.hlsl")) return p.wstring();
    return MUON_SHADER_DIR;
}

// The repository (the folder that holds muonbloom/ and engine/): two levels above the exe when it sits in
// engine/build - so that a copy of the folder on another machine runs as it is - else where it was built.
static std::wstring rootDir()
{
    fs::path p = exeDir().parent_path().parent_path();
    if (fs::exists(p / L"muonbloom" / L"show.py") && fs::exists(p / L"engine" / L"worker.py")) return p.wstring();
    return MUON_ROOT_DIR;
}

struct Args {
    std::vector<std::wstring> v;
    Args(int argc, wchar_t** argv) : v(argv, argv + argc) {}
    bool flag(const wchar_t* name) const { return std::find(v.begin(), v.end(), name) != v.end(); }
    std::wstring str(const wchar_t* name, const std::wstring& def) const
    {
        auto it = std::find(v.begin(), v.end(), name);
        return it != v.end() && it + 1 != v.end() ? *(it + 1) : def;
    }
    double num(const wchar_t* name, double def) const
    {
        auto it = std::find(v.begin(), v.end(), name);
        return it != v.end() && it + 1 != v.end() ? _wtof((it + 1)->c_str()) : def;
    }
};

// An option made of two whole numbers (--raster 3000x1688, --picture-at 11,272). False when it is there and
// is not that; left as it is when it is not there.
static bool pairArg(const Args& a, const wchar_t* name, wchar_t sep, int& x, int& y)
{
    if (!a.flag(name)) return true;
    std::wstring v = a.str(name, L"");
    size_t j = v.find(sep);
    if (j == std::wstring::npos) return false;
    x = _wtoi(v.substr(0, j).c_str());
    y = _wtoi(v.substr(j + 1).c_str());
    return true;
}

static PlayerOptions playerOptions(const Args& a)
{
    PlayerOptions o;
    o.workers = (int)a.num(L"--workers", o.workers);
    o.lead = (int)a.num(L"--lead", o.lead);
    o.fps = a.num(L"--fps", o.fps);
    o.root = a.str(L"--root", rootDir());
    // a "python" folder beside muonbloom/ (engine/package.py makes one): that interpreter, nothing to install
    fs::path own = fs::path(o.root) / L"python" / L"python.exe";
    o.python = a.str(L"--python", fs::exists(own) ? own.wstring() : o.python);
    o.shaders = shaderDir();
    o.debug = a.flag(L"--debug");
    return o;
}

static void sleepUntil(double t)
{
    for (;;) {
        double left = t - now();
        if (left <= 0) return;
        if (left > 0.002) Sleep(1);
        else YieldProcessor();
    }
}

// Wait for the workers, then have each of them build the scenes that play between t0 and t1.
static bool warmUp(Player& p, double t0, double t1, std::string& err)
{
    double c = now();
    while (p.pool->ready() < p.pool->workers()) {
        p.pool->pump();
        if (p.pool->dead()) { err = "a Python worker stopped at start-up (see above)"; return false; }
        if (now() - c > 120.0) { err = "the Python workers did not start"; return false; }
        Sleep(5);
    }
    printf("%d workers ready in %.1f s\n", p.pool->workers(), now() - c);
    c = now();
    p.warmAll(t0, t1);
    while (p.warmStep(p.pool->workers())) {
        p.pool->pump();
        Sleep(2);
    }
    printf("scenes built in %.1f s\n", now() - c);
    return true;
}

static float percentile(std::vector<float> v, double q)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min<double>((double)v.size() - 1, q * v.size())];
}

static int cmdBench(const Args& a)
{
    Player p;
    std::string err;
    PlayerOptions o = playerOptions(a);
    if (!p.start(o, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    printf("gpu: %ls\n", p.gpu.name.c_str());
    double from = a.num(L"--from", 0.0), to = a.num(L"--to", 1e9);
    if (!warmUp(p, from, to, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (!p.pool->looks.empty()) to = std::min(to, p.pool->looks.back().t1);
    printf("%d workers, %d frames ahead, %.0f fps\n", o.workers, o.lead, o.fps);
    printf("%-13s %7s %8s %6s %7s   %s\n", "look", "frames", "in time", "late", "dropped", "scene ms: median / 95 % / max    draw ms: cpu, gpu median / gpu max");

    timeBeginPeriod(1);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    PlayerStats total;
    std::vector<float> allMs;
    for (auto& l : p.pool->looks) {
        double t0 = std::max(l.t0, from), t1 = std::min(l.t1, to);
        if (t1 <= t0) continue;
        p.stats.reset();
        p.pool->stats.reset();
        std::vector<float> cpuMs, gpuMs;
        double start = now();
        for (;;) {
            double t = t0 + (now() - start);
            if (t >= t1) break;
            double c = now();
            p.gpu.timeBegin();
            bool drew = p.tick(t);
            p.gpu.timeEnd();
            if (drew) {
                cpuMs.push_back((float)((now() - c) * 1e3));
                p.gpu.finish();
                gpuMs.push_back((float)p.gpu.gpuMs());
            } else {
                sleepUntil(std::min(now() + 0.0005, start + (p.frameAt(t) + 1) / o.fps - t0));
            }
        }
        const PlayerStats& s = p.stats;
        auto& ms = p.pool->stats.ms;
        printf("%-13s %7llu %7.1f%% %6llu %7llu   %6.1f / %5.1f / %5.1f           %5.2f, %5.2f / %5.2f%s\n", l.name.c_str(),
               (unsigned long long)s.ticks, s.ticks ? 100.0 * s.fresh / s.ticks : 0.0, (unsigned long long)s.late, (unsigned long long)s.stale,
               percentile(ms, 0.5), percentile(ms, 0.95), percentile(ms, 1.0), percentile(cpuMs, 0.5), percentile(gpuMs, 0.5), percentile(gpuMs, 1.0),
               p.pool->stats.failed ? "   SCENE ERRORS" : "");
        fflush(stdout);
        total.ticks += s.ticks;
        total.fresh += s.fresh;
        total.late += s.late;
        total.stale += s.stale;
        allMs.insert(allMs.end(), ms.begin(), ms.end());
    }
    timeEndPeriod(1);
    printf("\nwhole run: %llu frames, %.2f %% in time, %llu late, %llu dropped; worker median %.1f ms, 95 %% %.1f ms, max %.1f ms\n",
           (unsigned long long)total.ticks, total.ticks ? 100.0 * total.fresh / total.ticks : 0.0, (unsigned long long)total.late,
           (unsigned long long)total.stale, percentile(allMs, 0.5), percentile(allMs, 0.95), percentile(allMs, 1.0));
    if (!p.pool->lastError.empty()) printf("last scene error (frame %lld): %s\n", (long long)p.pool->lastErrorFrame, p.pool->lastError.c_str());
    return 0;
}

static int cmdRender(const Args& a)
{
    if (a.v.size() < 4) {
        fprintf(stderr, "usage: muonengine render <frames.mbdl> <out prefix> [--float] [--repeat N] [--lift V] [--display WxH] [--debug]\n");
        return 2;
    }
    bool asFloat = a.flag(L"--float");
    int repeat = std::max(1, (int)a.num(L"--repeat", 1));
    std::ifstream f(a.v[2], std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %ls\n", a.v[2].c_str()); return 1; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    Gpu g;
    std::string err;
    if (!g.create(a.flag(L"--debug"), err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Renderer r;
    double t0 = now();
    if (!r.init(g.dev.Get(), g.ctx.Get(), shaderDir(), err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    r.lift((float)a.num(L"--lift", 0.0));
    g.finish();
    printf("gpu: %ls, start-up %.0f ms\n", g.name.c_str(), (now() - t0) * 1e3);

    // --display WxH: a target of the size of a display, to send it what the output of `live` would
    LiveOptions lo;
    int dw = 0, dh = 0;
    ComPtr<ID3D11Texture2D> display;
    ComPtr<ID3D11RenderTargetView> displayRtv;
    const std::wstring cardFile = a.str(L"--card-file", L"");
    if (!pairArg(a, L"--display", L'x', dw, dh) || !pairArg(a, L"--raster", L'x', lo.rasterW, lo.rasterH)
        || !pairArg(a, L"--raster-at", L',', lo.rasterX, lo.rasterY) || !pairArg(a, L"--picture-at", L',', lo.picX, lo.picY)) {
        fprintf(stderr, "--display WxH, --raster WxH, --raster-at X,Y, --picture-at X,Y\n");
        return 2;
    }
    if (dw > 0 && dh > 0 && !asFloat) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = (UINT)dw;
        td.Height = (UINT)dh;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(g.dev->CreateTexture2D(&td, nullptr, &display)) || FAILED(g.dev->CreateRenderTargetView(display.Get(), nullptr, &displayRtv))) {
            fprintf(stderr, "cannot create a %d x %d target\n", dw, dh);
            return 1;
        }
        if (!cardFile.empty() && !r.loadCard(cardFile, lo.rasterW, lo.rasterH)) {
            fprintf(stderr, "cannot read the test card %ls (%d x %d, raw BGRA)\n", cardFile.c_str(), lo.rasterW, lo.rasterH);
            return 1;
        }
    }

    size_t off = 0;
    int n = 0;
    std::vector<uint8_t> px;
    while (off + dl::HEADER <= data.size()) {
        uint32_t bytes;
        memcpy(&bytes, &data[off + 8], 4);
        if (bytes < dl::HEADER || off + bytes > data.size()) { fprintf(stderr, "frame %d: truncated file\n", n); return 1; }
        if (!r.render(&data[off], bytes, asFloat, err)) { fprintf(stderr, "frame %d: %s\n", n, err.c_str()); return 1; }
        g.finish();                                  // first draw: buffers and targets are created
        double c = now();
        for (int k = 0; k < repeat; k++) r.render(&data[off], bytes, asFloat, err);
        g.finish();
        double ms = (now() - c) * 1e3 / repeat;
        const FrameInfo& i = r.info();
        printf("frame %d  t %.3f  %.2f ms  segs %u (%u passes) dots %u splats %u rects %u ops %u text %u post %u%s%s\n", n, i.t, ms,
               i.segs, i.passes, i.dots, i.splats, i.rects, i.lightops, i.textinst, i.postops,
               i.missing_glyphs ? "  MISSING GLYPHS" : "", i.flags ? "  FLAGS" : "");
        Target& t = asFloat ? r.outF() : r.out();
        if (!g.readback(t.tex.Get(), asFloat ? 16 : 4, px)) { fprintf(stderr, "frame %d: cannot read the picture back\n", n); return 1; }
        wchar_t name[32];
        swprintf(name, 32, L"%04d.%ls", n, asFloat ? L"rgbaf" : L"rgba");
        std::ofstream o(a.v[3] + name, std::ios::binary);
        o.write((const char*)px.data(), (std::streamsize)px.size());
        if (!o) { fprintf(stderr, "cannot write %ls%ls\n", a.v[3].c_str(), name); return 1; }
        if (display) {
            if (!cardFile.empty() && !r.card(true, lo.picX, lo.picY)) { fprintf(stderr, "the picture does not fit in the card at %d, %d\n", lo.picX, lo.picY); return 1; }
            r.overlay({});
            const bool exact = r.output(display.Get(), displayRtv.Get(), dw, dh, lo.rasterW, lo.rasterH, lo.rasterX, lo.rasterY, lo.picX, lo.picY);
            if (!g.readback(display.Get(), 4, px)) { fprintf(stderr, "frame %d: cannot read the display back\n", n); return 1; }
            swprintf(name, 32, L"%04d.display", n);
            std::ofstream d(a.v[3] + name, std::ios::binary);
            d.write((const char*)px.data(), (std::streamsize)px.size());
            if (!d) { fprintf(stderr, "cannot write %ls%ls\n", a.v[3].c_str(), name); return 1; }
            printf("         a %d x %d display gets it %s\n", dw, dh, exact ? "pixel for pixel" : "scaled to fit");
        }
        off += bytes;
        n++;
    }
    return 0;
}

int wmain(int argc, wchar_t** argv)
{
    Args a(argc, argv);
    std::wstring cmd = argc >= 2 ? argv[1] : L"";
    if (cmd == L"render") return cmdRender(a);
    if (cmd == L"bench") return cmdBench(a);
    if (cmd == L"spoutcheck") {
        std::string name;
        for (wchar_t c : a.str(L"--name", L"")) name += (char)c;
        return runSpoutCheck(name.c_str(), a.str(L"--dump", L"").c_str());
    }
    if (cmd == L"live") {
        PlayerOptions o = playerOptions(a);
        LiveOptions lo;
        lo.from = a.num(L"--from", 0.0);
        lo.offset = a.num(L"--offset", 0.0) / 1000.0;
        lo.paused = a.flag(L"--paused");
        lo.loop = a.flag(L"--loop");
        lo.towers = a.flag(L"--towers");
        lo.quitAfter = a.num(L"--quit-after", 0.0);
        lo.volume = (float)a.num(L"--volume", 1.0);
        lo.oscPort = (int)a.num(L"--osc", lo.oscPort);
        std::wstring det = a.str(L"--detectors", L"scripted");
        lo.detectors.clear();
        for (wchar_t c : det) lo.detectors += (char)c;
        if (a.flag(L"--no-spout")) lo.spout.clear();
        else {
            std::wstring n = a.str(L"--spout", L"MuonBloom");
            lo.spout.clear();
            for (wchar_t c : n) lo.spout += (char)(c < 128 ? c : '_');
        }
        lo.bar = !a.flag(L"--no-bar");
        std::string narrowed;
        for (wchar_t c : a.str(L"--det-prefix", L"/muon")) narrowed += (char)(c < 128 ? c : '_');
        while (narrowed.size() > 1 && narrowed.back() == '/') narrowed.pop_back();
        lo.detPrefix = narrowed;
        // --det-level 0.25 (the three detectors) or --det-level 0.2,0.3,0.25 (L, C, R)
        if (std::find(a.v.begin(), a.v.end(), L"--det-level") != a.v.end()) {
            std::wstring list = a.str(L"--det-level", L"");
            float v[3] = { -1.0f, -1.0f, -1.0f };
            int n = 0;
            size_t i = 0;
            while (i <= list.size() && n < 3) {
                size_t j = list.find(L',', i);
                if (j == std::wstring::npos) j = list.size();
                v[n++] = (float)_wtof(list.substr(i, j - i).c_str());
                i = j + 1;
            }
            if (n == 1) v[1] = v[2] = v[0];
            if (n == 2 || !(v[0] > 0.0f && v[0] <= 1.0f && v[1] > 0.0f && v[1] <= 1.0f && v[2] > 0.0f && v[2] <= 1.0f)) {
                fprintf(stderr, "--det-level: one level for the three detectors, or three (L,C,R), each over 0 and at most 1\n");
                return 2;
            }
            for (int k = 0; k < 3; k++) lo.detLevel[k] = v[k];
        }
        // --osc-allow 10.0.0.5,10.0.0.6 : who may send play / pause / seek / reload (default: this machine only);
        // --osc-allow any : anybody on the network
        if (std::find(a.v.begin(), a.v.end(), L"--osc-allow") != a.v.end()) {
            lo.oscAllow.clear();
            std::string list;
            for (wchar_t c : a.str(L"--osc-allow", L"")) list += (char)(c < 128 ? c : '_');
            if (list != "any") {
                lo.oscAllow.push_back(0x7F000001u);
                size_t i = 0;
                while (i <= list.size()) {
                    size_t j = list.find(',', i);
                    if (j == std::string::npos) j = list.size();
                    uint32_t ip = OscIn::ipv4(list.substr(i, j - i));
                    if (ip) lo.oscAllow.push_back(ip);
                    else if (j > i) fprintf(stderr, "--osc-allow: '%s' is not an IPv4 address (ignored)\n", list.substr(i, j - i).c_str());
                    i = j + 1;
                }
            }
        }
        lo.log = a.str(L"--log", (fs::path(o.root) / L"engine" / L"out" / L"engine.log").wstring());
        if (a.flag(L"--no-log")) lo.log.clear();
        lo.position = a.str(L"--position-file", L"");
        // the output: --output 2 | \\.\DISPLAY2 | auto | off, --raster 3000x1688, --picture-at 11,272, --raster-at 0,0
        for (wchar_t c : a.str(L"--output", L"")) lo.output += (char)(c < 128 ? c : '_');
        if (!pairArg(a, L"--raster", L'x', lo.rasterW, lo.rasterH) || !pairArg(a, L"--picture-at", L',', lo.picX, lo.picY)
            || !pairArg(a, L"--raster-at", L',', lo.rasterX, lo.rasterY) || lo.rasterW <= 0 || lo.rasterH <= 0) {
            fprintf(stderr, "--raster WxH (the delivery raster), --picture-at X,Y (the picture of the show in it), --raster-at X,Y (the raster on the display)\n");
            return 2;
        }
        if (a.flag(L"--lift")) lo.lift = (float)std::clamp(a.num(L"--lift", 0.0), 0.0, 3.0);
        lo.card = a.flag(L"--card");
        // The sound of the show is played by Ableton, and the time comes from there (/muonbloom/time). For work
        // at the desk the engine can play sound itself, and then follows it: --sound (the stems of the
        // previews) or --audio FILE. (--no-audio is what it does by default now; still accepted.)
        if (!a.flag(L"--no-audio")) {
            for (size_t k = 0; k + 1 < a.v.size(); k++)
                if (a.v[k] == L"--audio") lo.audio.push_back(a.v[k + 1]);
            if (lo.audio.empty() && a.flag(L"--sound")) {                     // the stems of the previews
                std::vector<fs::path> wanted;
                for (const wchar_t* f : { L"muon bloom Mixed v1 scene 10 edit NO MUONS SOUNDS.wav", L"muon bloom Mixed v1 just muons.wav" })
                    wanted.push_back(fs::path(o.root) / L".." / L"audio" / f);
                for (auto& p : wanted)
                    if (fs::exists(p)) lo.audio.push_back(fs::weakly_canonical(p).wstring());
                if (lo.audio.size() != wanted.size()) {
                    // asked for sound and starting mute without a word would be the worst outcome: stop here
                    fprintf(stderr, "--sound: the stems of the previews were not found:\n");
                    for (auto& p : wanted) fprintf(stderr, "    %ls%s\n", fs::weakly_canonical(p).c_str(), fs::exists(p) ? "" : "   <- missing");
                    fprintf(stderr, "give the files with --audio a.wav --audio b.wav, or start without --sound\n");
                    return 2;
                }
            }
        }
        return runLive(o, lo);
    }
    fprintf(stderr, "muonengine live [--from S] [--paused] [--loop] [--sound] [--no-spout] [--offset MS] [--detectors live] [--output N]\n"
                    "muonengine bench [--from S] [--to S] [--workers N] [--lead N] [--fps F]\n"
                    "muonengine render <frames.mbdl> <out prefix> [--float] [--repeat N]\n");
    return 2;
}
