// What the preview window writes and asks: a small bitmap font drawn with the rectangles of the window
// overlay (the window has no other way to show text: time code, button names, the comment being typed),
// the comments file, the snapshots, and the dialog that chooses the sound files. Never in the Spout output.
#pragma once
#include <windows.h>
#include <commdlg.h>
#include <objbase.h>
#include <wincodec.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "renderer.h"

namespace gui {

// ASCII 32 .. 126, 5 x 7: five columns per character, bit 0 = top row
static const unsigned char FONT[95][5] = {
    { 0x00, 0x00, 0x00, 0x00, 0x00 }, { 0x00, 0x00, 0x5F, 0x00, 0x00 }, { 0x00, 0x07, 0x00, 0x07, 0x00 }, { 0x14, 0x7F, 0x14, 0x7F, 0x14 },
    { 0x24, 0x2A, 0x7F, 0x2A, 0x12 }, { 0x23, 0x13, 0x08, 0x64, 0x62 }, { 0x36, 0x49, 0x55, 0x22, 0x50 }, { 0x00, 0x05, 0x03, 0x00, 0x00 },
    { 0x00, 0x1C, 0x22, 0x41, 0x00 }, { 0x00, 0x41, 0x22, 0x1C, 0x00 }, { 0x14, 0x08, 0x3E, 0x08, 0x14 }, { 0x08, 0x08, 0x3E, 0x08, 0x08 },
    { 0x00, 0x50, 0x30, 0x00, 0x00 }, { 0x08, 0x08, 0x08, 0x08, 0x08 }, { 0x00, 0x60, 0x60, 0x00, 0x00 }, { 0x20, 0x10, 0x08, 0x04, 0x02 },
    { 0x3E, 0x51, 0x49, 0x45, 0x3E }, { 0x00, 0x42, 0x7F, 0x40, 0x00 }, { 0x42, 0x61, 0x51, 0x49, 0x46 }, { 0x21, 0x41, 0x45, 0x4B, 0x31 },
    { 0x18, 0x14, 0x12, 0x7F, 0x10 }, { 0x27, 0x45, 0x45, 0x45, 0x39 }, { 0x3C, 0x4A, 0x49, 0x49, 0x30 }, { 0x01, 0x71, 0x09, 0x05, 0x03 },
    { 0x36, 0x49, 0x49, 0x49, 0x36 }, { 0x06, 0x49, 0x49, 0x29, 0x1E }, { 0x00, 0x36, 0x36, 0x00, 0x00 }, { 0x00, 0x56, 0x36, 0x00, 0x00 },
    { 0x08, 0x14, 0x22, 0x41, 0x00 }, { 0x14, 0x14, 0x14, 0x14, 0x14 }, { 0x00, 0x41, 0x22, 0x14, 0x08 }, { 0x02, 0x01, 0x51, 0x09, 0x06 },
    { 0x32, 0x49, 0x79, 0x41, 0x3E }, { 0x7E, 0x11, 0x11, 0x11, 0x7E }, { 0x7F, 0x49, 0x49, 0x49, 0x36 }, { 0x3E, 0x41, 0x41, 0x41, 0x22 },
    { 0x7F, 0x41, 0x41, 0x22, 0x1C }, { 0x7F, 0x49, 0x49, 0x49, 0x41 }, { 0x7F, 0x09, 0x09, 0x09, 0x01 }, { 0x3E, 0x41, 0x49, 0x49, 0x7A },
    { 0x7F, 0x08, 0x08, 0x08, 0x7F }, { 0x00, 0x41, 0x7F, 0x41, 0x00 }, { 0x20, 0x40, 0x41, 0x3F, 0x01 }, { 0x7F, 0x08, 0x14, 0x22, 0x41 },
    { 0x7F, 0x40, 0x40, 0x40, 0x40 }, { 0x7F, 0x02, 0x0C, 0x02, 0x7F }, { 0x7F, 0x04, 0x08, 0x10, 0x7F }, { 0x3E, 0x41, 0x41, 0x41, 0x3E },
    { 0x7F, 0x09, 0x09, 0x09, 0x06 }, { 0x3E, 0x41, 0x51, 0x21, 0x5E }, { 0x7F, 0x09, 0x19, 0x29, 0x46 }, { 0x46, 0x49, 0x49, 0x49, 0x31 },
    { 0x01, 0x01, 0x7F, 0x01, 0x01 }, { 0x3F, 0x40, 0x40, 0x40, 0x3F }, { 0x1F, 0x20, 0x40, 0x20, 0x1F }, { 0x3F, 0x40, 0x38, 0x40, 0x3F },
    { 0x63, 0x14, 0x08, 0x14, 0x63 }, { 0x07, 0x08, 0x70, 0x08, 0x07 }, { 0x61, 0x51, 0x49, 0x45, 0x43 }, { 0x00, 0x7F, 0x41, 0x41, 0x00 },
    { 0x02, 0x04, 0x08, 0x10, 0x20 }, { 0x00, 0x41, 0x41, 0x7F, 0x00 }, { 0x04, 0x02, 0x01, 0x02, 0x04 }, { 0x40, 0x40, 0x40, 0x40, 0x40 },
    { 0x00, 0x01, 0x02, 0x04, 0x00 }, { 0x20, 0x54, 0x54, 0x54, 0x78 }, { 0x7F, 0x48, 0x44, 0x44, 0x38 }, { 0x38, 0x44, 0x44, 0x44, 0x20 },
    { 0x38, 0x44, 0x44, 0x48, 0x7F }, { 0x38, 0x54, 0x54, 0x54, 0x18 }, { 0x08, 0x7E, 0x09, 0x01, 0x02 }, { 0x0C, 0x52, 0x52, 0x52, 0x3E },
    { 0x7F, 0x08, 0x04, 0x04, 0x78 }, { 0x00, 0x44, 0x7D, 0x40, 0x00 }, { 0x20, 0x40, 0x44, 0x3D, 0x00 }, { 0x7F, 0x10, 0x28, 0x44, 0x00 },
    { 0x00, 0x41, 0x7F, 0x40, 0x00 }, { 0x7C, 0x04, 0x18, 0x04, 0x78 }, { 0x7C, 0x08, 0x04, 0x04, 0x78 }, { 0x38, 0x44, 0x44, 0x44, 0x38 },
    { 0x7C, 0x14, 0x14, 0x14, 0x08 }, { 0x08, 0x14, 0x14, 0x18, 0x7C }, { 0x7C, 0x08, 0x04, 0x04, 0x08 }, { 0x48, 0x54, 0x54, 0x54, 0x20 },
    { 0x04, 0x3F, 0x44, 0x40, 0x20 }, { 0x3C, 0x40, 0x40, 0x20, 0x7C }, { 0x1C, 0x20, 0x40, 0x20, 0x1C }, { 0x3C, 0x40, 0x30, 0x40, 0x3C },
    { 0x44, 0x28, 0x10, 0x28, 0x44 }, { 0x0C, 0x50, 0x50, 0x50, 0x3C }, { 0x44, 0x64, 0x54, 0x4C, 0x44 }, { 0x00, 0x08, 0x36, 0x41, 0x00 },
    { 0x00, 0x00, 0x7F, 0x00, 0x00 }, { 0x00, 0x41, 0x36, 0x08, 0x00 }, { 0x08, 0x04, 0x08, 0x10, 0x08 },
};

// Width of n characters, px = size of one pixel of the font (a character is 5 wide, 1 apart).
inline float textWidth(size_t n, float px) { return n ? (float)(n * 6 - 1) * px : 0.0f; }

// Text with its top left corner at (x, y), 7 px high. Each lit run of a column is one rectangle.
inline void text(std::vector<Renderer::Over>& o, float x, float y, const std::string& s, float px, float r, float g, float b, float a = 1.0f)
{
    for (unsigned char c : s) {
        const unsigned char* col = FONT[(c >= 32 && c <= 126 ? c : '?') - 32];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7;) {
                if (!(col[i] >> j & 1)) { j++; continue; }
                int j0 = j;
                while (j < 7 && (col[i] >> j & 1)) j++;
                o.push_back({ x + i * px, y + j0 * px, x + (i + 1) * px, y + j * px, r, g, b, a });
            }
        }
        x += 6.0f * px;
    }
}

// Show time as in the timings sheet: MM:SS:FF, FF = frames.
inline std::string timecode(double t, double fps)
{
    long long f = (long long)(t * fps + 1e-6);
    if (f < 0) f = 0;
    const long long n = (long long)(fps + 0.5);
    char b[32];
    snprintf(b, sizeof b, "%02lld:%02lld:%02lld", f / n / 60, f / n % 60, f % n);
    return b;
}

inline std::string utf8(const std::wstring& w)
{
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// The comments written in the window while the show is looked at: one line each in <root>/comments.txt,
//     [ ] MM:SS:FF | scene | when it was written | the comment
// "[x]" once the comment has been dealt with (whoever applies it changes the mark). Lines starting with #
// are not comments. A snapshot is a comment too: its line says "snapshot" and the file of the picture, which
// is there to be drawn and written on.
struct Comments {
    struct Mark { double t; bool done; };
    std::filesystem::path path;
    std::vector<Mark> marks;                // where the comments are, for the time bar

    void load(double fps)
    {
        marks.clear();
        std::ifstream f(path);
        std::string ln;
        while (std::getline(f, ln)) {
            int m, s, fr;
            if (ln.size() < 12 || ln[0] != '[' || ln[2] != ']' || sscanf(ln.c_str() + 3, " %d:%d:%d", &m, &s, &fr) != 3) continue;
            marks.push_back({ m * 60.0 + s + fr / fps, ln[1] != ' ' });
        }
    }

    bool add(double t, double fps, const std::string& scene, const std::wstring& what)
    {
        std::error_code ec;
        const bool fresh = !std::filesystem::exists(path, ec);
        std::ofstream f(path, std::ios::app | std::ios::binary);
        if (!f) return false;
        if (fresh)
            f << "# Muon Bloom - comments written in the engine window (key C, or the COMMENT button).\n"
                 "# [ ] MM:SS:FF | scene | when it was written | the comment      (FF = frames, [x] = done)\n"
                 "# A line that says \"snapshot\" names a picture taken there (key S, or the SNAPSHOT button): draw and write on it.\n";
        time_t now = time(nullptr);
        tm lt;
        localtime_s(&lt, &now);
        char when[32];
        strftime(when, sizeof when, "%Y-%m-%d %H:%M", &lt);
        f << "[ ] " << timecode(t, fps) << " | " << scene << " | " << when << " | " << utf8(what) << "\n";
        f.flush();
        if (!f) return false;
        f.close();
        load(fps);
        return true;
    }
};

// A picture (BGRA, rows from the top) written as a PNG file: a snapshot. Packing a picture of the size of
// the show takes a few tenths of a second: call it from a thread of its own.
inline bool savePng(const std::wstring& path, const unsigned char* bgra, unsigned w, unsigned h)
{
    std::vector<unsigned char> bgr((size_t)w * h * 3);
    for (size_t k = 0, n = (size_t)w * h; k < n; k++) {
        bgr[3 * k] = bgra[4 * k];
        bgr[3 * k + 1] = bgra[4 * k + 1];
        bgr[3 * k + 2] = bgra[4 * k + 2];
    }
    const bool com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
    bool ok = false;
    {
        ComPtr<IWICImagingFactory> factory;
        ComPtr<IWICStream> stream;
        ComPtr<IWICBitmapEncoder> enc;
        ComPtr<IWICBitmapFrameEncode> frame;
        WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
        ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))
          && SUCCEEDED(factory->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))
          && SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))
          && SUCCEEDED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache)) && SUCCEEDED(enc->CreateNewFrame(&frame, nullptr))
          && SUCCEEDED(frame->Initialize(nullptr)) && SUCCEEDED(frame->SetSize(w, h)) && SUCCEEDED(frame->SetPixelFormat(&fmt))
          && IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)
          && SUCCEEDED(frame->WritePixels(h, w * 3, (UINT)bgr.size(), bgr.data())) && SUCCEEDED(frame->Commit()) && SUCCEEDED(enc->Commit());
    }
    if (com) CoUninitialize();
    if (!ok) DeleteFileW(path.c_str());             // (half a file is worse than none)
    return ok;
}

// The dialog that chooses the sound: one WAV file or several (stems that start together).
inline bool chooseWav(HWND owner, const std::wstring& dir, std::vector<std::wstring>& files)
{
    std::vector<wchar_t> buf(32768, 0);
    OPENFILENAMEW of = { sizeof of };
    of.hwndOwner = owner;
    of.lpstrFilter = L"Sound (WAV)\0*.wav\0All files\0*.*\0";
    of.lpstrFile = buf.data();
    of.nMaxFile = (DWORD)buf.size();
    of.lpstrInitialDir = dir.empty() ? nullptr : dir.c_str();
    of.lpstrTitle = L"Sound of the show: one WAV file, or several stems that start together";
    of.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_ALLOWMULTISELECT | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;
    if (!GetOpenFileNameW(&of)) return false;
    files.clear();
    std::wstring first = buf.data();
    const wchar_t* p = buf.data() + first.size() + 1;
    if (!*p) files.push_back(first);                    // one file: its whole path
    else for (; *p; p += wcslen(p) + 1) files.push_back((std::filesystem::path(first) / p).wstring());      // folder, then names
    return !files.empty();
}

}  // namespace gui
