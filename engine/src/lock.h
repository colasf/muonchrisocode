// A copy of the show that is tied to this show and to a date: what engine/package.py --expires builds.
//
//   MUON_EXPIRES=YYYYMMDD     the engine does not start after that day (the date of the machine), and stops if
//                             it is running when the day ends. Once it has seen a later date it stays closed,
//                             also when the clock of the machine is set back (a mark in the registry of the user).
//   MUON_LOCK_HASH="<hex>"    SHA-256 of the files of the show as they were packaged (the scenes, the workers,
//                             the shaders, the fonts, the data): the engine only plays those. A file changed,
//                             added or taken away and it does not start; the same before every reload. The
//                             settings of the place (data/towers.json, engine/output.json, detectors.json) are
//                             not in it. tools: engine/package.py computes the same number (lock_hash).
//
// Without the two (engine\build.bat) nothing here does anything: the engine of the desk.
#pragma once

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef MUON_LOCK_HASH
#include <bcrypt.h>
#endif

namespace lock {

#if defined(MUON_EXPIRES) || defined(MUON_LOCK_HASH)
inline constexpr bool on = true;
#else
inline constexpr bool on = false;
#endif

// True after the last day of this copy.
inline bool expired()
{
#ifdef MUON_EXPIRES
    const wchar_t* key = L"Software\\Tyrell\\MuonBloom";
    const DWORD day = (DWORD)(MUON_EXPIRES);
    bool mark = false;
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, key, 0, KEY_READ, &k) == ERROR_SUCCESS) {
        DWORD v = 0, n = sizeof v;
        mark = RegQueryValueExW(k, L"state", nullptr, nullptr, (BYTE*)&v, &n) == ERROR_SUCCESS && v == day;
        RegCloseKey(k);
    }
    SYSTEMTIME st;
    GetLocalTime(&st);
    const bool late = (DWORD)st.wYear * 10000u + st.wMonth * 100u + st.wDay > day;
    if (late && !mark && RegCreateKeyExW(HKEY_CURRENT_USER, key, 0, nullptr, 0, KEY_WRITE, nullptr, &k, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(k, L"state", 0, REG_DWORD, (const BYTE*)&day, sizeof day);
        RegCloseKey(k);
    }
    return late || mark;
#else
    return false;
#endif
}

// The files of the show are the ones that were packaged. Byte code left beside them is removed first (Python
// would run it instead of the file it was made from).
inline bool intact(const std::filesystem::path& root, std::string& why)
{
#ifdef MUON_LOCK_HASH
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const wchar_t* d : { L"muonbloom/__pycache__", L"muonbloom/scenes/__pycache__", L"engine/__pycache__", L"tools/__pycache__" })
        fs::remove_all(root / d, ec);
    auto ext = [](const fs::path& p) {
        std::wstring e = p.extension().wstring();
        for (auto& c : e) c = (wchar_t)towlower(c);
        return e;
    };
    std::vector<std::string> files;
    auto dir = [&](const char* d, std::initializer_list<const wchar_t*> exts) {
        for (auto& e : fs::directory_iterator(root / d, ec)) {
            if (!e.is_regular_file(ec)) continue;
            const std::wstring x = ext(e.path());
            if (exts.size() && std::find_if(exts.begin(), exts.end(), [&](const wchar_t* w) { return x == w; }) == exts.end()) continue;
            std::string name = std::string(d) + "/";
            for (wchar_t c : e.path().filename().wstring()) name += c < 128 ? (char)c : '?';      // (the names of the show are plain)
            files.push_back(name);
        }
    };
    dir("muonbloom", { L".py", L".ttf" });
    dir("muonbloom/scenes", { L".py" });
    dir("engine/shaders", {});
    dir("data", { L".npz" });
    for (const char* f : { "engine/worker.py", "engine/detectors.py", "tools/test_card.py", "subtitletimecode.txt" }) files.push_back(f);
    std::sort(files.begin(), files.end());

    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h = nullptr;
    unsigned char out[32] = {};
    bool ok = BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0 && BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) >= 0;
    std::vector<char> buf;
    for (size_t i = 0; ok && i < files.size(); i++) {
        std::ifstream f(root / fs::path(std::wstring(files[i].begin(), files[i].end())), std::ios::binary);
        buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (!f.is_open()) buf.clear();                      // (a missing file: its name with nothing after it)
        const unsigned long long n = buf.size();
        ok = BCryptHashData(h, (PUCHAR)files[i].c_str(), (ULONG)files[i].size() + 1, 0) >= 0
             && BCryptHashData(h, (PUCHAR)&n, 8, 0) >= 0 && (!n || BCryptHashData(h, (PUCHAR)buf.data(), (ULONG)n, 0) >= 0);
    }
    ok = ok && BCryptFinishHash(h, out, 32, 0) >= 0;
    if (h) BCryptDestroyHash(h);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    char hex[65] = {};
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", out[i]);
    if (!ok || std::string(hex) != MUON_LOCK_HASH) {
        why = "this copy of the engine only plays MUON : BLOOM as it was delivered: a file of the show was changed, added or is missing";
        return false;
    }
#else
    (void)root;
    (void)why;
#endif
    return true;
}

// May this copy run? False: why not.
inline bool allowed(const std::filesystem::path& root, std::string& why)
{
    if (expired()) {
#ifdef MUON_EXPIRES
        char b[160];
        snprintf(b, sizeof b, "this copy of MUON : BLOOM was made to run until %04d-%02d-%02d: it does not start any more",
                 (int)(MUON_EXPIRES) / 10000, (int)(MUON_EXPIRES) / 100 % 100, (int)(MUON_EXPIRES) % 100);
        why = b;
#endif
        return false;
    }
    return intact(root, why);
}

}   // namespace lock
