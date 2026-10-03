// OSC over UDP, receiving side: the time of the show, the three detectors, orders for the transport.
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct OscMsg {
    std::string addr;
    std::vector<double> args;           // numbers only (f, i, d, h, T, F), in order; always finite
    uint32_t from = 0;                  // IPv4 address of the sender, host byte order (127.0.0.1 = 0x7F000001)
};

class OscIn {
public:
    ~OscIn() { close(); }

    bool open(int port)
    {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w)) return false;
        mSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (mSock == INVALID_SOCKET) return false;
        u_long nb = 1;
        ioctlsocket(mSock, FIONBIO, &nb);
        sockaddr_in a = {};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((u_short)port);
        if (bind(mSock, (sockaddr*)&a, sizeof a)) { close(); return false; }
        return true;
    }

    void close()
    {
        if (mSock != INVALID_SOCKET) closesocket(mSock);
        mSock = INVALID_SOCKET;
    }

    // Everything that has arrived since the last call.
    void poll(std::vector<OscMsg>& out)
    {
        if (mSock == INVALID_SOCKET) return;
        char buf[4096];
        for (;;) {
            sockaddr_in src = {};
            int sl = sizeof src;
            int n = recvfrom(mSock, buf, sizeof buf, 0, (sockaddr*)&src, &sl);
            if (n <= 0) return;                     // nothing left (or a datagram larger than the buffer: dropped)
            size_t first = out.size();
            parse((const uint8_t*)buf, (size_t)n, out, 0);
            for (size_t k = first; k < out.size(); k++) out[k].from = ntohl(src.sin_addr.s_addr);
        }
    }

    // "127.0.0.1" -> 0x7F000001; 0 if it is not an IPv4 address.
    static uint32_t ipv4(const std::string& s)
    {
        in_addr a = {};
        return inet_pton(AF_INET, s.c_str(), &a) == 1 ? ntohl(a.s_addr) : 0;
    }

private:
    static uint32_t be32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }

    static void parse(const uint8_t* p, size_t n, std::vector<OscMsg>& out, int depth)
    {
        if (n >= 16 && !memcmp(p, "#bundle", 8)) {
            size_t i = 16;
            while (i + 4 <= n && depth < 4) {
                uint32_t len = be32(p + i);
                i += 4;
                if (len > n - i) return;
                parse(p + i, len, out, depth + 1);
                i += len;
            }
            return;
        }
        if (!n || p[0] != '/') return;
        size_t a = strnlen((const char*)p, n);
        if (a == n) return;
        OscMsg m;
        m.addr.assign((const char*)p, a);
        size_t i = (a + 4) & ~(size_t)3;
        if (i < n && p[i] == ',') {
            size_t tl = strnlen((const char*)p + i, n - i);
            if (i + tl == n) return;
            size_t d = i + ((tl + 4) & ~(size_t)3);
            for (size_t k = 1; k < tl; k++) {
                char t = (char)p[i + k];
                if (t == 'f' || t == 'i') {
                    if (d + 4 > n) return;
                    uint32_t v = be32(p + d);
                    d += 4;
                    float f;
                    if (t == 'f') memcpy(&f, &v, 4);
                    else f = (float)(int32_t)v;
                    if (!std::isfinite(f)) return;  // a NaN (a division by zero upstream) must never reach the clock
                    m.args.push_back(f);
                } else if (t == 'd' || t == 'h') {
                    if (d + 8 > n) return;
                    uint64_t v = ((uint64_t)be32(p + d) << 32) | be32(p + d + 4);
                    d += 8;
                    double f;
                    if (t == 'd') memcpy(&f, &v, 8);
                    else f = (double)(int64_t)v;
                    if (!std::isfinite(f) || !std::isfinite((float)f)) return;
                    m.args.push_back(f);
                } else if (t == 'T' || t == 'F') {
                    m.args.push_back(t == 'T' ? 1.0 : 0.0);
                } else if (t == 's') {
                    if (d >= n) return;
                    size_t sl = strnlen((const char*)p + d, n - d);
                    d += (sl + 4) & ~(size_t)3;
                } else {
                    break;                          // a type this engine has no use for
                }
            }
        }
        out.push_back(std::move(m));
    }

    SOCKET mSock = INVALID_SOCKET;
};

// The detector values, with the show time they belong to, in shared memory: the scene workers read them
// (engine/detectors.py). One entry per poll that brought something; `set` says which detectors spoke in it
// (bit 0 = L, 1 = C, 2 = R), so that a detector that only sends a message when it is hit can be told from
// one that streams its value.
class DetectorRing {
public:
    static constexpr uint32_t CAPACITY = 1 << 20;   // about 17 minutes at 1000 entries a second: a whole show
    struct Entry { double t; float v[3]; uint32_t set; };

    bool create(const wchar_t* name)
    {
        size_t bytes = 64 + (size_t)CAPACITY * sizeof(Entry);
        mMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, (DWORD)bytes, name);
        mMem = mMap ? (uint8_t*)MapViewOfFile(mMap, FILE_MAP_ALL_ACCESS, 0, 0, 0) : nullptr;
        if (!mMem) return false;
        memset(mMem, 0, 64);
        memcpy(mMem, "MBDET\0\0", 8);
        memcpy(mMem + 16, &CAPACITY, 4);
        mName = name;
        writeLevels();
        return true;
    }

    ~DetectorRing()
    {
        if (mMem) UnmapViewOfFile(mMem);
        if (mMap) CloseHandle(mMap);
    }

    // Detector k (0 = L, 1 = C, 2 = R) has the value v at show time t.
    void set(int k, float v, double t)
    {
        if (!mMem || k < 0 || k > 2 || !std::isfinite(v) || !std::isfinite(t)) return;
        mLast[k] = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
        mSet |= 1u << k;
        mT = t;
    }
    // Write what set() gathered (one entry for the messages of one poll).
    void commit()
    {
        if (!mMem || !mSet) return;
        Entry e = { mT, { mLast[0], mLast[1], mLast[2] }, mSet };
        memcpy(mMem + 64 + (size_t)(mCount % CAPACITY) * sizeof(Entry), &e, sizeof e);
        mCount++;
        memcpy(mMem + 8, &mCount, 8);               // the count last: a reader never sees a half-written entry
        mSet = 0;
    }
    void clear()                                    // after a jump of the clock: the past is not the past any more
    {
        mCount = 0;
        mSet = 0;
        if (mMem) memcpy(mMem + 8, &mCount, 8);
        if (mMem) mEpoch++, memcpy(mMem + 20, &mEpoch, 4);
        writeLevels();
    }
    // The trigger level of each detector (0..1): a value that rises above it is a hit. It is written in the
    // stream, between the values, so that every worker changes level on the same value; and in the header,
    // for a worker that starts when the beginning of the stream is gone.
    void levels(const float lv[3])
    {
        for (int k = 0; k < 3; k++) mLevel[k] = lv[k] < 0.001f ? 0.001f : lv[k] > 1.0f ? 1.0f : lv[k];
        writeLevels();
    }
    const float* levels() const { return mLevel; }
    const std::wstring& name() const { return mName; }
    uint64_t count() const { return mCount; }
    const float* last() const { return mLast; }

private:
    void writeLevels()
    {
        if (!mMem) return;
        memcpy(mMem + 24, mLevel, 12);
        Entry e = { mT, { mLevel[0], mLevel[1], mLevel[2] }, CONFIG };
        memcpy(mMem + 64 + (size_t)(mCount % CAPACITY) * sizeof(Entry), &e, sizeof e);
        mCount++;
        memcpy(mMem + 8, &mCount, 8);
    }

    static constexpr uint32_t CONFIG = 0x80000000u; // in Entry::set: the entry holds the trigger levels, not values
    float mLevel[3] = { 0.10f, 0.10f, 0.10f };
    HANDLE mMap = nullptr;
    uint8_t* mMem = nullptr;
    std::wstring mName;
    uint64_t mCount = 0;
    uint32_t mEpoch = 0;
    uint32_t mSet = 0;
    float mLast[3] = { 0, 0, 0 };
    double mT = 0;
};
