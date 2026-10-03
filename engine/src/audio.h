// Sound of the show: the stems are played together (XAudio2) and their position is the clock of the picture.
#pragma once
#include <windows.h>
#include <xaudio2.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

class Audio : private IXAudio2EngineCallback {
public:
    ~Audio() { close(); }

    // Read the WAV files (PCM), all starting at show time 0. They stay in memory for good.
    bool load(const std::vector<std::wstring>& files, std::string& err)
    {
        mStems.clear();
        for (auto& f : files) {
            Stem s;
            if (!load(f, s, err)) { mStems.clear(); return false; }
            mStems.push_back(std::move(s));
        }
        return true;
    }

    // Open the sound output of the machine (again: after it was lost, call this to get it back).
    bool openDevice()
    {
        closeDevice();
        if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) { /* already initialised by somebody else: fine */ }
        if (FAILED(XAudio2Create(&mXa, 0, XAUDIO2_DEFAULT_PROCESSOR))) { mXa.Reset(); return false; }
        if (FAILED(mXa->CreateMasteringVoice(&mMaster))) {
            mMaster = nullptr;
            mXa.Reset();
            return false;
        }
        mXa->RegisterForCallbacks(this);
        mLost = false;
        return true;
    }

    void closeDevice()
    {
        stop();
        if (mXa) mXa->UnregisterForCallbacks(this);
        if (mMaster) mMaster->DestroyVoice();
        mMaster = nullptr;
        mXa.Reset();
    }

    void close()
    {
        closeDevice();
        mStems.clear();
    }

    bool loaded() const { return !mStems.empty(); }             // there is sound to play
    bool ok() const { return !mStems.empty() && mXa; }          // ... and an output to play it on
    bool lost() const { return mLost; }                         // the output reported a fatal error since openDevice()

    double duration() const
    {
        double d = 0;
        for (auto& s : mStems) d = std::max(d, (double)(s.pcm.size() / s.fmt.nBlockAlign) / s.fmt.nSamplesPerSec);
        return d;
    }

    // Start every stem at show time t, on the same sample. Returns the number of stems that could not start.
    int play(double t)
    {
        stop();
        if (!mXa) return (int)mStems.size();
        mStart = t;
        int failed = 0;
        for (auto& s : mStems) {
            uint64_t frames = s.pcm.size() / s.fmt.nBlockAlign;
            uint64_t begin = (uint64_t)(std::max(t, 0.0) * s.fmt.nSamplesPerSec);
            if (begin >= frames) continue;
            if (FAILED(mXa->CreateSourceVoice(&s.voice, &s.fmt))) { s.voice = nullptr; failed++; continue; }
            XAUDIO2_BUFFER b = {};
            b.Flags = XAUDIO2_END_OF_STREAM;
            b.AudioBytes = (UINT32)s.pcm.size();
            b.pAudioData = s.pcm.data();
            b.PlayBegin = (UINT32)begin;
            s.voice->SubmitSourceBuffer(&b);
            s.voice->SetVolume(s.volume * mVolume);
            s.voice->Start(0, 1);
        }
        mXa->CommitChanges(1);
        mOn = true;
        return failed;
    }

    void stop()
    {
        for (auto& s : mStems)
            if (s.voice) {
                s.voice->Stop();
                s.voice->DestroyVoice();
                s.voice = nullptr;
            }
        mOn = false;
    }

    // Show time of the sound being played. False when nothing plays (stopped, or past the end).
    bool position(double& t) const
    {
        if (!mOn || mLost) return false;
        for (auto& s : mStems) {
            if (!s.voice) continue;
            XAUDIO2_VOICE_STATE st;
            s.voice->GetState(&st, 0);
            if (!st.BuffersQueued) continue;
            t = mStart + (double)st.SamplesPlayed / s.fmt.nSamplesPerSec;
            return true;
        }
        return false;
    }

    void volume(float v) { mVolume = v; }           // of everything; takes effect at the next play()

private:
    struct Stem {
        std::vector<uint8_t> pcm;
        WAVEFORMATEX fmt = {};
        IXAudio2SourceVoice* voice = nullptr;
        float volume = 1.0f;
    };

    // IXAudio2EngineCallback (called on XAudio2's own thread)
    void STDMETHODCALLTYPE OnProcessingPassStart() override {}
    void STDMETHODCALLTYPE OnProcessingPassEnd() override {}
    void STDMETHODCALLTYPE OnCriticalError(HRESULT) override { mLost = true; }

    static bool load(const std::wstring& path, Stem& s, std::string& err)
    {
        std::string name;
        for (wchar_t c : path) name += (char)(c < 128 ? c : '?');
        std::ifstream f(path, std::ios::binary);
        char id[4];
        uint32_t size;
        if (!f || !f.read(id, 4) || memcmp(id, "RIFF", 4) || !f.read((char*)&size, 4) || !f.read(id, 4) || memcmp(id, "WAVE", 4)) {
            err = "cannot read the sound file " + name;
            return false;
        }
        bool haveFmt = false;
        while (f.read(id, 4) && f.read((char*)&size, 4)) {
            if (!memcmp(id, "fmt ", 4)) {
                uint8_t raw[40] = {};
                f.read((char*)raw, std::min<uint32_t>(size, sizeof raw));
                if (size > sizeof raw) f.seekg(size - sizeof raw, std::ios::cur);
                memcpy(&s.fmt, raw, 16);
                s.fmt.cbSize = 0;
                if (s.fmt.wFormatTag == 0xFFFE) s.fmt.wFormatTag = raw[24];      // WAVE_FORMAT_EXTENSIBLE: PCM or float
                haveFmt = s.fmt.wFormatTag == WAVE_FORMAT_PCM || s.fmt.wFormatTag == 3;
            } else if (!memcmp(id, "data", 4)) {
                s.pcm.resize(size);
                f.read((char*)s.pcm.data(), size);
                s.pcm.resize((size_t)f.gcount());
                break;
            } else {
                f.seekg(size, std::ios::cur);
            }
            if (size & 1) f.seekg(1, std::ios::cur);
        }
        if (!haveFmt || s.pcm.empty() || !s.fmt.nBlockAlign || !s.fmt.nSamplesPerSec) {
            err = "the sound file " + name + " is not a PCM WAV";
            return false;
        }
        s.pcm.resize(s.pcm.size() - s.pcm.size() % s.fmt.nBlockAlign);
        return true;
    }

    Microsoft::WRL::ComPtr<IXAudio2> mXa;
    IXAudio2MasteringVoice* mMaster = nullptr;
    std::vector<Stem> mStems;
    double mStart = 0;
    float mVolume = 1.0f;
    bool mOn = false;
    std::atomic<bool> mLost{ false };
};

// Show time. It runs on the machine's own timer and is pulled gently onto the sound: the position the
// sound card reports moves in steps of 10 ms, too coarse to pick frames with.
struct Clock {
    bool running = false;
    double t = 0, wall = 0;
    double eMax = -1e9;
    int n = 0;

    static double wallNow()
    {
        LARGE_INTEGER c, f;
        QueryPerformanceCounter(&c);
        QueryPerformanceFrequency(&f);
        return (double)c.QuadPart / (double)f.QuadPart;
    }
    double now() const { return running ? t + (wallNow() - wall) : t; }
    void set(double v)
    {
        t = v;
        wall = wallNow();
        eMax = -1e9;
        n = 0;
    }
    void start()
    {
        wall = wallNow();
        running = true;
    }
    void stop()
    {
        t = now();
        running = false;
    }
    // audio = the position the sound reports now. It is always a little behind (it only moves when a block
    // has been played): the largest value seen over half a second is the true one.
    void follow(double audio)
    {
        double e = audio - now();
        if (e > 0.08 || e < -0.08) { set(audio); return; }
        eMax = std::max(eMax, e);
        if (++n >= 30) {
            t += 0.5 * eMax;
            eMax = -1e9;
            n = 0;
        }
    }
};

// Watches the position the sound reports: a sound output that disappears (a cable, a driver) can leave its
// buffer queued with a position that no longer moves, and a clock that follows it would replay the same
// instant for ever. A position that stands still is not followed; after STALL seconds the sound is given
// up (the show goes on, on the machine's timer) until it is opened again.
struct SoundWatch {
    static constexpr double HOLD = 0.06, STALL = 0.4;
    double last = -1e18, movedAt = 0;

    void reset() { last = -1e18; movedAt = Clock::wallNow(); }
    // 0: follow this position, 1: it is not moving, ignore it for now, 2: the sound has stalled
    int check(double pos)
    {
        double w = Clock::wallNow();
        if (pos != last) { last = pos; movedAt = w; return 0; }
        return w - movedAt < HOLD ? 0 : w - movedAt < STALL ? 1 : 2;
    }
};
