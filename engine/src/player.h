// The show, live: the workers compute the frames to come, the renderer draws the one that is due.
#pragma once
#include <cmath>
#include <memory>
#include <string>

#include "gpu.h"
#include "pool.h"
#include "renderer.h"

struct PlayerOptions {
    int workers = 6;
    int lead = 6;                       // frames asked ahead of the clock (a scene may take 60 ms: 4 frames is too short)
    double fps = 60.0;
    size_t bufBytes = 48u << 20;        // room for one draw list
    std::wstring python = L"python";
    std::wstring root;                  // folder of the repository (holds engine/worker.py and muonbloom/)
    std::wstring shaders;
    std::wstring workerArgs;            // more arguments for engine/worker.py
    bool debug = false;
};

struct PlayerStats {
    uint64_t ticks = 0;                 // frame periods that went by
    uint64_t fresh = 0;                 // ... with their own frame on screen in time
    uint64_t late = 0;                  // ... with their own frame, but after the start of the period
    uint64_t stale = 0;                 // ... that kept an older picture (a dropped frame)
    void reset() { *this = PlayerStats(); }
};

class Player {
public:
    Gpu gpu;
    Renderer renderer;
    std::unique_ptr<Pool> pool;         // the workers in charge
    PlayerOptions opt;
    PlayerStats stats;
    std::string error;                  // last error of the renderer (cleared by whoever reports it)
    std::string reloadError;            // why the last reload was given up
    // Frames drawn without something they asked for, since the caller last cleared them: a post-process the
    // recorder does not know (the picture lacks the effect), glyphs the renderer never received.
    uint64_t unknownPost = 0, missingGlyphs = 0;

    bool start(const PlayerOptions& o, std::string& err)
    {
        opt = o;
        if (!gpu.create(o.debug, err)) return false;
        if (!renderer.init(gpu.dev.Get(), gpu.ctx.Get(), o.shaders, err)) return false;
        pool = std::make_unique<Pool>();
        return pool->start(o.workers, o.python, o.root + L"/engine/worker.py", o.workerArgs, o.bufBytes, 2, err);
    }

    int64_t frameAt(double t) const { return (int64_t)std::floor(t * opt.fps + 1e-6); }

    // Call as often as possible with the show time. Returns true when a new picture has been drawn
    // (renderer.out()).
    bool tick(double t)
    {
        const int64_t n = frameAt(t);
        pool->pump();
        auto ingest = [this](const Pool::Frame& f) { if (f.index != mDrawn) renderer.ingest(f.data, f.bytes); };
        // frames asked before a jump of the clock and finished since: never drawn, but their glyphs are taken
        pool->orphans([this](const Pool::Frame& f) { renderer.ingest(f.data, f.bytes); });
        if (mTick >= 0 && (n < mTick || n > mTick + (int64_t)opt.fps)) {     // the clock jumped
            pool->flush(ingest);
            mShown = -1;
            mTick = -1;
        }
        if (n != mTick) {
            if (mTick >= 0) {
                int64_t gone = n - mTick;                                  // usually 1
                stats.ticks += gone;
                if (mShown == mTick) (mLate ? stats.late : stats.fresh)++;
                else stats.stale++;
                if (gone > 1) stats.stale += gone - 1;
            }
            pool->release(n - 1, ingest);
            mTick = n;
            mTickStart = now();
            mLate = false;
        }
        pool->schedule(n, n + opt.lead, opt.fps);
        if (mShown >= n) return false;
        const Pool::Frame* f = pool->newest(n);
        if (!f || f->index <= mShown) return false;
        mDrawn = f->index;
        if (!renderer.render(f->data, f->bytes, false, error)) return false;
        if (renderer.info().flags & dl::POST_UNKNOWN) unknownPost++;
        missingGlyphs += renderer.info().missing_glyphs;
        mShown = f->index;
        mLate = now() - mTickStart > 0.003;
        pool->release(mShown, ingest);                                  // its buffer, and those of the frames it overtook
        return true;
    }

    // The picture on screen is not the one of the clock any more (the towers moved ...): draw it again.
    void invalidate() { mShown = -1; }
    int64_t shown() const { return mShown; }

    // ---- reload: a new set of workers takes over once it is ready -----------------------------------
    // Called when a file of the show has changed. The workers in charge keep drawing with the code they
    // have; if the new ones cannot start (a syntax error ...) or cannot draw the scene that is playing,
    // nothing changes on screen. Called during a reload, it starts that reload again (the files changed
    // once more: what was being started is already old).
    void reload()
    {
        reloadError.clear();
        mNext.reset();
        mNext = std::make_unique<Pool>();
        std::string err;
        if (!mNext->start(opt.workers, opt.python, opt.root + L"/engine/worker.py", opt.workerArgs, opt.bufBytes, 2, err)) {
            reloadError = err;
            mNext.reset();
        }
        mNextWarmed = false;
    }
    bool reloading() const { return mNext != nullptr; }
    static constexpr double START_LIMIT = 90.0;     // seconds a new set of workers may take to import the show

    // Call every frame with the show time. Returns true when the new workers have just taken over.
    bool reloadStep(double t)
    {
        if (!mNext) return false;
        mNext->pump();
        if (mNext->dead()) {
            reloadError = "the new code does not start (see the console): still running the previous one";
            mNext.reset();
            return false;
        }
        if (mNext->ready() < mNext->workers()) {
            if (mNext->age() > START_LIMIT) {       // never said hello: without this, nothing would ever restart
                reloadError = "the new scene workers did not start in time: still running the previous ones";
                mNext.reset();
            }
            return false;
        }
        if (!mNextWarmed) {                         // build the scene that is playing before taking over
            for (int k = 0; k < mNext->workers(); k++) mNext->warm(k, t);
            mNextWarmed = true;
            return false;
        }
        for (int k = 0; k < mNext->workers(); k++)
            if (!mNext->idle(k)) return false;
        if (mNext->warmFailed) {                    // scene modules are imported when first drawn: this is where
            reloadError = "the scene that is playing fails with the new code (" + mNext->warmError    // a broken one shows
                        + "): still running the previous one";
            mNext.reset();
            return false;
        }
        pool->flush([this](const Pool::Frame& f) { renderer.ingest(f.data, f.bytes); });
        pool = std::move(mNext);
        mShown = -1;
        mTick = -1;
        mWarm.clear();
        mWarming.clear();
        return true;
    }

    // ---- warm-up: the scenes are built before they are needed ----------------------------------------
    // Give every worker the list of the scenes to build (two times per look).
    void warmAll(double from = 0.0, double to = 1e9)
    {
        std::vector<double> times;
        for (auto& l : pool->looks)
            if (l.t1 > from && l.t0 < to) {
                times.push_back(std::max(l.t0, from) + 0.5);
                times.push_back(0.5 * (std::max(l.t0, from) + std::min(l.t1, to)));
            }
        mWarm.assign(pool->workers(), times);
    }
    // Send the next warm-up orders, to at most `atOnce` workers at a time (the others keep drawing).
    // Returns true while there is something left to build.
    bool warmStep(int atOnce)
    {
        const int n = pool->workers();
        if ((int)mWarm.size() != n) return false;
        mWarming.resize(n, false);
        int busy = 0, left = 0;
        for (int k = 0; k < n; k++) {
            if (mWarming[k] && pool->idle(k)) mWarming[k] = false;
            busy += mWarming[k];
        }
        for (int k = n - 1; k >= 0; k--) {          // the last workers first: the first ones are given the frames
            if (!mWarm[k].empty() && busy < atOnce && pool->warm(k, mWarm[k].back())) {
                mWarm[k].pop_back();
                mWarming[k] = true;
                busy++;
            }
            left += (int)mWarm[k].size() + (mWarming[k] ? 1 : 0);
        }
        return left > 0;
    }

private:
    int64_t mTick = -1;                 // frame period the clock is in
    int64_t mShown = -1;                // frame on screen
    int64_t mDrawn = -1;
    double mTickStart = 0;
    bool mLate = false;
    std::unique_ptr<Pool> mNext;        // the workers being started by a reload
    bool mNextWarmed = false;
    std::vector<std::vector<double>> mWarm;
    std::vector<bool> mWarming;
};
