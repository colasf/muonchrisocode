// The scene workers: Python processes (engine/worker.py) that run the scenes of the show and write their
// draw lists in shared memory. The scenes are pure functions of the show time, so the frames to come are
// computed in parallel, a few frames ahead of the clock: that is what makes 60 fps out of scenes that take
// up to 30 ms each.
#pragma once
#include <windows.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct PoolStats {
    uint64_t done = 0, failed = 0;
    double msSum = 0, msMax = 0;        // time a worker took for a frame (scene + recording)
    std::vector<float> ms;              // every frame since the last reset
    void reset() { *this = PoolStats(); }
};

class Pool {
public:
    struct Frame {                      // a frame a worker has finished
        int64_t index = -1;
        const uint8_t* data = nullptr;
        size_t bytes = 0;
        float ms = 0;
    };

    // Seconds a worker may take to answer before it is taken for hung and stopped (a scene that builds its
    // world for a new tower placement takes up to half a minute; a warm-up builds several).
    static constexpr double FRAME_LIMIT = 60.0, WARM_LIMIT = 240.0;

    ~Pool() { stop(); }

    // n workers with `bufs` buffers of bufBytes each. python = the interpreter, script = engine/worker.py.
    // extra = more arguments for the script.
    bool start(int n, const std::wstring& python, const std::wstring& script, const std::wstring& extra, size_t bufBytes, int bufs,
               std::string& err);
    void stop();

    void pump();                        // read what the workers said since the last call; stop the hung ones
    int ready() const;                  // workers that have answered "H"
    int workers() const { return (int)mW.size(); }
    int dead() const { return mDead; }  // workers that have stopped (an import that failed, a crash, hung and stopped)
    int hung() const { return mHung; }  // ... of which: stopped by pump() because they did not answer
    double age() const;                 // seconds since start()

    // Ask for the frames first..last (frame k is the show at time k / fps) as far as there are idle workers
    // and free buffers. Frames before `first` that were never asked are skipped.
    void schedule(int64_t first, int64_t last, double fps);
    // Warm up: an idle worker builds the scene of time t. False if nobody was idle.
    bool warm(int worker, double t);
    bool idle(int worker) const;

    // The finished frame with the highest index <= upTo, or nullptr. It stays valid until release().
    const Frame* newest(int64_t upTo) const;
    // Free the buffers of the finished frames with index <= upTo; `each` is called on every one first
    // (the renderer must see every blob once: glyphs travel only in the first frame that uses them).
    template <class F> void release(int64_t upTo, F each)
    {
        for (auto& s : mSlots)
            if (s.state == READY && s.frame.index <= upTo) {
                each(s.frame);
                s.state = FREE;
            }
    }
    // Forget everything that was asked (after a jump of the clock): what arrives late is not drawn.
    template <class F> void flush(F each)
    {
        for (auto& s : mSlots) {
            if (s.state == READY || s.state == ORPHAN) each(s.frame);
            if (s.state == BUSY) s.orphan = true;
            else s.state = FREE;
        }
        mNext = -1;
    }
    // The frames that arrived after a flush: not drawn, but `each` must still see them (their glyphs).
    template <class F> void orphans(F each)
    {
        for (auto& s : mSlots)
            if (s.state == ORPHAN) {
                each(s.frame);
                s.state = FREE;
            }
    }

    struct Look { double t0, t1; std::string name; };
    std::vector<Look> looks;            // the scenes of the show, as the workers give them

    PoolStats stats;
    std::string lastError;              // message of the last frame that failed
    int64_t lastErrorFrame = -1;
    int warmFailed = 0;                 // warm-ups that raised since start() ...
    std::string warmError;              // ... and what the last one said

private:
    enum State { FREE, BUSY, READY, ORPHAN };
    struct Slot {
        State state = FREE;
        bool orphan = false;            // asked before a flush: not to be drawn when it arrives
        uint64_t job = 0;
        Frame frame;
    };
    struct Worker {
        HANDLE process = nullptr, toChild = nullptr, fromChild = nullptr, mapping = nullptr;
        uint8_t* mem = nullptr;
        std::thread reader;
        bool hello = false;
        int busy = 0;                   // orders not answered yet
        int warming = 0;                // ... of which warm-ups
        double since = 0;               // when the oldest of them was sent (or the last answer came)
        bool killed = false;
    };
    struct Msg { int worker; std::string line; };

    void send(int worker, const std::string& line);

    std::vector<Worker> mW;
    std::vector<Slot> mSlots;           // worker * mBufs + buffer
    int mBufs = 0;
    int mDead = 0, mHung = 0;
    size_t mBufBytes = 0;
    HANDLE mJob = nullptr;
    std::mutex mLock;
    std::deque<Msg> mInbox;
    uint64_t mJobId = 0;
    int64_t mNext = -1;                 // next frame to ask
    double mStarted = 0;
};
