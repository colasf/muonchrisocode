#include "pool.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double wall()
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
}

bool Pool::start(int n, const std::wstring& python, const std::wstring& script, const std::wstring& extra, size_t bufBytes, int bufs,
                 std::string& err)
{
    stop();
    mBufs = bufs;
    mBufBytes = bufBytes;
    mStarted = wall();
    static int generations = 0;                     // a reload starts new workers while the old ones still draw
    const int generation = generations++;
    mJob = CreateJobObjectW(nullptr, nullptr);      // the workers die with the engine, whatever happens to it
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(mJob, JobObjectExtendedLimitInformation, &li, sizeof li);

    mW.resize(n);
    mSlots.assign((size_t)n * bufs, Slot());
    for (int k = 0; k < n; k++) {
        Worker& w = mW[k];
        wchar_t name[96];
        swprintf(name, 96, L"muonengine-%lu-%d-%d", GetCurrentProcessId(), generation, k);
        uint64_t total = (uint64_t)bufBytes * bufs;
        w.mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(total >> 32), (DWORD)total, name);
        w.mem = w.mapping ? (uint8_t*)MapViewOfFile(w.mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0) : nullptr;
        if (!w.mem) { err = "cannot create the shared memory of a worker"; return false; }

        SECURITY_ATTRIBUTES sa = { sizeof sa, nullptr, TRUE };
        HANDLE inRead, inWrite, outRead, outWrite;
        if (!CreatePipe(&inRead, &inWrite, &sa, 0) || !CreatePipe(&outRead, &outWrite, &sa, 1 << 16)) { err = "cannot create a pipe"; return false; }
        SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);

        wchar_t cmd[2048];
        swprintf(cmd, 2048, L"\"%ls\" \"%ls\" --shm %ls --size %zu --bufs %d%ls", python.c_str(), script.c_str(), name, bufBytes, bufs, extra.c_str());
        STARTUPINFOW si = { sizeof si };
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = inRead;
        si.hStdOutput = outWrite;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION pi = {};
        BOOL ok = CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &si, &pi);
        CloseHandle(inRead);
        CloseHandle(outWrite);
        if (!ok) {
            char b[200];
            snprintf(b, sizeof b, "cannot start the Python worker (error %lu): is '%ls' the interpreter?", GetLastError(), python.c_str());
            err = b;
            CloseHandle(inWrite);
            CloseHandle(outRead);
            return false;
        }
        AssignProcessToJobObject(mJob, pi.hProcess);
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        w.process = pi.hProcess;
        w.toChild = inWrite;
        w.fromChild = outRead;
        w.reader = std::thread([this, k, outRead] {
            std::string acc;
            char buf[4096];
            DWORD got;
            while (ReadFile(outRead, buf, sizeof buf, &got, nullptr) && got) {
                acc.append(buf, got);
                size_t nl;
                while ((nl = acc.find('\n')) != std::string::npos) {
                    std::string line = acc.substr(0, nl);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    acc.erase(0, nl + 1);
                    std::lock_guard<std::mutex> g(mLock);
                    mInbox.push_back({ k, std::move(line) });
                }
            }
            std::lock_guard<std::mutex> g(mLock);
            mInbox.push_back({ k, "X" });           // the worker is gone
        });
    }
    return true;
}

void Pool::stop()
{
    for (auto& w : mW) {
        if (w.toChild) {
            DWORD n;
            WriteFile(w.toChild, "Q\n", 2, &n, nullptr);
            CloseHandle(w.toChild);
        }
    }
    for (auto& w : mW) {
        if (w.process) {
            TerminateProcess(w.process, 0);         // nothing to save: do not wait for Python to wind down
            CloseHandle(w.process);
        }
        if (w.reader.joinable()) w.reader.join();
        if (w.fromChild) CloseHandle(w.fromChild);
        if (w.mem) UnmapViewOfFile(w.mem);
        if (w.mapping) CloseHandle(w.mapping);
    }
    mW.clear();
    mSlots.clear();
    mDead = mHung = 0;
    warmFailed = 0;
    warmError.clear();
    looks.clear();
    {
        std::lock_guard<std::mutex> g(mLock);
        mInbox.clear();
    }
    if (mJob) CloseHandle(mJob);
    mJob = nullptr;
    mNext = -1;
}

void Pool::send(int worker, const std::string& line)
{
    DWORD n;
    WriteFile(mW[worker].toChild, line.data(), (DWORD)line.size(), &n, nullptr);
}

void Pool::pump()
{
    std::deque<Msg> in;
    {
        std::lock_guard<std::mutex> g(mLock);
        in.swap(mInbox);
    }
    for (auto& m : in) {
        Worker& w = mW[m.worker];
        const char* s = m.line.c_str();
        if (s[0] == 'H') {
            w.hello = true;
            if (looks.empty()) {
                size_t i = 1;
                while (i < m.line.size()) {
                    size_t j = m.line.find(' ', i + 1);
                    if (j == std::string::npos) j = m.line.size();
                    std::string tok = m.line.substr(i + 1, j - i - 1);
                    size_t a = tok.find(':'), b = tok.find(':', a + 1);
                    if (a != std::string::npos && b != std::string::npos)
                        looks.push_back({ atof(tok.c_str()), atof(tok.c_str() + a + 1), tok.substr(b + 1) });
                    i = j;
                }
            }
        } else if (s[0] == 'W') {
            w.busy = std::max(0, w.busy - 1);
            w.warming = std::max(0, w.warming - 1);
            w.since = wall();
            const char* e = strstr(s, " E ");       // "W <ms> E <message>": the scene of that time cannot be built
            if (e) {
                warmFailed++;
                warmError = e + 3;
            }
        } else if (s[0] == 'D' || s[0] == 'E') {
            w.busy = std::max(0, w.busy - 1);
            w.since = wall();
            char* end;
            uint64_t job = strtoull(s + 2, &end, 10);
            for (int b = 0; b < mBufs; b++) {
                Slot& sl = mSlots[(size_t)m.worker * mBufs + b];
                if (sl.state != BUSY || sl.job != job) continue;
                if (s[0] == 'D') {
                    sl.frame.bytes = std::min<size_t>(strtoul(end, &end, 10), mBufBytes);
                    sl.frame.ms = strtof(end, nullptr);
                    // asked before a flush: not drawn, but kept until the renderer has taken its glyphs
                    // (the worker has marked them as sent)
                    sl.state = sl.orphan ? ORPHAN : READY;
                    if (!sl.orphan) {
                        stats.done++;
                        stats.msSum += sl.frame.ms;
                        stats.msMax = std::max(stats.msMax, (double)sl.frame.ms);
                        stats.ms.push_back(sl.frame.ms);
                    }
                } else {
                    if (s[0] == 'E') {
                        stats.failed++;
                        lastError = end + (*end == ' ');
                        lastErrorFrame = sl.frame.index;
                    }
                    sl.state = FREE;
                }
                sl.orphan = false;
            }
        } else if (s[0] == 'X') {
            mDead++;
            w.hello = false;
            w.busy = 1 << 20;           // never idle again
        }
    }
    // A worker that does not answer (a scene that loops for ever, a deadlock) would never be noticed: only
    // one that dies is. Stop it, so that it counts as dead and the set is started again.
    const double t = wall();
    for (auto& w : mW) {
        if (w.killed || !w.process || w.busy <= 0 || w.busy >= (1 << 20)) continue;
        if (t - w.since > (w.warming ? WARM_LIMIT : FRAME_LIMIT)) {
            w.killed = true;
            mHung++;
            TerminateProcess(w.process, 3);         // its reader thread then says "X"
        }
    }
}

double Pool::age() const
{
    return wall() - mStarted;
}

int Pool::ready() const
{
    int n = 0;
    for (auto& w : mW) n += w.hello && w.busy < (1 << 20);
    return n;
}

bool Pool::idle(int worker) const
{
    return mW[worker].hello && mW[worker].busy == 0;
}

bool Pool::warm(int worker, double t)
{
    if (!idle(worker)) return false;
    char line[64];
    snprintf(line, sizeof line, "W %.17g\n", t);
    if (mW[worker].busy++ == 0) mW[worker].since = wall();
    mW[worker].warming++;
    send(worker, line);
    return true;
}

void Pool::schedule(int64_t first, int64_t last, double fps)
{
    if (mNext < first) mNext = first;
    for (int k = 0; k < (int)mW.size() && mNext <= last; k++) {
        if (!idle(k)) continue;
        for (int b = 0; b < mBufs; b++) {
            Slot& sl = mSlots[(size_t)k * mBufs + b];
            if (sl.state != FREE) continue;
            sl.state = BUSY;
            sl.orphan = false;
            sl.job = ++mJobId;
            sl.frame.index = mNext;
            sl.frame.data = mW[k].mem + (size_t)b * mBufBytes;
            sl.frame.bytes = 0;
            // the time with all its digits: Python must get exactly k / fps (int(t * 30) depends on the last one)
            char line[96];
            snprintf(line, sizeof line, "F %llu %.17g %d\n", (unsigned long long)sl.job, (double)mNext / fps, b);
            if (mW[k].busy++ == 0) mW[k].since = wall();
            send(k, line);
            mNext++;
            break;
        }
    }
}

const Pool::Frame* Pool::newest(int64_t upTo) const
{
    const Frame* best = nullptr;
    for (auto& s : mSlots)
        if (s.state == READY && s.frame.index <= upTo && (!best || s.frame.index > best->index)) best = &s.frame;
    return best;
}
