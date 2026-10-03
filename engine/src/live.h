#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "player.h"

struct LiveOptions {
    // WAV stems the engine plays itself, all starting at show time 0. Empty (the show): Ableton plays the
    // sound and the time comes by OSC (/muonbloom/time). Not empty (work at the desk): that sound is the clock.
    std::vector<std::wstring> audio;
    std::string spout = "MuonBloom";    // name of the Spout sender ("" = none)
    double from = 0.0;                  // show time to start at
    double offset = 0.0;                // seconds the picture is drawn ahead of the clock (to make up for the delay of the output chain)
    int oscPort = 9000;                 // UDP port for OSC (0 = none)
    // Who may give orders to the transport (play, pause, seek, reload) over OSC: IPv4 addresses, host byte
    // order. Empty = anybody on the network. The detector values are taken from anybody.
    std::vector<uint32_t> oscAllow = { 0x7F000001u };
    std::string detectors = "scripted"; // scripted: the hits of the previews; live: from OSC; both
    std::string detPrefix = "/muon";    // OSC address of the detectors: <prefix>/L, /C, /R, or <prefix> with three floats
    // Trigger level of the detectors L, C, R (0..1): a value that rises above it is a hit. Negative = not
    // given: the levels of engine/detectors.json are used, else 0.10. Changed while the show runs with the
    // keys (D, 1 2 3 0, Up / Down) or by OSC (/muonbloom/level), and then written to that file.
    float detLevel[3] = { -1.0f, -1.0f, -1.0f };
    bool paused = false;
    bool towers = false;                // start in the tower placement tool
    bool bar = true;                    // the time bar under the picture, in the preview window
    double quitAfter = 0.0;             // seconds of play after which the program stops by itself (tests); 0 = never
    float volume = 1.0f;
    bool loop = false;
    std::wstring log;                   // file the messages of the engine are also written to ("" = none)
    std::wstring position;              // file the show time is written to every second ("" = none): run.bat starts
                                        // the engine again from there after a crash
};

// Exit codes of the live player: 0 = stopped by somebody, 2 = it cannot start as it is set up (a restart
// would not help), 3 = it failed while running (starting it again is the thing to do).
int runLive(const PlayerOptions& o, const LiveOptions& lo);

// Receive the Spout sender `name` ("" = the active one) for a moment and say what arrives.
int runSpoutCheck(const char* name, const wchar_t* dump);
