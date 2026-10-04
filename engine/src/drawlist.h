// The draw-list blob: one frame of the show as recorded by muonbloom/drawlist.py (the format is described
// there; this file mirrors it). Little endian, the same bytes in a file and in shared memory.
#pragma once
#include <cstdint>
#include <cstring>

namespace dl {

constexpr uint32_t VERSION = 1;
constexpr size_t HEADER = 256;
constexpr size_t SECTION_TABLE = 96;

enum Section { STATES, SEGS, DOTS, SPLATS, RECTS, LIGHTOPS, OCCL, TEXTOPS, CHARS, GLYPHS, INVERT, POSTOPS, NOGLOW, NSEC };
// NOGLOW came without a new VERSION: a blob made before it has zeros there (no box).

enum Flags : uint32_t { POST_UNKNOWN = 1, PALETTE = 2 };

#pragma pack(push, 1)
struct Header {
    char magic[4];              // "MBDL"
    uint32_t version, bytes, flags;
    double t;                   // show time, seconds
    uint32_t W, H;              // pixels
    float s;                    // W / 2978
    float exposure, bloom_gain, text_gain;
    float bloom_w[8];
    uint32_t dither_seed, frame;
};
struct SectionRef { uint32_t offset, count, bytes; };

struct State { float vz, vcx, vcy, vsx, vsy, pad0, cx0, cy0, cx1, cy1, pad1, pad2; };
struct Seg { float x0, y0, x1, y1, i0, i1, width, spacing, meta; };
struct Dot { float x, y, r, i, meta; };
struct Splat { float x, y, w, layer; };
struct Rect { float x0, y0, x1, y1, i, meta; };
struct LightOp { int32_t x0, y0, x1, y1; float fw, fr; uint32_t nseg, ndot, nsplat, nrect; };
struct Box { int32_t x0, y0, x1, y1; };
struct TextOp { uint32_t kind; int32_t a, b, c, d; float cw, cr, fw, fr; uint32_t off, cnt, pad; };
struct GlyphDef { uint32_t font, code; int32_t adv, ox, oy; uint32_t w, h; };   // followed by w * h bytes, padded to 4
struct PostOp { uint32_t kind; int32_t x0, y0, x1, y1; float p0, p1, p2; };
#pragma pack(pop)

enum TextKind : uint32_t { T_RECT = 0, T_RUN = 1, T_VRUN = 2 };
enum PostKind : uint32_t { P_ROLL_X = 1, P_REPEAT, P_SMEAR_Y, P_SMEAR_X, P_TO_RED, P_TO_WHITE, P_ROLL_Y };

// A blob, checked and ready to read.
struct View {
    const uint8_t* p = nullptr;
    size_t size = 0;
    const Header* h = nullptr;
    SectionRef sec[NSEC] = {};

    bool open(const uint8_t* data, size_t n)
    {
        if (n < HEADER || memcmp(data, "MBDL", 4) != 0) return false;
        h = (const Header*)data;
        if (h->version != VERSION || h->bytes > n) return false;
        p = data;
        size = h->bytes;
        static const size_t stride[NSEC] = { sizeof(State), sizeof(Seg), sizeof(Dot), sizeof(Splat), sizeof(Rect), sizeof(LightOp),
                                             sizeof(Box), sizeof(TextOp), 4, 0, sizeof(Box), sizeof(PostOp), sizeof(Box) };
        for (int k = 0; k < NSEC; k++) {
            memcpy(&sec[k], data + SECTION_TABLE + 12 * k, 12);
            if ((size_t)sec[k].offset + sec[k].bytes > size) return false;
            if (stride[k] && (size_t)sec[k].count * stride[k] != sec[k].bytes) return false;
        }
        return true;
    }
    template <class T> const T* at(Section s) const { return (const T*)(p + sec[s].offset); }
    uint32_t count(Section s) const { return sec[s].count; }
};

}  // namespace dl
