#include <stdint.h>
#include <stdio.h>

typedef struct __attribute__((packed)) {
    uint16_t ctrl;
    uint32_t seq;
    uint16_t payload_len;
} FrameHdr;

static uint64_t run_codec(int64_t iters) {
    uint64_t s = 987654321ULL;
    uint64_t acc = 0ULL;
    const uint64_t MASK = 9223372036854775807ULL;
    for (int64_t i = 0; i < iters; i++) {
        s = (s + 1140071481932319848ULL) & MASK;
        uint16_t ver = (uint16_t)(s & 15ULL);
        uint16_t kind = (uint16_t)((s >> 4) & 15ULL);
        uint16_t flags = (uint16_t)((s >> 8) & 255ULL);
        uint16_t raw = (uint16_t)(ver | (kind << 4) | (flags << 8));
        FrameHdr hdr;
        hdr.ctrl = raw;
        hdr.seq = (uint32_t)((s >> 16) & 4294967295ULL);
        hdr.payload_len = (uint16_t)((s >> 48) & 65535ULL);
        uint64_t d_ver = hdr.ctrl & 15ULL;
        uint64_t d_kind = (hdr.ctrl >> 4) & 15ULL;
        uint64_t d_flags = (hdr.ctrl >> 8) & 255ULL;
        uint64_t mixed = (d_ver << 24) ^ (d_kind << 16) ^ (d_flags << 8) ^ (uint64_t)hdr.seq ^ (uint64_t)hdr.payload_len;
        acc = ((acc ^ mixed) * 16777619ULL) & MASK;
    }
    return acc;
}

int main(void) {
    printf("%lld\n", (long long)run_codec(20000000));
    return 0;
}
