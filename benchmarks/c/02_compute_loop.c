#include <stdint.h>
#include <stdio.h>

static uint64_t compute_mix(int64_t iters) {
    uint64_t x = 123456789ULL;
    uint64_t acc = 0ULL;
    const uint64_t MASK = 9223372036854775807ULL;
    for (int64_t i = 0; i < iters; i++) {
        x = (x + 1140071481932319848ULL) & MASK;
        uint64_t z = x;
        z = ((z ^ (z >> 30)) * 490278349ULL) & MASK;
        z = ((z ^ (z >> 27)) * 16777619ULL) & MASK;
        z = z ^ (z >> 31);
        acc = (acc ^ z) & MASK;
    }
    return acc;
}

int main(void) {
    printf("%lld\n", (long long)compute_mix(100000000));
    return 0;
}
