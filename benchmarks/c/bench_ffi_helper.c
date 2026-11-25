#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    int64_t x, y, z;
} Vec3;

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

__attribute__((noinline))
int64_t tn_bench_c_hash(int64_t x) {
    return ((x * 6364136223846793005LL + 1442695040888963407LL) >> 17) ^ x;
}

#ifdef STANDALONE_C_BENCH
int main(void) {
    const int64_t FFI = 1000000;
    int64_t t0 = now_ns();
    volatile int64_t a = 42;
    int64_t acc = 42;
    for (int64_t i = 0; i < FFI; i++) {
        acc = tn_bench_c_hash(acc + i);
    }
    a = acc;
    int64_t d_ffi = now_ns() - t0;

    const int64_t ITERS = 50000000;
    int64_t t1 = now_ns();
    Vec3 p = { 1, 2, 3 };
    for (int64_t i = 0; i < ITERS; i++) {
        p.x = p.x + p.y + 1;
        p.y = p.y ^ (p.x >> 3);
        p.z = p.z + p.y - (i & 7);
    }
    int64_t d_st = now_ns() - t1;
    printf("C_FFI_1M ms=%.3f ns_per_call=%.2f checksum=%lld\n", (double)d_ffi / 1e6, (double)d_ffi / (double)FFI, (long long)a);
    printf("C_MUT_STRUCT_50M ms=%.3f ns_per_iter=%.2f checksum=%lld\n", (double)d_st / 1e6, (double)d_st / (double)ITERS, (long long)(p.x + p.y + p.z));
    return 0;
}
#endif
