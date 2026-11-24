#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint64_t seed;
    int64_t iters;
    uint64_t out;
} Work;

static void* worker_fn(void* arg) {
    Work* w = (Work*)arg;
    uint64_t x = w->seed;
    uint64_t acc = 0ULL;
    const uint64_t MASK = 9223372036854775807ULL;
    for (int64_t i = 0; i < w->iters; i++) {
        x = (x + 1140071481932319848ULL) & MASK;
        uint64_t z = x;
        z = ((z ^ (z >> 30)) * 490278349ULL) & MASK;
        z = ((z ^ (z >> 27)) * 16777619ULL) & MASK;
        z = z ^ (z >> 31);
        acc = (acc ^ z) & MASK;
    }
    w->out = acc;
    return NULL;
}

int main(void) {
    const uint64_t seeds[4] = {100000007ULL, 200000033ULL, 300000007ULL, 400000009ULL};
    Work works[4];
    pthread_t th[4];
    for (int i = 0; i < 4; i++) {
        works[i].seed = seeds[i];
        works[i].iters = 25000000;
        works[i].out = 0;
        pthread_create(&th[i], NULL, worker_fn, &works[i]);
    }
    uint64_t total = 0;
    const uint64_t MASK = 9223372036854775807ULL;
    for (int i = 0; i < 4; i++) {
        pthread_join(th[i], NULL);
        total = (total ^ works[i].out) & MASK;
    }
    printf("%lld\n", (long long)total);
    return 0;
}
