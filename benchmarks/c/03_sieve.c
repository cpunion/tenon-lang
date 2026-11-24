#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t sieve_count(int64_t limit) {
    int64_t* flags = (int64_t*)malloc((size_t)(limit + 1) * sizeof(int64_t));
    for (int64_t i = 0; i <= limit; i++) flags[i] = 1;
    flags[0] = 0;
    flags[1] = 0;
    for (int64_t p = 2; p * p <= limit; p++) {
        if (flags[p] == 1) {
            for (int64_t m = p * p; m <= limit; m += p) {
                flags[m] = 0;
            }
        }
    }
    int64_t count = 0;
    for (int64_t i = 2; i <= limit; i++) {
        count += flags[i];
    }
    free(flags);
    return count;
}

int main(void) {
    int64_t total = 0;
    for (int r = 0; r < 5; r++) {
        total += sieve_count(1000000);
    }
    printf("%lld\n", (long long)total);
    return 0;
}
