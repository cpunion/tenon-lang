#include <stdint.h>
#include <stdio.h>

typedef double v4d __attribute__((vector_size(32)));

int main(void) {
    const int N = 1024;
    const int REPS = 50000;
    static v4d a[1024];
    static v4d b[1024];

    for (int i = 0; i < N; i++) {
        a[i] = (v4d){
            ((i + 1) & 15) * 0.25,
            ((i + 3) & 15) * 0.25,
            ((i + 5) & 15) * 0.25,
            ((i + 7) & 15) * 0.25,
        };
        b[i] = (v4d){
            ((i * 3 + 2) & 15) * 0.25,
            ((i * 3 + 4) & 15) * 0.25,
            ((i * 3 + 6) & 15) * 0.25,
            ((i * 3 + 8) & 15) * 0.25,
        };
    }

    double total = 0.0;
    for (int r = 0; r < REPS; r++) {
        for (int j = 0; j < N; j++) {
            v4d p = a[j] * b[(j + r) & 1023];
            total += p[0] + p[1] + p[2] + p[3];
        }
    }

    printf("%lld\n", (long long)(total * 16.0));
    return 0;
}
