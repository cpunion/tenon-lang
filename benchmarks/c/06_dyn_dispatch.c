// Benchmark 06 (C): 5,000,000 Dynamic Interface Vtable Dispatches
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const uint64_t MASK = 9223372036854775807ULL;

typedef struct {
    int64_t (*apply)(const void* self, int64_t x);
} OpVtable;

typedef struct {
    void* data;
    const OpVtable* vt;
} DynOp;

typedef struct { int64_t k; } OpData;

static int64_t add_apply(const void* s, int64_t x) {
    const OpData* d = (const OpData*)s;
    return (int64_t)(((uint64_t)x + (uint64_t)d->k) & MASK);
}
static int64_t xor_apply(const void* s, int64_t x) {
    const OpData* d = (const OpData*)s;
    return (int64_t)(((uint64_t)x ^ (uint64_t)d->k) & MASK);
}
static int64_t mul_apply(const void* s, int64_t x) {
    const OpData* d = (const OpData*)s;
    return (int64_t)(((uint64_t)x * (uint64_t)d->k) & MASK);
}
static int64_t shift_apply(const void* s, int64_t x) {
    const OpData* d = (const OpData*)s;
    return (int64_t)(((uint64_t)x ^ ((uint64_t)x >> (uint64_t)d->k)) & MASK);
}

static const OpVtable VT_ADD   = { add_apply };
static const OpVtable VT_XOR   = { xor_apply };
static const OpVtable VT_MUL   = { mul_apply };
static const OpVtable VT_SHIFT = { shift_apply };

int main(void) {
    OpData* d0 = (OpData*)malloc(sizeof(OpData)); d0->k = 17;
    OpData* d1 = (OpData*)malloc(sizeof(OpData)); d1->k = 2654435761LL;
    OpData* d2 = (OpData*)malloc(sizeof(OpData)); d2->k = 16777619;
    OpData* d3 = (OpData*)malloc(sizeof(OpData)); d3->k = 13;
    DynOp* ops = (DynOp*)malloc(4 * sizeof(DynOp));
    ops[0] = (DynOp){ d0, &VT_ADD };
    ops[1] = (DynOp){ d1, &VT_XOR };
    ops[2] = (DynOp){ d2, &VT_MUL };
    ops[3] = (DynOp){ d3, &VT_SHIFT };

    // Volatile pointer prevents constant-folding the entire vtable array at compile time
    DynOp* volatile vops = ops;
    int64_t acc = 1234567;
    for (int i = 0; i < 5000000; i++) {
        DynOp op = vops[i & 3];
        acc = op.vt->apply(op.data, acc);
    }
    printf("%lld\n", (long long)acc);
    free(ops); free(d0); free(d1); free(d2); free(d3);
    return 0;
}
