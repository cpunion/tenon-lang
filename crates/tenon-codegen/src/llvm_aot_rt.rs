pub(crate) const TENON_AOT_RUNTIME_C: &str = r#"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t rc;
    int64_t tag;
    int64_t len;
    int64_t cap;
    uint64_t rc_mask;
    int64_t fields[];
} TenonObj;

static int64_t g_live = 0;
static int64_t g_allocs = 0;
static int64_t g_reuse = 0;
static int64_t g_cow = 0;
static int64_t g_polls = 0;
static int64_t g_suspends = 0;

void tenon_rt_assert_fail(int64_t line) {
    fprintf(stderr, "TENON ASSERTION FAILED at line %lld\n", (long long)line);
    exit(1);
}

void tenon_rt_overflow_trap(int64_t line) {
    fprintf(stderr, "TENON INTEGER OVERFLOW TRAP at line %lld\n", (long long)line);
    exit(1);
}

int64_t tenon_rt_print_int(int64_t v) {
    printf("%lld\n", (long long)v);
    return v;
}

#define TENON_FREELIST_MAX_CAP 8
#define TENON_FREELIST_LIMIT 512

static TenonObj* g_freelist[TENON_FREELIST_MAX_CAP + 1] = {0};
static int g_freelist_len[TENON_FREELIST_MAX_CAP + 1] = {0};

/* Tier-2 Chunked 64 KiB Slab Arena (ADR-0001 D09: `@arena` zero-malloc bulk scope) */
#define TENON_ARENA_SLAB_SIZE 65536
typedef struct TenonSlab {
    struct TenonSlab* prev;
    size_t used;
    size_t cap;
    uint8_t data[];
} TenonSlab;

typedef struct {
    TenonSlab* slab;
    size_t used;
} TenonArenaMark;

static TenonSlab* g_arena_slab = NULL;
static int g_arena_depth = 0;
static TenonArenaMark g_arena_marks[64];

static TenonSlab* tenon_slab_new(size_t min_bytes, TenonSlab* prev) {
    size_t cap = min_bytes > TENON_ARENA_SLAB_SIZE ? min_bytes : TENON_ARENA_SLAB_SIZE;
    TenonSlab* s = (TenonSlab*)malloc(sizeof(TenonSlab) + cap);
    if (!s) abort();
    s->prev = prev;
    s->used = 0;
    s->cap = cap;
    return s;
}

int64_t tenon_arena_enter(void) {
    int idx = g_arena_depth;
    if (idx < 64) {
        g_arena_marks[idx].slab = g_arena_slab;
        g_arena_marks[idx].used = g_arena_slab ? g_arena_slab->used : 0;
    }
    g_arena_depth++;
    return (int64_t)idx;
}

void tenon_arena_exit(int64_t mark) {
    if (g_arena_depth <= 0) return;
    g_arena_depth--;
    int idx = (int)mark;
    if (idx >= 0 && idx < 64) {
        TenonSlab* target = g_arena_marks[idx].slab;
        size_t saved_used = g_arena_marks[idx].used;
        while (g_arena_slab && g_arena_slab != target) {
            TenonSlab* prev = g_arena_slab->prev;
            free(g_arena_slab);
            g_arena_slab = prev;
        }
        if (g_arena_slab) {
            g_arena_slab->used = saved_used;
        }
    }
    if (g_arena_depth == 0 && g_arena_slab) {
        while (g_arena_slab->prev) {
            TenonSlab* prev = g_arena_slab->prev;
            free(g_arena_slab);
            g_arena_slab = prev;
        }
        g_arena_slab->used = 0;
    }
}

static inline TenonObj* tenon_arena_alloc_obj(int64_t tag, int64_t lu, int64_t cap) {
    size_t bytes = (sizeof(TenonObj) + (size_t)cap * sizeof(int64_t) + 7u) & ~7u;
    if (!g_arena_slab || g_arena_slab->used + bytes > g_arena_slab->cap) {
        g_arena_slab = tenon_slab_new(bytes, g_arena_slab);
    }
    TenonObj* obj = (TenonObj*)(g_arena_slab->data + g_arena_slab->used);
    g_arena_slab->used += bytes;
    obj->rc = -2; /* Immortal-in-arena sentinel: rc_inc/rc_dec skip in 1 branch, bulk-freed on arena_exit */
    obj->tag = tag;
    obj->len = lu;
    obj->cap = cap;
    obj->rc_mask = 0;
    return obj;
}

#ifdef TENON_USE_SYNC
void tenon_drop_mutex_handle(int64_t raw);
void tenon_drop_channel_handle(int64_t raw, int is_rc);
#endif

static inline TenonObj* tenon_rc_alloc_raw(int64_t tag, int64_t lu, int64_t cap) {
    TenonObj* obj;
#ifdef TENON_THREAD_SAFE
    obj = (TenonObj*)malloc(sizeof(TenonObj) + (size_t)cap * sizeof(int64_t));
    if (!obj) abort();
    obj->rc = 1;
    obj->tag = tag;
    obj->len = lu;
    obj->cap = cap;
    obj->rc_mask = 0;
    __atomic_add_fetch(&g_live, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_allocs, 1, __ATOMIC_RELAXED);
    return obj;
#else
    if (__builtin_expect(g_arena_depth > 0, 0)) {
        return tenon_arena_alloc_obj(tag, lu, cap);
    }
    if (__builtin_expect(cap >= 1 && cap <= TENON_FREELIST_MAX_CAP && g_freelist[cap] != NULL, 1)) {
        obj = g_freelist[cap];
        g_freelist[cap] = *(TenonObj**)obj;
        g_freelist_len[cap]--;
    } else {
        obj = (TenonObj*)malloc(sizeof(TenonObj) + (size_t)cap * sizeof(int64_t));
        if (!obj) abort();
    }
    obj->rc = 1;
    obj->tag = tag;
    obj->len = lu;
    obj->cap = cap;
    obj->rc_mask = 0;
    g_live++;
    g_allocs++;
    return obj;
#endif
}

int64_t tenon_rc_alloc(int64_t tag, int64_t len) {
    int64_t lu = len < 0 ? 0 : len;
    int64_t cap = lu < 1 ? 1 : lu;
    TenonObj* obj = tenon_rc_alloc_raw(tag, lu, cap);
    if (cap == 1) {
        obj->fields[0] = 0;
    } else if (cap == 2) {
        obj->fields[0] = 0;
        obj->fields[1] = 0;
    } else {
        memset(obj->fields, 0, (size_t)cap * sizeof(int64_t));
    }
    return (int64_t)(uintptr_t)obj;
}

void tenon_rc_set_rc_mask(int64_t ptr, int64_t mask) {
    if (ptr == 0) return;
    ((TenonObj*)(uintptr_t)ptr)->rc_mask = (uint64_t)mask;
}

int64_t tenon_rc_inc(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
#ifdef TENON_THREAD_SAFE
    if (__builtin_expect(__atomic_load_n(&obj->rc, __ATOMIC_RELAXED) > 0, 1)) {
        __atomic_add_fetch(&obj->rc, 1, __ATOMIC_RELAXED);
    }
#else
    if (__builtin_expect(obj->rc > 0, 1)) {
        obj->rc++;
    }
#endif
    return ptr;
}

void tenon_rc_dec(int64_t ptr);

__attribute__((noinline))
static void tenon_rc_free_slow(TenonObj* obj) {
    uint64_t mask = obj->rc_mask;
    if (mask == UINT64_MAX) {
        for (int64_t i = 0; i < obj->len; i++) {
            if (obj->fields[i] != 0) tenon_rc_dec(obj->fields[i]);
        }
    } else if (mask != 0) {
        int64_t n = obj->len < 64 ? obj->len : 64;
        for (int64_t i = 0; i < n; i++) {
            if ((mask & (1ULL << i)) && obj->fields[i] != 0) {
                tenon_rc_dec(obj->fields[i]);
            }
        }
    }
#ifdef TENON_USE_SYNC
    if (obj->tag == 14) {
        tenon_drop_mutex_handle(obj->fields[0]);
    } else if (obj->tag == 15) {
        tenon_drop_channel_handle(obj->fields[0], obj->fields[1] != 0);
    }
#endif
#ifdef TENON_THREAD_SAFE
    __atomic_sub_fetch(&g_live, 1, __ATOMIC_RELAXED);
    free(obj);
#else
    g_live--;
    int64_t cap = obj->cap;
    if (__builtin_expect(cap >= 1 && cap <= TENON_FREELIST_MAX_CAP && g_freelist_len[cap] < TENON_FREELIST_LIMIT, 1)) {
        *(TenonObj**)obj = g_freelist[cap];
        g_freelist[cap] = obj;
        g_freelist_len[cap]++;
        return;
    }
    free(obj);
#endif
}

void tenon_rc_free_obj(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return;
    tenon_rc_free_slow((TenonObj*)(uintptr_t)ptr);
}

void tenon_rc_dec(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
#ifdef TENON_THREAD_SAFE
    if (__builtin_expect(__atomic_load_n(&obj->rc, __ATOMIC_RELAXED) > 0, 1)) {
        if (__builtin_expect(__atomic_sub_fetch(&obj->rc, 1, __ATOMIC_ACQ_REL) <= 0, 0)) {
            tenon_rc_free_slow(obj);
        }
    }
#else
    if (__builtin_expect(obj->rc > 0, 1)) {
        if (__builtin_expect(--obj->rc <= 0, 0)) {
            tenon_rc_free_slow(obj);
        }
    }
#endif
}

__attribute__((noinline))
static int64_t tenon_rc_cow_clone_slow(TenonObj* obj) {
#ifdef TENON_THREAD_SAFE
    __atomic_add_fetch(&g_cow, 1, __ATOMIC_RELAXED);
#else
    g_cow++;
#endif
    int is_byte_payload = (obj->tag == 5 || obj->tag == 10 || obj->tag == 12 || obj->tag == 16);
    if (is_byte_payload) {
        int64_t extra = obj->tag == 5 ? 1 : 0;
        int64_t words = (obj->len + extra + 7) / 8;
        if (words < 1) words = 1;
        TenonObj* fresh = tenon_rc_alloc_raw(obj->tag, obj->len, words);
        if (obj->len > 0) memcpy(fresh->fields, obj->fields, (size_t)obj->len);
        if (extra) ((uint8_t*)fresh->fields)[obj->len] = 0;
        tenon_rc_dec((int64_t)(uintptr_t)obj);
        return (int64_t)(uintptr_t)fresh;
    }
    if (obj->tag == 17 || obj->tag == 18 || obj->tag == 19) {
        int64_t eb = (obj->tag == 19) ? 4 : ((obj->tag == 17) ? 16 : 32);
        int64_t total_bytes = obj->len * eb;
        int64_t words = (total_bytes + 7) / 8;
        if (words < 2) words = 2;
        TenonObj* fresh = tenon_rc_alloc_raw(obj->tag, obj->len, words);
        if (total_bytes > 0) memcpy(fresh->fields, obj->fields, (size_t)total_bytes);
        tenon_rc_dec((int64_t)(uintptr_t)obj);
        return (int64_t)(uintptr_t)fresh;
    }
    int64_t cap = obj->len < 1 ? 1 : obj->len;
    TenonObj* fresh = tenon_rc_alloc_raw(obj->tag, obj->len, cap);
    uint64_t mask = obj->rc_mask;
    fresh->rc_mask = mask;
    memcpy(fresh->fields, obj->fields, (size_t)obj->len * sizeof(int64_t));
    if (mask == UINT64_MAX) {
        for (int64_t i = 0; i < obj->len; i++) {
            if (obj->fields[i] != 0) tenon_rc_inc(obj->fields[i]);
        }
    } else if (mask != 0) {
        int64_t n = obj->len < 64 ? obj->len : 64;
        for (int64_t i = 0; i < n; i++) {
            if ((mask & (1ULL << i)) && obj->fields[i] != 0) {
                tenon_rc_inc(obj->fields[i]);
            }
        }
    }
    tenon_rc_dec((int64_t)(uintptr_t)obj);
    return (int64_t)(uintptr_t)fresh;
}

int64_t tenon_rc_ensure_unique(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
#ifdef TENON_THREAD_SAFE
    if (__builtin_expect(__atomic_load_n(&obj->rc, __ATOMIC_ACQUIRE) == 1, 1)) {
        return ptr;
    }
#else
    if (__builtin_expect(obj->rc == 1, 1)) {
        return ptr;
    }
#endif
    return tenon_rc_cow_clone_slow(obj);
}

int64_t tenon_rc_get_field(int64_t ptr, int64_t idx) {
    if (__builtin_expect(ptr == 0 || idx < 0, 0)) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (__builtin_expect(idx >= obj->len, 0)) return 0;
    return obj->fields[idx];
}

void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val) {
    if (__builtin_expect(ptr == 0 || idx < 0, 0)) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (__builtin_expect(idx < obj->len, 1)) {
        uint64_t mask = obj->rc_mask;
        if (__builtin_expect(mask == 0, 1)) {
            obj->fields[idx] = val;
            return;
        }
        int64_t old = obj->fields[idx];
        obj->fields[idx] = val;
        if ((mask == UINT64_MAX || (idx < 64 && (mask & (1ULL << idx)))) && old != 0) {
            tenon_rc_dec(old);
        }
    }
}

void tenon_rc_set_field_raw(int64_t ptr, int64_t idx, int64_t val) {
    if (ptr == 0 || idx < 0) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (idx < obj->len) {
        obj->fields[idx] = val;
    }
}

int64_t tenon_rc_get_tag(int64_t ptr) {
    if (ptr == 0) return -1;
    return ((TenonObj*)(uintptr_t)ptr)->tag;
}

int64_t tenon_rc_get_len(int64_t ptr) {
    if (ptr == 0) return 0;
    return ((TenonObj*)(uintptr_t)ptr)->len;
}

int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len) {
    int64_t blen = byte_len < 0 ? 0 : byte_len;
    int64_t words = (blen + 8) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(5, blen, words);
    if (blen > 0 && data_ptr != 0) {
        memcpy((void*)obj->fields, (const void*)(uintptr_t)data_ptr, (size_t)blen);
    }
    ((uint8_t*)obj->fields)[blen] = 0;
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_text_eq(int64_t a, int64_t b) {
    if (a == b) return 1;
    if (a == 0 || b == 0) return 0;
    TenonObj* ha = (TenonObj*)(uintptr_t)a;
    TenonObj* hb = (TenonObj*)(uintptr_t)b;
    if (ha->len != hb->len) return 0;
    return memcmp((const void*)ha->fields, (const void*)hb->fields, (size_t)ha->len) == 0 ? 1 : 0;
}

int64_t tenon_byte_at(int64_t s, int64_t idx) {
    if (s == 0 || idx < 0) return -1;
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    if (idx >= hdr->len) return -1;
    return (int64_t)((const uint8_t*)hdr->fields)[idx];
}

int64_t tenon_str_slice(int64_t s, int64_t start, int64_t end) {
    if (s == 0) return tenon_text_new(0, 0);
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    int64_t len = hdr->len;
    int64_t st = start < 0 ? 0 : (start > len ? len : start);
    int64_t en = end < 0 ? 0 : (end > len ? len : end);
    if (st >= en) return tenon_text_new(0, 0);
    const uint8_t* base = (const uint8_t*)hdr->fields;
    return tenon_text_new((int64_t)(uintptr_t)(base + st), en - st);
}

int64_t tenon_byte_to_text(int64_t b) {
    uint8_t ch = (uint8_t)(b & 0xFF);
    return tenon_text_new((int64_t)(uintptr_t)&ch, 1);
}

int64_t tenon_str_concat(int64_t a, int64_t b) {
    int64_t la = a ? ((TenonObj*)(uintptr_t)a)->len : 0;
    int64_t lb = b ? ((TenonObj*)(uintptr_t)b)->len : 0;
    int64_t total = la + lb;
    int64_t words = (total + 8) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(5, total, words);
    uint8_t* dst = (uint8_t*)obj->fields;
    if (la > 0) memcpy(dst, (const void*)((TenonObj*)(uintptr_t)a)->fields, (size_t)la);
    if (lb > 0) memcpy(dst + la, (const void*)((TenonObj*)(uintptr_t)b)->fields, (size_t)lb);
    dst[total] = 0;
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_int_to_float(int64_t v) {
    union { double d; int64_t i; } u;
    u.d = (double)v;
    return u.i;
}

int64_t tenon_float_to_int(int64_t bits) {
    union { double d; int64_t i; } u;
    u.i = bits;
    return (int64_t)u.d;
}

int64_t tenon_float_to_text(int64_t bits) {
    union { double d; int64_t i; } u;
    u.i = bits;
    double d = (u.d == 0.0) ? 0.0 : u.d;
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%.15g", d);
    if (n < 0) n = 0;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    return tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)n);
}

int64_t tenon_parse_float(int64_t s) {
    union { double d; int64_t i; } u;
    if (s == 0) { u.d = NAN; return u.i; }
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    size_t n = (size_t)hdr->len;
    char tmp[128];
    char* buf = n < sizeof(tmp) ? tmp : (char*)malloc(n + 1);
    if (n > 0) memcpy(buf, (const void*)hdr->fields, n);
    buf[n] = '\0';
    char* end = NULL;
    double v = strtod(buf, &end);
    while (end && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end++;
    int ok = (end != buf && end && *end == '\0');
    if (buf != tmp) free(buf);
    u.d = ok ? v : NAN;
    return u.i;
}

int64_t tenon_float_is_nan(int64_t bits) {
    union { double d; int64_t i; } u;
    u.i = bits;
    return isnan(u.d) ? 1 : 0;
}

int64_t tenon_float_bits(int64_t bits) {
    union { double d; int64_t i; } u;
    u.i = bits;
    if (u.d == 0.0) { u.d = 0.0; return u.i; }
    return bits;
}

typedef struct {
    uint8_t* data;
    size_t len;
    size_t cap;
    uint8_t inline_buf[128];
} TenonBuf;

#ifdef TENON_THREAD_SAFE
static _Thread_local TenonBuf* g_buf_cache = NULL;
#else
static TenonBuf* g_buf_cache = NULL;
#endif

int64_t tenon_buf_new(void) {
    TenonBuf* b;
    if (g_buf_cache) {
        b = g_buf_cache;
        g_buf_cache = NULL;
    } else {
        b = (TenonBuf*)malloc(sizeof(TenonBuf));
    }
    b->data = b->inline_buf;
    b->cap = sizeof(b->inline_buf);
    b->len = 0;
    return (int64_t)(uintptr_t)b;
}

__attribute__((noinline))
static void tenon_buf_grow(TenonBuf* b, size_t need) {
    size_t nc = b->cap * 2;
    while (nc < need) nc *= 2;
    if (b->data == b->inline_buf) {
        uint8_t* heap = (uint8_t*)malloc(nc);
        if (b->len > 0) memcpy(heap, b->inline_buf, b->len);
        b->data = heap;
    } else {
        b->data = (uint8_t*)realloc(b->data, nc);
    }
    b->cap = nc;
}

int64_t tenon_buf_push_str(int64_t h, int64_t s) {
    if (__builtin_expect(h == 0 || s == 0, 0)) return 0;
    TenonBuf* b = (TenonBuf*)(uintptr_t)h;
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    size_t n = (size_t)hdr->len;
    if (n > 0) {
        if (__builtin_expect(b->len + n > b->cap, 0)) tenon_buf_grow(b, b->len + n);
        memcpy(b->data + b->len, (const void*)hdr->fields, n);
        b->len += n;
    }
    return (int64_t)b->len;
}

int64_t tenon_buf_push_byte(int64_t h, int64_t byte_val) {
    if (__builtin_expect(h == 0, 0)) return 0;
    TenonBuf* b = (TenonBuf*)(uintptr_t)h;
    if (__builtin_expect(b->len + 1 > b->cap, 0)) tenon_buf_grow(b, b->len + 1);
    b->data[b->len++] = (uint8_t)(byte_val & 0xFF);
    return (int64_t)b->len;
}

int64_t tenon_buf_finish(int64_t h) {
    if (__builtin_expect(h == 0, 0)) return tenon_text_new(0, 0);
    TenonBuf* b = (TenonBuf*)(uintptr_t)h;
    int64_t res = tenon_text_new((int64_t)(uintptr_t)b->data, (int64_t)b->len);
    if (b->data != b->inline_buf) free(b->data);
    if (!g_buf_cache) {
        g_buf_cache = b;
    } else {
        free(b);
    }
    return res;
}

int64_t tenon_list_push(int64_t ptr, int64_t val) {
    if (ptr == 0) {
        int64_t f = tenon_rc_alloc(4, 1);
        tenon_rc_set_field(f, 0, val);
        return f;
    }
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    int64_t old_len = obj->len;
    uint64_t mask = obj->rc_mask;
#ifdef TENON_THREAD_SAFE
    int shared = __atomic_load_n(&obj->rc, __ATOMIC_ACQUIRE) > 1;
#else
    int shared = obj->rc > 1;
#endif
    if (!shared && old_len < obj->cap) {
        g_reuse++;
        obj->fields[old_len] = val;
        obj->len = old_len + 1;
        return ptr;
    }
    int64_t new_cap = obj->cap * 2;
    if (new_cap < old_len + 1) new_cap = old_len + 1;
    int64_t fp = tenon_rc_alloc(4, new_cap);
    TenonObj* fresh = (TenonObj*)(uintptr_t)fp;
    fresh->len = old_len + 1;
    fresh->rc_mask = mask;
    for (int64_t i = 0; i < old_len; i++) {
        fresh->fields[i] = obj->fields[i];
        if (shared && mask == UINT64_MAX && obj->fields[i] != 0) {
            tenon_rc_inc(obj->fields[i]);
        }
    }
    fresh->fields[old_len] = val;
    if (shared) {
        g_cow++;
    } else {
        g_reuse++;
        obj->rc_mask = 0;
    }
    tenon_rc_dec(ptr);
    return fp;
}

int64_t tenon_list_filled(int64_t len, int64_t init_val, int64_t elem_is_rc) {
    int64_t n = len < 0 ? 0 : len;
    int64_t cap = n < 1 ? 1 : n;
    TenonObj* fresh = tenon_rc_alloc_raw(4, n, cap);
    if (elem_is_rc != 0) {
        fresh->rc_mask = UINT64_MAX;
        if (init_val != 0 && n > 0) {
            TenonObj* iv = (TenonObj*)(uintptr_t)init_val;
#ifdef TENON_THREAD_SAFE
            if (__atomic_load_n(&iv->rc, __ATOMIC_RELAXED) > 0) __atomic_add_fetch(&iv->rc, n, __ATOMIC_RELAXED);
#else
            if (iv->rc > 0) iv->rc += n;
#endif
        }
    }
    if (init_val == 0) {
        memset(fresh->fields, 0, (size_t)n * sizeof(int64_t));
    } else {
        for (int64_t i = 0; i < n; i++) {
            fresh->fields[i] = init_val;
        }
    }
    return (int64_t)(uintptr_t)fresh;
}

int64_t tenon_flat_list_filled(int64_t len, int64_t elem_bytes, const void* init_ptr) {
    int64_t n = len < 0 ? 0 : len;
    int64_t tag = (elem_bytes == 4) ? 19 : ((elem_bytes == 16) ? 17 : 18);
    int64_t total_bytes = n * elem_bytes;
    int64_t words = (total_bytes + 7) / 8;
    if (words < 4) words = 4;
    TenonObj* fresh = tenon_rc_alloc_raw(tag, n, words);
    uint8_t* dst = (uint8_t*)fresh->fields;
    if (init_ptr == NULL) {
        if (total_bytes > 0) memset(dst, 0, (size_t)total_bytes);
    } else {
        for (int64_t i = 0; i < n; i++) {
            memcpy(dst + (size_t)(i * elem_bytes), init_ptr, (size_t)elem_bytes);
        }
    }
    return (int64_t)(uintptr_t)fresh;
}

int64_t tenon_flat_list_push(int64_t ptr, int64_t elem_bytes, const void* val_ptr) {
    int64_t tag = (elem_bytes == 4) ? 19 : ((elem_bytes == 16) ? 17 : 18);
    if (ptr == 0) {
        return tenon_flat_list_filled(1, elem_bytes, val_ptr);
    }
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    int64_t old_len = obj->len;
    int64_t cap_elems = (obj->cap * 8) / elem_bytes;
#ifdef TENON_THREAD_SAFE
    int shared = __atomic_load_n(&obj->rc, __ATOMIC_ACQUIRE) > 1;
#else
    int shared = obj->rc > 1;
#endif
    if (!shared && old_len < cap_elems) {
        g_reuse++;
        memcpy((uint8_t*)obj->fields + (size_t)(old_len * elem_bytes), val_ptr, (size_t)elem_bytes);
        obj->len = old_len + 1;
        return ptr;
    }
    int64_t new_elems = cap_elems * 2;
    if (new_elems < old_len + 1) new_elems = old_len + 1;
    if (new_elems < 4) new_elems = 4;
    int64_t new_words = (new_elems * elem_bytes + 7) / 8;
    TenonObj* fresh = tenon_rc_alloc_raw(tag, old_len + 1, new_words);
    if (old_len > 0) {
        memcpy(fresh->fields, obj->fields, (size_t)(old_len * elem_bytes));
    }
    memcpy((uint8_t*)fresh->fields + (size_t)(old_len * elem_bytes), val_ptr, (size_t)elem_bytes);
    if (shared) g_cow++; else g_reuse++;
    tenon_rc_dec(ptr);
    return (int64_t)(uintptr_t)fresh;
}

int64_t tenon_list_pop_shrink(int64_t ptr) {
    if (ptr == 0) return 0;
    int64_t uniq = tenon_rc_ensure_unique(ptr);
    TenonObj* hdr = (TenonObj*)(uintptr_t)uniq;
    if (hdr->len > 0) {
        if (hdr->tag == 17 || hdr->tag == 18 || hdr->tag == 19) { hdr->len -= 1; return uniq; }
        int64_t last_idx = hdr->len - 1;
        int64_t old = hdr->fields[last_idx];
        hdr->fields[last_idx] = 0;
        hdr->len = last_idx;
        if (hdr->rc_mask == UINT64_MAX && old != 0) {
            tenon_rc_dec(old);
        }
    }
    return uniq;
}

int64_t tenon_async_frame_alloc(int64_t total_slots, int64_t poll_fn_ptr) {
    int64_t slots = total_slots < 4 ? 4 : total_slots;
    int64_t f = tenon_rc_alloc(6, slots);
    tenon_rc_set_field(f, 0, 0);
    tenon_rc_set_field(f, 1, poll_fn_ptr);
    tenon_rc_set_field(f, 2, 0);
    tenon_rc_set_field(f, 3, 0);
    return f;
}

static int64_t tenon_async_yield_poll(int64_t frame) {
    int64_t st = tenon_rc_get_field(frame, 0);
    if (st == 0) {
        tenon_rc_set_field(frame, 0, 1);
        g_suspends++;
        return 0;
    } else {
        int64_t val = tenon_rc_get_field(frame, 4);
        tenon_rc_set_field(frame, 2, val);
        tenon_rc_set_field(frame, 0, -1);
        return 1;
    }
}

int64_t tenon_async_yield_create(int64_t val) {
    int64_t f = tenon_async_frame_alloc(5, (int64_t)(uintptr_t)&tenon_async_yield_poll);
    tenon_rc_set_field(f, 4, val);
    return f;
}

int64_t tenon_async_poll_frame(int64_t frame) {
    if (frame == 0) return 1;
    g_polls++;
    int64_t pp = tenon_rc_get_field(frame, 1);
    if (pp == 0) return 1;
    int64_t (*poll_fn)(int64_t) = (int64_t (*)(int64_t))(uintptr_t)pp;
    return poll_fn(frame);
}

int64_t tenon_async_block_on(int64_t frame) {
    for (;;) {
        if (tenon_async_poll_frame(frame) != 0) {
            int64_t res = tenon_rc_get_field(frame, 2);
            tenon_rc_dec(frame);
            return res;
        }
    }
}

extern int64_t tenon_fn_main(void);
"#;

pub(crate) const TENON_AOT_MAIN_STATS_C: &str = r#"
int main(void) {
    int64_t res = tenon_fn_main();
    printf("main() => %lld\n", (long long)res);
    if (g_allocs > 0 || g_polls > 0) {
        printf("Perceus/Async stats: allocs=%lld, unique_reuse_hits=%lld, cow_clones=%lld, live_objects=%lld, async_polls=%lld, async_suspends=%lld\n",
               (long long)g_allocs, (long long)g_reuse, (long long)g_cow,
               (long long)g_live, (long long)g_polls, (long long)g_suspends);
    }
    return g_live == 0 ? 0 : 2;
}
"#;

pub(crate) const TENON_AOT_MAIN_CLI_C: &str = r#"
int main(void) {
    return (int)tenon_fn_main();
}
"#;

pub(crate) const TENON_AOT_OS_C: &str = r#"
#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_stdin_eof = 0;
static int g_file_ok = 1;
static int64_t g_exec_exit_code = 0;

static char* tenon_cstr_copy(int64_t s) {
    if (s == 0) {
        char* empty = (char*)malloc(1);
        empty[0] = '\0';
        return empty;
    }
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    size_t n = (size_t)hdr->len;
    char* out = (char*)malloc(n + 1);
    if (n > 0) memcpy(out, (const void*)hdr->fields, n);
    out[n] = '\0';
    return out;
}

int64_t tenon_sys_write_stdout(int64_t s) {
    if (s == 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    if (hdr->len > 0) {
        fwrite((const void*)hdr->fields, 1, (size_t)hdr->len, stdout);
        fflush(stdout);
    }
    return hdr->len;
}

int64_t tenon_sys_read_line(void) {
    size_t cap = 256, len = 0;
    char* buf = (char*)malloc(cap);
    int ch;
    while ((ch = fgetc(stdin)) != EOF) {
        if (ch == '\n') break;
        if (len + 1 >= cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
        }
        buf[len++] = (char)ch;
    }
    if (ch == EOF && len == 0) {
        g_stdin_eof = 1;
        free(buf);
        return tenon_text_new(0, 0);
    }
    g_stdin_eof = 0;
    if (len > 0 && buf[len - 1] == '\r') len--;
    int64_t res = tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)len);
    free(buf);
    return res;
}

int64_t tenon_sys_is_eof(void) { return g_stdin_eof ? 1 : 0; }

int64_t tenon_sys_env(int64_t key) {
    char* k = tenon_cstr_copy(key);
    const char* v = getenv(k);
    free(k);
    if (!v) return tenon_text_new(0, 0);
    return tenon_text_new((int64_t)(uintptr_t)v, (int64_t)strlen(v));
}

int64_t tenon_sys_cwd(void) {
    char buf[4096];
    if (!getcwd(buf, sizeof(buf))) return tenon_text_new((int64_t)(uintptr_t)".", 1);
    return tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)strlen(buf));
}

int64_t tenon_sys_read_file(int64_t path) {
    char* p = tenon_cstr_copy(path);
    FILE* f = fopen(p, "rb");
    if (!f) {
        g_file_ok = 0;
        char msg[256];
        int ml = snprintf(msg, sizeof(msg), "%s", strerror(errno));
        free(p);
        return tenon_text_new((int64_t)(uintptr_t)msg, (int64_t)ml);
    }
    free(p);
    size_t cap = 4096, len = 0;
    char* buf = (char*)malloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
        }
    }
    fclose(f);
    g_file_ok = 1;
    int64_t res = tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)len);
    free(buf);
    return res;
}

int64_t tenon_sys_file_ok(void) { return g_file_ok ? 1 : 0; }

int64_t tenon_sys_write_file(int64_t path, int64_t content) {
    char* p = tenon_cstr_copy(path);
    FILE* f = fopen(p, "wb");
    free(p);
    if (!f) {
        g_file_ok = 0;
        return 0;
    }
    int ok = 1;
    if (content != 0) {
        TenonObj* hdr = (TenonObj*)(uintptr_t)content;
        if (hdr->len > 0) {
            ok = fwrite((const void*)hdr->fields, 1, (size_t)hdr->len, f) == (size_t)hdr->len;
        }
    }
    if (fclose(f) != 0) ok = 0;
    g_file_ok = ok;
    return ok ? 1 : 0;
}

int64_t tenon_sys_exec(int64_t cmd) {
    char* c = tenon_cstr_copy(cmd);
    size_t clen = strlen(c);
    char* wrapped = (char*)malloc(clen + 16);
    snprintf(wrapped, clen + 16, "%s 2>&1", c);
    free(c);
    FILE* fp = popen(wrapped, "r");
    free(wrapped);
    if (!fp) {
        g_exec_exit_code = -1;
        return tenon_text_new(0, 0);
    }
    size_t cap = 4096, len = 0;
    char* buf = (char*)malloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len, fp)) > 0) {
        len += n;
        if (len == cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
        }
    }
    int st = pclose(fp);
    if (WIFEXITED(st)) g_exec_exit_code = WEXITSTATUS(st);
    else g_exec_exit_code = -1;
    int64_t res = tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)len);
    free(buf);
    return res;
}

int64_t tenon_sys_exec_exit_code(void) { return g_exec_exit_code; }
"#;

pub(crate) const TENON_AOT_HTTP_C: &str = r#"
#include <curl/curl.h>
#include <pthread.h>

typedef struct HttpLineNode {
    char* data;
    size_t len;
    struct HttpLineNode* next;
} HttpLineNode;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    HttpLineNode* head;
    HttpLineNode* tail;
    int done;
    int eof;
    int closed;
    char* err;
    char* method;
    char* url;
    char* headers;
    char* body;
    size_t body_len;
    char* line_buf;
    size_t line_len;
    size_t line_cap;
    char err_preview[4096];
    size_t err_preview_len;
    pthread_t th;
} TenonHttpStream;

static void http_push_line(TenonHttpStream* st, const char* data, size_t len) {
    HttpLineNode* node = (HttpLineNode*)malloc(sizeof(HttpLineNode));
    node->data = (char*)malloc(len + 1);
    if (len > 0) memcpy(node->data, data, len);
    node->data[len] = '\0';
    node->len = len;
    node->next = NULL;
    pthread_mutex_lock(&st->mu);
    if (!st->closed) {
        if (st->tail) st->tail->next = node;
        else st->head = node;
        st->tail = node;
        pthread_cond_signal(&st->cv);
    } else {
        free(node->data);
        free(node);
    }
    pthread_mutex_unlock(&st->mu);
}

static size_t tenon_curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    TenonHttpStream* st = (TenonHttpStream*)userdata;
    if (st->err_preview_len < sizeof(st->err_preview) - 1) {
        size_t take = sizeof(st->err_preview) - 1 - st->err_preview_len;
        if (total < take) take = total;
        memcpy(st->err_preview + st->err_preview_len, ptr, take);
        st->err_preview_len += take;
        st->err_preview[st->err_preview_len] = '\0';
    }
    if (st->line_len + total > st->line_cap) {
        size_t nc = st->line_cap ? st->line_cap * 2 : 1024;
        while (nc < st->line_len + total) nc *= 2;
        st->line_buf = (char*)realloc(st->line_buf, nc);
        st->line_cap = nc;
    }
    memcpy(st->line_buf + st->line_len, ptr, total);
    st->line_len += total;

    size_t start = 0;
    for (size_t i = 0; i < st->line_len; i++) {
        if (st->line_buf[i] == '\n') {
            size_t llen = i - start;
            if (llen > 0 && st->line_buf[start + llen - 1] == '\r') llen--;
            http_push_line(st, st->line_buf + start, llen);
            start = i + 1;
        }
    }
    if (start > 0) {
        size_t rem = st->line_len - start;
        if (rem > 0) memmove(st->line_buf, st->line_buf + start, rem);
        st->line_len = rem;
    }
    pthread_mutex_lock(&st->mu);
    int closed = st->closed;
    pthread_mutex_unlock(&st->mu);
    return closed ? 0 : total;
}

static void* tenon_http_thread(void* arg) {
    TenonHttpStream* st = (TenonHttpStream*)arg;
    CURL* curl = curl_easy_init();
    if (!curl) {
        pthread_mutex_lock(&st->mu);
        st->err = strdup("curl_easy_init failed");
        st->done = 1;
        pthread_cond_signal(&st->cv);
        pthread_mutex_unlock(&st->mu);
        return NULL;
    }
    struct curl_slist* slist = NULL;
    char* p = st->headers;
    while (p && *p) {
        char* nl = strchr(p, '\n');
        size_t l = nl ? (size_t)(nl - p) : strlen(p);
        while (l > 0 && (p[l - 1] == '\r' || p[l - 1] == ' ')) l--;
        if (l > 0) {
            char* h = (char*)malloc(l + 1);
            memcpy(h, p, l);
            h[l] = '\0';
            if (strchr(h, ':')) slist = curl_slist_append(slist, h);
            free(h);
        }
        if (!nl) break;
        p = nl + 1;
    }
    curl_easy_setopt(curl, CURLOPT_URL, st->url);
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, st->method[0] ? st->method : "POST");
    if (slist) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist);
    if (st->body_len > 0) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, st->body);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)st->body_len);
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, tenon_curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, st);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (st->line_len > 0) {
        size_t llen = st->line_len;
        if (llen > 0 && st->line_buf[llen - 1] == '\r') llen--;
        http_push_line(st, st->line_buf, llen);
    }
    pthread_mutex_lock(&st->mu);
    if (res != CURLE_OK && !st->closed) {
        const char* em = curl_easy_strerror(res);
        size_t el = strlen(em) + 32;
        st->err = (char*)malloc(el);
        snprintf(st->err, el, "http request failed: %s", em);
    } else if (http_code >= 400 && !st->err) {
        size_t el = st->err_preview_len + 32;
        st->err = (char*)malloc(el);
        snprintf(st->err, el, "HTTP %ld: %s", http_code, st->err_preview);
    }
    st->done = 1;
    pthread_cond_signal(&st->cv);
    pthread_mutex_unlock(&st->mu);

    if (slist) curl_slist_free_all(slist);
    curl_easy_cleanup(curl);
    return NULL;
}

int64_t tenon_sys_http_open(int64_t method, int64_t url, int64_t headers, int64_t body) {
    TenonHttpStream* st = (TenonHttpStream*)calloc(1, sizeof(TenonHttpStream));
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    st->method = tenon_cstr_copy(method);
    st->url = tenon_cstr_copy(url);
    st->headers = tenon_cstr_copy(headers);
    if (body != 0) {
        TenonObj* hdr = (TenonObj*)(uintptr_t)body;
        st->body_len = (size_t)hdr->len;
        st->body = (char*)malloc(st->body_len + 1);
        if (st->body_len > 0) memcpy(st->body, (const void*)hdr->fields, st->body_len);
        st->body[st->body_len] = '\0';
    }
    pthread_create(&st->th, NULL, tenon_http_thread, st);
    return (int64_t)(uintptr_t)st;
}

int64_t tenon_sys_http_sse_open(int64_t url, int64_t bearer, int64_t body) {
    char* b = tenon_cstr_copy(bearer);
    size_t bl = strlen(b);
    size_t hl = bl + 128;
    char* h = (char*)malloc(hl);
    if (bl > 0) snprintf(h, hl, "Content-Type: application/json\nAccept: text/event-stream\nAuthorization: Bearer %s\n", b);
    else snprintf(h, hl, "Content-Type: application/json\nAccept: text/event-stream\n");
    free(b);
    int64_t m = tenon_text_new((int64_t)(uintptr_t)"POST", 4);
    int64_t hs = tenon_text_new((int64_t)(uintptr_t)h, (int64_t)strlen(h));
    free(h);
    int64_t res = tenon_sys_http_open(m, url, hs, body);
    tenon_rc_dec(m);
    tenon_rc_dec(hs);
    return res;
}

int64_t tenon_sys_http_sse_next_line(int64_t handle) {
    if (handle == 0) return tenon_text_new(0, 0);
    TenonHttpStream* st = (TenonHttpStream*)(uintptr_t)handle;
    pthread_mutex_lock(&st->mu);
    while (!st->head && !st->done) {
        pthread_cond_wait(&st->cv, &st->mu);
    }
    if (!st->head) {
        st->eof = 1;
        pthread_mutex_unlock(&st->mu);
        return tenon_text_new(0, 0);
    }
    HttpLineNode* node = st->head;
    st->head = node->next;
    if (!st->head) st->tail = NULL;
    pthread_mutex_unlock(&st->mu);
    int64_t res = tenon_text_new((int64_t)(uintptr_t)node->data, (int64_t)node->len);
    free(node->data);
    free(node);
    return res;
}

int64_t tenon_sys_http_sse_eof(int64_t handle) {
    if (handle == 0) return 1;
    TenonHttpStream* st = (TenonHttpStream*)(uintptr_t)handle;
    pthread_mutex_lock(&st->mu);
    int e = st->eof;
    pthread_mutex_unlock(&st->mu);
    return e ? 1 : 0;
}

int64_t tenon_sys_http_sse_error(int64_t handle) {
    if (handle == 0) return tenon_text_new(0, 0);
    TenonHttpStream* st = (TenonHttpStream*)(uintptr_t)handle;
    pthread_mutex_lock(&st->mu);
    int64_t res = st->err ? tenon_text_new((int64_t)(uintptr_t)st->err, (int64_t)strlen(st->err)) : tenon_text_new(0, 0);
    pthread_mutex_unlock(&st->mu);
    return res;
}

int64_t tenon_sys_http_sse_close(int64_t handle) {
    if (handle == 0) return 0;
    TenonHttpStream* st = (TenonHttpStream*)(uintptr_t)handle;
    pthread_mutex_lock(&st->mu);
    st->closed = 1;
    pthread_mutex_unlock(&st->mu);
    pthread_join(st->th, NULL);
    while (st->head) {
        HttpLineNode* n = st->head;
        st->head = n->next;
        free(n->data);
        free(n);
    }
    free(st->err);
    free(st->method);
    free(st->url);
    free(st->headers);
    free(st->body);
    free(st->line_buf);
    pthread_cond_destroy(&st->cv);
    pthread_mutex_destroy(&st->mu);
    free(st);
    return 0;
}
"#;


pub(crate) const TENON_NONATOMIC_RC_IR_HELPERS: &str = r#"
define internal i64 @__tn_rc_inc(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %rc = load i64, ptr %obj, align 8, !tbaa !6
  %pos = icmp sgt i64 %rc, 0
  br i1 %pos, label %inc, label %done
inc:
  %nrc = add nsw i64 %rc, 1
  store i64 %nrc, ptr %obj, align 8, !tbaa !6
  br label %done
done:
  ret i64 %p
}

define internal void @__tn_rc_dec(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %rc = load i64, ptr %obj, align 8, !tbaa !6
  %pos = icmp sgt i64 %rc, 0
  br i1 %pos, label %dec, label %done
dec:
  %nrc = add nsw i64 %rc, -1
  store i64 %nrc, ptr %obj, align 8, !tbaa !6
  %zero = icmp sle i64 %nrc, 0
  br i1 %zero, label %free, label %done
free:
  tail call void @tenon_rc_free_obj(i64 %p)
  br label %done
done:
  ret void
}

define internal i64 @__tn_rc_ensure_unique(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %ret0
chk:
  %obj = inttoptr i64 %p to ptr
  %rc = load i64, ptr %obj, align 8, !tbaa !6
  %uniq = icmp eq i64 %rc, 1
  br i1 %uniq, label %retp, label %slow
slow:
  %cow = tail call i64 @tenon_rc_ensure_unique(i64 %p)
  ret i64 %cow
retp:
  ret i64 %p
ret0:
  ret i64 0
}
"#;

pub(crate) const TENON_INLINE_IR_HELPERS: &str = r#"
define internal void @__tn_rc_set_rc_mask(i64 %p, i64 %mask) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %do, label %done
do:
  %obj = inttoptr i64 %p to ptr
  %mp = getelementptr inbounds i8, ptr %obj, i64 32
  store i64 %mask, ptr %mp, align 8, !tbaa !9
  br label %done
done:
  ret void
}

define internal i64 @__tn_cstruct_data_ptr(i64 %p) alwaysinline nounwind willreturn memory(none) {
entry:
  %is0 = icmp eq i64 %p, 0
  %d = add i64 %p, 40
  %res = select i1 %is0, i64 0, i64 %d
  ret i64 %res
}

define internal i64 @__tn_slot_get(i64 %p, i64 %idx) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %v = load i64, ptr %fp, align 8, !tbaa !10
  ret i64 %v
}

define internal void @__tn_slot_set_prim(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  store i64 %val, ptr %fp, align 8, !tbaa !10
  ret void
}

define internal void @__tn_slot_set_rc(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %old = load i64, ptr %fp, align 8, !tbaa !10
  store i64 %val, ptr %fp, align 8, !tbaa !10
  call void @__tn_rc_dec(i64 %old)
  ret void
}

define internal i64 @__tn_list_get(i64 %p, i64 %idx) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %oob
chk:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %oob
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %v = load i64, ptr %fp, align 8, !tbaa !10
  ret i64 %v
oob:
  ret i64 0
}

define internal i64 @__tn_list_get_nonnull(i64 %p, i64 %idx) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %oob
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %v = load i64, ptr %fp, align 8, !tbaa !10
  ret i64 %v
oob:
  ret i64 0
}

define internal i64 @__tn_list_get_unchecked(i64 %p, i64 %idx) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %v = load i64, ptr %fp, align 8, !tbaa !10
  ret i64 %v
}

define internal void @__tn_list_set_prim_unchecked(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  store i64 %val, ptr %fp, align 8, !tbaa !10
  ret void
}

define internal void @__tn_list_set_prim(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %done
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  store i64 %val, ptr %fp, align 8, !tbaa !10
  br label %done
done:
  ret void
}

define internal void @__tn_list_set_prim_nonnull(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %done
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  store i64 %val, ptr %fp, align 8, !tbaa !10
  br label %done
done:
  ret void
}

define internal void @__tn_list_set_rc(i64 %p, i64 %idx, i64 %val) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %done
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %fp = getelementptr inbounds i64, ptr %base, i64 %idx
  %old = load i64, ptr %fp, align 8, !tbaa !10
  store i64 %val, ptr %fp, align 8, !tbaa !10
  call void @__tn_rc_dec(i64 %old)
  br label %done
done:
  ret void
}

define internal i64 @__tn_rc_len(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %do, label %ret0
do:
  %obj = inttoptr i64 %p to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  ret i64 %len
ret0:
  ret i64 0
}

define internal i64 @__tn_rc_tag(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %do, label %retm1
do:
  %obj = inttoptr i64 %p to ptr
  %tp = getelementptr inbounds i8, ptr %obj, i64 8
  %tag = load i64, ptr %tp, align 8, !tbaa !7
  ret i64 %tag
retm1:
  ret i64 -1
}

define internal i64 @__tn_byte_at(i64 %s, i64 %idx) alwaysinline {
entry:
  %nz = icmp ne i64 %s, 0
  br i1 %nz, label %chk, label %oob
chk:
  %obj = inttoptr i64 %s to ptr
  %lp = getelementptr inbounds i8, ptr %obj, i64 16
  %len = load i64, ptr %lp, align 8, !tbaa !8
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %inb, label %oob
inb:
  %base = getelementptr inbounds i8, ptr %obj, i64 40
  %bp = getelementptr inbounds i8, ptr %base, i64 %idx
  %b = load i8, ptr %bp, align 1, !tbaa !10
  %ext = zext i8 %b to i64
  ret i64 %ext
oob:
  ret i64 -1
}

define internal i64 @__tn_buf_push_byte(i64 %h, i64 %byte_val) alwaysinline {
entry:
  %nz = icmp ne i64 %h, 0
  br i1 %nz, label %chk, label %ret0
chk:
  %bp = inttoptr i64 %h to ptr
  %lp = getelementptr inbounds i8, ptr %bp, i64 8
  %cp = getelementptr inbounds i8, ptr %bp, i64 16
  %len = load i64, ptr %lp, align 8
  %cap = load i64, ptr %cp, align 8
  %ok = icmp ult i64 %len, %cap
  br i1 %ok, label %fast, label %slow
fast:
  %dp = load ptr, ptr %bp, align 8
  %dst = getelementptr inbounds i8, ptr %dp, i64 %len
  %b8 = trunc i64 %byte_val to i8
  store i8 %b8, ptr %dst, align 1
  %nlen = add nuw i64 %len, 1
  store i64 %nlen, ptr %lp, align 8
  ret i64 %nlen
slow:
  %r = tail call i64 @tenon_buf_push_byte(i64 %h, i64 %byte_val)
  ret i64 %r
ret0:
  ret i64 0
}

define internal i64 @__tn_text_eq(i64 %a, i64 %b) alwaysinline {
entry:
  %same = icmp eq i64 %a, %b
  br i1 %same, label %ret1, label %chk_nz
chk_nz:
  %anz = icmp ne i64 %a, 0
  %bnz = icmp ne i64 %b, 0
  %both = and i1 %anz, %bnz
  br i1 %both, label %chk_len, label %ret0
chk_len:
  %pa = inttoptr i64 %a to ptr
  %pb = inttoptr i64 %b to ptr
  %la_p = getelementptr inbounds i8, ptr %pa, i64 16
  %lb_p = getelementptr inbounds i8, ptr %pb, i64 16
  %la = load i64, ptr %la_p, align 8, !tbaa !8
  %lb = load i64, ptr %lb_p, align 8, !tbaa !8
  %leq = icmp eq i64 %la, %lb
  br i1 %leq, label %cmp_mem, label %ret0
cmp_mem:
  %ba = getelementptr inbounds i8, ptr %pa, i64 40
  %bb = getelementptr inbounds i8, ptr %pb, i64 40
  %mc = call i32 @memcmp(ptr %ba, ptr %bb, i64 %la)
  %eq = icmp eq i32 %mc, 0
  %res = zext i1 %eq to i64
  ret i64 %res
ret1:
  ret i64 1
ret0:
  ret i64 0
}
"#;

pub(crate) const TENON_TBAA_METADATA: &str = r#"
!0 = !{!"Tenon TBAA"}
!1 = !{!"tn_rc", !0, i64 0}
!2 = !{!"tn_tag", !0, i64 0}
!3 = !{!"tn_len", !0, i64 0}
!4 = !{!"tn_mask", !0, i64 0}
!5 = !{!"tn_field", !0, i64 0}
!6 = !{!1, !1, i64 0}
!7 = !{!2, !2, i64 0}
!8 = !{!3, !3, i64 0}
!9 = !{!4, !4, i64 0}
!10 = !{!5, !5, i64 0}
"#;

