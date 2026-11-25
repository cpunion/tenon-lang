//! Modular AOT C Runtime Sections (Systems Core, POSIX Sync/Threads, POSIX TCP Net),
//! Atomic Inline LLVM IR Helpers (only emitted when threading is used!), and C99 Header Emitter.

use std::collections::BTreeMap;
use tenon_syntax::*;

pub(crate) const TENON_AOT_SYSTEMS_CORE_C: &str = r#"
/* ===== Tenon Systems Core C Runtime (Bytes, BytesSlice, CStruct, RawMem, F32) ===== */

int64_t tenon_clz(int64_t x) {
    uint64_t u = (uint64_t)x;
    return u == 0 ? 64 : (int64_t)__builtin_clzll(u);
}
int64_t tenon_ctz(int64_t x) {
    uint64_t u = (uint64_t)x;
    return u == 0 ? 64 : (int64_t)__builtin_ctzll(u);
}
int64_t tenon_popcnt(int64_t x) {
    return (int64_t)__builtin_popcountll((uint64_t)x);
}
int64_t tenon_bswap16(int64_t x) {
    return (int64_t)__builtin_bswap16((uint16_t)x);
}
int64_t tenon_bswap32(int64_t x) {
    return (int64_t)__builtin_bswap32((uint32_t)x);
}
int64_t tenon_bswap64(int64_t x) {
    return (int64_t)__builtin_bswap64((uint64_t)x);
}
int64_t tenon_rotl64(int64_t x, int64_t n) {
    uint64_t u = (uint64_t)x;
    unsigned k = (unsigned)n & 63u;
    return k == 0 ? x : (int64_t)((u << k) | (u >> (64u - k)));
}
int64_t tenon_rotr64(int64_t x, int64_t n) {
    uint64_t u = (uint64_t)x;
    unsigned k = (unsigned)n & 63u;
    return k == 0 ? x : (int64_t)((u >> k) | (u << (64u - k)));
}

int64_t tenon_f32_to_text(int64_t bits) {
    union { float f; uint32_t u; } u;
    u.u = (uint32_t)bits;
    float v = (u.f == 0.0f) ? 0.0f : u.f;
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%.7g", (double)v);
    if (n < 0) n = 0;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    return tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)n);
}

int64_t tenon_parse_f32(int64_t s) {
    union { float f; uint32_t u; } u;
    if (s == 0) { u.f = (float)NAN; return (int64_t)u.u; }
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    size_t n = (size_t)hdr->len;
    char tmp[128];
    char* buf = n < sizeof(tmp) ? tmp : (char*)malloc(n + 1);
    if (n > 0) memcpy(buf, (const void*)hdr->fields, n);
    buf[n] = '\0';
    char* end = NULL;
    float v = strtof(buf, &end);
    while (end && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end++;
    int ok = (end != buf && end && *end == '\0');
    if (buf != tmp) free(buf);
    u.f = ok ? v : (float)NAN;
    return (int64_t)u.u;
}

int64_t tenon_cstruct_alloc(int64_t byte_size) {
    int64_t bs = byte_size < 0 ? 0 : byte_size;
    int64_t words = (bs + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(12, bs, words);
    memset(obj->fields, 0, (size_t)words * sizeof(int64_t));
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_cstruct_data_ptr(int64_t obj) {
    if (obj == 0) return 0;
    return (int64_t)(uintptr_t)((TenonObj*)(uintptr_t)obj)->fields;
}

int64_t tenon_raw_alloc(int64_t byte_size) {
    size_t bs = byte_size < 1 ? 1 : (size_t)byte_size;
    void* p = calloc(1, bs);
    return (int64_t)(uintptr_t)p;
}

int64_t tenon_raw_free(int64_t ptr, int64_t byte_size) {
    (void)byte_size;
    if (ptr != 0) free((void*)(uintptr_t)ptr);
    return 0;
}

int64_t tenon_raw_memcpy(int64_t dst, int64_t src, int64_t n) {
    if (dst != 0 && src != 0 && n > 0) {
        memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, (size_t)n);
    }
    return dst;
}

int64_t tenon_raw_memset(int64_t dst, int64_t val, int64_t n) {
    if (dst != 0 && n > 0) {
        memset((void*)(uintptr_t)dst, (int)(val & 0xFF), (size_t)n);
    }
    return dst;
}

int64_t tenon_raw_memcmp(int64_t a, int64_t b, int64_t n) {
    if (a == b || n <= 0) return 0;
    if (a == 0) return -1;
    if (b == 0) return 1;
    int c = memcmp((const void*)(uintptr_t)a, (const void*)(uintptr_t)b, (size_t)n);
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

int64_t tenon_bytes_new(int64_t cap_bytes) {
    int64_t cb = cap_bytes < 0 ? 0 : cap_bytes;
    int64_t words = (cb + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(10, 0, words);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_bytes_filled(int64_t len_bytes, int64_t fill_byte) {
    int64_t lb = len_bytes < 0 ? 0 : len_bytes;
    int64_t words = (lb + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(10, lb, words);
    if (lb > 0) memset(obj->fields, (int)(fill_byte & 0xFF), (size_t)lb);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_bytes_len(int64_t b) {
    return b == 0 ? 0 : ((TenonObj*)(uintptr_t)b)->len;
}

int64_t tenon_bytes_get(int64_t b, int64_t idx) {
    if (b == 0 || idx < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (idx >= hdr->len) return 0;
    return (int64_t)((const uint8_t*)hdr->fields)[idx];
}

int64_t tenon_bytes_set(int64_t b, int64_t idx, int64_t byte_val) {
    if (b == 0 || idx < 0) return b;
    int64_t uniq = tenon_rc_ensure_unique(b);
    TenonObj* hdr = (TenonObj*)(uintptr_t)uniq;
    if (idx < hdr->len) {
        ((uint8_t*)hdr->fields)[idx] = (uint8_t)(byte_val & 0xFF);
    }
    return uniq;
}

int64_t tenon_bytes_push(int64_t b, int64_t byte_val) {
    uint8_t bv = (uint8_t)(byte_val & 0xFF);
    if (b == 0) return tenon_bytes_filled(1, (int64_t)bv);
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    int64_t old_len = hdr->len;
    int64_t cap_bytes = hdr->cap * 8;
#ifdef TENON_THREAD_SAFE
    int is_uniq = tenon_rc_is_unique_mt(hdr);
#else
    int is_uniq = hdr->rc == 1;
#endif
    if (is_uniq && old_len < cap_bytes) {
        ((uint8_t*)hdr->fields)[old_len] = bv;
        hdr->len = old_len + 1;
        return b;
    }
    int64_t new_cap = (old_len >= cap_bytes) ? (cap_bytes * 2) : (old_len + 16);
    if (new_cap < old_len + 1) new_cap = old_len + 1;
    if (new_cap < 16) new_cap = 16;
    int64_t new_words = (new_cap + 7) / 8;
    TenonObj* fresh = tenon_rc_alloc_raw(10, old_len + 1, new_words);
    uint8_t* dst = (uint8_t*)fresh->fields;
    if (old_len > 0) memcpy(dst, (const void*)hdr->fields, (size_t)old_len);
    dst[old_len] = bv;
    tenon_rc_dec(b);
    return (int64_t)(uintptr_t)fresh;
}

int64_t tenon_bytes_read_u16_le(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 2 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    return (int64_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int64_t tenon_bytes_read_u16_be(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 2 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    return (int64_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

int64_t tenon_bytes_read_u32_le(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 4 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return (int64_t)v;
}

int64_t tenon_bytes_read_u32_be(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 4 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    uint32_t v = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    return (int64_t)v;
}

int64_t tenon_bytes_read_u64_le(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 8 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i] << (i * 8));
    return (int64_t)v;
}

int64_t tenon_bytes_read_u64_be(int64_t b, int64_t off) {
    if (b == 0 || off < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    if (off + 8 > hdr->len) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields + off;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | (uint64_t)p[i];
    return (int64_t)v;
}

int64_t tenon_bytes_write_u16_le(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 2 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint16_t v = (uint16_t)val;
        p[0] = (uint8_t)(v & 0xFF);
        p[1] = (uint8_t)((v >> 8) & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_write_u16_be(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 2 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint16_t v = (uint16_t)val;
        p[0] = (uint8_t)((v >> 8) & 0xFF);
        p[1] = (uint8_t)(v & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_write_u32_le(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 4 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint32_t v = (uint32_t)val;
        p[0] = (uint8_t)(v & 0xFF);
        p[1] = (uint8_t)((v >> 8) & 0xFF);
        p[2] = (uint8_t)((v >> 16) & 0xFF);
        p[3] = (uint8_t)((v >> 24) & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_write_u32_be(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 4 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint32_t v = (uint32_t)val;
        p[0] = (uint8_t)((v >> 24) & 0xFF);
        p[1] = (uint8_t)((v >> 16) & 0xFF);
        p[2] = (uint8_t)((v >> 8) & 0xFF);
        p[3] = (uint8_t)(v & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_write_u64_le(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 8 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint64_t v = (uint64_t)val;
        for (int i = 0; i < 8; i++) p[i] = (uint8_t)((v >> (i * 8)) & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_write_u64_be(int64_t b, int64_t off, int64_t val) {
    int64_t u = tenon_rc_ensure_unique(b);
    if (u == 0 || off < 0) return u;
    TenonObj* hdr = (TenonObj*)(uintptr_t)u;
    if (off + 8 <= hdr->len) {
        uint8_t* p = (uint8_t*)hdr->fields + off;
        uint64_t v = (uint64_t)val;
        for (int i = 0; i < 8; i++) p[i] = (uint8_t)((v >> ((7 - i) * 8)) & 0xFF);
    }
    return u;
}

int64_t tenon_bytes_slice(int64_t owner, int64_t start, int64_t end) {
    int64_t olen = owner == 0 ? 0 : ((TenonObj*)(uintptr_t)owner)->len;
    int64_t st = start < 0 ? 0 : (start > olen ? olen : start);
    int64_t en = end < st ? st : (end > olen ? olen : end);
    TenonObj* sl = tenon_rc_alloc_raw(11, 3, 3);
    sl->rc_mask = 1;
    sl->fields[0] = tenon_rc_inc(owner);
    sl->fields[1] = st;
    sl->fields[2] = en - st;
    return (int64_t)(uintptr_t)sl;
}

int64_t tenon_slice_len(int64_t sl) {
    return sl == 0 ? 0 : ((TenonObj*)(uintptr_t)sl)->fields[2];
}

int64_t tenon_slice_get(int64_t sl, int64_t idx) {
    if (sl == 0 || idx < 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)sl;
    int64_t owner = hdr->fields[0];
    int64_t off = hdr->fields[1];
    int64_t len = hdr->fields[2];
    if (owner == 0 || idx >= len) return 0;
    return (int64_t)((const uint8_t*)((TenonObj*)(uintptr_t)owner)->fields)[off + idx];
}

int64_t tenon_bytes_to_text(int64_t b) {
    if (b == 0) return tenon_text_new(0, 0);
    TenonObj* hdr = (TenonObj*)(uintptr_t)b;
    return tenon_text_new((int64_t)(uintptr_t)hdr->fields, hdr->len);
}

int64_t tenon_text_to_bytes(int64_t s) {
    if (s == 0) return tenon_bytes_new(0);
    TenonObj* hdr = (TenonObj*)(uintptr_t)s;
    int64_t lb = hdr->len;
    int64_t words = (lb + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(10, lb, words);
    if (lb > 0) memcpy(obj->fields, (const void*)hdr->fields, (size_t)lb);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_bytes_ptr(int64_t b) {
    if (b == 0) return 0;
    return (int64_t)(uintptr_t)((TenonObj*)(uintptr_t)b)->fields;
}

int64_t tenon_list_data_ptr(int64_t xs) {
    if (xs == 0) return 0;
    return (int64_t)(uintptr_t)((TenonObj*)(uintptr_t)xs)->fields;
}

int64_t tenon_gpu_gid_x(void) { return 0; }
int64_t tenon_gpu_gid_y(void) { return 0; }
int64_t tenon_gpu_gid_z(void) { return 0; }
int64_t tenon_gpu_lid_x(void) { return 0; }
int64_t tenon_gpu_lid_y(void) { return 0; }
int64_t tenon_gpu_wgid_x(void) { return 0; }
int64_t tenon_gpu_wgid_y(void) { return 0; }
int64_t tenon_gpu_barrier(void) { return 0; }
int64_t tenon_gpu_shared_f32(int64_t n) { int64_t z = 0; return tenon_flat_list_filled(n, 4, &z); }
int64_t tenon_gpu_shared_f16(int64_t n) { int64_t z = 0; return tenon_flat_list_filled(n, 2, &z); }
static inline float tn_u32_to_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline int64_t tn_f32_to_i64(float f) { uint32_t u; memcpy(&u, &f, 4); return (int64_t)u; }
int64_t tenon_gpu_exp(int64_t x) { return tn_f32_to_i64(expf(tn_u32_to_f32((uint32_t)x))); }
int64_t tenon_gpu_sqrt(int64_t x) { return tn_f32_to_i64(sqrtf(tn_u32_to_f32((uint32_t)x))); }
int64_t tenon_gpu_rsqrt(int64_t x) { return tn_f32_to_i64(1.0f / sqrtf(tn_u32_to_f32((uint32_t)x))); }
int64_t tenon_gpu_cos(int64_t x) { return tn_f32_to_i64(cosf(tn_u32_to_f32((uint32_t)x))); }
int64_t tenon_gpu_sin(int64_t x) { return tn_f32_to_i64(sinf(tn_u32_to_f32((uint32_t)x))); }
int64_t tenon_gpu_pow(int64_t a, int64_t b) { return tn_f32_to_i64(powf(tn_u32_to_f32((uint32_t)a), tn_u32_to_f32((uint32_t)b))); }
int64_t tenon_gpu_tanh(int64_t x) { return tn_f32_to_i64(tanhf(tn_u32_to_f32((uint32_t)x))); }

int64_t tenon_text_cstr_ptr(int64_t s) {
    if (s == 0) return (int64_t)(uintptr_t)"";
    return (int64_t)(uintptr_t)((TenonObj*)(uintptr_t)s)->fields;
}

int64_t tenon_cstr_to_text(int64_t ptr) {
    if (ptr == 0) return tenon_text_new(0, 0);
    const char* cs = (const char*)(uintptr_t)ptr;
    return tenon_text_new(ptr, (int64_t)strlen(cs));
}
"#;

pub(crate) const TENON_AOT_SYNC_C: &str = r#"
/* ===== Tenon POSIX Multi-Threading, Atomics, Mutex & MPMC Channel ===== */
#include <pthread.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int64_t closure;
    int64_t result;
    pthread_t th;
} TenonThreadRec;

static void* tenon_thread_trampoline(void* arg) {
    TenonThreadRec* tr = (TenonThreadRec*)arg;
    int64_t cl = tr->closure;
    int64_t fn_addr = ((TenonObj*)(uintptr_t)cl)->fields[0];
    int64_t res = 0;
    if (fn_addr != 0) {
        int64_t (*fn)(int64_t) = (int64_t (*)(int64_t))(uintptr_t)fn_addr;
        res = fn(cl);
    }
    tenon_rc_dec(cl);
    tr->result = res;
    return NULL;
}

int64_t tenon_thread_spawn(int64_t closure_ptr) {
    if (closure_ptr == 0) return 0;
    tenon_rc_mark_shared(closure_ptr);
    TenonThreadRec* tr = (TenonThreadRec*)calloc(1, sizeof(TenonThreadRec));
    tr->closure = tenon_rc_inc(closure_ptr);
    pthread_create(&tr->th, NULL, tenon_thread_trampoline, tr);
    return (int64_t)(uintptr_t)tr;
}

int64_t tenon_thread_join(int64_t handle) {
    if (handle == 0) return 0;
    TenonThreadRec* tr = (TenonThreadRec*)(uintptr_t)handle;
    pthread_join(tr->th, NULL);
    int64_t res = tr->result;
    free(tr);
    return res;
}

int64_t tenon_thread_sleep_ms(int64_t ms) {
    if (ms > 0) {
        struct timespec ts;
        ts.tv_sec = ms / 1000;
        ts.tv_nsec = (ms % 1000) * 1000000L;
        nanosleep(&ts, NULL);
    }
    return 0;
}

int64_t tenon_thread_hw_concurrency(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int64_t)n : 1;
}

static inline int tenon_decode_ord(int64_t o) {
    switch (o) {
        case 0: return __ATOMIC_RELAXED;
        case 1: return __ATOMIC_ACQUIRE;
        case 2: return __ATOMIC_RELEASE;
        case 3: return __ATOMIC_ACQ_REL;
        default: return __ATOMIC_SEQ_CST;
    }
}
static inline int tenon_decode_load_ord(int64_t o) {
    if (o == 0) return __ATOMIC_RELAXED;
    if (o == 1 || o == 3) return __ATOMIC_ACQUIRE;
    return __ATOMIC_SEQ_CST;
}
static inline int tenon_decode_store_ord(int64_t o) {
    if (o == 0) return __ATOMIC_RELAXED;
    if (o == 2 || o == 3) return __ATOMIC_RELEASE;
    return __ATOMIC_SEQ_CST;
}

int64_t tenon_atomic_new(int64_t init_val) {
    TenonObj* obj = tenon_rc_alloc_raw(13, 1, 1);
    __atomic_store_n(&obj->fields[0], init_val, __ATOMIC_SEQ_CST);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_atomic_load(int64_t a, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    return __atomic_load_n(&obj->fields[0], tenon_decode_load_ord(ord));
}

int64_t tenon_atomic_store(int64_t a, int64_t val, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    __atomic_store_n(&obj->fields[0], val, tenon_decode_store_ord(ord));
    return 0;
}

int64_t tenon_atomic_fetch_add(int64_t a, int64_t delta, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    return __atomic_fetch_add(&obj->fields[0], delta, tenon_decode_ord(ord));
}

int64_t tenon_atomic_fetch_sub(int64_t a, int64_t delta, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    return __atomic_fetch_sub(&obj->fields[0], delta, tenon_decode_ord(ord));
}

int64_t tenon_atomic_cas(int64_t a, int64_t expected, int64_t desired, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    int64_t exp = expected;
    return __atomic_compare_exchange_n(&obj->fields[0], &exp, desired, 0, tenon_decode_ord(ord), tenon_decode_load_ord(ord)) ? 1 : 0;
}

int64_t tenon_atomic_swap(int64_t a, int64_t val, int64_t ord) {
    if (a == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)a;
    return __atomic_exchange_n(&obj->fields[0], val, tenon_decode_ord(ord));
}

void tenon_drop_mutex_handle(int64_t raw) {
    if (raw != 0) {
        pthread_mutex_t* mu = (pthread_mutex_t*)(uintptr_t)raw;
        pthread_mutex_destroy(mu);
        free(mu);
    }
}

int64_t tenon_mutex_new(int64_t init_val, int64_t elem_is_rc) {
    pthread_mutex_t* mu = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mu, NULL);
    TenonObj* obj = tenon_rc_alloc_raw(14, 2, 2);
    if (elem_is_rc != 0) {
        obj->rc_mask = 2;
        if (init_val != 0) { tenon_rc_mark_shared(init_val); tenon_rc_inc(init_val); }
    }
    obj->fields[0] = (int64_t)(uintptr_t)mu;
    obj->fields[1] = init_val;
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_mutex_lock(int64_t m) {
    if (m != 0) {
        pthread_mutex_lock((pthread_mutex_t*)(uintptr_t)((TenonObj*)(uintptr_t)m)->fields[0]);
    }
    return 0;
}

int64_t tenon_mutex_unlock(int64_t m) {
    if (m != 0) {
        pthread_mutex_unlock((pthread_mutex_t*)(uintptr_t)((TenonObj*)(uintptr_t)m)->fields[0]);
    }
    return 0;
}

int64_t tenon_mutex_get(int64_t m) {
    if (m == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)m;
    int64_t v = obj->fields[1];
    if ((obj->rc_mask & 2) && v != 0) tenon_rc_inc(v);
    return v;
}

int64_t tenon_mutex_set(int64_t m, int64_t val) {
    if (m == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)m;
    int is_rc = (obj->rc_mask & 2) != 0;
    int64_t old = obj->fields[1];
    if (is_rc && val != 0) { tenon_rc_mark_shared(val); tenon_rc_inc(val); }
    obj->fields[1] = val;
    if (is_rc && old != 0) tenon_rc_dec(old);
    return 0;
}

typedef struct TenonChanNode {
    int64_t val;
    struct TenonChanNode* next;
} TenonChanNode;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    TenonChanNode* head;
    TenonChanNode* tail;
    int closed;
} TenonChanInner;

void tenon_drop_channel_handle(int64_t raw, int is_rc) {
    if (raw == 0) return;
    TenonChanInner* ch = (TenonChanInner*)(uintptr_t)raw;
    TenonChanNode* n = ch->head;
    while (n) {
        TenonChanNode* nxt = n->next;
        if (is_rc && n->val != 0) tenon_rc_dec(n->val);
        free(n);
        n = nxt;
    }
    pthread_cond_destroy(&ch->cv);
    pthread_mutex_destroy(&ch->mu);
    free(ch);
}

int64_t tenon_chan_new(int64_t elem_is_rc) {
    TenonChanInner* ch = (TenonChanInner*)calloc(1, sizeof(TenonChanInner));
    pthread_mutex_init(&ch->mu, NULL);
    pthread_cond_init(&ch->cv, NULL);
    TenonObj* obj = tenon_rc_alloc_raw(15, 2, 2);
    obj->fields[0] = (int64_t)(uintptr_t)ch;
    obj->fields[1] = elem_is_rc != 0 ? 1 : 0;
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_chan_send(int64_t ch_ptr, int64_t val) {
    if (ch_ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ch_ptr;
    TenonChanInner* ch = (TenonChanInner*)(uintptr_t)obj->fields[0];
    int is_rc = obj->fields[1] != 0;
    pthread_mutex_lock(&ch->mu);
    if (ch->closed) {
        pthread_mutex_unlock(&ch->mu);
        return 0;
    }
    if (is_rc && val != 0) { tenon_rc_mark_shared(val); tenon_rc_inc(val); }
    TenonChanNode* node = (TenonChanNode*)malloc(sizeof(TenonChanNode));
    node->val = val;
    node->next = NULL;
    if (ch->tail) ch->tail->next = node;
    else ch->head = node;
    ch->tail = node;
    pthread_cond_signal(&ch->cv);
    pthread_mutex_unlock(&ch->mu);
    return 1;
}

static int64_t tenon_make_option_aot(int has_val, int64_t val, int is_rc) {
    if (has_val) {
        TenonObj* opt = tenon_rc_alloc_raw(3, 1, 1);
        if (is_rc) opt->rc_mask = 1;
        opt->fields[0] = val;
        return (int64_t)(uintptr_t)opt;
    }
    return tenon_rc_alloc(2, 0);
}

int64_t tenon_chan_recv(int64_t ch_ptr) {
    if (ch_ptr == 0) return tenon_make_option_aot(0, 0, 0);
    TenonObj* obj = (TenonObj*)(uintptr_t)ch_ptr;
    TenonChanInner* ch = (TenonChanInner*)(uintptr_t)obj->fields[0];
    int is_rc = obj->fields[1] != 0;
    pthread_mutex_lock(&ch->mu);
    while (!ch->head && !ch->closed) {
        pthread_cond_wait(&ch->cv, &ch->mu);
    }
    if (ch->head) {
        TenonChanNode* n = ch->head;
        ch->head = n->next;
        if (!ch->head) ch->tail = NULL;
        int64_t v = n->val;
        free(n);
        pthread_mutex_unlock(&ch->mu);
        return tenon_make_option_aot(1, v, is_rc);
    }
    pthread_mutex_unlock(&ch->mu);
    return tenon_make_option_aot(0, 0, is_rc);
}

int64_t tenon_chan_try_recv(int64_t ch_ptr) {
    if (ch_ptr == 0) return tenon_make_option_aot(0, 0, 0);
    TenonObj* obj = (TenonObj*)(uintptr_t)ch_ptr;
    TenonChanInner* ch = (TenonChanInner*)(uintptr_t)obj->fields[0];
    int is_rc = obj->fields[1] != 0;
    pthread_mutex_lock(&ch->mu);
    if (ch->head) {
        TenonChanNode* n = ch->head;
        ch->head = n->next;
        if (!ch->head) ch->tail = NULL;
        int64_t v = n->val;
        free(n);
        pthread_mutex_unlock(&ch->mu);
        return tenon_make_option_aot(1, v, is_rc);
    }
    pthread_mutex_unlock(&ch->mu);
    return tenon_make_option_aot(0, 0, is_rc);
}

int64_t tenon_chan_close(int64_t ch_ptr) {
    if (ch_ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ch_ptr;
    TenonChanInner* ch = (TenonChanInner*)(uintptr_t)obj->fields[0];
    pthread_mutex_lock(&ch->mu);
    ch->closed = 1;
    pthread_cond_broadcast(&ch->cv);
    pthread_mutex_unlock(&ch->mu);
    return 0;
}

/* ===== Tenon Lock-Free P-Core SPMC Batch-Stealing M:N Preemptive Scheduler (Zero cgo Overhead!) ===== */
#include <sched.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#define TN_MAX_WORKERS  16
#define TN_INJ_SHARDS   8
#define TN_INJ_MASK     (TN_INJ_SHARDS - 1)
#define TN_MAX_QUEUES   (TN_MAX_WORKERS + TN_INJ_SHARDS + 1)
#define TN_DEQUE_CAP    4096
#define TN_DEQUE_MASK   (TN_DEQUE_CAP - 1)
#define TN_STEAL_BATCH  32
#define TN_MAX_SPINNING 1

/* 64-byte cache-line aligned per task: eliminates 3-way false sharing between main spawner, worker & joiner! */
typedef struct __attribute__((aligned(64))) {
    int64_t closure;
    volatile int64_t result;
    volatile int32_t done;
    volatile int32_t refcount; /* 2 = runner + joiner */
} TenonTask;

/* Cache-line-isolated Lock-Free Monotonic SPMC FIFO Ring + Private Batch Cache!
 * Mathematically ABA-free & safe for adaptive batch-stealing: BOTH owner and thieves consume from `head` via CAS,
 * while ONLY the owner writes `ring[tail]` and monotonically advances `tail` (tail NEVER decreases)!
 * Producer caches `cached_head` in its own cache line so push NEVER touches contended `head` on the hot path! */
typedef struct __attribute__((aligned(64))) {
    volatile int64_t head; /* monotonic consumer cursor (CAS by owner & thieves) */
    char _pad0[56];
    volatile int64_t tail; /* monotonic producer cursor (owner-only release store) */
    int64_t cached_head;   /* producer-private shadow of head (zero cross-core read on push!) */
    char _pad1[48];
    volatile uint64_t task_epoch;
    volatile uint64_t last_seen_epoch;
    volatile int32_t busy;
    volatile int32_t sleeping;
    int id;
    int priv_count;
    int64_t local_completed;
    int64_t local_steals;
    pthread_mutex_t wake_mu;
    pthread_cond_t  wake_cv;
    pthread_t th;
    TenonTask* priv[TN_STEAL_BATCH];
    TenonTask* ring[TN_DEQUE_CAP];
} TenonWorker;

/* Exported global preemption poll flag checked by LLVM loop back-edges (0.25ns predicted-not-taken!) */
volatile uint32_t tenon_preempt_flag = 0;

static TenonWorker g_workers[TN_MAX_QUEUES];
static int g_num_workers = 0;
static int g_num_queues = 0;
static volatile int g_sched_inited = 0;
static volatile int g_workers_started = 0;
static volatile int g_main_claimed = 0;
static uint64_t g_main_stripe_seq = 0;
static pthread_mutex_t g_sched_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_ext_push_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_sysmon_th;
static volatile int64_t __attribute__((aligned(64))) g_sched_sleeping = 0;
static volatile int64_t __attribute__((aligned(64))) g_sched_spinning = 0;
static volatile int64_t __attribute__((aligned(64))) g_sched_preemptions = 0;

/* Fast lock-free slab ring pool for TenonTask (64B-aligned slots eliminate false sharing!) */
#define TN_TASK_POOL_CAP 65536
#define TN_TASK_POOL_MASK (TN_TASK_POOL_CAP - 1)
static TenonTask __attribute__((aligned(64))) g_task_pool[TN_TASK_POOL_CAP];
static volatile uint64_t __attribute__((aligned(64))) g_task_pool_cursor = 0;
static uint64_t g_main_task_cursor = 0;

static inline TenonTask* tn_task_alloc_main(void) {
    uint64_t idx = (g_main_task_cursor++) & 32767u;
    TenonTask* slot = &g_task_pool[idx];
    if (__builtin_expect(__atomic_load_n(&slot->refcount, __ATOMIC_ACQUIRE) == 0, 1)) {
        slot->done = 0;
        slot->result = 0;
        __atomic_store_n(&slot->refcount, 2, __ATOMIC_RELEASE);
        return slot;
    }
    TenonTask* heap_t = (TenonTask*)calloc(1, sizeof(TenonTask));
    heap_t->refcount = 2;
    return heap_t;
}

static inline TenonTask* tn_task_alloc(void) {
    uint64_t idx = 32768u + (__atomic_fetch_add(&g_task_pool_cursor, 1, __ATOMIC_RELAXED) & 32767u);
    TenonTask* slot = &g_task_pool[idx];
    int32_t expected = 0;
    if (__builtin_expect(__atomic_compare_exchange_n(&slot->refcount, &expected, 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED), 1)) {
        slot->done = 0;
        slot->result = 0;
        return slot;
    }
    TenonTask* heap_t = (TenonTask*)calloc(1, sizeof(TenonTask));
    heap_t->refcount = 2;
    return heap_t;
}
static pthread_key_t g_sched_tls_key;
static int g_sched_tls_ready = 0;

static inline int tn_tls_get_worker_id(void) {
    if (!g_sched_tls_ready) return -1;
    uintptr_t v = (uintptr_t)pthread_getspecific(g_sched_tls_key);
    return (int)(v & 0xFFFFu) - 1;
}
static inline int tn_tls_get_preempt_depth(void) {
    if (!g_sched_tls_ready) return 0;
    uintptr_t v = (uintptr_t)pthread_getspecific(g_sched_tls_key);
    return (int)(v >> 16);
}
static inline void tn_tls_set(int wid, int depth) {
    if (!g_sched_tls_ready) return;
    uintptr_t v = ((uintptr_t)(wid + 1) & 0xFFFFu) | ((uintptr_t)(depth & 0xFFFF) << 16);
    pthread_setspecific(g_sched_tls_key, (void*)v);
}

static inline void tn_task_release(TenonTask* t) {
    if (__builtin_expect(__atomic_load_n(&t->refcount, __ATOMIC_ACQUIRE) == 1, 1)) {
        __atomic_store_n(&t->refcount, 0, __ATOMIC_RELEASE);
        if (t < &g_task_pool[0] || t >= &g_task_pool[TN_TASK_POOL_CAP]) {
            free(t);
        }
        return;
    }
    if (__atomic_sub_fetch(&t->refcount, 1, __ATOMIC_ACQ_REL) == 0) {
        if (t < &g_task_pool[0] || t >= &g_task_pool[TN_TASK_POOL_CAP]) {
            free(t);
        }
    }
}

/* Monotonic SPMC FIFO Push (single owner: 1 store + 1 release store, ZERO CAS, ZERO read of contended head!) */
static inline int tn_deque_push_owner(TenonWorker* w, TenonTask* t) {
    int64_t b = __atomic_load_n(&w->tail, __ATOMIC_RELAXED);
    if (__builtin_expect(b - w->cached_head >= TN_DEQUE_CAP - 64, 0)) {
        w->cached_head = __atomic_load_n(&w->head, __ATOMIC_ACQUIRE);
        if (b - w->cached_head >= TN_DEQUE_CAP - 64) return 0;
    }
    w->ring[b & TN_DEQUE_MASK] = t;
    __atomic_store_n(&w->tail, b + 1, __ATOMIC_RELEASE);
    return 1;
}

/* Monotonic SPMC Adaptive Batch Pop/Steal (100% race-free & ABA-free):
 * - Massive bursts (avail >= 32): grabs 32-task batch in ONE CAS (32x less bus traffic!)
 * - Cache-line bursts (8..31): grabs 8-task batch (1 full 64B cache line of pointers!)
 * - Small heavy tasks or tail (< 8): grabs 1 task at a time -> ZERO head-of-line blocking / stragglers! */
static inline TenonTask* tn_ring_take_batch(TenonWorker* src, TenonWorker* dst, int is_steal) {
    for (;;) {
        int64_t top = __atomic_load_n(&src->head, __ATOMIC_ACQUIRE);
        int64_t bot = __atomic_load_n(&src->tail, __ATOMIC_ACQUIRE);
        int64_t avail = bot - top;
        if (avail <= 0) return NULL;
        int64_t take = 1;
        if (dst != NULL && avail > 1) {
            if (is_steal) {
                take = avail >= 32 ? TN_STEAL_BATCH : (avail > 16 ? 8 : 1);
            } else {
                take = avail > TN_STEAL_BATCH ? TN_STEAL_BATCH : avail;
            }
            int64_t free_priv = TN_STEAL_BATCH - dst->priv_count;
            if (take - 1 > free_priv) take = free_priv + 1;
            if (take < 1) take = 1;
        }
        TenonTask* buf[TN_STEAL_BATCH];
        for (int64_t i = 0; i < take; i++) {
            buf[i] = src->ring[(top + i) & TN_DEQUE_MASK];
        }
        if (__atomic_compare_exchange_n(&src->head, &top, top + take, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            if (dst != NULL && take > 1) {
                for (int64_t i = 1; i < take; i++) {
                    dst->priv[dst->priv_count++] = buf[i];
                }
            }
            if (is_steal) {
                if (dst != NULL) dst->local_steals += take;
                else __atomic_fetch_add(&g_workers[g_num_workers + TN_INJ_SHARDS].local_steals, take, __ATOMIC_RELAXED);
            }
            return buf[0];
        }
        /* Contended CAS: if stealing, move on to another shard/victim immediately */
        if (is_steal) return NULL;
    }
}

static inline TenonTask* tn_deque_pop_owner(TenonWorker* w) {
    if (__builtin_expect(w->priv_count > 0, 1)) {
        return w->priv[--w->priv_count];
    }
    if (__atomic_load_n(&w->head, __ATOMIC_RELAXED) >= __atomic_load_n(&w->tail, __ATOMIC_RELAXED)) {
        return NULL;
    }
    return tn_ring_take_batch(w, w, 0);
}

static inline TenonTask* tn_deque_steal_batch(TenonWorker* victim, TenonWorker* thief) {
    if (__atomic_load_n(&victim->head, __ATOMIC_RELAXED) >= __atomic_load_n(&victim->tail, __ATOMIC_RELAXED)) {
        return NULL;
    }
    return tn_ring_take_batch(victim, thief, 1);
}

static inline int tn_sched_any_work(void) {
    for (int i = 0; i < g_num_queues; i++) {
        if (__atomic_load_n(&g_workers[i].head, __ATOMIC_ACQUIRE) < __atomic_load_n(&g_workers[i].tail, __ATOMIC_ACQUIRE)) {
            return 1;
        }
    }
    return 0;
}

static TenonTask* tn_sched_find_work(int my_id) {
    TenonWorker* self_w = (my_id >= 0 && my_id < g_num_workers) ? &g_workers[my_id] : NULL;
    if (self_w != NULL) {
        TenonTask* local = tn_deque_pop_owner(self_w);
        if (local) return local;
    }
    /* Only workers batch-steal into priv[]; main thread steals 1 at a time so workers never starve! */
    TenonWorker* steal_dst = self_w;
    /* 1. Probe the 4 stealable main-thread injection shards FIRST (rotated by worker id -> 4x less contention!) */
    int shard_base = (my_id >= 0 ? my_id : 0);
    for (int k = 0; k < TN_INJ_SHARDS; k++) {
        int vid = g_num_workers + ((shard_base + k) & TN_INJ_MASK);
        TenonTask* stolen = tn_deque_steal_batch(&g_workers[vid], steal_dst);
        if (stolen) return stolen;
    }
    /* 2. Probe external fallback queue */
    TenonTask* ext_t = tn_deque_steal_batch(&g_workers[g_num_workers + TN_INJ_SHARDS], steal_dst);
    if (ext_t) return ext_t;
    /* 3. Steal from peer worker rings */
    int start = (my_id >= 0 ? my_id + 1 : 0);
    for (int i = 0; i < g_num_workers; i++) {
        int vid = (start + i) % g_num_workers;
        if (vid == my_id) continue;
        TenonTask* stolen = tn_deque_steal_batch(&g_workers[vid], steal_dst);
        if (stolen) return stolen;
    }
    return NULL;
}

static void tn_sched_run_task(TenonTask* t, int wid) {
    if (wid >= 0 && wid < g_num_workers) {
        __atomic_store_n(&g_workers[wid].task_epoch, g_workers[wid].task_epoch + 1, __ATOMIC_RELAXED);
        __atomic_store_n(&g_workers[wid].busy, 1, __ATOMIC_RELEASE);
    }
    int64_t cl = t->closure;
    int64_t fn_addr = ((TenonObj*)(uintptr_t)cl)->fields[0];
    int64_t res = 0;
    if (fn_addr != 0) {
        /* Executes DIRECTLY on the worker's 8MB native OS stack: ZERO cgo stack switch! */
        int64_t (*fn)(int64_t) = (int64_t (*)(int64_t))(uintptr_t)fn_addr;
        res = fn(cl);
    }
    tenon_rc_dec(cl);
    t->result = res;
    __atomic_store_n(&t->done, 1, __ATOMIC_RELEASE);
    if (wid >= 0 && wid <= g_num_workers) {
        g_workers[wid].local_completed++;
        if (wid < g_num_workers) {
            __atomic_store_n(&g_workers[wid].busy, 0, __ATOMIC_RELEASE);
        }
    } else {
        __atomic_fetch_add(&g_workers[g_num_workers + TN_INJ_SHARDS].local_completed, 1, __ATOMIC_RELAXED);
    }
    tn_task_release(t);
}

/* Go-style background sysmon preemption watchdog (500us quantum) */
static void* tn_sysmon_loop(void* arg) {
    (void)arg;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 500000L; /* 0.5 ms quantum */
    for (;;) {
        nanosleep(&ts, NULL);
        int need_preempt = 0;
        for (int i = 0; i < g_num_workers; i++) {
            if (__atomic_load_n(&g_workers[i].busy, __ATOMIC_ACQUIRE)) {
                uint64_t ep = __atomic_load_n(&g_workers[i].task_epoch, __ATOMIC_RELAXED);
                if (ep == g_workers[i].last_seen_epoch) {
                    need_preempt = 1;
                } else {
                    g_workers[i].last_seen_epoch = ep;
                }
            }
        }
        if (need_preempt) {
            __atomic_store_n(&tenon_preempt_flag, 1, __ATOMIC_RELAXED);
        }
    }
    return NULL;
}

static inline uint64_t tn_fast_ticks(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static inline void tn_wake_worker(TenonWorker* w) {
    if (__builtin_expect(__atomic_load_n(&w->sleeping, __ATOMIC_RELAXED) != 0, 0)) {
        if (__atomic_exchange_n(&w->sleeping, 0, __ATOMIC_ACQ_REL) != 0) {
            __atomic_fetch_sub(&g_sched_sleeping, 1, __ATOMIC_RELEASE);
            pthread_mutex_lock(&w->wake_mu);
            pthread_cond_signal(&w->wake_cv);
            pthread_mutex_unlock(&w->wake_mu);
        }
    }
}

static inline void tn_wake_one_sleeper(void) {
    if (__builtin_expect(__atomic_load_n(&g_sched_sleeping, __ATOMIC_ACQUIRE) > 0, 0)) {
        for (int i = 0; i < g_num_workers; i++) {
            if (__atomic_load_n(&g_workers[i].sleeping, __ATOMIC_RELAXED)) {
                if (__atomic_exchange_n(&g_workers[i].sleeping, 0, __ATOMIC_ACQ_REL) != 0) {
                    __atomic_fetch_sub(&g_sched_sleeping, 1, __ATOMIC_RELEASE);
                    pthread_mutex_lock(&g_workers[i].wake_mu);
                    pthread_cond_signal(&g_workers[i].wake_cv);
                    pthread_mutex_unlock(&g_workers[i].wake_mu);
                    return;
                }
            }
        }
    }
}

static void* tn_worker_loop(void* arg) {
    TenonWorker* w = (TenonWorker*)arg;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    tn_tls_set(w->id, 0);
    __atomic_fetch_add(&g_workers_started, 1, __ATOMIC_RELEASE);
    /* ARM64 cntvct_el0 runs at 24 MHz (24,000 ticks = 1.0 ms); other arches use ns (1,000,000 = 1.0 ms) */
#if defined(__aarch64__)
    const uint64_t idle_park_ticks = 24000ULL;
#else
    const uint64_t idle_park_ticks = 1000000ULL;
#endif
    for (;;) {
        TenonTask* t = tn_deque_pop_owner(w);
        if (!t) t = tn_sched_find_work(w->id);
        if (t) {
            /* Handoff wake: if no other worker is spinning and more work is waiting, wake 1 sleeper! */
            if (__builtin_expect(__atomic_load_n(&g_sched_spinning, __ATOMIC_RELAXED) == 0 &&
                                 __atomic_load_n(&g_sched_sleeping, __ATOMIC_RELAXED) > 0, 0)) {
                if (w->priv_count > 0 || tn_sched_any_work()) {
                    tn_wake_one_sleeper();
                }
            }
            tn_sched_run_task(t, w->id);
            continue;
        }
        /* Go-style nmspinning cap (<= TN_MAX_SPINNING=2 active spinners):
         * Prevents 11 idle P-cores from pounding main()'s injection tail cache lines! */
        int is_spinner = 0;
        int64_t cur_sp = __atomic_load_n(&g_sched_spinning, __ATOMIC_RELAXED);
        while (cur_sp < TN_MAX_SPINNING) {
            if (__atomic_compare_exchange_n(&g_sched_spinning, &cur_sp, cur_sp + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
                is_spinner = 1;
                break;
            }
        }
        if (is_spinner) {
            uint64_t idle_start = tn_fast_ticks();
            uint32_t backoff = 32;
            for (;;) {
                t = tn_sched_find_work(w->id);
                if (t) {
                    int64_t rem = __atomic_sub_fetch(&g_sched_spinning, 1, __ATOMIC_ACQ_REL);
                    if (rem == 0 && __atomic_load_n(&g_sched_sleeping, __ATOMIC_RELAXED) > 0 && tn_sched_any_work()) {
                        tn_wake_one_sleeper();
                    }
                    tn_sched_run_task(t, w->id);
                    break;
                }
                if ((tn_fast_ticks() - idle_start) >= idle_park_ticks) {
                    __atomic_sub_fetch(&g_sched_spinning, 1, __ATOMIC_ACQ_REL);
                    break;
                }
                for (uint32_t s = 0; s < backoff; s++) {
#if defined(__aarch64__)
                    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__)
                    __asm__ __volatile__("pause" ::: "memory");
#endif
                }
                if (backoff < 256u) backoff <<= 1;
            }
            if (t) continue;
        }
        /* Park cleanly in kernel sleep with Dekker SEQ_CST barrier (mathematically lost-wakeup-free!) */
        pthread_mutex_lock(&w->wake_mu);
        __atomic_store_n(&w->sleeping, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_add(&g_sched_sleeping, 1, __ATOMIC_SEQ_CST);
        if (!tn_sched_any_work()) {
            pthread_cond_wait(&w->wake_cv, &w->wake_mu);
        }
        if (__atomic_exchange_n(&w->sleeping, 0, __ATOMIC_ACQ_REL) != 0) {
            __atomic_fetch_sub(&g_sched_sleeping, 1, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&w->wake_mu);
    }
    return NULL;
}

__attribute__((noinline))
static void tn_sched_init_slow(void) {
    pthread_mutex_lock(&g_sched_init_mu);
    if (!g_sched_inited) {
#ifdef __APPLE__
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        pthread_key_create(&g_sched_tls_key, NULL);
        g_sched_tls_ready = 1;
        long n = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __APPLE__
        /* Pin worker count strictly to Apple Silicon P-cores (hw.perflevel0.physicalcpu)
         * so NO worker is ever demoted onto slow E-cores (zero stragglers & zero oversubscription)! */
        int32_t perf_cpus = 0;
        size_t perf_sz = sizeof(perf_cpus);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cpus, &perf_sz, NULL, 0) == 0 && perf_cpus >= 2) {
            n = (long)perf_cpus;
        }
#endif
        if (n > 2) n = n - 1; /* Reserve 1 P-core for main spawner/joiner */
        if (n < 2) n = 2;
        if (n > TN_MAX_WORKERS) n = TN_MAX_WORKERS;
        g_num_workers = (int)n;
        g_num_queues = g_num_workers + TN_INJ_SHARDS + 1;
        for (int i = 0; i < g_num_queues; i++) {
            g_workers[i].head = 0;
            g_workers[i].tail = 0;
            g_workers[i].cached_head = 0;
            g_workers[i].task_epoch = 0;
            g_workers[i].last_seen_epoch = 0;
            g_workers[i].busy = 0;
            g_workers[i].sleeping = 0;
            g_workers[i].priv_count = 0;
            g_workers[i].local_completed = 0;
            g_workers[i].local_steals = 0;
            g_workers[i].id = i;
            pthread_mutex_init(&g_workers[i].wake_mu, NULL);
            pthread_cond_init(&g_workers[i].wake_cv, NULL);
        }
        for (int i = 0; i < g_num_workers; i++) {
            pthread_create(&g_workers[i].th, NULL, tn_worker_loop, &g_workers[i]);
            pthread_detach(g_workers[i].th);
        }
        while (__atomic_load_n(&g_workers_started, __ATOMIC_ACQUIRE) < g_num_workers) {
            sched_yield();
        }
        pthread_create(&g_sysmon_th, NULL, tn_sysmon_loop, NULL);
        pthread_detach(g_sysmon_th);
        __atomic_store_n(&g_sched_inited, 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&g_sched_init_mu);
}

static inline void tn_sched_ensure_init(void) {
    if (__builtin_expect(__atomic_load_n(&g_sched_inited, __ATOMIC_ACQUIRE) == 0, 0)) {
        tn_sched_init_slow();
    }
}

static inline int tn_sched_get_or_claim_wid(void) {
    int wid = tn_tls_get_worker_id();
    if (__builtin_expect(wid >= 0, 1)) return wid;
    int exp = 0;
    if (__atomic_compare_exchange_n(&g_main_claimed, &exp, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        tn_tls_set(g_num_workers, 0);
        return g_num_workers;
    }
    return -1;
}

/* Called on LLVM loop back-edge slow path when `tenon_preempt_flag != 0` */
void tenon_sched_preempt_point(void) {
    if (!__atomic_load_n(&g_sched_inited, __ATOMIC_RELAXED)) return;
    __atomic_store_n(&tenon_preempt_flag, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_sched_preemptions, 1, __ATOMIC_RELAXED);
    int wid = tn_tls_get_worker_id();
    int depth = tn_tls_get_preempt_depth();
    if (depth < 4) {
        tn_tls_set(wid, depth + 1);
        TenonTask* next = tn_sched_find_work(wid);
        if (next) {
            tn_sched_run_task(next, wid);
        } else {
            sched_yield();
        }
        tn_tls_set(wid, depth);
    } else {
        sched_yield();
    }
}

int64_t tenon_sched_spawn_owned(int64_t closure_ptr) {
    if (closure_ptr == 0) return 0;
    tenon_rc_mark_shared(closure_ptr);
    tn_sched_ensure_init();
    int wid = tn_sched_get_or_claim_wid();
    TenonTask* t = __builtin_expect(wid == g_num_workers, 1) ? tn_task_alloc_main() : tn_task_alloc();
    t->closure = closure_ptr; /* Ownership moved directly: ZERO rc_inc + ZERO caller rc_dec! */
    int pushed = 0;
    uint64_t seq = 0;
    if (__builtin_expect(wid == g_num_workers, 1)) {
        /* Stripe 16 consecutive tasks across 8 stealable injection shards so main()'s tail stays L1-exclusive! */
        seq = g_main_stripe_seq++;
        int shard = (int)((seq >> 4) & TN_INJ_MASK);
        pushed = tn_deque_push_owner(&g_workers[g_num_workers + shard], t);
    } else if (wid >= 0 && wid < g_num_workers) {
        pushed = tn_deque_push_owner(&g_workers[wid], t);
    } else {
        pthread_mutex_lock(&g_ext_push_mu);
        pushed = tn_deque_push_owner(&g_workers[g_num_workers + TN_INJ_SHARDS], t);
        pthread_mutex_unlock(&g_ext_push_mu);
    }
    if (__builtin_expect(!pushed, 0)) {
        tn_sched_run_task(t, wid);
    } else {
        int64_t slp = __atomic_load_n(&g_sched_sleeping, __ATOMIC_ACQUIRE);
        if (__builtin_expect(slp >= g_num_workers ||
                             (slp > 0 && ((seq & 63u) == 0) &&
                              __atomic_load_n(&g_sched_spinning, __ATOMIC_ACQUIRE) == 0), 0)) {
            /* Zero-contention Go wakep(): instant wake when all workers sleep; otherwise sampled every 64th spawn,
             * while running workers perform parallel tree handoff wakes off main()'s hot path! */
            tn_wake_one_sleeper();
        }
    }
    return (int64_t)(uintptr_t)t;
}

int64_t tenon_sched_spawn(int64_t closure_ptr) {
    if (closure_ptr == 0) return 0;
    return tenon_sched_spawn_owned(tenon_rc_inc(closure_ptr));
}

int64_t tenon_sched_join(int64_t handle) {
    if (handle == 0) return 0;
    TenonTask* t = (TenonTask*)(uintptr_t)handle;
    if (__builtin_expect(__atomic_load_n(&t->done, __ATOMIC_ACQUIRE) != 0, 1)) {
        int64_t res = t->result;
        tn_task_release(t);
        return res;
    }
    int wid = tn_sched_get_or_claim_wid();
    while (!__atomic_load_n(&t->done, __ATOMIC_ACQUIRE)) {
        TenonTask* help = tn_sched_find_work(wid);
        if (help) {
            tn_sched_run_task(help, wid);
        } else {
#if defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__)
            __asm__ __volatile__("pause" ::: "memory");
#else
            sched_yield();
#endif
        }
    }
    int64_t res = t->result;
    tn_task_release(t);
    return res;
}

int64_t tenon_sched_yield(void) {
    tenon_sched_preempt_point();
    return 0;
}

int64_t tenon_sched_workers(void) {
    tn_sched_ensure_init();
    return (int64_t)g_num_workers;
}

int64_t tenon_sched_preemptions(void) {
    return __atomic_load_n(&g_sched_preemptions, __ATOMIC_RELAXED);
}

int64_t tenon_sched_steals(void) {
    int64_t sum = 0;
    for (int i = 0; i < g_num_queues; i++) {
        sum += __atomic_load_n(&g_workers[i].local_steals, __ATOMIC_RELAXED);
    }
    return sum;
}

int64_t tenon_sched_tasks_completed(void) {
    int64_t sum = 0;
    for (int i = 0; i < g_num_queues; i++) {
        sum += __atomic_load_n(&g_workers[i].local_completed, __ATOMIC_RELAXED);
    }
    return sum;
}
"#;

pub(crate) const TENON_AOT_NET_C: &str = r#"
/* ===== Tenon POSIX Non-Blocking TCP Sockets, kqueue/epoll Reactor & Clock ===== */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/epoll.h>
#endif

#ifdef TENON_LLI_MODE
static _Atomic int tenon_last_eagain = 0;
#else
static _Thread_local int tenon_last_eagain = 0;
#endif

int64_t tenon_time_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

int64_t tenon_time_now_ms(void) {
    return tenon_time_now_ns() / 1000000LL;
}

int64_t tenon_net_last_would_block(void) {
    return tenon_last_eagain ? 1 : 0;
}

static int tenon_parse_ipv4_port(int64_t addr_text, struct sockaddr_in* sa) {
    if (addr_text == 0) return 0;
    TenonObj* hdr = (TenonObj*)(uintptr_t)addr_text;
    char buf[256];
    size_t n = (size_t)hdr->len;
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, (const void*)hdr->fields, n);
    buf[n] = '\0';
    char* colon = strrchr(buf, ':');
    if (!colon) return 0;
    *colon = '\0';
    int port = atoi(colon + 1);
    memset(sa, 0, sizeof(*sa));
    sa->sin_family = AF_INET;
    sa->sin_port = htons((uint16_t)port);
    if (buf[0] == '\0' || strcmp(buf, "0.0.0.0") == 0) {
        sa->sin_addr.s_addr = htonl(INADDR_ANY);
        return 1;
    }
    if (strcmp(buf, "localhost") == 0) {
        inet_pton(AF_INET, "127.0.0.1", &sa->sin_addr);
        return 1;
    }
    return inet_pton(AF_INET, buf, &sa->sin_addr) == 1;
}

int64_t tenon_net_tcp_listen(int64_t addr_text) {
    struct sockaddr_in sa;
    if (!tenon_parse_ipv4_port(addr_text, &sa)) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
    if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0 || listen(fd, 1024) < 0) {
        close(fd);
        return -1;
    }
    return (int64_t)fd;
}

int64_t tenon_net_tcp_local_port(int64_t fd) {
    if (fd < 0) return -1;
    struct sockaddr_in sa;
    socklen_t sl = sizeof(sa);
    if (getsockname((int)fd, (struct sockaddr*)&sa, &sl) < 0) return -1;
    return (int64_t)ntohs(sa.sin_port);
}

int64_t tenon_net_tcp_accept(int64_t fd) {
    tenon_last_eagain = 0;
    if (fd < 0) return -1;
    int cfd = accept((int)fd, NULL, NULL);
    if (cfd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) tenon_last_eagain = 1;
        return -1;
    }
    int one = 1;
    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return (int64_t)cfd;
}

int64_t tenon_net_tcp_connect(int64_t addr_text) {
    struct sockaddr_in sa;
    if (!tenon_parse_ipv4_port(addr_text, &sa)) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (connect(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        close(fd);
        return -1;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return (int64_t)fd;
}

int64_t tenon_net_set_nonblocking(int64_t fd, int64_t nb) {
    if (fd < 0) return 0;
    int flags = fcntl((int)fd, F_GETFL, 0);
    if (flags < 0) return 0;
    if (nb) flags |= O_NONBLOCK;
    else flags &= ~O_NONBLOCK;
    return fcntl((int)fd, F_SETFL, flags) == 0 ? 1 : 0;
}

int64_t tenon_net_set_nodelay(int64_t fd, int64_t nd) {
    if (fd < 0) return 0;
    int val = nd ? 1 : 0;
    return setsockopt((int)fd, IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val)) == 0 ? 1 : 0;
}

int64_t tenon_net_read(int64_t fd, int64_t max_bytes) {
    tenon_last_eagain = 0;
    if (fd < 0 || max_bytes <= 0) return tenon_bytes_new(0);
    size_t cap = (size_t)(max_bytes > 1048576 ? 1048576 : max_bytes);
    uint8_t stack_buf[4096];
    uint8_t* tmp = cap <= sizeof(stack_buf) ? stack_buf : (uint8_t*)malloc(cap);
    ssize_t n = recv((int)fd, tmp, cap, 0);
    if (n <= 0) {
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) tenon_last_eagain = 1;
        if (tmp != stack_buf) free(tmp);
        return tenon_bytes_new(0);
    }
    int64_t words = ((int64_t)n + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(10, (int64_t)n, words);
    memcpy(obj->fields, tmp, (size_t)n);
    if (tmp != stack_buf) free(tmp);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_net_read_text(int64_t fd, int64_t max_bytes) {
    tenon_last_eagain = 0;
    if (fd < 0 || max_bytes <= 0) return tenon_text_new(0, 0);
    size_t cap = (size_t)(max_bytes > 1048576 ? 1048576 : max_bytes);
    uint8_t stack_buf[8192];
    uint8_t* tmp = cap <= sizeof(stack_buf) ? stack_buf : (uint8_t*)malloc(cap);
    ssize_t n = recv((int)fd, tmp, cap, 0);
    if (n <= 0) {
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) tenon_last_eagain = 1;
        if (tmp != stack_buf) free(tmp);
        return tenon_text_new(0, 0);
    }
    int64_t words = ((int64_t)n + 1 + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(4, (int64_t)n, words);
    memcpy(obj->fields, tmp, (size_t)n);
    ((char*)obj->fields)[n] = '\0';
    if (tmp != stack_buf) free(tmp);
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_net_write(int64_t fd, int64_t bytes_obj) {
    if (fd < 0 || bytes_obj == 0) return -1;
    TenonObj* hdr = (TenonObj*)(uintptr_t)bytes_obj;
    if (hdr->len <= 0) return 0;
    const uint8_t* p = (const uint8_t*)hdr->fields;
    size_t rem = (size_t)hdr->len;
    size_t total = 0;
    while (rem > 0) {
        ssize_t n = send((int)fd, p + total, rem, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return total > 0 ? (int64_t)total : -1;
        }
        if (n == 0) break;
        total += (size_t)n;
        rem -= (size_t)n;
    }
    return (int64_t)total;
}

int64_t tenon_net_write_text(int64_t fd, int64_t text_obj) {
    return tenon_net_write(fd, text_obj);
}

int64_t tenon_net_close(int64_t fd) {
    if (fd >= 0) close((int)fd);
    return 0;
}

/* ===== OS Event Reactor (kqueue / epoll) ===== */
int64_t tenon_reactor_new(void) {
#ifdef __APPLE__
    return (int64_t)kqueue();
#elif defined(__linux__)
    return (int64_t)epoll_create1(EPOLL_CLOEXEC);
#else
    return -1;
#endif
}

int64_t tenon_reactor_add(int64_t rfd, int64_t fd, int64_t mask, int64_t token) {
    if (rfd < 0 || fd < 0) return 0;
#ifdef __APPLE__
    struct kevent ch[2];
    int n = 0;
    if (mask & 1) {
        EV_SET(&ch[n], (uintptr_t)fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, (void*)(uintptr_t)token);
        n++;
    }
    if (mask & 2) {
        EV_SET(&ch[n], (uintptr_t)fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, (void*)(uintptr_t)token);
        n++;
    }
    if (n == 0) return 1;
    return kevent((int)rfd, ch, n, NULL, 0, NULL) >= 0 ? 1 : 0;
#elif defined(__linux__)
    struct epoll_event ev;
    ev.events = ((mask & 1) ? EPOLLIN : 0) | ((mask & 2) ? EPOLLOUT : 0);
    ev.data.u64 = (uint64_t)token;
    return epoll_ctl((int)rfd, EPOLL_CTL_ADD, (int)fd, &ev) == 0 ? 1 : 0;
#else
    (void)mask; (void)token;
    return 0;
#endif
}

int64_t tenon_reactor_del(int64_t rfd, int64_t fd) {
    if (rfd < 0 || fd < 0) return 0;
#ifdef __APPLE__
    struct kevent ch[2];
    EV_SET(&ch[0], (uintptr_t)fd, EVFILT_READ, EV_DELETE, 0, 0, NULL);
    EV_SET(&ch[1], (uintptr_t)fd, EVFILT_WRITE, EV_DELETE, 0, 0, NULL);
    kevent((int)rfd, ch, 2, NULL, 0, NULL);
    return 1;
#elif defined(__linux__)
    epoll_ctl((int)rfd, EPOLL_CTL_DEL, (int)fd, NULL);
    return 1;
#else
    return 0;
#endif
}

int64_t tenon_reactor_wait(int64_t rfd, int64_t timeout_ms, int64_t max_events) {
    int cap = (int)(max_events < 1 ? 1 : (max_events > 1024 ? 1024 : max_events));
#ifdef __APPLE__
    struct kevent evs[1024];
    struct timespec ts;
    struct timespec* tsp = NULL;
    if (timeout_ms >= 0) {
        ts.tv_sec = (time_t)(timeout_ms / 1000);
        ts.tv_nsec = (long)((timeout_ms % 1000) * 1000000L);
        tsp = &ts;
    }
    int n = kevent((int)rfd, NULL, 0, evs, cap, tsp);
    if (n < 0) n = 0;
    TenonObj* list = tenon_rc_alloc_raw(5, (int64_t)n, (int64_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) {
        int64_t rev = 0;
        if (evs[i].filter == EVFILT_READ) rev |= 1;
        if (evs[i].filter == EVFILT_WRITE) rev |= 2;
        if (evs[i].flags & (EV_EOF | EV_ERROR)) rev |= 4;
        int64_t tok = (int64_t)(uintptr_t)evs[i].udata;
        list->fields[i] = (tok << 8) | rev;
    }
    return (int64_t)(uintptr_t)list;
#elif defined(__linux__)
    struct epoll_event evs[1024];
    int n = epoll_wait((int)rfd, evs, cap, (int)timeout_ms);
    if (n < 0) n = 0;
    TenonObj* list = tenon_rc_alloc_raw(5, (int64_t)n, (int64_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) {
        int64_t rev = 0;
        if (evs[i].events & EPOLLIN) rev |= 1;
        if (evs[i].events & EPOLLOUT) rev |= 2;
        if (evs[i].events & (EPOLLHUP | EPOLLERR)) rev |= 4;
        list->fields[i] = ((int64_t)evs[i].data.u64 << 8) | rev;
    }
    return (int64_t)(uintptr_t)list;
#else
    (void)rfd; (void)timeout_ms; (void)cap;
    return (int64_t)(uintptr_t)tenon_rc_alloc_raw(5, 0, 1);
#endif
}

int64_t tenon_reactor_close(int64_t rfd) {
    if (rfd >= 0) close((int)rfd);
    return 0;
}
"#;

/// Biased Non-Atomic + Atomic Perceus RC inline LLVM IR helpers — ONLY emitted when the program uses threads/atomics!
/// Thread-local objects (`owner_tid == tenon_tls_owner_tid`) execute 0 atomic instructions (~0.5ns vs ~5ns)!
pub(crate) const TENON_ATOMIC_RC_IR_HELPERS: &str = r#"
@tenon_tls_owner_tid = external thread_local global i32

define internal i64 @__tn_rc_inc(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %rc = load i64, ptr %obj, align 8
  %hi = lshr i64 %rc, 32
  %own = trunc i64 %hi to i32
  %me = load i32, ptr @tenon_tls_owner_tid, align 4
  %mine = icmp eq i32 %own, %me
  br i1 %mine, label %fast_inc, label %shared_chk
fast_inc:
  %nxt = add i64 %rc, 1
  store i64 %nxt, ptr %obj, align 8
  br label %done
shared_chk:
  %src = load atomic i64, ptr %obj monotonic, align 8
  %pos = icmp sgt i64 %src, 0
  br i1 %pos, label %atomic_inc, label %done
atomic_inc:
  %old = atomicrmw add ptr %obj, i64 1 monotonic, align 8
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
  %rc = load i64, ptr %obj, align 8
  %hi = lshr i64 %rc, 32
  %own = trunc i64 %hi to i32
  %me = load i32, ptr @tenon_tls_owner_tid, align 4
  %mine = icmp eq i32 %own, %me
  br i1 %mine, label %fast_dec, label %shared_chk
fast_dec:
  %cnt = trunc i64 %rc to i32
  %is1 = icmp ule i32 %cnt, 1
  br i1 %is1, label %free, label %fast_sub
fast_sub:
  %nxt = sub i64 %rc, 1
  store i64 %nxt, ptr %obj, align 8
  br label %done
shared_chk:
  %src = load atomic i64, ptr %obj acquire, align 8
  %pos = icmp sgt i64 %src, 0
  br i1 %pos, label %shared_live, label %done
shared_live:
  %scnt = trunc i64 %src to i32
  %sis1 = icmp eq i32 %scnt, 1
  br i1 %sis1, label %free, label %atomic_dec
atomic_dec:
  %old = atomicrmw sub ptr %obj, i64 1 acq_rel, align 8
  %ocnt = trunc i64 %old to i32
  %was1 = icmp ule i32 %ocnt, 1
  br i1 %was1, label %free, label %done
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
  %rc = load i64, ptr %obj, align 8
  %hi = lshr i64 %rc, 32
  %own = trunc i64 %hi to i32
  %me = load i32, ptr @tenon_tls_owner_tid, align 4
  %mine = icmp eq i32 %own, %me
  br i1 %mine, label %fast_uniq, label %shared_uniq
fast_uniq:
  %cnt = trunc i64 %rc to i32
  %u1 = icmp eq i32 %cnt, 1
  br i1 %u1, label %retp, label %slow
shared_uniq:
  %src = load atomic i64, ptr %obj acquire, align 8
  %scnt = trunc i64 %src to i32
  %su1 = icmp eq i32 %scnt, 1
  br i1 %su1, label %retp, label %slow
slow:
  %cow = tail call i64 @tenon_rc_ensure_unique(i64 %p)
  ret i64 %cow
retp:
  ret i64 %p
ret0:
  ret i64 0
}
"#;

/// Emit a clean C99 header (`.h`) for all `@repr(C)`/`@packed`/`@align` records,
/// `union`s, `bitfield`s, and `@export` functions in `program`.
pub fn emit_c_header(program: &Program) -> String {
    let mut aliases: BTreeMap<String, TypeExpr> = BTreeMap::new();
    let mut bitfield_backing: BTreeMap<String, ScalarKind> = BTreeMap::new();
    for item in &program.items {
        match item {
            Item::TypeAlias(ta) => {
                aliases.insert(ta.name.clone(), ta.base.clone());
            }
            Item::Bitfield(bf) => {
                if let TypeExpr::Named { name, .. } = &bf.backing_ty {
                    if let Some(sk) = ScalarKind::from_name(name) {
                        bitfield_backing.insert(bf.name.clone(), sk);
                    }
                }
            }
            _ => {}
        }
    }

    let mut out = String::new();
    out.push_str("/* Auto-generated C99 FFI header by Tenon compiler */\n");
    out.push_str("#pragma once\n");
    out.push_str("#include <stdbool.h>\n");
    out.push_str("#include <stddef.h>\n");
    out.push_str("#include <stdint.h>\n\n");
    out.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");

    // 1. Bitfields
    for item in &program.items {
        if let Item::Bitfield(bf) = item {
            let cty = c_type_str(&bf.backing_ty, &aliases, &bitfield_backing, None);
            out.push_str(&format!("typedef {} {};\n", cty, bf.name));
        }
    }

    // 2. `@repr(C)` / `@packed` / `@align` records & `union`s
    for item in &program.items {
        match item {
            Item::Record(r) if r.type_params.is_empty() && is_c_layout_attrs(&r.attrs) => {
                let mut attrs_c = String::new();
                if has_packed_attr(&r.attrs) {
                    attrs_c.push_str(" __attribute__((packed))");
                }
                if let Some(al) = explicit_align_attr(&r.attrs) {
                    attrs_c.push_str(&format!(" __attribute__((aligned({})))", al));
                }
                out.push_str(&format!("typedef struct{} {} {{\n", attrs_c, r.name));
                for f in &r.fields {
                    out.push_str(&format!(
                        "    {};\n",
                        c_field_decl(&f.name, &f.ty, &aliases, &bitfield_backing)
                    ));
                }
                out.push_str(&format!("}} {};\n\n", r.name));
            }
            Item::Union(u) => {
                let mut attrs_c = String::new();
                if let Some(al) = explicit_align_attr(&u.attrs) {
                    attrs_c.push_str(&format!(" __attribute__((aligned({})))", al));
                }
                out.push_str(&format!("typedef union{} {} {{\n", attrs_c, u.name));
                for f in &u.fields {
                    out.push_str(&format!(
                        "    {};\n",
                        c_field_decl(&f.name, &f.ty, &aliases, &bitfield_backing)
                    ));
                }
                out.push_str(&format!("}} {};\n\n", u.name));
            }
            _ => {}
        }
    }

    // 3. `@export` functions
    for item in &program.items {
        if let Item::Fn(f) = item {
            if let Some(cname) = extract_export_name(&f.attrs, &f.name) {
                let ret_c = match &f.ret_ty {
                    Some(rt) => c_type_str(rt, &aliases, &bitfield_backing, None),
                    None => "void".into(),
                };
                let params_c: Vec<String> = if f.params.is_empty() {
                    vec!["void".into()]
                } else {
                    f.params
                        .iter()
                        .map(|p| c_field_decl(&p.name, &p.ty, &aliases, &bitfield_backing))
                        .collect()
                };
                out.push_str(&format!(
                    "extern {} {}({});\n",
                    ret_c,
                    cname,
                    params_c.join(", ")
                ));
            }
        }
    }

    out.push_str("\n#ifdef __cplusplus\n}\n#endif\n");
    out
}

fn resolve_c_alias(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    if let TypeExpr::Named { name, .. } = ty {
        if let Some(b) = aliases.get(name) {
            return resolve_c_alias(b, aliases);
        }
    }
    ty.clone()
}

fn c_field_decl(
    name: &str,
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
    bitfields: &BTreeMap<String, ScalarKind>,
) -> String {
    let resolved = resolve_c_alias(ty, aliases);
    if let TypeExpr::Array { elem, len, .. } = resolved {
        let et = c_type_str(&elem, aliases, bitfields, None);
        return format!("{} {}[{}]", et, name, len);
    }
    format!("{} {}", c_type_str(&resolved, aliases, bitfields, None), name)
}

fn c_type_str(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
    bitfields: &BTreeMap<String, ScalarKind>,
    _unused: Option<()>,
) -> String {
    let resolved = resolve_c_alias(ty, aliases);
    match resolved {
        TypeExpr::Named { name, args, .. } => {
            if bitfields.contains_key(&name) {
                return name;
            }
            match name.as_str() {
                "Int8" | "CChar" => "int8_t".into(),
                "Int16" => "int16_t".into(),
                "Int32" | "CInt" => "int32_t".into(),
                "Int64" | "Int" | "ISize" => "int64_t".into(),
                "UInt8" | "Byte" => "uint8_t".into(),
                "UInt16" => "uint16_t".into(),
                "UInt32" | "CUInt" => "uint32_t".into(),
                "UInt64" | "USize" | "CSize" => "uint64_t".into(),
                "Float32" => "float".into(),
                "Float64" | "Float" => "double".into(),
                "Bool" => "bool".into(),
                "CVoid" => "void".into(),
                "CStr" => "const char*".into(),
                "Ptr" => {
                    let inner = args
                        .first()
                        .map(|a| c_type_str(a, aliases, bitfields, None))
                        .unwrap_or_else(|| "void".into());
                    format!("const {}*", inner)
                }
                "MutPtr" => {
                    let inner = args
                        .first()
                        .map(|a| c_type_str(a, aliases, bitfields, None))
                        .unwrap_or_else(|| "void".into());
                    format!("{}*", inner)
                }
                other => other.to_string(),
            }
        }
        TypeExpr::Array { elem, .. } => {
            format!("{}*", c_type_str(&elem, aliases, bitfields, None))
        }
        _ => "int64_t".into(),
    }
}
