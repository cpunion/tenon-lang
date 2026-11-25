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
    int is_uniq = __atomic_load_n(&hdr->rc, __ATOMIC_ACQUIRE) == 1;
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
        if (init_val != 0) tenon_rc_inc(init_val);
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
    if (is_rc && val != 0) tenon_rc_inc(val);
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
    if (is_rc && val != 0) tenon_rc_inc(val);
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

/// Atomic Perceus RC inline LLVM IR helpers — ONLY emitted when the program uses threads/atomics!
pub(crate) const TENON_ATOMIC_RC_IR_HELPERS: &str = r#"
define internal i64 @__tn_rc_inc(i64 %p) alwaysinline {
entry:
  %nz = icmp ne i64 %p, 0
  br i1 %nz, label %chk, label %done
chk:
  %obj = inttoptr i64 %p to ptr
  %rc = load atomic i64, ptr %obj monotonic, align 8
  %pos = icmp sgt i64 %rc, 0
  br i1 %pos, label %inc, label %done
inc:
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
  %rc = load atomic i64, ptr %obj monotonic, align 8
  %pos = icmp sgt i64 %rc, 0
  br i1 %pos, label %dec, label %done
dec:
  %old = atomicrmw sub ptr %obj, i64 1 acq_rel, align 8
  %was1 = icmp sle i64 %old, 1
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
  %rc = load atomic i64, ptr %obj acquire, align 8
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
