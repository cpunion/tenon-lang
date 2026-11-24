use crate::lower_interfaces::{DynInterfaceLayout, prepare_program_for_codegen};
use crate::perceus::{
    RC_MASK_ALL, TAG_DYN, TAG_ERR, TAG_LIST, TAG_NONE, TAG_OK, TAG_RECORD_BASE, TAG_SOME,
};
use crate::{
    CORE_SYSCALLS, generic_ty, is_rc_type_expr, is_text_type_expr, named_ty, resolve_alias_ty,
    split_async_segments,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tenon_syntax::*;

static AOT_TMP_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct BuildArtifact {
    pub output_path: String,
    pub llvm_ir_bytes: usize,
    pub binary_bytes: u64,
    pub opt_level: String,
}

#[derive(Debug, Clone)]
struct LlvmFieldLayout {
    name: String,
    ty: TypeExpr,
    preds: Vec<Expr>,
    is_rc: bool,
}

#[derive(Debug, Clone)]
struct LlvmRecordLayout {
    tag: i64,
    fields: Vec<LlvmFieldLayout>,
    rc_mask: i64,
    where_clauses: Vec<Expr>,
}

#[derive(Debug, Clone)]
struct LlvmEnumVariantLayout {
    enum_name: String,
    payload_tys: Vec<TypeExpr>,
}

#[derive(Debug, Clone)]
struct LlvmFnMeta {
    params_inout: Vec<bool>,
    param_tys: Vec<TypeExpr>,
    ret_ty: Option<TypeExpr>,
    returns_rc: bool,
}

const TENON_AOT_RUNTIME_C: &str = r#"
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

int64_t tenon_rt_print_int(int64_t v) {
    printf("%lld\n", (long long)v);
    return v;
}

#define TENON_FREELIST_MAX_CAP 8
#define TENON_FREELIST_LIMIT 512

static TenonObj* g_freelist[TENON_FREELIST_MAX_CAP + 1] = {0};
static int g_freelist_len[TENON_FREELIST_MAX_CAP + 1] = {0};

static inline TenonObj* tenon_rc_alloc_raw(int64_t tag, int64_t lu, int64_t cap) {
    TenonObj* obj;
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
    if (__builtin_expect(obj->rc > 0, 1)) {
        obj->rc++;
    }
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
    g_live--;
    int64_t cap = obj->cap;
    if (__builtin_expect(cap >= 1 && cap <= TENON_FREELIST_MAX_CAP && g_freelist_len[cap] < TENON_FREELIST_LIMIT, 1)) {
        *(TenonObj**)obj = g_freelist[cap];
        g_freelist[cap] = obj;
        g_freelist_len[cap]++;
        return;
    }
    free(obj);
}

void tenon_rc_free_obj(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return;
    tenon_rc_free_slow((TenonObj*)(uintptr_t)ptr);
}

void tenon_rc_dec(int64_t ptr) {
    if (__builtin_expect(ptr == 0, 0)) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (__builtin_expect(obj->rc > 0, 1)) {
        if (__builtin_expect(--obj->rc <= 0, 0)) {
            tenon_rc_free_slow(obj);
        }
    }
}

__attribute__((noinline))
static int64_t tenon_rc_cow_clone_slow(TenonObj* obj) {
    g_cow++;
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
    if (__builtin_expect(obj->rc == 1, 1)) {
        return ptr;
    }
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
    int64_t words = (blen + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(5, blen, words);
    if (blen > 0 && data_ptr != 0) {
        memcpy((void*)obj->fields, (const void*)(uintptr_t)data_ptr, (size_t)blen);
    }
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
    int64_t words = (total + 7) / 8;
    if (words < 1) words = 1;
    TenonObj* obj = tenon_rc_alloc_raw(5, total, words);
    uint8_t* dst = (uint8_t*)obj->fields;
    if (la > 0) memcpy(dst, (const void*)((TenonObj*)(uintptr_t)a)->fields, (size_t)la);
    if (lb > 0) memcpy(dst + la, (const void*)((TenonObj*)(uintptr_t)b)->fields, (size_t)lb);
    return (int64_t)(uintptr_t)obj;
}

typedef struct {
    uint8_t* data;
    size_t len;
    size_t cap;
    uint8_t inline_buf[128];
} TenonBuf;

static TenonBuf* g_buf_cache = NULL;

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
    int shared = obj->rc > 1;
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
            if (iv->rc > 0) iv->rc += n;
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

int64_t tenon_list_pop_shrink(int64_t ptr) {
    if (ptr == 0) return 0;
    int64_t uniq = tenon_rc_ensure_unique(ptr);
    TenonObj* hdr = (TenonObj*)(uintptr_t)uniq;
    if (hdr->len > 0) {
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

const TENON_AOT_MAIN_STATS_C: &str = r#"
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

const TENON_AOT_MAIN_CLI_C: &str = r#"
int main(void) {
    return (int)tenon_fn_main();
}
"#;

const TENON_AOT_OS_C: &str = r#"
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

const TENON_AOT_HTTP_C: &str = r#"
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

fn ir_uses_os_syscalls(ir: &str) -> bool {
    ir.contains("call i64 @tenon_sys_")
}

fn ir_uses_http_syscalls(ir: &str) -> bool {
    ir.contains("call i64 @tenon_sys_http_")
}

pub fn compile_to_native_binary(
    program: &Program,
    output_path: &Path,
    emit_ll_path: Option<&Path>,
    opt_level: &str,
) -> Result<BuildArtifact, String> {
    let ir = emit_llvm_ir(program)?;
    let pid = std::process::id();
    let seq = AOT_TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let tmp_ll = std::env::temp_dir().join(format!("tenon_aot_{}_{}.ll", pid, seq));
    let tmp_rt = std::env::temp_dir().join(format!("tenon_rt_{}_{}.c", pid, seq));

    fs::write(&tmp_ll, &ir).map_err(|e| e.to_string())?;
    let uses_os = ir_uses_os_syscalls(&ir);
    let uses_http = ir_uses_http_syscalls(&ir);
    let mut rt_c = String::with_capacity(TENON_AOT_RUNTIME_C.len() + 8192);
    rt_c.push_str(TENON_AOT_RUNTIME_C);
    if uses_os {
        rt_c.push_str(TENON_AOT_OS_C);
    }
    if uses_http {
        rt_c.push_str(TENON_AOT_HTTP_C);
    }
    if uses_os {
        rt_c.push_str(TENON_AOT_MAIN_CLI_C);
    } else {
        rt_c.push_str(TENON_AOT_MAIN_STATS_C);
    }
    fs::write(&tmp_rt, &rt_c).map_err(|e| e.to_string())?;
    if let Some(p) = emit_ll_path {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(p, &ir).map_err(|e| e.to_string())?;
    }
    if let Some(parent) = output_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let opt_flag = match opt_level {
        "O0" | "-O0" => "-O0",
        "O2" | "-O2" => "-O2",
        _ => "-O3",
    };

    let mut cmd = Command::new("clang");
    cmd.arg(opt_flag);
    if opt_flag == "-O3" {
        cmd.arg("-flto=thin");
    }
    cmd.args([
        "-Wno-override-module",
        tmp_ll.to_str().unwrap(),
        tmp_rt.to_str().unwrap(),
    ]);
    if uses_http {
        cmd.args(["-lcurl", "-lpthread"]);
    }
    cmd.args(["-o", output_path.to_str().unwrap()]);

    let out = cmd
        .output()
        .map_err(|e| format!("failed to invoke LLVM clang: {}", e))?;

    let _ = fs::remove_file(&tmp_ll);
    let _ = fs::remove_file(&tmp_rt);

    if !out.status.success() {
        return Err(format!(
            "LLVM clang compilation failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let binary_bytes = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    Ok(BuildArtifact {
        output_path: output_path.display().to_string(),
        llvm_ir_bytes: ir.len(),
        binary_bytes,
        opt_level: opt_flag.to_string(),
    })
}

const TENON_INLINE_IR_HELPERS: &str = r#"
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

const TENON_TBAA_METADATA: &str = r#"
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

pub fn emit_llvm_ir(program: &Program) -> Result<String, String> {
    let prepared = prepare_program_for_codegen(program);
    let program = &prepared.program;
    let vtable_ifaces: BTreeMap<String, String> = prepared
        .vtables
        .iter()
        .map(|v| (v.sym_name.clone(), v.interface.clone()))
        .collect();
    let dyn_interfaces = &prepared.dyn_interfaces;
    let mut alias_preds: BTreeMap<String, Expr> = BTreeMap::new();
    let mut alias_bases: BTreeMap<String, TypeExpr> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            alias_bases.insert(ta.name.clone(), ta.base.clone());
            if let Some(p) = &ta.predicate {
                alias_preds.insert(ta.name.clone(), p.clone());
            }
        }
    }

    let record_names: BTreeSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Record(r) if r.type_params.is_empty() => Some(r.name.clone()),
            _ => None,
        })
        .collect();
    let enum_names: BTreeSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect();

    let mut records: BTreeMap<String, LlvmRecordLayout> = BTreeMap::new();
    let mut field_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut enum_tags: BTreeMap<String, i64> = BTreeMap::new();
    let mut enum_variants: BTreeMap<String, LlvmEnumVariantLayout> = BTreeMap::new();
    enum_tags.insert("Err".into(), TAG_ERR);
    enum_tags.insert("Ok".into(), TAG_OK);
    enum_tags.insert("None".into(), TAG_NONE);
    enum_tags.insert("Some".into(), TAG_SOME);

    let mut next_rec_tag = TAG_RECORD_BASE;
    let mut next_enum_tag = 10i64;
    for item in &program.items {
        match item {
            Item::Record(r) if r.type_params.is_empty() => {
                let mut fields = Vec::new();
                let mut rc_mask = 0u64;
                for (idx, f) in r.fields.iter().enumerate() {
                    let tname = f.ty.display_name();
                    let mut preds = Vec::new();
                    if let Some(ap) = alias_preds.get(&tname) {
                        preds.push(ap.clone());
                    }
                    if let Some(fp) = &f.predicate {
                        preds.push(fp.clone());
                    }
                    let is_rc =
                        is_rc_type_expr(&f.ty, &alias_bases, &record_names, &enum_names);
                    if is_rc && idx < 64 {
                        rc_mask |= 1u64 << idx;
                    }
                    fields.push(LlvmFieldLayout {
                        name: f.name.clone(),
                        ty: f.ty.clone(),
                        preds,
                        is_rc,
                    });
                    field_to_idx.entry(f.name.clone()).or_insert(idx);
                }
                records.insert(
                    r.name.clone(),
                    LlvmRecordLayout {
                        tag: next_rec_tag,
                        fields,
                        rc_mask: rc_mask as i64,
                        where_clauses: r.where_clauses.clone(),
                    },
                );
                next_rec_tag += 1;
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    let payload_tys: Vec<TypeExpr> =
                        v.fields.iter().map(|f| f.ty.clone()).collect();
                    let vl = LlvmEnumVariantLayout {
                        enum_name: e.name.clone(),
                        payload_tys,
                    };
                    enum_tags.insert(v.name.clone(), next_enum_tag);
                    enum_tags.insert(format!("{}.{}", e.name, v.name), next_enum_tag);
                    enum_variants.insert(v.name.clone(), vl.clone());
                    enum_variants.insert(format!("{}.{}", e.name, v.name), vl);
                    next_enum_tag += 1;
                }
            }
            _ => {}
        }
    }

    let sync_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();
    let async_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();

    let mut fn_metas: BTreeMap<String, LlvmFnMeta> = BTreeMap::new();
    let mut syscall_map: BTreeMap<String, &'static str> = BTreeMap::new();
    for spec in CORE_SYSCALLS {
        syscall_map.insert(spec.tenon_name.to_string(), spec.c_symbol);
        fn_metas.insert(
            spec.tenon_name.to_string(),
            LlvmFnMeta {
                params_inout: vec![false; spec.arity],
                param_tys: vec![named_ty("Any"); spec.arity],
                ret_ty: Some(if spec.returns_text {
                    named_ty("Text")
                } else {
                    named_ty("Int")
                }),
                returns_rc: spec.returns_rc,
            },
        );
    }
    for f in &sync_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: f.params.iter().map(|p| p.is_inout).collect(),
                param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret_ty: f.ret_ty.clone(),
                returns_rc: f.ret_ty.as_ref().is_some_and(|t| {
                    is_rc_type_expr(t, &alias_bases, &record_names, &enum_names)
                }),
            },
        );
    }
    for f in &async_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: vec![false; f.params.len()],
                param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret_ty: f.ret_ty.clone(),
                returns_rc: true,
            },
        );
    }
    for (rname, layout) in &records {
        if !layout.fields.is_empty() {
            fn_metas.insert(
                format!("{}__parse", rname),
                LlvmFnMeta {
                    params_inout: vec![false; layout.fields.len()],
                    param_tys: layout.fields.iter().map(|f| f.ty.clone()).collect(),
                    ret_ty: Some(generic_ty("Result", vec![named_ty(rname), named_ty("Int")])),
                    returns_rc: true,
                },
            );
        }
    }

    let mut string_globals: Vec<(String, String)> = Vec::new();
    let mut string_intern: BTreeMap<String, String> = BTreeMap::new();
    let mut unit_enum_tags: BTreeSet<i64> = BTreeSet::new();
    let mut fn_defs = String::new();

    for f in &sync_fns {
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            dyn_interfaces,
            &vtable_ifaces,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
        );
        fn_defs.push_str(&cg.emit_sync_fn(f)?);
        fn_defs.push('\n');
    }

    for (rname, layout) in &records {
        if layout.fields.is_empty() {
            continue;
        }
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            dyn_interfaces,
            &vtable_ifaces,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
        );
        fn_defs.push_str(&cg.emit_record_parser(rname, layout)?);
        fn_defs.push('\n');
    }

    for f in &async_fns {
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            dyn_interfaces,
            &vtable_ifaces,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
        );
        fn_defs.push_str(&cg.emit_async_fn(f)?);
        fn_defs.push('\n');
    }

    let mut ir = String::new();
    ir.push_str("; Generated by Tenon LLVM 22 AOT Release Backend\n\n");
    ir.push_str("declare void @llvm.assume(i1)\n");
    ir.push_str("declare i32 @memcmp(ptr, ptr, i64)\n");
    ir.push_str("declare void @tenon_rt_assert_fail(i64) cold\n");
    ir.push_str("declare i64 @tenon_rt_print_int(i64)\n");
    ir.push_str("declare i64 @tenon_rc_alloc(i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_rc_mask(i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_inc(i64)\n");
    ir.push_str("declare void @tenon_rc_dec(i64)\n");
    ir.push_str("declare void @tenon_rc_free_obj(i64) cold\n");
    ir.push_str("declare i64 @tenon_rc_ensure_unique(i64) cold\n");
    ir.push_str("declare i64 @tenon_rc_get_field(i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_field(i64, i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_field_raw(i64, i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_tag(i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_len(i64)\n");
    ir.push_str("declare i64 @tenon_text_new(i64, i64)\n");
    ir.push_str("declare i64 @tenon_text_eq(i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_push(i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_filled(i64, i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_pop_shrink(i64)\n");
    ir.push_str("declare i64 @tenon_async_frame_alloc(i64, i64)\n");
    ir.push_str("declare i64 @tenon_async_yield_create(i64)\n");
    ir.push_str("declare i64 @tenon_async_poll_frame(i64)\n");
    ir.push_str("declare i64 @tenon_async_block_on(i64)\n");
    for spec in CORE_SYSCALLS {
        let params = vec!["i64"; spec.arity].join(", ");
        ir.push_str(&format!("declare i64 @{}({})\n", spec.c_symbol, params));
    }
    ir.push_str(TENON_INLINE_IR_HELPERS);
    ir.push('\n');

    for (gname, text) in &string_globals {
        let blen = text.len();
        let words = ((blen + 7) / 8).max(1);
        if blen == 0 {
            ir.push_str(&format!(
                "@{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, [1 x i8] }} {{ i64 -1, i64 5, i64 0, i64 1, i64 0, [1 x i8] zeroinitializer }}, align 8\n",
                gname
            ));
        } else {
            let escaped = escape_llvm_bytes(text.as_bytes());
            ir.push_str(&format!(
                "@{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, [{} x i8] }} {{ i64 -1, i64 5, i64 {}, i64 {}, i64 0, [{} x i8] c\"{}\" }}, align 8\n",
                gname, blen, blen, words, blen, escaped
            ));
        }
    }
    if !string_globals.is_empty() {
        ir.push('\n');
    }

    for &tag in &unit_enum_tags {
        ir.push_str(&format!(
            "@.unit_enum.{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, i64 }} {{ i64 -1, i64 {}, i64 0, i64 1, i64 0, i64 0 }}, align 8\n",
            tag, tag
        ));
    }
    if !unit_enum_tags.is_empty() {
        ir.push('\n');
    }

    for vt in &prepared.vtables {
        let entries: Vec<String> = vt
            .fn_names
            .iter()
            .map(|fn_name| format!("ptr @tenon_fn_{}", sanitize(fn_name)))
            .collect();
        ir.push_str(&format!(
            "@{} = private constant [{} x ptr] [{}]\n",
            sanitize(&vt.sym_name),
            entries.len(),
            entries.join(", ")
        ));
    }
    if !prepared.vtables.is_empty() {
        ir.push('\n');
    }

    ir.push_str(&fn_defs);
    ir.push_str(TENON_TBAA_METADATA);
    Ok(ir)
}

fn escape_llvm_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        if (0x20..=0x7E).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\{:02X}", b));
        }
    }
    out
}

struct LlvmLoopCtx {
    hdr_lbl: String,
    exit_lbl: String,
    rc_scope_depth: usize,
    temp_scope_depth: usize,
}

struct LlvmSavedLexical {
    var_ptrs: BTreeMap<String, String>,
    var_types: BTreeMap<String, TypeExpr>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
}

struct LlvmFnGen<'a> {
    alias_bases: &'a BTreeMap<String, TypeExpr>,
    records: &'a BTreeMap<String, LlvmRecordLayout>,
    record_names: &'a BTreeSet<String>,
    enum_names: &'a BTreeSet<String>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    enum_variants: &'a BTreeMap<String, LlvmEnumVariantLayout>,
    fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
    syscall_map: &'a BTreeMap<String, &'static str>,
    dyn_interfaces: &'a BTreeMap<String, DynInterfaceLayout>,
    vtable_ifaces: &'a BTreeMap<String, String>,
    string_globals: &'a mut Vec<(String, String)>,
    string_intern: &'a mut BTreeMap<String, String>,
    unit_enum_tags: &'a mut BTreeSet<i64>,
    tmp: usize,
    lbl: usize,
    slot_id: usize,
    entry_allocas: String,
    code: String,
    var_ptrs: BTreeMap<String, String>,
    var_types: BTreeMap<String, TypeExpr>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
    always_unique_vars: BTreeSet<String>,
    rc_scopes: Vec<Vec<(String, String)>>,
    temp_rc_scopes: Vec<Vec<String>>,
    loop_stack: Vec<LlvmLoopCtx>,
    fn_returns_rc: bool,
    fn_ret_ty: Option<TypeExpr>,
    self_var: Option<String>,
    terminated: bool,
}

impl<'a> LlvmFnGen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        alias_bases: &'a BTreeMap<String, TypeExpr>,
        records: &'a BTreeMap<String, LlvmRecordLayout>,
        record_names: &'a BTreeSet<String>,
        enum_names: &'a BTreeSet<String>,
        field_to_idx: &'a BTreeMap<String, usize>,
        enum_tags: &'a BTreeMap<String, i64>,
        enum_variants: &'a BTreeMap<String, LlvmEnumVariantLayout>,
        fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
        syscall_map: &'a BTreeMap<String, &'static str>,
        dyn_interfaces: &'a BTreeMap<String, DynInterfaceLayout>,
        vtable_ifaces: &'a BTreeMap<String, String>,
        string_globals: &'a mut Vec<(String, String)>,
        string_intern: &'a mut BTreeMap<String, String>,
        unit_enum_tags: &'a mut BTreeSet<i64>,
    ) -> Self {
        Self {
            alias_bases,
            records,
            record_names,
            enum_names,
            field_to_idx,
            enum_tags,
            enum_variants,
            fn_metas,
            syscall_map,
            dyn_interfaces,
            vtable_ifaces,
            string_globals,
            string_intern,
            unit_enum_tags,
            tmp: 0,
            lbl: 0,
            slot_id: 0,
            entry_allocas: String::new(),
            code: String::new(),
            var_ptrs: BTreeMap::new(),
            var_types: BTreeMap::new(),
            inout_vars: BTreeSet::new(),
            owned_rc_vars: BTreeSet::new(),
            borrowed_rc_vars: BTreeSet::new(),
            always_unique_vars: BTreeSet::new(),
            rc_scopes: Vec::new(),
            temp_rc_scopes: Vec::new(),
            loop_stack: Vec::new(),
            fn_returns_rc: false,
            fn_ret_ty: None,
            self_var: None,
            terminated: false,
        }
    }

    fn save_lexical(&self) -> LlvmSavedLexical {
        LlvmSavedLexical {
            var_ptrs: self.var_ptrs.clone(),
            var_types: self.var_types.clone(),
            owned_rc_vars: self.owned_rc_vars.clone(),
            borrowed_rc_vars: self.borrowed_rc_vars.clone(),
        }
    }

    fn restore_lexical(&mut self, saved: LlvmSavedLexical) {
        self.var_ptrs = saved.var_ptrs;
        self.var_types = saved.var_types;
        self.owned_rc_vars = saved.owned_rc_vars;
        self.borrowed_rc_vars = saved.borrowed_rc_vars;
    }

    fn fresh_tmp(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }

    fn fresh_lbl(&mut self, prefix: &str) -> String {
        self.lbl += 1;
        format!("{}_{}", prefix, self.lbl)
    }

    fn fresh_var_slot(&mut self, name: &str) -> String {
        self.slot_id += 1;
        let slot = format!("%v_{}_{}", sanitize(name), self.slot_id);
        self.entry_allocas
            .push_str(&format!("  {} = alloca i64\n  store i64 0, ptr {}\n", slot, slot));
        self.var_ptrs.insert(name.to_string(), slot.clone());
        slot
    }

    fn ensure_var_slot(&mut self, name: &str) -> String {
        if let Some(p) = self.var_ptrs.get(name) {
            return p.clone();
        }
        self.fresh_var_slot(name)
    }

    fn intern_string_global(&mut self, s: &str) -> String {
        if let Some(g) = self.string_intern.get(s) {
            return g.clone();
        }
        let gname = format!(".str.{}", self.string_globals.len());
        self.string_globals.push((gname.clone(), s.to_string()));
        self.string_intern.insert(s.to_string(), gname.clone());
        gname
    }

    fn emit_unit_enum_const(&mut self, tag: i64) -> String {
        self.unit_enum_tags.insert(tag);
        let o = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = ptrtoint ptr @.unit_enum.{} to i64\n",
            o, tag
        ));
        o
    }

    fn flush_temps(&mut self, temps: &[String]) {
        for v in temps.iter().rev() {
            self.code
                .push_str(&format!("  call void @__tn_rc_dec(i64 {})\n", v));
        }
    }

    fn emit_cleanup_scopes_from(&mut self, from_depth: usize, skip_slot: Option<&str>) {
        let scopes: Vec<Vec<(String, String)>> = self.rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for (_, slot) in scope.iter().rev() {
                if skip_slot == Some(slot.as_str()) {
                    continue;
                }
                let h = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                    h, slot, h
                ));
            }
        }
    }

    fn emit_cleanup_temp_scopes_from(&mut self, from_depth: usize) {
        let scopes: Vec<Vec<String>> = self.temp_rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for val in scope.iter().rev() {
                self.code
                    .push_str(&format!("  call void @__tn_rc_dec(i64 {})\n", val));
            }
        }
    }

    fn find_owned_slot_in_scopes(&self, slot: &str, from_depth: usize) -> bool {
        self.rc_scopes[from_depth..]
            .iter()
            .any(|sc| sc.iter().any(|(_, s)| s == slot))
    }

    fn type_is_rc(&self, ty: &TypeExpr) -> bool {
        is_rc_type_expr(ty, self.alias_bases, self.record_names, self.enum_names)
    }

    fn list_elem_ty(&self, ty: &TypeExpr) -> Option<TypeExpr> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        if let TypeExpr::Named { name, args, .. } = resolved {
            if name == "List" && !args.is_empty() {
                return Some(args[0].clone());
            }
        }
        None
    }

    fn infer_expr_type(&self, expr: &Expr) -> TypeExpr {
        self.infer_expr_type_with(expr, &BTreeMap::new())
    }

    fn infer_expr_type_with(&self, expr: &Expr, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(ty) = extra_vars.get(s) {
                    return ty.clone();
                }
                if let Some(ty) = self.var_types.get(s) {
                    return ty.clone();
                }
                if let Some(vl) = self.enum_variants.get(s) {
                    return named_ty(&vl.enum_name);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, .. } => named_ty(name),
            ExprKind::List(elems) => {
                if let Some(first) = elems.first() {
                    generic_ty("List", vec![self.infer_expr_type_with(first, extra_vars)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::Binary { op: BinOp::Concat, .. } => named_ty("Text"),
            ExprKind::Binary {
                op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or,
                ..
            } => named_ty("Bool"),
            ExprKind::Binary { .. } => named_ty("Int"),
            ExprKind::Unary { op: UnOp::Not, .. } => named_ty("Bool"),
            ExprKind::Unary { op: UnOp::Neg, .. } => named_ty("Int"),
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        return named_ty(&vl.enum_name);
                    }
                }
                let obj_ty = resolve_alias_ty(&self.infer_expr_type_with(target, extra_vars), self.alias_bases);
                if let TypeExpr::Named { name: rname, .. } = obj_ty {
                    if let Some(layout) = self.records.get(&rname) {
                        if let Some(fl) = layout.fields.iter().find(|f| f.name == *field) {
                            return fl.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type_with(target, extra_vars);
                self.list_elem_ty(&tt).unwrap_or_else(|| named_ty("Any"))
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return named_ty(&vl.enum_name);
                        }
                    }
                    let recv_ty = resolve_alias_ty(
                        &self.infer_expr_type_with(target, extra_vars),
                        self.alias_bases,
                    );
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Ident(f) => {
                    if f == "__box_dyn" {
                        if let Some(a1) = args.get(1) {
                            if let ExprKind::Str(vsym) = &a1.value.kind {
                                if let Some(iface) = self.vtable_ifaces.get(vsym) {
                                    return TypeExpr::Dyn {
                                        interface: iface.clone(),
                                        args: vec![],
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if f == "Ok" && args.len() == 1 {
                        return generic_ty("Result", vec![self.infer_expr_type_with(&args[0].value, extra_vars), named_ty("Any")]);
                    }
                    if f == "Err" && args.len() == 1 {
                        return generic_ty("Result", vec![named_ty("Any"), self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "Some" && args.len() == 1 {
                        return generic_ty("Option", vec![self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "push" && args.len() == 2 {
                        let lt = self.infer_expr_type_with(&args[0].value, extra_vars);
                        if self.list_elem_ty(&lt).is_some_and(|e| e.display_name() != "Any") {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_filled" && args.len() == 2 {
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_pop_shrink" && args.len() == 1 {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if let Some(meta) = self.fn_metas.get(f) {
                        if let Some(rt) = &meta.ret_ty {
                            return rt.clone();
                        }
                    }
                    if let Some(vl) = self.enum_variants.get(f) {
                        return named_ty(&vl.enum_name);
                    }
                    named_ty("Any")
                }
                _ => named_ty("Any"),
            },
            ExprKind::If { then_block, else_branch, .. } => {
                let then_ty = self.infer_block_type_with(then_block, extra_vars);
                if then_ty.display_name() != "Any" {
                    return then_ty;
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_type_with(eb, extra_vars);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_type_with(scrutinee, extra_vars);
                for arm in arms {
                    let mut arm_vars = extra_vars.clone();
                    for p in &arm.patterns {
                        if let Pattern::Variant { name, binding: Some(bname), .. } = p {
                            if bname != "_" {
                                arm_vars.insert(
                                    bname.clone(),
                                    self.infer_match_binding_type(&st, name),
                                );
                            }
                        }
                    }
                    let t = self.infer_expr_type_with(&arm.body, &arm_vars);
                    if t.display_name() != "Any" {
                        return t;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(blk) => self.infer_block_type_with(blk, extra_vars),
            _ => named_ty("Any"),
        }
    }

    fn infer_block_type_with(&self, blk: &Block, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        let mut local = extra_vars.clone();
        for s in &blk.stmts {
            if let Stmt::Let { name, ty, value, .. } = s {
                let lt = ty.clone().unwrap_or_else(|| self.infer_expr_type_with(value, &local));
                local.insert(name.clone(), lt);
            }
        }
        blk.tail
            .as_deref()
            .map(|t| self.infer_expr_type_with(t, &local))
            .unwrap_or_else(|| named_ty("Any"))
    }

    fn resolve_field_layout(
        &self,
        obj_expr: &Expr,
        field: &str,
    ) -> Result<(usize, Option<LlvmFieldLayout>), String> {
        let obj_ty = resolve_alias_ty(&self.infer_expr_type(obj_expr), self.alias_bases);
        if let TypeExpr::Named { name: rname, .. } = obj_ty {
            if let Some(layout) = self.records.get(&rname) {
                if let Some((pos, fl)) = layout
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == field)
                {
                    return Ok((pos, Some(fl.clone())));
                }
            }
        }
        let idx = self
            .field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown field `{}` in LLVM codegen", field))?;
        Ok((idx, None))
    }

    fn is_rc_ident(&self, s: &str) -> bool {
        self.owned_rc_vars.contains(s)
            || self.borrowed_rc_vars.contains(s)
            || self.var_types.get(s).is_some_and(|t| self.type_is_rc(t))
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        if self.type_is_rc(&self.infer_expr_type(expr)) {
            return true;
        }
        match &expr.kind {
            ExprKind::RecordLit { .. } | ExprKind::List(_) | ExprKind::Str(_) => true,
            ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::Ident(s) => {
                s == "None"
                    || self.is_rc_ident(s)
                    || (self.enum_tags.contains_key(s) && !self.var_ptrs.contains_key(s))
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_tags.contains_key(&format!("{}.{}", ename, field)) {
                        return true;
                    }
                }
                if let Ok((_, Some(fl))) = self.resolve_field_layout(target, field) {
                    return fl.is_rc;
                }
                false
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type(target);
                self.list_elem_ty(&tt).is_some_and(|et| self.type_is_rc(&et))
            }
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => {
                    matches!(
                        f.as_str(),
                        "Ok" | "Err" | "Some" | "push" | "list_filled" | "list_pop_shrink" | "yield_val" | "__box_dyn"
                    )
                        || (self.enum_tags.contains_key(f) && !self.fn_metas.contains_key(f))
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { target, field } => {
                    if field == "parse"
                        || matches!(&target.kind, ExprKind::Ident(e) if self.enum_tags.contains_key(&format!("{}.{}", e, field)))
                    {
                        return true;
                    }
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.returns_rc;
                            }
                        }
                    }
                    false
                }
                _ => false,
            },
            ExprKind::If { then_block, else_branch, .. } => {
                else_branch.is_some()
                    && (then_block.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t))
                        || else_branch.as_deref().is_some_and(|e| self.expr_produces_rc(e)))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.expr_produces_rc(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t)),
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        let ty = self.infer_expr_type(expr);
        if is_text_type_expr(&ty, self.alias_bases) {
            return true;
        }
        match &expr.kind {
            ExprKind::Str(_) | ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_text_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_text_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_text_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_text_expr(t)),
            _ => false,
        }
    }

    fn infer_match_binding_type(&self, scrutinee_ty: &TypeExpr, variant_name: &str) -> TypeExpr {
        let base = resolve_alias_ty(scrutinee_ty, self.alias_bases);
        if variant_name == "Some" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Option" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Ok" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Err" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && args.len() >= 2 {
                    return args[1].clone();
                }
            }
        }
        if let Some(vl) = self.enum_variants.get(variant_name) {
            if let Some(pty) = vl.payload_tys.first() {
                return pty.clone();
            }
        }
        named_ty("Any")
    }

    fn emit_sync_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.fresh_var_slot(&p.name);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
            self.var_types.insert(p.name.clone(), p.ty.clone());
            if self.type_is_rc(&p.ty) {
                self.borrowed_rc_vars.insert(p.name.clone());
            }
            if p.is_inout {
                self.inout_vars.insert(p.name.clone());
            }
        }

        self.fn_returns_rc = self.fn_metas.get(&f.name).is_some_and(|m| m.returns_rc);
        self.fn_ret_ty = f.ret_ty.clone();
        self.always_unique_vars = collect_always_unique_locals(&f.body);

        let ret_val = self.lower_block(&f.body, self.fn_returns_rc, f.ret_ty.as_ref())?;
        if !self.terminated {
            self.code.push_str(&format!("  ret i64 {}\n", ret_val));
        }

        let linkage = if f.name == "main" { "" } else { "internal " };
        Ok(format!(
            "define {}i64 @tenon_fn_{}({}) {{\nentry:\n{}{}}}\n",
            linkage,
            sanitize(&f.name),
            param_list.join(", "),
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_record_parser(
        &mut self,
        rname: &str,
        layout: &LlvmRecordLayout,
    ) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, fl) in layout.fields.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.fresh_var_slot(&fl.name);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
            self.var_types.insert(fl.name.clone(), fl.ty.clone());
            if fl.is_rc {
                self.borrowed_rc_vars.insert(fl.name.clone());
            }
        }

        for (idx, fl) in layout.fields.iter().enumerate() {
            for pred in &fl.preds {
                self.self_var = Some(fl.name.clone());
                let mut temps = Vec::new();
                let cv = self.lower_expr_borrowed(pred, &mut temps)?;
                self.flush_temps(&temps);
                self.self_var = None;
                let cond = self.fresh_tmp();
                let fail_l = self.fresh_lbl("pfail");
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                    cond, cv, cond, ok_l, fail_l
                ));
                self.code.push_str(&format!("{}:\n", fail_l));
                let eo = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
                    eo, TAG_ERR, eo, idx + 1, eo
                ));
                self.code.push_str(&format!("{}:\n", ok_l));
            }
        }

        for wc in &layout.where_clauses {
            let mut temps = Vec::new();
            let cv = self.lower_expr_borrowed(wc, &mut temps)?;
            self.flush_temps(&temps);
            let cond = self.fresh_tmp();
            let fail_l = self.fresh_lbl("wfail");
            let ok_l = self.fresh_lbl("wok");
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cond, cv, cond, ok_l, fail_l
            ));
            self.code.push_str(&format!("{}:\n", fail_l));
            let eo = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 99)\n  ret i64 {}\n",
                eo, TAG_ERR, eo, eo
            ));
            self.code.push_str(&format!("{}:\n", ok_l));
        }

        let rec = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
            rec,
            layout.tag,
            layout.fields.len()
        ));
        if layout.rc_mask != 0 {
            self.code.push_str(&format!(
                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                rec, layout.rc_mask
            ));
        }
        for (idx, fl) in layout.fields.iter().enumerate() {
            let slot = self.var_ptrs[&fl.name].clone();
            let mut fv = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = load i64, ptr {}\n", fv, slot));
            if fl.is_rc {
                let inc = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = call i64 @__tn_rc_inc(i64 {})\n", inc, fv));
                fv = inc;
            }
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                rec, idx, fv
            ));
        }
        let ok_obj = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_rc_set_rc_mask(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
            ok_obj, TAG_OK, ok_obj, ok_obj, rec, ok_obj
        ));

        Ok(format!(
            "define internal i64 @tenon_fn_{}__parse({}) {{\nentry:\n{}{}}}\n",
            sanitize(rname),
            param_list.join(", "),
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_async_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut frame_slots: BTreeMap<String, i64> = BTreeMap::new();
        let mut next_slot = 4i64;
        for p in &f.params {
            frame_slots.insert(p.name.clone(), next_slot);
            next_slot += 1;
        }
        for s in &f.body.stmts {
            if let Stmt::Let { name, .. } = s {
                if !frame_slots.contains_key(name) {
                    frame_slots.insert(name.clone(), next_slot);
                    next_slot += 1;
                }
            }
        }
        let segments = split_async_segments(&f.body, &mut frame_slots, &mut next_slot);
        let total_slots = next_slot;
        let sname = sanitize(&f.name);

        let mut ctor_params = Vec::new();
        let mut ctor_body = String::new();
        ctor_body.push_str(&format!(
            "  %pp = ptrtoint ptr @tenon_fn_{}__poll to i64\n  %fr = call i64 @tenon_async_frame_alloc(i64 {}, i64 %pp)\n",
            sname, total_slots
        ));
        for (i, p) in f.params.iter().enumerate() {
            ctor_params.push(format!("i64 %arg_{}", i));
            ctor_body.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %fr, i64 {}, i64 %arg_{})\n",
                frame_slots[&p.name], i
            ));
        }
        ctor_body.push_str("  ret i64 %fr\n");
        let ctor_ir = format!(
            "define internal i64 @tenon_fn_{}({}) {{\nentry:\n{}}}\n\n",
            sname,
            ctor_params.join(", "),
            ctor_body
        );

        for (vname, &slot) in &frame_slots {
            let vp = self.ensure_var_slot(vname);
            let lv = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 %frame, i64 {})\n  store i64 {}, ptr {}\n",
                lv, slot, lv, vp
            ));
        }
        let cur_st = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = call i64 @__tn_slot_get(i64 %frame, i64 0)\n", cur_st));

        let n_awaits = segments.await_steps.len();
        let mut start_lbls = Vec::new();
        let mut poll_lbls = Vec::new();
        for s in 0..n_awaits {
            start_lbls.push(format!("st_start_{}", s));
            poll_lbls.push(format!("st_poll_{}", s));
        }
        let final_lbl = "st_final".to_string();

        self.code.push_str(&format!("  switch i64 {}, label %{} [\n", cur_st, final_lbl));
        for s in 0..n_awaits {
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s, start_lbls[s]));
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s + 1, poll_lbls[s]));
        }
        self.code.push_str("  ]\n");

        for (s, step) in segments.await_steps.iter().enumerate() {
            self.code.push_str(&format!("{}:\n", start_lbls[s]));
            self.terminated = false;
            let pre_blk = Block {
                stmts: step.pre_stmts.clone(),
                tail: None,
                span: f.span,
            };
            let _ = self.lower_block(&pre_blk, false, None)?;
            for (vname, &slot) in &frame_slots {
                let vp = self.var_ptrs[vname].clone();
                let lv = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 %frame, i64 {}, i64 {})\n",
                    lv, vp, slot, lv
                ));
            }
            let cframe = self.lower_expr_owned(&step.child_expr, None)?;
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %frame, i64 3, i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                cframe,
                2 * s + 1,
                poll_lbls[s]
            ));

            self.code.push_str(&format!("{}:\n", poll_lbls[s]));
            self.terminated = false;
            let cf = self.fresh_tmp();
            let rdy = self.fresh_tmp();
            let cmp = self.fresh_tmp();
            let pend_l = self.fresh_lbl("pend");
            let res_l = self.fresh_lbl("res");
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 %frame, i64 3)\n  {} = call i64 @tenon_async_poll_frame(i64 {})\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cf, rdy, cf, cmp, rdy, cmp, res_l, pend_l
            ));
            self.code.push_str(&format!("{}:\n  ret i64 0\n", pend_l));
            self.code.push_str(&format!("{}:\n", res_l));
            let cres = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 {}, i64 2)\n  call void @__tn_rc_dec(i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 3, i64 0)\n",
                cres, cf, cf
            ));
            if let Some(bname) = &step.bind_var {
                let slot = frame_slots[bname];
                let vp = self.var_ptrs[bname].clone();
                self.code.push_str(&format!(
                    "  call void @__tn_slot_set_prim(i64 %frame, i64 {}, i64 {})\n  store i64 {}, ptr {}\n",
                    slot, cres, cres, vp
                ));
            }
            let next_lbl = if s + 1 < n_awaits {
                start_lbls[s + 1].as_str()
            } else {
                final_lbl.as_str()
            };
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                2 * (s + 1),
                next_lbl
            ));
        }

        self.code.push_str(&format!("{}:\n", final_lbl));
        self.terminated = false;
        let final_blk = Block {
            stmts: segments.final_stmts.clone(),
            tail: segments.final_tail.clone().map(Box::new),
            span: f.span,
        };
        let fval = self.lower_block(&final_blk, true, None)?;
        self.code.push_str(&format!(
            "  call void @__tn_slot_set_prim(i64 %frame, i64 2, i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 -1)\n  ret i64 1\n",
            fval
        ));

        let poll_ir = format!(
            "define internal i64 @tenon_fn_{}__poll(i64 %frame) {{\nentry:\n{}{}}}\n",
            sname, self.entry_allocas, self.code
        );
        Ok(format!("{}{}", ctor_ir, poll_ir))
    }

    fn load_ident_val(&mut self, s: &str) -> Result<String, String> {
        let slot = self
            .var_ptrs
            .get(s)
            .cloned()
            .ok_or_else(|| format!("undefined variable `{}` in LLVM codegen", s))?;
        if self.inout_vars.contains(s) {
            let ai = self.fresh_tmp();
            let ap = self.fresh_tmp();
            let v = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                ai, slot, ap, ai, v, ap
            ));
            Ok(v)
        } else {
            let v = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
            Ok(v)
        }
    }

    fn lower_block(
        &mut self,
        blk: &Block,
        want_owned_tail: bool,
        expected_tail_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let saved = self.save_lexical();
        let scope_idx = self.rc_scopes.len();
        self.rc_scopes.push(Vec::new());

        let mut last = "0".to_string();
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, ..
                } => {
                    let effective_ty = ty.clone().unwrap_or_else(|| self.infer_expr_type(value));
                    let produces_rc = self.type_is_rc(&effective_ty) || self.expr_produces_rc(value);
                    let v = if produces_rc {
                        self.lower_expr_owned(value, ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let val = self.lower_expr_borrowed(value, &mut temps)?;
                        self.flush_temps(&temps);
                        val
                    };
                    let slot = self.fresh_var_slot(name);
                    self.code
                        .push_str(&format!("  store i64 {}, ptr {}\n", v, slot));

                    self.var_types.insert(name.clone(), effective_ty);
                    self.owned_rc_vars.remove(name);
                    self.borrowed_rc_vars.remove(name);
                    if produces_rc {
                        self.owned_rc_vars.insert(name.clone());
                        self.rc_scopes[scope_idx].push((name.clone(), slot));
                    }
                }
                Stmt::Assign {
                    target, op, value, ..
                } => match &target.kind {
                    ExprKind::Ident(name) => {
                        let consuming_push = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                                && args.len() == 2
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_pop = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "list_pop_shrink")
                                && args.len() == 1
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_in_place = consuming_push || consuming_pop;
                        let target_ty = self.var_types.get(name).cloned();
                        let is_inout = self.inout_vars.contains(name);
                        let target_is_rc = self.owned_rc_vars.contains(name);
                        let inout_is_rc = is_inout
                            && target_ty.as_ref().is_some_and(|t| self.type_is_rc(t));
                        let rhs = if consuming_push {
                            let ExprKind::Call { args, .. } = &value.kind else {
                                unreachable!()
                            };
                            let elem_is_rc = target_ty
                                .as_ref()
                                .and_then(|t| self.list_elem_ty(t))
                                .is_some_and(|et| self.type_is_rc(&et))
                                || self.expr_produces_rc(&args[1].value);
                            let xs_val = self.load_ident_val(name)?;
                            let val_v = if elem_is_rc {
                                self.code.push_str(&format!(
                                    "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                    xs_val, RC_MASK_ALL as i64
                                ));
                                self.lower_expr_owned(&args[1].value, None)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_list_push(i64 {}, i64 {})\n",
                                res, xs_val, val_v
                            ));
                            res
                        } else if consuming_pop {
                            let xs_val = self.load_ident_val(name)?;
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_list_pop_shrink(i64 {})\n",
                                res, xs_val
                            ));
                            res
                        } else if (target_is_rc || inout_is_rc) && *op == AssignOp::Set {
                            self.lower_expr_owned(value, target_ty.as_ref())?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let slot = self
                            .var_ptrs
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("undefined variable `{}` in LLVM assign", name))?;
                        if target_is_rc && !consuming_in_place && !is_inout {
                            let old = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  store i64 {}, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                                old, slot, rhs, slot, old
                            ));
                        } else if is_inout {
                            let ai = self.fresh_tmp();
                            let ap = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n",
                                ai, slot, ap, ai
                            ));
                            if inout_is_rc && !consuming_in_place && *op == AssignOp::Set {
                                let old = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  store i64 {}, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                                    old, ap, rhs, ap, old
                                ));
                            } else {
                                let new_v = match op {
                                    AssignOp::Set => rhs,
                                    AssignOp::AddSet => {
                                        let cur = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = add i64 {}, {}\n",
                                            cur, ap, res, cur, rhs
                                        ));
                                        res
                                    }
                                    AssignOp::SubSet => {
                                        let cur = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = sub i64 {}, {}\n",
                                            cur, ap, res, cur, rhs
                                        ));
                                        res
                                    }
                                };
                                self.code
                                    .push_str(&format!("  store i64 {}, ptr {}\n", new_v, ap));
                            }
                        } else {
                            let new_v = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => {
                                    let cur = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = add i64 {}, {}\n",
                                        cur, slot, res, cur, rhs
                                    ));
                                    res
                                }
                                AssignOp::SubSet => {
                                    let cur = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = sub i64 {}, {}\n",
                                        cur, slot, res, cur, rhs
                                    ));
                                    res
                                }
                            };
                            self.code
                                .push_str(&format!("  store i64 {}, ptr {}\n", new_v, slot));
                        }
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                            let field_is_rc = fl.as_ref().is_some_and(|f| f.is_rc);
                            let field_ty = fl.as_ref().map(|f| &f.ty);
                            let rhs = if field_is_rc && *op == AssignOp::Set {
                                self.lower_expr_owned(value, field_ty)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            let slot = self.var_ptrs[vname].clone();
                            let is_inout = self.inout_vars.contains(vname);
                            let cur_h = if is_inout {
                                let addr_i = self.fresh_tmp();
                                let addr_p = self.fresh_tmp();
                                let val = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                                    addr_i, slot, addr_p, addr_i, val, addr_p
                                ));
                                val
                            } else {
                                let val = self.fresh_tmp();
                                self.code
                                    .push_str(&format!("  {} = load i64, ptr {}\n", val, slot));
                                val
                            };
                            let uniq = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                uniq, cur_h
                            ));
                            if is_inout {
                                let addr_i = self.fresh_tmp();
                                let addr_p = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                    addr_i, slot, addr_p, addr_i, uniq, addr_p
                                ));
                            } else {
                                self.code
                                    .push_str(&format!("  store i64 {}, ptr {}\n", uniq, slot));
                            }
                            let new_fv = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => {
                                    let cf = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n  {} = add i64 {}, {}\n",
                                        cf, uniq, fidx, res, cf, rhs
                                    ));
                                    res
                                }
                                AssignOp::SubSet => {
                                    let cf = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n  {} = sub i64 {}, {}\n",
                                        cf, uniq, fidx, res, cf, rhs
                                    ));
                                    res
                                }
                            };
                            let setter = if *op != AssignOp::Set {
                                "@__tn_slot_set_prim"
                            } else if let Some(ref f) = fl {
                                if f.is_rc {
                                    "@__tn_slot_set_rc"
                                } else {
                                    "@__tn_slot_set_prim"
                                }
                            } else {
                                "@tenon_rc_set_field"
                            };
                            self.code.push_str(&format!(
                                "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                setter, uniq, fidx, new_fv
                            ));
                        }
                    }
                    ExprKind::Index {
                        target: list_expr,
                        index,
                    } => {
                        let mut idx_temps = Vec::new();
                        let idx_v = self.lower_expr_borrowed(index, &mut idx_temps)?;
                        self.flush_temps(&idx_temps);
                        match &list_expr.kind {
                            ExprKind::Ident(vname) => {
                                let elem_ty = self
                                    .var_types
                                    .get(vname)
                                    .and_then(|t| self.list_elem_ty(t));
                                let elem_is_rc = elem_ty
                                    .as_ref()
                                    .is_some_and(|et| self.type_is_rc(et))
                                    || self.expr_produces_rc(value);
                                let rhs = if elem_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, elem_ty.as_ref())?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let cur_list = self.load_ident_val(vname)?;
                                let is_always_uniq = self.always_unique_vars.contains(vname);
                                let uniq_list = if is_always_uniq {
                                    cur_list
                                } else {
                                    let u = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        u, cur_list
                                    ));
                                    let vslot = self.var_ptrs[vname].clone();
                                    if self.inout_vars.contains(vname) {
                                        let ai = self.fresh_tmp();
                                        let ap = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            ai, vslot, ap, ai, u, ap
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            u, vslot
                                        ));
                                    }
                                    u
                                };
                                if elem_is_rc {
                                    self.code.push_str(&format!(
                                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                        uniq_list, RC_MASK_ALL as i64
                                    ));
                                }
                                let getter = if is_always_uniq {
                                    "@__tn_list_get_nonnull"
                                } else {
                                    "@__tn_list_get"
                                };
                                let new_val = match op {
                                    AssignOp::Set => rhs,
                                    AssignOp::AddSet => {
                                        let ce = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 {}(i64 {}, i64 {})\n  {} = add i64 {}, {}\n",
                                            ce, getter, uniq_list, idx_v, res, ce, rhs
                                        ));
                                        res
                                    }
                                    AssignOp::SubSet => {
                                        let ce = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 {}(i64 {}, i64 {})\n  {} = sub i64 {}, {}\n",
                                            ce, getter, uniq_list, idx_v, res, ce, rhs
                                        ));
                                        res
                                    }
                                };
                                let setter = if *op != AssignOp::Set {
                                    if is_always_uniq { "@__tn_list_set_prim_nonnull" } else { "@__tn_list_set_prim" }
                                } else if elem_is_rc {
                                    "@__tn_list_set_rc"
                                } else if elem_ty
                                    .as_ref()
                                    .is_some_and(|et| et.display_name() != "Any")
                                {
                                    if is_always_uniq { "@__tn_list_set_prim_nonnull" } else { "@__tn_list_set_prim" }
                                } else {
                                    "@tenon_rc_set_field"
                                };
                                self.code.push_str(&format!(
                                    "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                    setter, uniq_list, idx_v, new_val
                                ));
                            }
                            ExprKind::Field { target: obj, field }
                                if matches!(&obj.kind, ExprKind::Ident(_)) =>
                            {
                                let ExprKind::Ident(vname) = &obj.kind else {
                                    unreachable!()
                                };
                                let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                                let elem_ty = fl.as_ref().and_then(|f| self.list_elem_ty(&f.ty));
                                let elem_is_rc = elem_ty
                                    .as_ref()
                                    .is_some_and(|et| self.type_is_rc(et))
                                    || self.expr_produces_rc(value);
                                let rhs = if elem_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, elem_ty.as_ref())?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let cur_obj = self.load_ident_val(vname)?;
                                let uniq_obj = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                    uniq_obj, cur_obj
                                ));
                                let vslot = self.var_ptrs[vname].clone();
                                if self.inout_vars.contains(vname) {
                                    let ai = self.fresh_tmp();
                                    let ap = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                        ai, vslot, ap, ai, uniq_obj, ap
                                    ));
                                } else {
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n",
                                        uniq_obj, vslot
                                    ));
                                }
                                let cur_list = self.fresh_tmp();
                                let uniq_list = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                                    cur_list, uniq_obj, fidx, uniq_list, cur_list, uniq_obj, fidx, uniq_list
                                ));
                                if elem_is_rc {
                                    self.code.push_str(&format!(
                                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                        uniq_list, RC_MASK_ALL as i64
                                    ));
                                }
                                let new_val = match op {
                                    AssignOp::Set => rhs,
                                    AssignOp::AddSet => {
                                        let ce = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_list_get(i64 {}, i64 {})\n  {} = add i64 {}, {}\n",
                                            ce, uniq_list, idx_v, res, ce, rhs
                                        ));
                                        res
                                    }
                                    AssignOp::SubSet => {
                                        let ce = self.fresh_tmp();
                                        let res = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_list_get(i64 {}, i64 {})\n  {} = sub i64 {}, {}\n",
                                            ce, uniq_list, idx_v, res, ce, rhs
                                        ));
                                        res
                                    }
                                };
                                let setter = if *op != AssignOp::Set {
                                    "@__tn_list_set_prim"
                                } else if elem_is_rc {
                                    "@__tn_list_set_rc"
                                } else if elem_ty
                                    .as_ref()
                                    .is_some_and(|et| et.display_name() != "Any")
                                {
                                    "@__tn_list_set_prim"
                                } else {
                                    "@tenon_rc_set_field"
                                };
                                self.code.push_str(&format!(
                                    "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                    setter, uniq_list, idx_v, new_val
                                ));
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                },
                Stmt::Mutate { body, .. } => {
                    let _ = self.lower_block(body, false, None)?;
                }
                Stmt::Return { value, .. } => {
                    let mut move_skip_slot: Option<String> = None;
                    let ret_ty_opt = self.fn_ret_ty.clone();
                    let v = match value {
                        Some(e) => {
                            if self.fn_returns_rc {
                                if let ExprKind::Ident(rname) = &e.kind {
                                    if !self.inout_vars.contains(rname) {
                                        if let Some(slot) = self.var_ptrs.get(rname).cloned() {
                                            if self.find_owned_slot_in_scopes(&slot, 0) {
                                                move_skip_slot = Some(slot.clone());
                                                let lv = self.fresh_tmp();
                                                self.code.push_str(&format!(
                                                    "  {} = load i64, ptr {}\n",
                                                    lv, slot
                                                ));
                                                lv
                                            } else {
                                                self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                            }
                                        } else {
                                            self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                        }
                                    } else {
                                        self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                    }
                                } else {
                                    self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                }
                            } else {
                                let mut temps = Vec::new();
                                let val = self.lower_expr_borrowed(e, &mut temps)?;
                                self.flush_temps(&temps);
                                val
                            }
                        }
                        None => "0".into(),
                    };
                    self.emit_cleanup_temp_scopes_from(0);
                    self.emit_cleanup_scopes_from(0, move_skip_slot.as_deref());
                    self.code.push_str(&format!("  ret i64 {}\n", v));
                    self.terminated = true;
                    last = v;
                    break;
                }
                Stmt::Break { .. } => {
                    let (exit_lbl, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.exit_lbl.clone(), l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`break` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.code.push_str(&format!("  br label %{}\n", exit_lbl));
                    self.terminated = true;
                    break;
                }
                Stmt::Continue { .. } => {
                    let (hdr_lbl, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.hdr_lbl.clone(), l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`continue` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.code.push_str(&format!("  br label %{}\n", hdr_lbl));
                    self.terminated = true;
                    break;
                }
                Stmt::Assert { cond, span } => {
                    let mut temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut temps)?;
                    self.flush_temps(&temps);
                    let ok = self.fresh_tmp();
                    let fl = self.fresh_lbl("afail");
                    let cl = self.fresh_lbl("acont");
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_assert_fail(i64 {})\n  br label %{}\n{}:\n",
                        ok, c, ok, cl, fl, fl, span.line, cl, cl
                    ));
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.fresh_lbl("whdr");
                    let lp = self.fresh_lbl("wbody");
                    let ex = self.fresh_lbl("wexit");
                    self.code.push_str(&format!("  br label %{}\n{}:\n", hdr, hdr));
                    let mut ctemps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
                    self.flush_temps(&ctemps);
                    let cmp = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                        cmp, c, cmp, lp, ex, lp
                    ));
                    self.loop_stack.push(LlvmLoopCtx {
                        hdr_lbl: hdr.clone(),
                        exit_lbl: ex.clone(),
                        rc_scope_depth: self.rc_scopes.len(),
                        temp_scope_depth: self.temp_rc_scopes.len(),
                    });
                    let _ = self.lower_block(body, false, None)?;
                    self.loop_stack.pop();
                    if !self.terminated {
                        self.code.push_str(&format!("  br label %{}\n", hdr));
                    }
                    self.terminated = false;
                    self.code.push_str(&format!("{}:\n", ex));
                }
                Stmt::Expr { expr, .. } => {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(expr, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
        }

        let mut move_skip_slot: Option<String> = None;
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                if want_owned_tail {
                    if let ExprKind::Ident(tname) = &tail.kind {
                        if !self.inout_vars.contains(tname) {
                            if let Some(slot) = self.var_ptrs.get(tname).cloned() {
                                if self.find_owned_slot_in_scopes(&slot, scope_idx) {
                                    move_skip_slot = Some(slot.clone());
                                    let lv = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n",
                                        lv, slot
                                    ));
                                    last = lv;
                                } else {
                                    last = self.lower_expr_owned(tail, expected_tail_ty)?;
                                }
                            } else {
                                last = self.lower_expr_owned(tail, expected_tail_ty)?;
                            }
                        } else {
                            last = self.lower_expr_owned(tail, expected_tail_ty)?;
                        }
                    } else {
                        last = self.lower_expr_owned(tail, expected_tail_ty)?;
                    }
                } else {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(tail, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
            if !self.terminated {
                self.emit_cleanup_scopes_from(scope_idx, move_skip_slot.as_deref());
            }
        }

        self.rc_scopes.pop();
        self.restore_lexical(saved);
        Ok(last)
    }

    fn lower_expr_owned(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let slot = self.var_ptrs[&sv].clone();
                        let v = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                        return Ok(v);
                    }
                }
                if s == "None" {
                    return Ok(self.emit_unit_enum_const(TAG_NONE));
                }
                if let Some(&tag) = self.enum_tags.get(s) {
                    if !self.var_ptrs.contains_key(s) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let v = self.load_ident_val(s)?;
                if self.is_rc_ident(s) {
                    let inc = self.fresh_tmp();
                    self.code
                        .push_str(&format!("  {} = call i64 @__tn_rc_inc(i64 {})\n", inc, v));
                    Ok(inc)
                } else {
                    Ok(v)
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let (idx, fl) = self.resolve_field_layout(target, field)?;
                let mut temps = Vec::new();
                let obj = self.lower_expr_borrowed(target, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, obj, idx
                ));
                if fl.as_ref().is_some_and(|f| f.is_rc) {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Index { target, index } => {
                let elem_is_rc = self
                    .list_elem_ty(&self.infer_expr_type(target))
                    .is_some_and(|et| self.type_is_rc(&et));
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let i = self.lower_expr_borrowed(index, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_list_get(i64 {}, i64 {})\n",
                    res, t, i
                ));
                if elem_is_rc {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => self.lower_if(cond, then_block, else_branch.as_deref(), true, expected_ty),
            ExprKind::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms, true, expected_ty)
            }
            ExprKind::Block(blk) => self.lower_block(blk, true, expected_ty),
            _ => self.lower_expr_fresh(expr, expected_ty),
        }
    }

    fn lower_expr_borrowed(
        &mut self,
        expr: &Expr,
        temps: &mut Vec<String>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(n.to_string()),
            ExprKind::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            ExprKind::Str(s) => {
                let gname = self.intern_string_global(s);
                let obj = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = ptrtoint ptr @{} to i64\n", obj, gname));
                Ok(obj)
            }
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let slot = self.var_ptrs[&sv].clone();
                        let v = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                        return Ok(v);
                    }
                }
                if s == "None" {
                    return Ok(self.emit_unit_enum_const(TAG_NONE));
                }
                if let Some(&tag) = self.enum_tags.get(s) {
                    if !self.var_ptrs.contains_key(s) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                self.load_ident_val(s)
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let (idx, _) = self.resolve_field_layout(target, field)?;
                let obj = self.lower_expr_borrowed(target, temps)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, obj, idx
                ));
                Ok(res)
            }
            ExprKind::Index { target, index } => {
                let is_nonnull = matches!(&target.kind, ExprKind::Ident(n) if self.always_unique_vars.contains(n));
                let t = self.lower_expr_borrowed(target, temps)?;
                let i = self.lower_expr_borrowed(index, temps)?;
                let res = self.fresh_tmp();
                let getter = if is_nonnull { "@__tn_list_get_nonnull" } else { "@__tn_list_get" };
                self.code.push_str(&format!(
                    "  {} = call i64 {}(i64 {}, i64 {})\n",
                    res, getter, t, i
                ));
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_if(cond, then_block, else_branch.as_deref(), want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            ExprKind::Match { scrutinee, arms } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_match(scrutinee, arms, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            ExprKind::Block(blk) => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_block(blk, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            _ => {
                let v = self.lower_expr_fresh(expr, None)?;
                if self.expr_produces_rc(expr) {
                    temps.push(v.clone());
                }
                Ok(v)
            }
        }
    }

    fn lower_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_branch: Option<&Expr>,
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let mut ctemps = Vec::new();
        let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
        self.flush_temps(&ctemps);

        let cmp = self.fresh_tmp();
        self.slot_id += 1;
        let res_slot = format!("%ifres_{}", self.slot_id);
        self.entry_allocas.push_str(&format!(
            "  {} = alloca i64\n  store i64 0, ptr {}\n",
            res_slot, res_slot
        ));
        let tl = self.fresh_lbl("then");
        let el = self.fresh_lbl("else");
        let ml = self.fresh_lbl("endif");
        self.code.push_str(&format!(
            "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
            cmp, c, cmp, tl, el, tl
        ));
        let tv = self.lower_block(then_block, want_owned, expected_ty)?;
        let then_term = self.terminated;
        if !then_term {
            self.code.push_str(&format!(
                "  store i64 {}, ptr {}\n  br label %{}\n",
                tv, res_slot, ml
            ));
        }
        self.terminated = false;
        self.code.push_str(&format!("{}:\n", el));
        let ev = match else_branch {
            Some(eb) => {
                if want_owned {
                    self.lower_expr_owned(eb, expected_ty)?
                } else {
                    let mut etemps = Vec::new();
                    let v = self.lower_expr_borrowed(eb, &mut etemps)?;
                    if !self.terminated {
                        self.flush_temps(&etemps);
                    }
                    v
                }
            }
            None => "0".into(),
        };
        let else_term = self.terminated;
        if !else_term {
            self.code.push_str(&format!(
                "  store i64 {}, ptr {}\n  br label %{}\n",
                ev, res_slot, ml
            ));
        }
        self.code.push_str(&format!("{}:\n", ml));
        if then_term && else_term {
            self.code.push_str("  unreachable\n");
            self.terminated = true;
            return Ok("0".into());
        }
        self.terminated = false;
        let out = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
        Ok(out)
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let scrutinee_ty = self.infer_expr_type(scrutinee);
        let is_variant = arms
            .iter()
            .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Variant { .. })));
        let is_str = arms
            .iter()
            .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Str(_, _))));

        let mut mtemps = Vec::new();
        let sv = self.lower_expr_borrowed(scrutinee, &mut mtemps)?;
        let temp_idx = self.temp_rc_scopes.len();
        self.temp_rc_scopes.push(mtemps);

        let cmp_val = if is_variant {
            let tg = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = call i64 @__tn_rc_tag(i64 {})\n", tg, sv));
            tg
        } else {
            sv.clone()
        };

        self.slot_id += 1;
        let res_slot = format!("%mres_{}", self.slot_id);
        self.entry_allocas.push_str(&format!(
            "  {} = alloca i64\n  store i64 0, ptr {}\n",
            res_slot, res_slot
        ));
        let end_l = self.fresh_lbl("mend");
        let mut any_reachable = false;

        for arm in arms {
            let is_wild = arm.guard.is_none()
                && arm
                    .patterns
                    .iter()
                    .any(|p| matches!(p, Pattern::Wildcard(_)));
            if is_wild {
                let v = if want_owned {
                    self.lower_expr_owned(&arm.body, expected_ty)?
                } else {
                    let mut atemps = Vec::new();
                    let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                    if !self.terminated {
                        self.flush_temps(&atemps);
                    }
                    av
                };
                if !self.terminated {
                    self.code.push_str(&format!(
                        "  store i64 {}, ptr {}\n  br label %{}\n",
                        v, res_slot, end_l
                    ));
                    any_reachable = true;
                }
                self.terminated = false;
                self.code.push_str(&format!("{}:\n", end_l));
                let mt = self.temp_rc_scopes.pop().unwrap();
                if !any_reachable {
                    self.code.push_str("  unreachable\n");
                    self.terminated = true;
                    return Ok("0".into());
                }
                self.flush_temps(&mt);
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
                return Ok(out);
            }

            let mut cond_var: Option<String> = None;
            let mut vbind: Option<(String, String)> = None;
            for p in &arm.patterns {
                let eq = match p {
                    Pattern::Wildcard(_) => {
                        let e = self.fresh_tmp();
                        self.code.push_str(&format!("  {} = icmp eq i64 1, 1\n", e));
                        e
                    }
                    Pattern::Int(n, _) => {
                        let e = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = icmp eq i64 {}, {}\n", e, cmp_val, n));
                        e
                    }
                    Pattern::Bool(b, _) => {
                        let e = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, {}\n",
                            e,
                            cmp_val,
                            if *b { 1 } else { 0 }
                        ));
                        e
                    }
                    Pattern::Str(s, _) if is_str => {
                        let gname = self.intern_string_global(s);
                        let ptxt = self.fresh_tmp();
                        let teq = self.fresh_tmp();
                        let e = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = ptrtoint ptr @{} to i64\n  {} = call i64 @__tn_text_eq(i64 {}, i64 {})\n  {} = icmp ne i64 {}, 0\n",
                            ptxt, gname, teq, cmp_val, ptxt, e, teq
                        ));
                        e
                    }
                    Pattern::Variant { name, binding, .. } => {
                        let t = *self
                            .enum_tags
                            .get(name)
                            .ok_or_else(|| format!("unknown variant `{}` in LLVM match", name))?;
                        if let Some(bn) = binding {
                            if bn != "_" {
                                vbind = Some((bn.clone(), name.clone()));
                            }
                        }
                        let e = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = icmp eq i64 {}, {}\n", e, cmp_val, t));
                        e
                    }
                    _ => continue,
                };
                cond_var = Some(match cond_var {
                    Some(prev) => {
                        let o = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = or i1 {}, {}\n", o, prev, eq));
                        o
                    }
                    None => eq,
                });
            }

            let mat_l = self.fresh_lbl("marm");
            let nxt_l = self.fresh_lbl("mnxt");
            let c = cond_var.unwrap_or_else(|| "false".into());
            self.code.push_str(&format!(
                "  br i1 {}, label %{}, label %{}\n{}:\n",
                c, mat_l, nxt_l, mat_l
            ));
            let saved_arm = self.save_lexical();
            if let Some((bn, vname)) = vbind {
                let pay = self.fresh_tmp();
                let bslot = self.fresh_var_slot(&bn);
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n  store i64 {}, ptr {}\n",
                    pay, sv, pay, bslot
                ));
                let bty = self.infer_match_binding_type(&scrutinee_ty, &vname);
                if self.type_is_rc(&bty) {
                    self.borrowed_rc_vars.insert(bn.clone());
                }
                self.var_types.insert(bn, bty);
            }
            if let Some(guard_expr) = &arm.guard {
                let mut gtemps = Vec::new();
                let gv = self.lower_expr_borrowed(guard_expr, &mut gtemps)?;
                self.flush_temps(&gtemps);
                let gc = self.fresh_tmp();
                let body_l = self.fresh_lbl("mgbody");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    gc, gv, gc, body_l, nxt_l, body_l
                ));
            }
            let av = if want_owned {
                self.lower_expr_owned(&arm.body, expected_ty)?
            } else {
                let mut atemps = Vec::new();
                let v = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                if !self.terminated {
                    self.flush_temps(&atemps);
                }
                v
            };
            self.restore_lexical(saved_arm);
            if !self.terminated {
                self.code.push_str(&format!(
                    "  store i64 {}, ptr {}\n  br label %{}\n",
                    av, res_slot, end_l
                ));
                any_reachable = true;
            }
            self.terminated = false;
            self.code.push_str(&format!("{}:\n", nxt_l));
        }

        self.code
            .push_str(&format!("  br label %{}\n{}:\n", end_l, end_l));
        let mt = self.temp_rc_scopes.remove(temp_idx);
        self.flush_temps(&mt);
        let out = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
        Ok(out)
    }

    fn lower_expr_fresh(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(n.to_string()),
            ExprKind::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            ExprKind::Str(s) => {
                let gname = self.intern_string_global(s);
                let obj = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = ptrtoint ptr @{} to i64\n", obj, gname));
                Ok(obj)
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}`", name))?;
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    layout.tag,
                    layout.fields.len()
                ));
                if layout.rc_mask != 0 {
                    self.code.push_str(&format!(
                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                        obj, layout.rc_mask
                    ));
                }
                for (fname, fexpr) in fields {
                    if let Some((idx, fl)) = layout
                        .fields
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.name == *fname)
                    {
                        let fv = if fl.is_rc {
                            self.lower_expr_owned(fexpr, Some(&fl.ty))?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        self.code.push_str(&format!(
                            "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                            obj, idx, fv
                        ));
                    }
                }
                Ok(obj)
            }
            ExprKind::List(elems) => {
                let exp_elem_ty = expected_ty.and_then(|t| self.list_elem_ty(t));
                let elem_is_rc = exp_elem_ty
                    .as_ref()
                    .is_some_and(|et| self.type_is_rc(et))
                    || elems.iter().any(|e| self.expr_produces_rc(e));
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    TAG_LIST,
                    elems.len()
                ));
                if elem_is_rc {
                    self.code.push_str(&format!(
                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                        obj, RC_MASK_ALL as i64
                    ));
                }
                for (idx, e) in elems.iter().enumerate() {
                    let ev = if elem_is_rc {
                        self.lower_expr_owned(e, exp_elem_ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(e, &mut temps)?;
                        self.flush_temps(&temps);
                        v
                    };
                    self.code.push_str(&format!(
                        "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                        obj, idx, ev
                    ));
                }
                Ok(obj)
            }
            ExprKind::Unary { op, expr } => {
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(expr, &mut temps)?;
                self.flush_temps(&temps);
                let r = self.fresh_tmp();
                match op {
                    UnOp::Neg => self.code.push_str(&format!("  {} = sub i64 0, {}\n", r, v)),
                    UnOp::Not => {
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                            c, v, r, c
                        ));
                    }
                }
                Ok(r)
            }
            ExprKind::Binary {
                op: BinOp::And,
                left,
                right,
            } => {
                self.slot_id += 1;
                let slot = format!("%and_{}", self.slot_id);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    slot, slot
                ));
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let lb = self.fresh_tmp();
                let rhs_l = self.fresh_lbl("and_rhs");
                let end_l = self.fresh_lbl("and_end");
                self.code.push_str(&format!(
                    "  store i64 0, ptr {}\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    slot, lb, l, lb, rhs_l, end_l, rhs_l
                ));
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rb = self.fresh_tmp();
                let rz = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  {} = zext i1 {} to i64\n  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                    rb, r, rz, rb, rz, slot, end_l, end_l
                ));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, slot));
                Ok(out)
            }
            ExprKind::Binary {
                op: BinOp::Or,
                left,
                right,
            } => {
                self.slot_id += 1;
                let slot = format!("%or_{}", self.slot_id);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    slot, slot
                ));
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let lb = self.fresh_tmp();
                let rhs_l = self.fresh_lbl("or_rhs");
                let end_l = self.fresh_lbl("or_end");
                self.code.push_str(&format!(
                    "  store i64 1, ptr {}\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    slot, lb, l, lb, end_l, rhs_l, rhs_l
                ));
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rb = self.fresh_tmp();
                let rz = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  {} = zext i1 {} to i64\n  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                    rb, r, rz, rb, rz, slot, end_l, end_l
                ));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, slot));
                Ok(out)
            }
            ExprKind::Binary { op, left, right } => {
                let tcmp = self.is_text_expr(left) || self.is_text_expr(right);
                let mut temps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut temps)?;
                let r = self.lower_expr_borrowed(right, &mut temps)?;
                if *op == BinOp::Concat {
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_str_concat(i64 {}, i64 {})\n",
                        res, l, r
                    ));
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if tcmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_text_eq(i64 {}, i64 {})\n",
                        eq, l, r
                    ));
                    self.flush_temps(&temps);
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let c = self.fresh_tmp();
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                        c, eq, res, c
                    ));
                    return Ok(res);
                }
                self.flush_temps(&temps);
                let res = self.fresh_tmp();
                match op {
                    BinOp::Add => self.code.push_str(&format!("  {} = add i64 {}, {}\n", res, l, r)),
                    BinOp::Sub => self.code.push_str(&format!("  {} = sub i64 {}, {}\n", res, l, r)),
                    BinOp::Mul => self.code.push_str(&format!("  {} = mul i64 {}, {}\n", res, l, r)),
                    BinOp::Div => self.code.push_str(&format!("  {} = sdiv i64 {}, {}\n", res, l, r)),
                    BinOp::Mod => self.code.push_str(&format!("  {} = srem i64 {}, {}\n", res, l, r)),
                    BinOp::BitAnd => self.code.push_str(&format!("  {} = and i64 {}, {}\n", res, l, r)),
                    BinOp::BitOr => self.code.push_str(&format!("  {} = or i64 {}, {}\n", res, l, r)),
                    BinOp::BitXor => self.code.push_str(&format!("  {} = xor i64 {}, {}\n", res, l, r)),
                    BinOp::Shl => self.code.push_str(&format!("  {} = shl i64 {}, {}\n", res, l, r)),
                    BinOp::Shr => self.code.push_str(&format!("  {} = lshr i64 {}, {}\n", res, l, r)),
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let pred = match op {
                            BinOp::Eq => "eq",
                            BinOp::Ne => "ne",
                            BinOp::Lt => "slt",
                            BinOp::Le => "sle",
                            BinOp::Gt => "sgt",
                            BinOp::Ge => "sge",
                            _ => unreachable!(),
                        };
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp {} i64 {}, {}\n  {} = zext i1 {} to i64\n",
                            c, pred, l, r, res, c
                        ));
                    }
                    other => return Err(format!("unsupported op {:?} in LLVM codegen", other)),
                }
                Ok(res)
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let mut temps = Vec::new();
                        let mut avals = Vec::new();
                        for a in args {
                            avals.push(format!(
                                "i64 {}",
                                self.lower_expr_borrowed(&a.value, &mut temps)?
                            ));
                        }
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_fn_{}__parse({})\n",
                            res,
                            sanitize(rname),
                            avals.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    Err("unsupported `.parse()` target in LLVM".into())
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        let key = format!("{}.{}", ename, field);
                        if let Some(&tag) = self.enum_tags.get(&key) {
                            let ptys = self
                                .enum_variants
                                .get(&key)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }
                    // Dynamic interface method dispatch: `d.method(args)` where `d: dyn Interface`!
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        let dm = self
                            .dyn_interfaces
                            .get(&interface)
                            .and_then(|l| l.methods.iter().find(|m| m.name == *field))
                            .cloned()
                            .ok_or_else(|| {
                                format!("unknown method `{}` on `dyn {}` in LLVM codegen", field, interface)
                            })?;
                        let mut temps = Vec::new();
                        let mut d_val = self.lower_expr_borrowed(target, &mut temps)?;
                        let mut call_args = Vec::new();
                        let mut inout_dyn_slot: Option<(String, String, Option<String>)> = None;

                        if dm.is_inout_self {
                            let d_uniq = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                d_uniq, d_val
                            ));
                            if let ExprKind::Ident(vname) = &target.kind {
                                if let Some(ptr) = self.var_ptrs.get(vname).cloned() {
                                    if self.inout_vars.contains(vname) {
                                        let slot_addr = self.fresh_tmp();
                                        let p = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            slot_addr, ptr, p, slot_addr, d_uniq, p
                                        ));
                                    } else {
                                        self.code
                                            .push_str(&format!("  store i64 {}, ptr {}\n", d_uniq, ptr));
                                    }
                                }
                            }
                            d_val = d_uniq.clone();
                            let data_val = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n",
                                data_val, d_val
                            ));
                            self.slot_id += 1;
                            let slot = format!("%dyn_inout_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!(
                                "  {} = alloca i64\n  store i64 0, ptr {}\n",
                                slot, slot
                            ));
                            let slot_i64 = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                data_val, slot, slot_i64, slot
                            ));
                            call_args.push(format!("i64 {}", slot_i64));
                            inout_dyn_slot = Some((d_uniq, slot, None));
                        } else {
                            let data_val = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n",
                                data_val, d_val
                            ));
                            call_args.push(format!("i64 {}", data_val));
                        }

                        for a in args {
                            let av = self.lower_expr_borrowed(&a.value, &mut temps)?;
                            call_args.push(format!("i64 {}", av));
                        }

                        let vt_i64 = self.fresh_tmp();
                        let vt_ptr = self.fresh_tmp();
                        let slot_ptr = self.fresh_tmp();
                        let fn_ptr = self.fresh_tmp();
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_slot_get(i64 {}, i64 1)\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds ptr, ptr {}, i64 {}\n  {} = load ptr, ptr {}\n  {} = call i64 {}({})\n",
                            vt_i64, d_val,
                            vt_ptr, vt_i64,
                            slot_ptr, vt_ptr, dm.slot_idx,
                            fn_ptr, slot_ptr,
                            res, fn_ptr, call_args.join(", ")
                        ));
                        if let Some((d_uniq, slot, _)) = inout_dyn_slot {
                            let updated = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n",
                                updated, slot, d_uniq, updated
                            ));
                        }
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    Err(format!(
                        "unsupported method call `{}` on `{}` in LLVM codegen",
                        field,
                        recv_ty.display_name()
                    ))
                }
                ExprKind::Ident(fname) => {
                    if fname == "__box_dyn" && args.len() == 3 {
                        let vsym = match &args[1].value.kind {
                            ExprKind::Str(s) => s.clone(),
                            _ => return Err("__box_dyn expects vtable symbol string".into()),
                        };
                        let data_is_rc = matches!(&args[2].value.kind, ExprKind::Int(n) if *n != 0);
                        let data_val = if data_is_rc {
                            self.lower_expr_owned(&args[0].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let obj = self.fresh_tmp();
                        let vt_i64 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 2)\n",
                            obj, TAG_DYN
                        ));
                        if data_is_rc {
                            self.code.push_str(&format!(
                                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 1)\n",
                                obj
                            ));
                        }
                        self.code.push_str(&format!(
                            "  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  {} = ptrtoint ptr @{} to i64\n  call void @__tn_slot_set_prim(i64 {}, i64 1, i64 {})\n",
                            obj, data_val,
                            vt_i64, sanitize(&vsym),
                            obj, vt_i64
                        ));
                        return Ok(obj);
                    }
                    if fname == "print_int" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rt_print_int(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "len" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_rc_len(i64 {})\n",
                            res, v
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "push" && args.len() == 2 {
                        let xs_ty = self.infer_expr_type(&args[0].value);
                        let elem_is_rc = self
                            .list_elem_ty(&xs_ty)
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.expr_produces_rc(&args[1].value);
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        let val = if elem_is_rc {
                            self.code.push_str(&format!(
                                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                xs, RC_MASK_ALL as i64
                            ));
                            self.lower_expr_owned(&args[1].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_push(i64 {}, i64 {})\n",
                            res, xs, val
                        ));
                        return Ok(res);
                    }
                    if fname == "list_filled" && args.len() == 2 {
                        let elem_is_rc = expected_ty
                            .and_then(|t| self.list_elem_ty(t))
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.type_is_rc(&self.infer_expr_type(&args[1].value))
                            || self.expr_produces_rc(&args[1].value);
                        let mut temps = Vec::new();
                        let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let init_v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_filled(i64 {}, i64 {}, i64 {})\n",
                            res,
                            count_v,
                            init_v,
                            if elem_is_rc { 1 } else { 0 }
                        ));
                        let lf_pos = self.fresh_tmp();
                        let lf_exp = self.fresh_tmp();
                        let lf_nz = self.fresh_tmp();
                        let lf_obj = self.fresh_tmp();
                        let lf_lp = self.fresh_tmp();
                        let lf_len = self.fresh_tmp();
                        let lf_eq = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp sgt i64 {}, 0\n  {} = select i1 {}, i64 {}, i64 0\n  {} = icmp ne i64 {}, 0\n  call void @llvm.assume(i1 {})\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp eq i64 {}, {}\n  call void @llvm.assume(i1 {})\n",
                            lf_pos, count_v,
                            lf_exp, lf_pos, count_v,
                            lf_nz, res,
                            lf_nz,
                            lf_obj, res,
                            lf_lp, lf_obj,
                            lf_len, lf_lp,
                            lf_eq, lf_len, lf_exp,
                            lf_eq
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "list_pop_shrink" && args.len() == 1 {
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_pop_shrink(i64 {})\n",
                            res, xs
                        ));
                        return Ok(res);
                    }
                    if fname == "yield_val" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_yield_create(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "block_on" && args.len() == 1 {
                        let f = self.lower_expr_owned(&args[0].value, None)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_block_on(i64 {})\n",
                            res, f
                        ));
                        return Ok(res);
                    }
                    if let Some(&csym) = self.syscall_map.get(fname) {
                        let mut temps = Vec::new();
                        let mut avals = Vec::new();
                        for a in args {
                            avals.push(format!(
                                "i64 {}",
                                self.lower_expr_borrowed(&a.value, &mut temps)?
                            ));
                        }
                        let res = self.fresh_tmp();
                        let target_sym = match csym {
                            "tenon_byte_at" => "__tn_byte_at",
                            "tenon_buf_push_byte" => "__tn_buf_push_byte",
                            _ => csym,
                        };
                        self.code.push_str(&format!(
                            "  {} = call i64 @{}({})\n",
                            res,
                            target_sym,
                            avals.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.fn_metas.contains_key(fname) {
                            let mut ptys = self
                                .enum_variants
                                .get(fname)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            if ptys.is_empty() {
                                if let Some(exp) = expected_ty {
                                    let resolved = resolve_alias_ty(exp, self.alias_bases);
                                    if let TypeExpr::Named { name: tn, args: targs, .. } = resolved {
                                        if fname == "Some" && tn == "Option" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Ok" && tn == "Result" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Err" && tn == "Result" && targs.len() >= 2 {
                                            ptys.push(targs[1].clone());
                                        }
                                    }
                                }
                            }
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }

                    let meta = self.fn_metas.get(fname).cloned();
                    let mut temps = Vec::new();
                    let mut avals = Vec::new();
                    let mut writebacks: Vec<(String, String)> = Vec::new();
                    let mut field_writebacks: Vec<(String, usize, String)> = Vec::new();
                    for (idx, a) in args.iter().enumerate() {
                        let is_inout = meta
                            .as_ref()
                            .and_then(|m| m.params_inout.get(idx).copied())
                            .unwrap_or(false);
                        let pty = meta.as_ref().and_then(|m| m.param_tys.get(idx));
                        if is_inout {
                            match &a.value.kind {
                                ExprKind::Ident(vn) => {
                                    let cur = self.load_ident_val(vn)?;
                                    self.slot_id += 1;
                                    let tmp_slot = format!("%io_{}_{}", self.slot_id, idx);
                                    self.entry_allocas
                                        .push_str(&format!("  {} = alloca i64\n", tmp_slot));
                                    let addr_i = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                        cur, tmp_slot, addr_i, tmp_slot
                                    ));
                                    avals.push(format!("i64 {}", addr_i));
                                    writebacks.push((vn.clone(), tmp_slot));
                                }
                                ExprKind::Field { target: obj, field }
                                    if matches!(&obj.kind, ExprKind::Ident(_)) =>
                                {
                                    let ExprKind::Ident(vn) = &obj.kind else {
                                        unreachable!()
                                    };
                                    let (fidx, _) = self.resolve_field_layout(obj, field)?;
                                    let cur_obj = self.load_ident_val(vn)?;
                                    let uniq_obj = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        uniq_obj, cur_obj
                                    ));
                                    let vslot = self.var_ptrs[vn].clone();
                                    if self.inout_vars.contains(vn) {
                                        let ai = self.fresh_tmp();
                                        let ap = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            ai, vslot, ap, ai, uniq_obj, ap
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            uniq_obj, vslot
                                        ));
                                    }
                                    let cur_fv = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                                        cur_fv, uniq_obj, fidx
                                    ));
                                    self.slot_id += 1;
                                    let tmp_slot = format!("%iof_{}_{}", self.slot_id, idx);
                                    self.entry_allocas
                                        .push_str(&format!("  {} = alloca i64\n", tmp_slot));
                                    let addr_i = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                        cur_fv, tmp_slot, addr_i, tmp_slot
                                    ));
                                    avals.push(format!("i64 {}", addr_i));
                                    field_writebacks.push((uniq_obj, fidx, tmp_slot));
                                }
                                _ => {
                                    return Err("inout arg must be variable or record field".into());
                                }
                            }
                        } else if matches!(&a.value.kind, ExprKind::List(es) if es.is_empty())
                            && pty.is_some()
                        {
                            let lv = self.lower_expr_owned(&a.value, pty)?;
                            temps.push(lv.clone());
                            avals.push(format!("i64 {}", lv));
                        } else {
                            let av = self.lower_expr_borrowed(&a.value, &mut temps)?;
                            avals.push(format!("i64 {}", av));
                        }
                    }
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_fn_{}({})\n",
                        res,
                        sanitize(fname),
                        avals.join(", ")
                    ));
                    self.flush_temps(&temps);
                    for (vn, tmp_slot) in writebacks {
                        let updated = self.fresh_tmp();
                        let vslot = self.var_ptrs[&vn].clone();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", updated, tmp_slot));
                        if self.inout_vars.contains(&vn) {
                            let ai = self.fresh_tmp();
                            let ap = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                ai, vslot, ap, ai, updated, ap
                            ));
                        } else {
                            self.code
                                .push_str(&format!("  store i64 {}, ptr {}\n", updated, vslot));
                        }
                    }
                    for (uniq_obj, fidx, tmp_slot) in field_writebacks {
                        let updated = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                            updated, tmp_slot, uniq_obj, fidx, updated
                        ));
                    }
                    Ok(res)
                }
                _ => Err("unsupported call target in LLVM codegen".into()),
            },
            _ => Ok("0".into()),
        }
    }

    fn lower_enum_ctor(
        &mut self,
        tag: i64,
        payload_tys: &[TypeExpr],
        args: &[CallArg],
    ) -> Result<String, String> {
        if args.is_empty() {
            return Ok(self.emit_unit_enum_const(tag));
        }
        let o = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
            o,
            tag,
            args.len()
        ));
        let mut rc_mask = 0u64;
        for (i, a) in args.iter().enumerate() {
            let pty = payload_tys.get(i);
            let is_rc = pty.is_some_and(|t| self.type_is_rc(t)) || self.expr_produces_rc(&a.value);
            let av = if is_rc {
                if i < 64 {
                    rc_mask |= 1u64 << i;
                }
                self.lower_expr_owned(&a.value, pty)?
            } else {
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(&a.value, &mut temps)?;
                self.flush_temps(&temps);
                v
            };
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                o, i, av
            ));
        }
        if rc_mask != 0 {
            self.code.push_str(&format!(
                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                o, rc_mask as i64
            ));
        }
        Ok(o)
    }
}

fn collect_always_unique_locals(body: &Block) -> BTreeSet<String> {
    let mut candidates: Vec<(usize, String)> = Vec::new();
    for (idx, stmt) in body.stmts.iter().enumerate() {
        if let Stmt::Let { name, value, .. } = stmt {
            let is_fresh_unique = match &value.kind {
                ExprKind::List(_) | ExprKind::RecordLit { .. } => true,
                ExprKind::Call { callee, .. } => matches!(&callee.kind, ExprKind::Ident(f) if f == "list_filled"),
                _ => false,
            };
            if is_fresh_unique {
                candidates.push((idx, name.clone()));
            }
        }
    }
    let mut result = BTreeSet::new();
    for (decl_idx, name) in candidates {
        let mut ok = true;
        for stmt in &body.stmts[(decl_idx + 1)..] {
            if !stmt_keeps_unique(stmt, &name) {
                ok = false;
                break;
            }
        }
        if ok {
            if let Some(tail) = &body.tail {
                if !expr_keeps_unique(tail, &name) {
                    ok = false;
                }
            }
        }
        if ok {
            result.insert(name);
        }
    }
    result
}

fn block_keeps_unique(blk: &Block, name: &str) -> bool {
    for stmt in &blk.stmts {
        if !stmt_keeps_unique(stmt, name) {
            return false;
        }
    }
    blk.tail.as_deref().is_none_or(|t| expr_keeps_unique(t, name))
}

fn stmt_keeps_unique(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Let { name: ln, value, .. } => ln != name && expr_keeps_unique(value, name),
        Stmt::Assign { target, op, value, .. } => match &target.kind {
            ExprKind::Ident(vn) if vn == name => {
                if *op != AssignOp::Set {
                    return false;
                }
                match &value.kind {
                    ExprKind::Call { callee, args }
                        if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                            && args.len() == 2
                            && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name) =>
                    {
                        expr_keeps_unique(&args[1].value, name)
                    }
                    ExprKind::Call { callee, args }
                        if matches!(&callee.kind, ExprKind::Ident(f) if f == "list_pop_shrink")
                            && args.len() == 1
                            && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name) =>
                    {
                        true
                    }
                    _ => false,
                }
            }
            ExprKind::Index { target: lt, index } => {
                let t_ok = matches!(&lt.kind, ExprKind::Ident(s) if s == name)
                    || expr_keeps_unique(lt, name);
                t_ok && expr_keeps_unique(index, name) && expr_keeps_unique(value, name)
            }
            ExprKind::Field { target: ot, .. } => {
                let t_ok = matches!(&ot.kind, ExprKind::Ident(s) if s == name)
                    || expr_keeps_unique(ot, name);
                t_ok && expr_keeps_unique(value, name)
            }
            _ => expr_keeps_unique(target, name) && expr_keeps_unique(value, name),
        },
        Stmt::Mutate { body, .. } => block_keeps_unique(body, name),
        Stmt::Return { value, .. } => value.as_ref().is_none_or(|e| expr_keeps_unique(e, name)),
        Stmt::Assert { cond, .. } => expr_keeps_unique(cond, name),
        Stmt::While { cond, body, .. } => {
            expr_keeps_unique(cond, name) && block_keeps_unique(body, name)
        }
        Stmt::Expr { expr, .. } => expr_keeps_unique(expr, name),
        Stmt::Break { .. } | Stmt::Continue { .. } => true,
    }
}

fn expr_keeps_unique(expr: &Expr, name: &str) -> bool {
    match &expr.kind {
        ExprKind::Ident(s) => s != name,
        ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::Str(_) => true,
        ExprKind::Index { target, index } => {
            let t_ok = matches!(&target.kind, ExprKind::Ident(s) if s == name)
                || expr_keeps_unique(target, name);
            t_ok && expr_keeps_unique(index, name)
        }
        ExprKind::Field { target, .. } => {
            matches!(&target.kind, ExprKind::Ident(s) if s == name)
                || expr_keeps_unique(target, name)
        }
        ExprKind::Call { callee, args } => {
            if matches!(&callee.kind, ExprKind::Ident(f) if f == "len")
                && args.len() == 1
                && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name)
            {
                return true;
            }
            expr_keeps_unique(callee, name)
                && args.iter().all(|a| expr_keeps_unique(&a.value, name))
        }
        ExprKind::Unary { expr: inner, .. } => expr_keeps_unique(inner, name),
        ExprKind::Binary { left, right, .. } => {
            expr_keeps_unique(left, name) && expr_keeps_unique(right, name)
        }
        ExprKind::RecordLit { fields, .. } => {
            fields.iter().all(|(_, e)| expr_keeps_unique(e, name))
        }
        ExprKind::List(es) => es.iter().all(|e| expr_keeps_unique(e, name)),
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            expr_keeps_unique(cond, name)
                && block_keeps_unique(then_block, name)
                && else_branch
                    .as_deref()
                    .is_none_or(|e| expr_keeps_unique(e, name))
        }
        ExprKind::Match { scrutinee, arms } => {
            if !expr_keeps_unique(scrutinee, name) {
                return false;
            }
            arms.iter().all(|a| {
                let shadows = a.patterns.iter().any(|p| {
                    matches!(p, Pattern::Variant { binding: Some(b), .. } if b == name)
                });
                !shadows
                    && a.guard.as_ref().is_none_or(|g| expr_keeps_unique(g, name))
                    && expr_keeps_unique(&a.body, name)
            })
        }
        ExprKind::Block(blk) => block_keeps_unique(blk, name),
        ExprKind::Await(e) => expr_keeps_unique(e, name),
        _ => false,
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn llvm_aot_compiles_and_runs_records_mutate_parse_and_stackless_async() {
        let src = r#"
pub type Year  = Int where 1 <= self && self <= 9999
pub type Month = Int where 1 <= self && self <= 12

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Year, m Month) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Year
    month Month
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

pub fn shift_prev(inout d Date) {
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}

pub async fn async_add(v Int) Int {
    let a = await yield_val(v + 5)
    let b = await yield_val(a + 7)
    b
}

pub fn main() Int {
    var d Date = Date { year = 2024, month = 2, day = 29 }
    shift_prev(d)
    assert d.year == 2023
    assert d.day == 28

    let ok_day = match Date.parse(2024, 2, 29) {
        Ok(dt) => dt.day
        Err(c) => -c
    }
    let bad_m = match Date.parse(2024, 13, 1) {
        Ok(dt) => dt.day
        Err(c) => -c
    }
    assert ok_day == 29
    assert bad_m == -2

    let ar = block_on(async_add(10))
    assert ar == 22
    ok_day + d.day + ar
}
"#;
        let prog = parse(src).unwrap();
        let bin = std::env::temp_dir().join(format!("tenon_llvm_aot_test_{}", std::process::id()));
        let art = compile_to_native_binary(&prog, &bin, None, "O3").unwrap();
        assert!(art.binary_bytes > 1024);

        let run = Command::new(&bin).output().unwrap();
        let _ = fs::remove_file(&bin);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "native LLVM binary failed: stdout={} stderr={}",
            stdout,
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(stdout.contains("main() => 79"), "unexpected stdout: {}", stdout);
        assert!(stdout.contains("live_objects=0"), "leaked objects: {}", stdout);
        assert!(stdout.contains("async_suspends=2"), "missing suspensions: {}", stdout);
    }

    #[test]
    fn llvm_aot_links_tenon_rt_and_runs_core_syscalls() {
        let src = r#"
pub record Msg {
    tag Text
    body Text
}

pub enum BoxVal {
    BEmpty
    BMsg(Msg)
    BList(List[Msg])
}

pub fn build_msg(t Text, b Text) Msg {
    Msg { tag = "[" ++ t ++ "]", body = b }
}

pub fn main() Int {
    var xs List[Msg] = []
    xs = push(xs, build_msg("OK", "hello tenon"))
    let bv = BList(xs)
    let out = match bv {
        BList(items) => {
            let m = items[0]
            m.tag ++ " " ++ str_slice(m.body, 0, 5)
        }
        BMsg(m) => m.tag
        BEmpty => ""
    }
    assert out == "[OK] hello"
    let _ = sys_write_stdout(out ++ "\n")
    0
}
"#;
        let prog = parse(src).unwrap();
        let bin = std::env::temp_dir().join(format!("tenon_llvm_rt_test_{}", std::process::id()));
        let art = compile_to_native_binary(&prog, &bin, None, "O2").unwrap();
        assert!(art.binary_bytes > 4096);

        let run = Command::new(&bin).output().unwrap();
        let _ = fs::remove_file(&bin);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "native LLVM+tenon_rt binary failed: stdout={} stderr={}",
            stdout,
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(stdout.trim(), "[OK] hello");
    }

    #[test]
    fn llvm_aot_interfaces_static_monomorph_and_dyn_vtable_zero_leaks() {
        let src = r#"
pub interface Greeter {
    pure fn greet(self, who Text) Text
}

pub interface Accumulator {
    fn add(inout self, delta Int) Int
    pure fn current(self) Int
}

pub record EnGreeter { prefix Text }
pub record ZhGreeter { prefix Text }

impl Greeter for EnGreeter {
    pure fn greet(self, who Text) Text { self.prefix ++ " " ++ who }
}

impl Greeter for ZhGreeter {
    pure fn greet(self, who Text) Text { self.prefix ++ ", " ++ who }
}

pub record Counter { val Int }

impl Accumulator for Counter {
    fn add(inout self, delta Int) Int {
        self.val += delta
        self.val
    }
    pure fn current(self) Int { self.val }
}

pub pure fn run_static[T Greeter](g T, name Text) Text {
    g.greet(name)
}

pub pure fn run_dyn(g dyn Greeter, name Text) Text {
    g.greet(name)
}

pub fn main() Int {
    let en = EnGreeter { prefix = "Hello" }
    let zh = ZhGreeter { prefix = "NiHao" }
    assert run_static(en, "Tenon") == "Hello Tenon"
    assert run_static(zh, "Tenon") == "NiHao, Tenon"

    var list List[dyn Greeter] = []
    list = push(list, en)
    list = push(list, zh)
    let b = buf_new()
    let _ = buf_push_str(b, run_dyn(list[0], "World"))
    let _ = buf_push_byte(b, 124)
    let _ = buf_push_str(b, list[1].greet("World"))
    let joined = buf_finish(b)
    assert joined == "Hello World|NiHao, World"

    var c dyn Accumulator = Counter { val = 10 }
    let c2 dyn Accumulator = c
    let v1 = c.add(5)
    assert v1 == 15
    assert c.current() == 15
    assert c2.current() == 10
    let _ = sys_write_stdout(joined ++ "\n")
    0
}
"#;
        let prog = parse(src).unwrap();
        let bin = std::env::temp_dir().join(format!("tenon_llvm_dyn_test_{}", std::process::id()));
        let art = compile_to_native_binary(&prog, &bin, None, "O3").unwrap();
        assert!(art.binary_bytes > 4096);

        let run = Command::new(&bin).output().unwrap();
        let _ = fs::remove_file(&bin);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "native LLVM dyn/static interface binary failed: stdout={} stderr={}",
            stdout,
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(stdout.trim(), "Hello World|NiHao, World");
    }
}
