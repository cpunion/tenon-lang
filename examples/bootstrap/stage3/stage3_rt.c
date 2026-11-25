// examples/bootstrap/stage3/stage3_rt.c — Minimal Zero-Rust C Runtime for Tenon Stage-3 Self-Hosted Compiler
//
// Provides uniform 64-bit value representation (`i64` for Int/Bool, heap pointer as `i64` for Text & List)
// so `tenon_selfhost.tn` can compile ITSELF -> Stage-3 -> Stage-4 with bit-exact SHA-256 fixed point!

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t tag; // 1 = Text, 2 = List
    int64_t len;
    int64_t cap;
    char data[];
} S3Obj;

static int g_argc = 0;
static char** g_argv = NULL;

int64_t s3_str_from_cstr(const char* s, int64_t n) {
    if (!s || n < 0) n = 0;
    S3Obj* o = (S3Obj*)malloc(sizeof(S3Obj) + (size_t)n + 1);
    o->tag = 1;
    o->len = n;
    o->cap = n;
    if (n > 0) memcpy(o->data, s, (size_t)n);
    o->data[n] = '\0';
    return (int64_t)(uintptr_t)o;
}

int64_t s3_str_concat(int64_t a, int64_t b) {
    S3Obj* sa = (S3Obj*)(uintptr_t)a;
    S3Obj* sb = (S3Obj*)(uintptr_t)b;
    int64_t la = sa ? sa->len : 0;
    int64_t lb = sb ? sb->len : 0;
    S3Obj* o = (S3Obj*)malloc(sizeof(S3Obj) + (size_t)(la + lb) + 1);
    o->tag = 1;
    o->len = la + lb;
    o->cap = la + lb;
    if (la > 0) memcpy(o->data, sa->data, (size_t)la);
    if (lb > 0) memcpy(o->data + la, sb->data, (size_t)lb);
    o->data[la + lb] = '\0';
    return (int64_t)(uintptr_t)o;
}

int64_t s3_str_eq(int64_t a, int64_t b) {
    if (a == b) return 1;
    S3Obj* sa = (S3Obj*)(uintptr_t)a;
    S3Obj* sb = (S3Obj*)(uintptr_t)b;
    if (!sa || !sb) return 0;
    if (sa->len != sb->len) return 0;
    if (sa->len == 0) return 1;
    return memcmp(sa->data, sb->data, (size_t)sa->len) == 0 ? 1 : 0;
}

int64_t s3_len(int64_t x) {
    if (x == 0) return 0;
    S3Obj* o = (S3Obj*)(uintptr_t)x;
    return o->len;
}

int64_t s3_byte_at(int64_t s, int64_t idx) {
    S3Obj* o = (S3Obj*)(uintptr_t)s;
    if (!o || idx < 0 || idx >= o->len) return 0;
    return (int64_t)(uint8_t)o->data[idx];
}

int64_t s3_str_slice(int64_t s, int64_t start, int64_t end) {
    S3Obj* o = (S3Obj*)(uintptr_t)s;
    if (!o) return s3_str_from_cstr("", 0);
    if (start < 0) start = 0;
    if (end > o->len) end = o->len;
    if (end <= start) return s3_str_from_cstr("", 0);
    return s3_str_from_cstr(o->data + start, end - start);
}

int64_t s3_int_to_text(int64_t v) {
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%lld", (long long)v);
    return s3_str_from_cstr(buf, n > 0 ? (int64_t)n : 0);
}

int64_t s3_list_new(void) {
    int64_t cap = 8;
    S3Obj* o = (S3Obj*)malloc(sizeof(S3Obj) + (size_t)cap * sizeof(int64_t));
    o->tag = 2;
    o->len = 0;
    o->cap = cap;
    return (int64_t)(uintptr_t)o;
}

int64_t s3_list_push(int64_t lst, int64_t val) {
    S3Obj* o = (S3Obj*)(uintptr_t)lst;
    if (!o) o = (S3Obj*)(uintptr_t)s3_list_new();
    if (o->len >= o->cap) {
        int64_t ncap = o->cap < 8 ? 8 : o->cap * 2;
        S3Obj* n = (S3Obj*)malloc(sizeof(S3Obj) + (size_t)ncap * sizeof(int64_t));
        n->tag = 2;
        n->len = o->len;
        n->cap = ncap;
        memcpy(n->data, o->data, (size_t)o->len * sizeof(int64_t));
        o = n;
    }
    int64_t* items = (int64_t*)o->data;
    items[o->len++] = val;
    return (int64_t)(uintptr_t)o;
}

int64_t s3_list_get(int64_t lst, int64_t idx) {
    S3Obj* o = (S3Obj*)(uintptr_t)lst;
    if (!o || idx < 0 || idx >= o->len) return 0;
    return ((int64_t*)o->data)[idx];
}

int64_t s3_list_set(int64_t lst, int64_t idx, int64_t val) {
    S3Obj* o = (S3Obj*)(uintptr_t)lst;
    if (!o || idx < 0 || idx >= o->len) return lst;
    ((int64_t*)o->data)[idx] = val;
    return (int64_t)(uintptr_t)o;
}

typedef struct {
    char* data;
    int64_t len;
    int64_t cap;
} S3Buf;

int64_t s3_buf_new(void) {
    S3Buf* b = (S3Buf*)malloc(sizeof(S3Buf));
    b->cap = 4096;
    b->len = 0;
    b->data = (char*)malloc((size_t)b->cap + 1);
    b->data[0] = '\0';
    return (int64_t)(uintptr_t)b;
}

int64_t s3_buf_push_str(int64_t buf, int64_t str_val) {
    S3Buf* b = (S3Buf*)(uintptr_t)buf;
    S3Obj* s = (S3Obj*)(uintptr_t)str_val;
    if (!b || !s || s->len <= 0) return 0;
    if (b->len + s->len > b->cap) {
        int64_t ncap = b->cap * 2;
        while (ncap < b->len + s->len) ncap *= 2;
        b->data = (char*)realloc(b->data, (size_t)ncap + 1);
        b->cap = ncap;
    }
    memcpy(b->data + b->len, s->data, (size_t)s->len);
    b->len += s->len;
    b->data[b->len] = '\0';
    return b->len;
}

int64_t s3_buf_push_byte(int64_t buf, int64_t ch) {
    S3Buf* b = (S3Buf*)(uintptr_t)buf;
    if (!b) return 0;
    if (b->len + 1 > b->cap) {
        int64_t ncap = b->cap * 2;
        b->data = (char*)realloc(b->data, (size_t)ncap + 1);
        b->cap = ncap;
    }
    b->data[b->len++] = (char)(ch & 0xFF);
    b->data[b->len] = '\0';
    return b->len;
}

int64_t s3_buf_finish(int64_t buf) {
    S3Buf* b = (S3Buf*)(uintptr_t)buf;
    if (!b) return s3_str_from_cstr("", 0);
    int64_t res = s3_str_from_cstr(b->data, b->len);
    free(b->data);
    free(b);
    return res;
}

int64_t s3_println(int64_t s) {
    S3Obj* o = (S3Obj*)(uintptr_t)s;
    if (o && o->len > 0) fwrite(o->data, 1, (size_t)o->len, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return 0;
}

int64_t s3_read_file(int64_t path_val) {
    S3Obj* p = (S3Obj*)(uintptr_t)path_val;
    if (!p || p->len <= 0) return s3_str_from_cstr("", 0);
    FILE* f = fopen(p->data, "rb");
    if (!f) return s3_str_from_cstr("", 0);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return s3_str_from_cstr("", 0); }
    char* tmp = (char*)malloc((size_t)sz + 1);
    size_t got = fread(tmp, 1, (size_t)sz, f);
    fclose(f);
    int64_t res = s3_str_from_cstr(tmp, (int64_t)got);
    free(tmp);
    return res;
}

int64_t s3_write_file(int64_t path_val, int64_t data_val) {
    S3Obj* p = (S3Obj*)(uintptr_t)path_val;
    S3Obj* d = (S3Obj*)(uintptr_t)data_val;
    if (!p || !d) return 0;
    FILE* f = fopen(p->data, "wb");
    if (!f) return 0;
    if (d->len > 0) fwrite(d->data, 1, (size_t)d->len, f);
    fclose(f);
    return 1;
}

int64_t s3_env(int64_t key_val) {
    S3Obj* k = (S3Obj*)(uintptr_t)key_val;
    if (!k || k->len <= 0) return s3_str_from_cstr("", 0);
    const char* v = getenv(k->data);
    return v ? s3_str_from_cstr(v, (int64_t)strlen(v)) : s3_str_from_cstr("", 0);
}

extern int64_t s3_main(void);

int main(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
    return (int)s3_main();
}
