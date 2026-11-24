// Benchmark 05 (C): 5,000x Nested JSON Parse + Field Access + Rebuild + Stringify (DOM AST)
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { J_NULL, J_BOOL, J_INT, J_STR, J_ARR, J_OBJ } JKind;
typedef struct JVal JVal;
typedef struct { char* key; JVal* val; } JEntry;
struct JVal {
    JKind kind;
    union {
        int b;
        int64_t i;
        struct { char* s; size_t len; } str;
        struct { JVal** items; size_t len; size_t cap; } arr;
        struct { JEntry* items; size_t len; size_t cap; } obj;
    } u;
};

static JVal* jnew(JKind k) {
    JVal* v = (JVal*)calloc(1, sizeof(JVal));
    v->kind = k;
    return v;
}

static void jfree(JVal* v) {
    if (!v) return;
    if (v->kind == J_STR) free(v->u.str.s);
    else if (v->kind == J_ARR) {
        for (size_t i = 0; i < v->u.arr.len; i++) jfree(v->u.arr.items[i]);
        free(v->u.arr.items);
    } else if (v->kind == J_OBJ) {
        for (size_t i = 0; i < v->u.obj.len; i++) {
            free(v->u.obj.items[i].key);
            jfree(v->u.obj.items[i].val);
        }
        free(v->u.obj.items);
    }
    free(v);
}

static void skip_ws(const char* s, size_t n, size_t* p) {
    while (*p < n && (s[*p] == ' ' || s[*p] == '\t' || s[*p] == '\n' || s[*p] == '\r')) (*p)++;
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static char* parse_str_raw(const char* s, size_t n, size_t* p, size_t* out_len) {
    (*p)++; // skip '"'
    size_t cap = 32, len = 0;
    char* buf = (char*)malloc(cap);
    while (*p < n) {
        char c = s[(*p)++];
        if (c == '"') { buf[len] = '\0'; *out_len = len; return buf; }
        if (len + 6 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
        if (c == '\\' && *p < n) {
            char e = s[(*p)++];
            if (e == 'n') buf[len++] = '\n';
            else if (e == 'r') buf[len++] = '\r';
            else if (e == 't') buf[len++] = '\t';
            else if (e == '"' || e == '\\' || e == '/') buf[len++] = e;
            else if (e == 'u' && *p + 4 <= n) {
                int cp = (hex_val(s[*p]) << 12) | (hex_val(s[*p + 1]) << 8) |
                         (hex_val(s[*p + 2]) << 4) | hex_val(s[*p + 3]);
                *p += 4;
                if (cp < 0x80) buf[len++] = (char)cp;
                else if (cp < 0x800) {
                    buf[len++] = (char)(0xC0 | (cp >> 6));
                    buf[len++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    buf[len++] = (char)(0xE0 | (cp >> 12));
                    buf[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    buf[len++] = (char)(0x80 | (cp & 0x3F));
                }
            } else buf[len++] = e;
        } else {
            buf[len++] = c;
        }
    }
    buf[len] = '\0';
    *out_len = len;
    return buf;
}

static JVal* parse_val(const char* s, size_t n, size_t* p) {
    skip_ws(s, n, p);
    if (*p >= n) return jnew(J_NULL);
    char c = s[*p];
    if (c == '"') {
        JVal* v = jnew(J_STR);
        v->u.str.s = parse_str_raw(s, n, p, &v->u.str.len);
        return v;
    }
    if (c == '[') {
        (*p)++;
        JVal* v = jnew(J_ARR);
        skip_ws(s, n, p);
        if (*p < n && s[*p] == ']') { (*p)++; return v; }
        while (*p < n) {
            if (v->u.arr.len == v->u.arr.cap) {
                v->u.arr.cap = v->u.arr.cap ? v->u.arr.cap * 2 : 8;
                v->u.arr.items = (JVal**)realloc(v->u.arr.items, v->u.arr.cap * sizeof(JVal*));
            }
            v->u.arr.items[v->u.arr.len++] = parse_val(s, n, p);
            skip_ws(s, n, p);
            if (*p < n && s[*p] == ',') { (*p)++; continue; }
            if (*p < n && s[*p] == ']') { (*p)++; break; }
            break;
        }
        return v;
    }
    if (c == '{') {
        (*p)++;
        JVal* v = jnew(J_OBJ);
        skip_ws(s, n, p);
        if (*p < n && s[*p] == '}') { (*p)++; return v; }
        while (*p < n) {
            skip_ws(s, n, p);
            size_t kl = 0;
            char* k = parse_str_raw(s, n, p, &kl);
            skip_ws(s, n, p);
            if (*p < n && s[*p] == ':') (*p)++;
            JVal* val = parse_val(s, n, p);
            if (v->u.obj.len == v->u.obj.cap) {
                v->u.obj.cap = v->u.obj.cap ? v->u.obj.cap * 2 : 8;
                v->u.obj.items = (JEntry*)realloc(v->u.obj.items, v->u.obj.cap * sizeof(JEntry));
            }
            v->u.obj.items[v->u.obj.len++] = (JEntry){ k, val };
            skip_ws(s, n, p);
            if (*p < n && s[*p] == ',') { (*p)++; continue; }
            if (*p < n && s[*p] == '}') { (*p)++; break; }
            break;
        }
        return v;
    }
    if (c == 't') { *p += 4; JVal* v = jnew(J_BOOL); v->u.b = 1; return v; }
    if (c == 'f') { *p += 5; JVal* v = jnew(J_BOOL); v->u.b = 0; return v; }
    if (c == 'n') { *p += 4; return jnew(J_NULL); }
    int neg = 0;
    if (c == '-') { neg = 1; (*p)++; }
    int64_t acc = 0;
    while (*p < n && s[*p] >= '0' && s[*p] <= '9') { acc = acc * 10 + (s[*p] - '0'); (*p)++; }
    JVal* v = jnew(J_INT);
    v->u.i = neg ? -acc : acc;
    return v;
}

static JVal* jget(const JVal* obj, const char* key) {
    if (!obj || obj->kind != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->u.obj.len; i++) {
        if (strcmp(obj->u.obj.items[i].key, key) == 0) return obj->u.obj.items[i].val;
    }
    return NULL;
}

typedef struct { char* data; size_t len; size_t cap; } Buf;
static void buf_push(Buf* b, const char* s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->cap ? b->cap * 2 : 64);
        while (b->cap < b->len + n + 1) b->cap *= 2;
        b->data = (char*)realloc(b->data, b->cap);
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
}

static void jwrite(Buf* b, const JVal* v) {
    char num[32];
    if (v->kind == J_NULL) buf_push(b, "null", 4);
    else if (v->kind == J_BOOL) buf_push(b, v->u.b ? "true" : "false", v->u.b ? 4 : 5);
    else if (v->kind == J_INT) { int l = snprintf(num, sizeof(num), "%lld", (long long)v->u.i); buf_push(b, num, (size_t)l); }
    else if (v->kind == J_STR) {
        buf_push(b, "\"", 1);
        buf_push(b, v->u.str.s, v->u.str.len);
        buf_push(b, "\"", 1);
    } else if (v->kind == J_OBJ) {
        buf_push(b, "{", 1);
        for (size_t i = 0; i < v->u.obj.len; i++) {
            if (i > 0) buf_push(b, ",", 1);
            buf_push(b, "\"", 1);
            buf_push(b, v->u.obj.items[i].key, strlen(v->u.obj.items[i].key));
            buf_push(b, "\":", 2);
            jwrite(b, v->u.obj.items[i].val);
        }
        buf_push(b, "}", 1);
    }
}

static void jobj_set(JVal* o, const char* k, JVal* val) {
    if (o->u.obj.len == o->u.obj.cap) {
        o->u.obj.cap = o->u.obj.cap ? o->u.obj.cap * 2 : 8;
        o->u.obj.items = (JEntry*)realloc(o->u.obj.items, o->u.obj.cap * sizeof(JEntry));
    }
    size_t kl = strlen(k);
    char* kc = (char*)malloc(kl + 1);
    memcpy(kc, k, kl + 1);
    o->u.obj.items[o->u.obj.len++] = (JEntry){ kc, val };
}

int main(void) {
    const char* sample = "{\"type\":\"response.output_item.done\",\"ok\":true,\"seq\":42,\"nums\":[10,20,30,40,50],\"item\":{\"id\":\"fc_123\",\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"src/main.tn\\\",\\\"content\\\":\\\"line1\\\\nline2 \\\\u4f60\\\\u597d\\\"}\"}}";
    size_t slen = strlen(sample);
    int64_t checksum = 0;
    for (int i = 0; i < 5000; i++) {
        size_t pos = 0;
        JVal* ev = parse_val(sample, slen, &pos);
        JVal* tv = jget(ev, "type");
        JVal* sv = jget(ev, "seq");
        JVal* nv = jget(ev, "nums");
        int64_t num2 = (nv && nv->kind == J_ARR && nv->u.arr.len > 2) ? nv->u.arr.items[2]->u.i : 0;
        int64_t seq = (sv && sv->kind == J_INT) ? sv->u.i : 0;

        JVal* built = jnew(J_OBJ);
        JVal* ts = jnew(J_STR);
        ts->u.str.len = tv ? tv->u.str.len : 0;
        ts->u.str.s = (char*)malloc(ts->u.str.len + 1);
        if (tv) memcpy(ts->u.str.s, tv->u.str.s, ts->u.str.len + 1);
        jobj_set(built, "type", ts);
        JVal* si = jnew(J_INT); si->u.i = seq + i; jobj_set(built, "seq", si);
        JVal* pi = jnew(J_INT); pi->u.i = num2; jobj_set(built, "pick", pi);
        JVal* ok = jnew(J_BOOL); ok->u.b = 1; jobj_set(built, "ok", ok);

        Buf b = {0};
        jwrite(&b, built);
        checksum += (int64_t)b.len + num2;
        free(b.data);
        jfree(built);
        jfree(ev);
    }
    printf("%lld\n", (long long)checksum);
    return 0;
}
