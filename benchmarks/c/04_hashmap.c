// Benchmark 04 (C): 50,000-key String HashMap (same compact open-addressing FNV-1a algorithm)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* key;
    size_t klen;
    char* val;
    size_t vlen;
    int64_t hash;
} Entry;

typedef struct {
    int64_t* buckets;
    size_t cap;
    Entry* entries;
    size_t entries_len;
    size_t entries_cap;
    size_t count;
    size_t used;
} TextMap;

static int64_t hash_str(const char* s, size_t n) {
    uint64_t h = 2166136261ULL;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ (uint8_t)s[i]) * 16777619ULL;
    }
    h = (h ^ (h >> 33)) * 490278349ULL;
    h = h ^ (h >> 29);
    return (int64_t)(h & 9223372036854775807ULL);
}

static char* dup_str(const char* s, size_t n) {
    char* p = (char*)malloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void map_init(TextMap* m) {
    memset(m, 0, sizeof(*m));
}

static void map_rehash(TextMap* m, size_t target_cap) {
    size_t cap = 8;
    while (cap < target_cap) cap <<= 1;
    size_t mask = cap - 1;
    int64_t* nb = (int64_t*)malloc(cap * sizeof(int64_t));
    for (size_t i = 0; i < cap; i++) nb[i] = -1;
    size_t ncap = m->count > 8 ? m->count * 2 : 16;
    Entry* ne = (Entry*)malloc(ncap * sizeof(Entry));
    size_t live = 0;
    for (size_t i = 0; i < m->entries_len; i++) {
        if (m->entries[i].hash >= 0) {
            size_t slot = (size_t)m->entries[i].hash & mask;
            while (nb[slot] != -1) slot = (slot + 1) & mask;
            nb[slot] = (int64_t)live;
            ne[live++] = m->entries[i];
        }
    }
    free(m->buckets);
    free(m->entries);
    m->buckets = nb;
    m->cap = cap;
    m->entries = ne;
    m->entries_len = live;
    m->entries_cap = ncap;
    m->count = live;
    m->used = live;
}

static void map_set(TextMap* m, const char* key, size_t klen, const char* val, size_t vlen) {
    if (m->cap == 0) {
        map_rehash(m, 8);
    } else if ((m->used + 1) * 10 >= m->cap * 7) {
        map_rehash(m, (m->count + 1) * 2 >= m->cap ? m->cap * 2 : m->cap);
    }
    size_t mask = m->cap - 1;
    int64_t h = hash_str(key, klen);
    size_t slot = (size_t)h & mask;
    int64_t first_tomb = -1;
    for (;;) {
        int64_t idx = m->buckets[slot];
        if (idx == -1) {
            size_t target = first_tomb >= 0 ? (size_t)first_tomb : slot;
            if (m->entries_len == m->entries_cap) {
                m->entries_cap = m->entries_cap ? m->entries_cap * 2 : 16;
                m->entries = (Entry*)realloc(m->entries, m->entries_cap * sizeof(Entry));
            }
            size_t ni = m->entries_len++;
            m->entries[ni] = (Entry){ dup_str(key, klen), klen, dup_str(val, vlen), vlen, h };
            m->buckets[target] = (int64_t)ni;
            m->count++;
            if (first_tomb < 0) m->used++;
            return;
        }
        if (idx == -2) {
            if (first_tomb < 0) first_tomb = (int64_t)slot;
        } else if (m->entries[idx].hash == h && m->entries[idx].klen == klen &&
                   memcmp(m->entries[idx].key, key, klen) == 0) {
            free(m->entries[idx].val);
            m->entries[idx].val = dup_str(val, vlen);
            m->entries[idx].vlen = vlen;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

static const Entry* map_get(const TextMap* m, const char* key, size_t klen) {
    if (m->count == 0 || m->cap == 0) return NULL;
    size_t mask = m->cap - 1;
    int64_t h = hash_str(key, klen);
    size_t slot = (size_t)h & mask;
    for (;;) {
        int64_t idx = m->buckets[slot];
        if (idx == -1) return NULL;
        if (idx >= 0 && m->entries[idx].hash == h && m->entries[idx].klen == klen &&
            memcmp(m->entries[idx].key, key, klen) == 0) {
            return &m->entries[idx];
        }
        slot = (slot + 1) & mask;
    }
}

static void map_remove(TextMap* m, const char* key, size_t klen) {
    if (m->count == 0 || m->cap == 0) return;
    size_t mask = m->cap - 1;
    int64_t h = hash_str(key, klen);
    size_t slot = (size_t)h & mask;
    for (;;) {
        int64_t idx = m->buckets[slot];
        if (idx == -1) return;
        if (idx >= 0 && m->entries[idx].hash == h && m->entries[idx].klen == klen &&
            memcmp(m->entries[idx].key, key, klen) == 0) {
            free(m->entries[idx].key);
            free(m->entries[idx].val);
            m->entries[idx].key = NULL;
            m->entries[idx].val = NULL;
            m->entries[idx].hash = -1;
            m->buckets[slot] = -2;
            m->count--;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

static void map_free(TextMap* m) {
    for (size_t i = 0; i < m->entries_len; i++) {
        if (m->entries[i].hash >= 0) {
            free(m->entries[i].key);
            free(m->entries[i].val);
        }
    }
    free(m->buckets);
    free(m->entries);
}

int main(void) {
    const int n = 50000;
    TextMap m;
    map_init(&m);
    char kbuf[32], vbuf[32];
    for (int i = 0; i < n; i++) {
        int kl = snprintf(kbuf, sizeof(kbuf), "key_%d", i);
        int vl = snprintf(vbuf, sizeof(vbuf), "val_%d", i * 3);
        map_set(&m, kbuf, (size_t)kl, vbuf, (size_t)vl);
    }
    for (int u = 0; u < 10000; u++) {
        int kl = snprintf(kbuf, sizeof(kbuf), "key_%d", u);
        int vl = snprintf(vbuf, sizeof(vbuf), "upd_%d", u);
        map_set(&m, kbuf, (size_t)kl, vbuf, (size_t)vl);
    }
    int64_t checksum = 0;
    for (int q = 0; q < n; q++) {
        int kl = snprintf(kbuf, sizeof(kbuf), "key_%d", q);
        const Entry* e = map_get(&m, kbuf, (size_t)kl);
        if (e) checksum += (int64_t)e->vlen;
    }
    for (int d = 0; d < n; d += 2) {
        int kl = snprintf(kbuf, sizeof(kbuf), "key_%d", d);
        map_remove(&m, kbuf, (size_t)kl);
    }
    for (int q = 0; q < n; q++) {
        int kl = snprintf(kbuf, sizeof(kbuf), "key_%d", q);
        const Entry* e = map_get(&m, kbuf, (size_t)kl);
        if (e) checksum += (int64_t)e->vlen;
    }
    checksum += (int64_t)m.count;
    map_free(&m);
    printf("%lld\n", (long long)checksum);
    return 0;
}
