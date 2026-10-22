#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline uint32_t hex_ceil_pow2(uint32_t x) {
    nếu (x <= 1) { trả_về 1; }
    số_nguyên p = 2;
    x--;
    trong_khi (x >>= 1) { p <<= 1; }
    trả_về p;
}

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    trả_về (num + den - 1) / den;
}

static inline int32_t hex_is_aligned(const void * addr, uint32_t align) {
    trả_về ((size_t) addr & (align - 1)) == 0;
}

static inline size_t hex_align_up(size_t v, size_t align) {
    trả_về hmx_ceil_div(v, align) * align;
}

static inline size_t hex_align_down(size_t v, size_t align) {
    trả_về (v / align) * align;
}

static inline int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    trả_về right_off <= chunk_size;
}

static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    trả_về m * ((n + m - 1) / m);
}

static inline size_t hex_smin(size_t a, size_t b) {
    trả_về a < b ? a : b;
}

static inline size_t hex_smax(size_t a, size_t b) {
    trả_về a > b ? a : b;
}

static inline void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static inline bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    nếu (a != 0 && b > SIZE_MAX / a) trả_về true;
    *out = a * b;
    trả_về false;
}

static inline bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    nếu (a > SIZE_MAX - b) trả_về true;
    *out = a + b;
    trả_về false;
}

static inline uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    trong_khi (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    trả_về a;
}

#endif // HEX_COMMON_H
