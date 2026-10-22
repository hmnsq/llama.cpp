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
    ако (x <= 1) { врати 1; }
    цео p = 2;
    x--;
    док (x >>= 1) { p <<= 1; }
    врати p;
}

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    врати (num + den - 1) / den;
}

static inline int32_t hex_is_aligned(const void * addr, uint32_t align) {
    врати ((size_t) addr & (align - 1)) == 0;
}

static inline size_t hex_align_up(size_t v, size_t align) {
    врати hmx_ceil_div(v, align) * align;
}

static inline size_t hex_align_down(size_t v, size_t align) {
    врати (v / align) * align;
}

static inline int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    врати right_off <= chunk_size;
}

static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    врати m * ((n + m - 1) / m);
}

static inline size_t hex_smin(size_t a, size_t b) {
    врати a < b ? a : b;
}

static inline size_t hex_smax(size_t a, size_t b) {
    врати a > b ? a : b;
}

static inline void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static inline bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ако (a != 0 && b > SIZE_MAX / a) врати true;
    *out = a * b;
    врати false;
}

static inline bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ако (a > SIZE_MAX - b) врати true;
    *out = a + b;
    врати false;
}

static inline uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    док (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    врати a;
}

#endif // HEX_COMMON_H
