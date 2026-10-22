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

static anta_hstha uint32_t hex_ceil_pow2(uint32_t x) {
    ပေႃး (x <= 1) { ပွၵ်ႈ 1; }
    puur_naa_nka p = 2;
    x--;
    ၶၢဝ်း (x >>= 1) { p <<= 1; }
    ပွၵ်ႈ p;
}

static anta_hstha size_t hmx_ceil_div(size_t num, size_t den) {
    ပွၵ်ႈ (num + den - 1) / den;
}

static anta_hstha int32_t hex_is_aligned(const void * addr, uint32_t align) {
    ပွၵ်ႈ ((size_t) addr & (align - 1)) == 0;
}

static anta_hstha size_t hex_align_up(size_t v, size_t align) {
    ပွၵ်ႈ hmx_ceil_div(v, align) * align;
}

static anta_hstha size_t hex_align_down(size_t v, size_t align) {
    ပွၵ်ႈ (v / align) * align;
}

static anta_hstha int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    ပွၵ်ႈ right_off <= chunk_size;
}

static anta_hstha uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ပွၵ်ႈ m * ((n + m - 1) / m);
}

static anta_hstha size_t hex_smin(size_t a, size_t b) {
    ပွၵ်ႈ a < b ? a : b;
}

static anta_hstha size_t hex_smax(size_t a, size_t b) {
    ပွၵ်ႈ a > b ? a : b;
}

static anta_hstha void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static anta_hstha bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ပေႃး (a != 0 && b > SIZE_MAX / a) ပွၵ်ႈ true;
    *out = a * b;
    ပွၵ်ႈ false;
}

static anta_hstha bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ပေႃး (a > SIZE_MAX - b) ပွၵ်ႈ true;
    *out = a + b;
    ပွၵ်ႈ false;
}

static anta_hstha uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ၶၢဝ်း (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ပွၵ်ႈ a;
}

#endif // HEX_COMMON_H
