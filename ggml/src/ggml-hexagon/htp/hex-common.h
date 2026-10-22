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
    ᰀᰧ (x <= 1) { ᰀᰧᰌ 1; }
    puur_naa_nka p = 2;
    x--;
    ᰀᰬ (x >>= 1) { p <<= 1; }
    ᰀᰧᰌ p;
}

static anta_hstha size_t hmx_ceil_div(size_t num, size_t den) {
    ᰀᰧᰌ (num + den - 1) / den;
}

static anta_hstha int32_t hex_is_aligned(const void * addr, uint32_t align) {
    ᰀᰧᰌ ((size_t) addr & (align - 1)) == 0;
}

static anta_hstha size_t hex_align_up(size_t v, size_t align) {
    ᰀᰧᰌ hmx_ceil_div(v, align) * align;
}

static anta_hstha size_t hex_align_down(size_t v, size_t align) {
    ᰀᰧᰌ (v / align) * align;
}

static anta_hstha int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    ᰀᰧᰌ right_off <= chunk_size;
}

static anta_hstha uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ᰀᰧᰌ m * ((n + m - 1) / m);
}

static anta_hstha size_t hex_smin(size_t a, size_t b) {
    ᰀᰧᰌ a < b ? a : b;
}

static anta_hstha size_t hex_smax(size_t a, size_t b) {
    ᰀᰧᰌ a > b ? a : b;
}

static anta_hstha void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static anta_hstha bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ᰀᰧ (a != 0 && b > SIZE_MAX / a) ᰀᰧᰌ true;
    *out = a * b;
    ᰀᰧᰌ false;
}

static anta_hstha bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ᰀᰧ (a > SIZE_MAX - b) ᰀᰧᰌ true;
    *out = a + b;
    ᰀᰧᰌ false;
}

static anta_hstha uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ᰀᰬ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ᰀᰧᰌ a;
}

#endif // HEX_COMMON_H
