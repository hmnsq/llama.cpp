#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t hex_ceil_pow2(uint32_t x) {
    ଯଦି (x <= 1) { ଫେରନ୍ତୁ 1; }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = 2;
    x--;
    ଯେତେବେଳେ (x >>= 1) { p <<= 1; }
    ଫେରନ୍ତୁ p;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t hmx_ceil_div(size_t ଅଙ୍କ, size_t den) {
    ଫେରନ୍ତୁ (ଅଙ୍କ + den - 1) / den;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32_t hex_is_aligned(ଅଚଳ ଶୂନ୍ୟ * addr, uint32_t align) {
    ଫେରନ୍ତୁ ((size_t) addr & (align - 1)) == 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t hex_align_up(size_t v, size_t align) {
    ଫେରନ୍ତୁ hmx_ceil_div(v, align) * align;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t hex_align_down(size_t v, size_t align) {
    ଫେରନ୍ତୁ (v / align) * align;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32_t hex_is_one_chunk(ଶୂନ୍ୟ * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t ବାମ_off  = (size_t) addr & (chunk_size - 1);
    uint32_t ଡାହାଣ_off = ବାମ_off + n;
    ଫେରନ୍ତୁ ଡାହାଣ_off <= chunk_size;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ଫେରନ୍ତୁ m * ((n + m - 1) / m);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t hex_smin(size_t a, size_t b) {
    ଫେରନ୍ତୁ a < b ? a : b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t hex_smax(size_t a, size_t b) {
    ଫେରନ୍ତୁ a > b ? a : b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_swap_ptr(ଶୂନ୍ୟ ** p1, ଶୂନ୍ୟ ** p2) {
    ଶୂନ୍ୟ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ଯଦି (a != 0 && b > SIZE_MAX / a) ଫେରନ୍ତୁ true;
    *out = a * b;
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ଯଦି (a > SIZE_MAX - b) ଫେରନ୍ତୁ true;
    *out = a + b;
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ଯେତେବେଳେ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ଫେରନ୍ତୁ a;
}

#endif // HEX_COMMON_H
