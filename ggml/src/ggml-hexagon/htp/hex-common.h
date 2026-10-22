#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

স্থির অন্তঃরেখ uint32_t hex_ceil_pow2(uint32_t x) {
    যদি (x <= 1) { ফেরত 1; }
    পূর্ণসংখ্যা p = 2;
    x--;
    যখন (x >>= 1) { p <<= 1; }
    ফেরত p;
}

স্থির অন্তঃরেখ size_t hmx_ceil_div(size_t অঙ্ক, size_t den) {
    ফেরত (অঙ্ক + den - 1) / den;
}

স্থির অন্তঃরেখ int32_t hex_is_aligned(ধ্রুবক শূন্য * addr, uint32_t align) {
    ফেরত ((size_t) addr & (align - 1)) == 0;
}

স্থির অন্তঃরেখ size_t hex_align_up(size_t v, size_t align) {
    ফেরত hmx_ceil_div(v, align) * align;
}

স্থির অন্তঃরেখ size_t hex_align_down(size_t v, size_t align) {
    ফেরত (v / align) * align;
}

স্থির অন্তঃরেখ int32_t hex_is_one_chunk(শূন্য * addr, uint32_t n, uint32_t chunk_আকার) {
    uint32_t বাম_off  = (size_t) addr & (chunk_আকার - 1);
    uint32_t ডান_off = বাম_off + n;
    ফেরত ডান_off <= chunk_আকার;
}

স্থির অন্তঃরেখ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ফেরত m * ((n + m - 1) / m);
}

স্থির অন্তঃরেখ size_t hex_smin(size_t a, size_t b) {
    ফেরত a < b ? a : b;
}

স্থির অন্তঃরেখ size_t hex_smax(size_t a, size_t b) {
    ফেরত a > b ? a : b;
}

স্থির অন্তঃরেখ শূন্য hex_swap_ptr(শূন্য ** p1, শূন্য ** p2) {
    শূন্য * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

স্থির অন্তঃরেখ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    যদি (a != 0 && b > SIZE_MAX / a) ফেরত true;
    *out = a * b;
    ফেরত false;
}

স্থির অন্তঃরেখ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    যদি (a > SIZE_MAX - b) ফেরত true;
    *out = a + b;
    ফেরত false;
}

স্থির অন্তঃরেখ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    যখন (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ফেরত a;
}

#endif // HEX_COMMON_H
