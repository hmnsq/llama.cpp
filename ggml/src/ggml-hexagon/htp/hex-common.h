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

স্থির অন্তঃস্থ uint32_t hex_ceil_pow2(uint32_t x) {
    যদি (x <= 1) { প্রত্যাগম 1; }
    পূর্ণাঙ্ক p = 2;
    x--;
    যাবত্ (x >>= 1) { p <<= 1; }
    প্রত্যাগম p;
}

স্থির অন্তঃস্থ size_t hmx_ceil_div(size_t অঙ্ক, size_t den) {
    প্রত্যাগম (অঙ্ক + den - 1) / den;
}

স্থির অন্তঃস্থ int32_t hex_is_aligned(ধ্রুব শূন্য * addr, uint32_t align) {
    প্রত্যাগম ((size_t) addr & (align - 1)) == 0;
}

স্থির অন্তঃস্থ size_t hex_align_up(size_t v, size_t align) {
    প্রত্যাগম hmx_ceil_div(v, align) * align;
}

স্থির অন্তঃস্থ size_t hex_align_down(size_t v, size_t align) {
    প্রত্যাগম (v / align) * align;
}

স্থির অন্তঃস্থ int32_t hex_is_one_chunk(শূন্য * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t বাম_off  = (size_t) addr & (chunk_size - 1);
    uint32_t ডান_off = বাম_off + n;
    প্রত্যাগম ডান_off <= chunk_size;
}

স্থির অন্তঃস্থ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    প্রত্যাগম m * ((n + m - 1) / m);
}

স্থির অন্তঃস্থ size_t hex_smin(size_t a, size_t b) {
    প্রত্যাগম a < b ? a : b;
}

স্থির অন্তঃস্থ size_t hex_smax(size_t a, size_t b) {
    প্রত্যাগম a > b ? a : b;
}

স্থির অন্তঃস্থ শূন্য hex_swap_ptr(শূন্য ** p1, শূন্য ** p2) {
    শূন্য * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

স্থির অন্তঃস্থ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    যদি (a != 0 && b > SIZE_MAX / a) প্রত্যাগম true;
    *out = a * b;
    প্রত্যাগম false;
}

স্থির অন্তঃস্থ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    যদি (a > SIZE_MAX - b) প্রত্যাগম true;
    *out = a + b;
    প্রত্যাগম false;
}

স্থির অন্তঃস্থ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    যাবত্ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    প্রত্যাগম a;
}

#endif // HEX_COMMON_H
