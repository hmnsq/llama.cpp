#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

ساكن مضمن uint32_t hex_ceil_pow2(uint32_t x) {
    ܐܢ (x <= 1) { ܦܢܐ 1; }
    صحيح p = 2;
    x--;
    ܟܡܐ (x >>= 1) { p <<= 1; }
    ܦܢܐ p;
}

ساكن مضمن size_t hmx_ceil_div(size_t num, size_t den) {
    ܦܢܐ (num + den - 1) / den;
}

ساكن مضمن int32_t hex_is_aligned(ثابت فراغ * addr, uint32_t align) {
    ܦܢܐ ((size_t) addr & (align - 1)) == 0;
}

ساكن مضمن size_t hex_align_up(size_t v, size_t align) {
    ܦܢܐ hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_down(size_t v, size_t align) {
    ܦܢܐ (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    ܦܢܐ right_off <= chunk_size;
}

ساكن مضمن uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ܦܢܐ m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    ܦܢܐ a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    ܦܢܐ a > b ? a : b;
}

ساكن مضمن فراغ hex_swap_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ܐܢ (a != 0 && b > SIZE_MAX / a) ܦܢܐ true;
    *out = a * b;
    ܦܢܐ false;
}

ساكن مضمن bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ܐܢ (a > SIZE_MAX - b) ܦܢܐ true;
    *out = a + b;
    ܦܢܐ false;
}

ساكن مضمن uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ܟܡܐ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ܦܢܐ a;
}

#endif // HEX_COMMON_H
