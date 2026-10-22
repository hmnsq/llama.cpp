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
    إذا (x <= 1) { إرجاع 1; }
    صحيح p = 2;
    x--;
    بينما (x >>= 1) { p <<= 1; }
    إرجاع p;
}

ساكن مضمن size_t hmx_ceil_div(size_t رقم, size_t den) {
    إرجاع (رقم + den - 1) / den;
}

ساكن مضمن int32_t hex_is_aligned(ثابت فراغ * addr, uint32_t align) {
    إرجاع ((size_t) addr & (align - 1)) == 0;
}

ساكن مضمن size_t hex_align_up(size_t v, size_t align) {
    إرجاع hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_down(size_t v, size_t align) {
    إرجاع (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t يسار_off  = (size_t) addr & (chunk_size - 1);
    uint32_t يمين_off = يسار_off + n;
    إرجاع يمين_off <= chunk_size;
}

ساكن مضمن uint32_t hex_round_up(uint32_t n, uint32_t m) {
    إرجاع m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    إرجاع a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    إرجاع a > b ? a : b;
}

ساكن مضمن فراغ hex_swap_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    إذا (a != 0 && b > SIZE_MAX / a) إرجاع true;
    *out = a * b;
    إرجاع false;
}

ساكن مضمن bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    إذا (a > SIZE_MAX - b) إرجاع true;
    *out = a + b;
    إرجاع false;
}

ساكن مضمن uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    بينما (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    إرجاع a;
}

#endif // HEX_COMMON_H
