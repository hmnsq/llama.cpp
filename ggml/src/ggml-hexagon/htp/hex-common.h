#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

நிலையான உள்வரி uint32_t hex_ceil_pow2(uint32_t x) {
    எனில் (x <= 1) { திரும்பு 1; }
    முழு p = 2;
    x--;
    போது (x >>= 1) { p <<= 1; }
    திரும்பு p;
}

நிலையான உள்வரி size_t hmx_ceil_div(size_t இலக்கம், size_t den) {
    திரும்பு (இலக்கம் + den - 1) / den;
}

நிலையான உள்வரி int32_t hex_is_aligned(மாறிலி வெறுமை * addr, uint32_t align) {
    திரும்பு ((size_t) addr & (align - 1)) == 0;
}

நிலையான உள்வரி size_t hex_align_up(size_t v, size_t align) {
    திரும்பு hmx_ceil_div(v, align) * align;
}

நிலையான உள்வரி size_t hex_align_down(size_t v, size_t align) {
    திரும்பு (v / align) * align;
}

நிலையான உள்வரி int32_t hex_is_one_chunk(வெறுமை * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t இடது_off  = (size_t) addr & (chunk_size - 1);
    uint32_t வலது_off = இடது_off + n;
    திரும்பு வலது_off <= chunk_size;
}

நிலையான உள்வரி uint32_t hex_round_up(uint32_t n, uint32_t m) {
    திரும்பு m * ((n + m - 1) / m);
}

நிலையான உள்வரி size_t hex_smin(size_t a, size_t b) {
    திரும்பு a < b ? a : b;
}

நிலையான உள்வரி size_t hex_smax(size_t a, size_t b) {
    திரும்பு a > b ? a : b;
}

நிலையான உள்வரி வெறுமை hex_swap_ptr(வெறுமை ** p1, வெறுமை ** p2) {
    வெறுமை * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

நிலையான உள்வரி bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    எனில் (a != 0 && b > SIZE_MAX / a) திரும்பு true;
    *out = a * b;
    திரும்பு false;
}

நிலையான உள்வரி bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    எனில் (a > SIZE_MAX - b) திரும்பு true;
    *out = a + b;
    திரும்பு false;
}

நிலையான உள்வரி uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    போது (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    திரும்பு a;
}

#endif // HEX_COMMON_H
