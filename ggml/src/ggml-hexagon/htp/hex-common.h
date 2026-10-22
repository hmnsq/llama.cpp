#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((माप_प्रकार)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

स्थिर ओळीत uint32_t hex_ceil_pow2(uint32_t x) {
    जर (x <= 1) { परत 1; }
    पूर्णांक p = 2;
    x--;
    पर्यंत (x >>= 1) { p <<= 1; }
    परत p;
}

स्थिर ओळीत माप_प्रकार hmx_ceil_div(माप_प्रकार अंक, माप_प्रकार den) {
    परत (अंक + den - 1) / den;
}

स्थिर ओळीत int32_t hex_is_aligned(स्थिरांक शून्य * addr, uint32_t align) {
    परत ((माप_प्रकार) addr & (align - 1)) == 0;
}

स्थिर ओळीत माप_प्रकार hex_align_up(माप_प्रकार v, माप_प्रकार align) {
    परत hmx_ceil_div(v, align) * align;
}

स्थिर ओळीत माप_प्रकार hex_align_down(माप_प्रकार v, माप_प्रकार align) {
    परत (v / align) * align;
}

स्थिर ओळीत int32_t hex_is_one_chunk(शून्य * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t डावा_off  = (माप_प्रकार) addr & (chunk_size - 1);
    uint32_t उजवा_off = डावा_off + n;
    परत उजवा_off <= chunk_size;
}

स्थिर ओळीत uint32_t hex_round_up(uint32_t n, uint32_t m) {
    परत m * ((n + m - 1) / m);
}

स्थिर ओळीत माप_प्रकार hex_smin(माप_प्रकार a, माप_प्रकार b) {
    परत a < b ? a : b;
}

स्थिर ओळीत माप_प्रकार hex_smax(माप_प्रकार a, माप_प्रकार b) {
    परत a > b ? a : b;
}

स्थिर ओळीत शून्य hex_swap_ptr(शून्य ** p1, शून्य ** p2) {
    शून्य * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

स्थिर ओळीत bool hex_mul_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    जर (a != 0 && b > SIZE_MAX / a) परत true;
    *out = a * b;
    परत false;
}

स्थिर ओळीत bool hex_add_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    जर (a > SIZE_MAX - b) परत true;
    *out = a + b;
    परत false;
}

स्थिर ओळीत uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    पर्यंत (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    परत a;
}

#endif // HEX_COMMON_H
