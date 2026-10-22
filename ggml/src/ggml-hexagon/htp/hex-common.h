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

स्थिर अन्तःस्थ uint32_t hex_ceil_pow2(uint32_t x) {
    यदि (x <= 1) { फर्काऊ 1; }
    पूर्णाङ्क p = 2;
    x--;
    जबसम्म (x >>= 1) { p <<= 1; }
    फर्काऊ p;
}

स्थिर अन्तःस्थ माप_प्रकार hmx_ceil_div(माप_प्रकार अङ्क, माप_प्रकार den) {
    फर्काऊ (अङ्क + den - 1) / den;
}

स्थिर अन्तःस्थ int32_t hex_is_aligned(अचर शून्य * addr, uint32_t align) {
    फर्काऊ ((माप_प्रकार) addr & (align - 1)) == 0;
}

स्थिर अन्तःस्थ माप_प्रकार hex_align_up(माप_प्रकार v, माप_प्रकार align) {
    फर्काऊ hmx_ceil_div(v, align) * align;
}

स्थिर अन्तःस्थ माप_प्रकार hex_align_down(माप_प्रकार v, माप_प्रकार align) {
    फर्काऊ (v / align) * align;
}

स्थिर अन्तःस्थ int32_t hex_is_one_chunk(शून्य * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t बायाँ_off  = (माप_प्रकार) addr & (chunk_size - 1);
    uint32_t दायाँ_off = बायाँ_off + n;
    फर्काऊ दायाँ_off <= chunk_size;
}

स्थिर अन्तःस्थ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    फर्काऊ m * ((n + m - 1) / m);
}

स्थिर अन्तःस्थ माप_प्रकार hex_smin(माप_प्रकार a, माप_प्रकार b) {
    फर्काऊ a < b ? a : b;
}

स्थिर अन्तःस्थ माप_प्रकार hex_smax(माप_प्रकार a, माप_प्रकार b) {
    फर्काऊ a > b ? a : b;
}

स्थिर अन्तःस्थ शून्य hex_swap_ptr(शून्य ** p1, शून्य ** p2) {
    शून्य * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

स्थिर अन्तःस्थ bool hex_mul_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    यदि (a != 0 && b > SIZE_MAX / a) फर्काऊ true;
    *out = a * b;
    फर्काऊ false;
}

स्थिर अन्तःस्थ bool hex_add_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    यदि (a > SIZE_MAX - b) फर्काऊ true;
    *out = a + b;
    फर्काऊ false;
}

स्थिर अन्तःस्थ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    जबसम्म (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    फर्काऊ a;
}

#endif // HEX_COMMON_H
