#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

स्थिर अन्तःस्थ uint32_t hex_ceil_pow2(uint32_t x) {
    जइ (x <= 1) { प्रत्यागम 1; }
    पूर्णाङ्क p = 2;
    x--;
    जाव (x >>= 1) { p <<= 1; }
    प्रत्यागम p;
}

स्थिर अन्तःस्थ size_t hmx_ceil_div(size_t अङ्कः, size_t den) {
    प्रत्यागम (अङ्कः + den - 1) / den;
}

स्थिर अन्तःस्थ int32_t hex_is_aligned(ध्रुव शून्य * addr, uint32_t align) {
    प्रत्यागम ((size_t) addr & (align - 1)) == 0;
}

स्थिर अन्तःस्थ size_t hex_align_up(size_t v, size_t align) {
    प्रत्यागम hmx_ceil_div(v, align) * align;
}

स्थिर अन्तःस्थ size_t hex_align_down(size_t v, size_t align) {
    प्रत्यागम (v / align) * align;
}

स्थिर अन्तःस्थ int32_t hex_is_one_chunk(शून्य * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t वामः_off  = (size_t) addr & (chunk_size - 1);
    uint32_t दक्षिणः_off = वामः_off + n;
    प्रत्यागम दक्षिणः_off <= chunk_size;
}

स्थिर अन्तःस्थ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    प्रत्यागम m * ((n + m - 1) / m);
}

स्थिर अन्तःस्थ size_t hex_smin(size_t a, size_t b) {
    प्रत्यागम a < b ? a : b;
}

स्थिर अन्तःस्थ size_t hex_smax(size_t a, size_t b) {
    प्रत्यागम a > b ? a : b;
}

स्थिर अन्तःस्थ शून्य hex_swap_ptr(शून्य ** p1, शून्य ** p2) {
    शून्य * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

स्थिर अन्तःस्थ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    जइ (a != 0 && b > SIZE_MAX / a) प्रत्यागम true;
    *out = a * b;
    प्रत्यागम false;
}

स्थिर अन्तःस्थ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    जइ (a > SIZE_MAX - b) प्रत्यागम true;
    *out = a + b;
    प्रत्यागम false;
}

स्थिर अन्तःस्थ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    जाव (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    प्रत्यागम a;
}

#endif // HEX_COMMON_H
