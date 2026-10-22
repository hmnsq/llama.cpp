#यदि_अपरिभाषितम् HEX_COMMON_H
#परिभाषय HEX_COMMON_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#यदि_अपरिभाषितम् SIZE_MAX
#परिभाषय SIZE_MAX ((size_t)-1)
#यदि_अन्तः

#यदि_अपरिभाषितम् MAX
#परिभाषय MAX(a, b) ((a) > (b) ? (a) : (b))
#यदि_अन्तः

#यदि_अपरिभाषितम् MIN
#परिभाषय MIN(a, b) ((a) < (b) ? (a) : (b))
#यदि_अन्तः

स्थिर अन्तःस्थ uint32_t hex_ceil_pow2(uint32_t x) {
    यदि (x <= 1) { प्रत्यागम 1; }
    पूर्णाङ्क p = 2;
    x--;
    यावत् (x >>= 1) { p <<= 1; }
    प्रत्यागम p;
}

स्थिर अन्तःस्थ size_t hmx_ceil_div(size_t अङ्कः, size_t den) {
    प्रत्यागम (अङ्कः + den - 1) / den;
}

स्थिर अन्तःस्थ int32_t hex_is_aligned(ध्रुव शून्य * addr, uint32_t align) {
    प्रत्यागम ((size_t) addr & (align - 1)) == 0;
}

स्थिर अन्तःस्थ size_t hex_align_ऊर्ध्वम्(size_t v, size_t align) {
    प्रत्यागम hmx_ceil_div(v, align) * align;
}

स्थिर अन्तःस्थ size_t hex_align_अधस्तात्(size_t v, size_t align) {
    प्रत्यागम (v / align) * align;
}

स्थिर अन्तःस्थ int32_t hex_is_one_chunk(शून्य * addr, uint32_t n, uint32_t chunk_आकार) {
    uint32_t वामः_off  = (size_t) addr & (chunk_आकार - 1);
    uint32_t दक्षिणः_off = वामः_off + n;
    प्रत्यागम दक्षिणः_off <= chunk_आकार;
}

स्थिर अन्तःस्थ uint32_t hex_आवर्तनम्_ऊर्ध्वम्(uint32_t n, uint32_t m) {
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

स्थिर अन्तःस्थ bool hex_mul_अतिप्रवाहः(size_t a, size_t b, size_t *बहिर्गमनम्) {
    यदि (a != 0 && b > SIZE_MAX / a) प्रत्यागम true;
    *बहिर्गमनम् = a * b;
    प्रत्यागम false;
}

स्थिर अन्तःस्थ bool hex_सङ्कलय_अतिप्रवाहः(size_t a, size_t b, size_t *बहिर्गमनम्) {
    यदि (a > SIZE_MAX - b) प्रत्यागम true;
    *बहिर्गमनम् = a + b;
    प्रत्यागम false;
}

स्थिर अन्तःस्थ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    यावत् (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    प्रत्यागम a;
}

#यदि_अन्तः // HEX_COMMON_H
