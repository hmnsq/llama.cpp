#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_COMMON_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_COMMON_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SIZE_MAX
#ವ್ಯಾಖ್ಯಾನಿಸು SIZE_MAX ((size_t)-1)
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MAX
#ವ್ಯಾಖ್ಯಾನಿಸು MAX(a, b) ((a) > (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MIN
#ವ್ಯಾಖ್ಯಾನಿಸು MIN(a, b) ((a) < (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t hex_ceil_pow2(uint32_t x) {
    ಆದರೆ (x <= 1) { ಹಿಂತಿರುಗಿ 1; }
    ಪೂರ್ಣಾಂಕ p = 2;
    x--;
    ಆಗ (x >>= 1) { p <<= 1; }
    ಹಿಂತಿರುಗಿ p;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hmx_ceil_div(size_t ಅಂಕೆ, size_t den) {
    ಹಿಂತಿರುಗಿ (ಅಂಕೆ + den - 1) / den;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hex_is_aligned(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, uint32_t align) {
    ಹಿಂತಿರುಗಿ ((size_t) addr & (align - 1)) == 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hex_align_ಮೇಲೆ(size_t v, size_t align) {
    ಹಿಂತಿರುಗಿ hmx_ceil_div(v, align) * align;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hex_align_ಕೆಳಕ್ಕೆ(size_t v, size_t align) {
    ಹಿಂತಿರುಗಿ (v / align) * align;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hex_is_one_chunk(ಶೂನ್ಯ * addr, uint32_t n, uint32_t chunk_ಗಾತ್ರ) {
    uint32_t ಎಡ_off  = (size_t) addr & (chunk_ಗಾತ್ರ - 1);
    uint32_t ಬಲ_off = ಎಡ_off + n;
    ಹಿಂತಿರುಗಿ ಬಲ_off <= chunk_ಗಾತ್ರ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t hex_ಸುತ್ತು_ಮೇಲೆ(uint32_t n, uint32_t m) {
    ಹಿಂತಿರುಗಿ m * ((n + m - 1) / m);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hex_smin(size_t a, size_t b) {
    ಹಿಂತಿರುಗಿ a < b ? a : b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hex_smax(size_t a, size_t b) {
    ಹಿಂತಿರುಗಿ a > b ? a : b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_ಅದಲುಬದಲು_ptr(ಶೂನ್ಯ ** p1, ಶೂನ್ಯ ** p2) {
    ಶೂನ್ಯ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool hex_mul_ಅತಿಪ್ರವಾಹ(size_t a, size_t b, size_t *ಹೊರಹೋಗು) {
    ಆದರೆ (a != 0 && b > SIZE_MAX / a) ಹಿಂತಿರುಗಿ true;
    *ಹೊರಹೋಗು = a * b;
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool hex_ಕೂಡಿಸು_ಅತಿಪ್ರವಾಹ(size_t a, size_t b, size_t *ಹೊರಹೋಗು) {
    ಆದರೆ (a > SIZE_MAX - b) ಹಿಂತಿರುಗಿ true;
    *ಹೊರಹೋಗು = a + b;
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(uint32_t a, uint32_t b) {
    ಆಗ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ಹಿಂತಿರುಗಿ a;
}

#ಆದರೆ_ಅಂತ್ಯ // HEX_COMMON_H
