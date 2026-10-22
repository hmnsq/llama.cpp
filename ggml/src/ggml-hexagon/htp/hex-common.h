#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t hex_ceil_pow2(uint32_t x) {
    ಯದಿ (x <= 1) { ಪ್ರತ್ಯಾಗಮ 1; }
    ಪೂರ್ಣಾಙ್ಕ p = 2;
    x--;
    ಯಾವತ್ (x >>= 1) { p <<= 1; }
    ಪ್ರತ್ಯಾಗಮ p;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t hmx_ceil_div(size_t ಅಂಕೆ, size_t den) {
    ಪ್ರತ್ಯಾಗಮ (ಅಂಕೆ + den - 1) / den;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int32_t hex_is_aligned(ಧ್ರುವ ಶೂನ್ಯ * addr, uint32_t align) {
    ಪ್ರತ್ಯಾಗಮ ((size_t) addr & (align - 1)) == 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t hex_align_up(size_t v, size_t align) {
    ಪ್ರತ್ಯಾಗಮ hmx_ceil_div(v, align) * align;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t hex_align_down(size_t v, size_t align) {
    ಪ್ರತ್ಯಾಗಮ (v / align) * align;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int32_t hex_is_one_chunk(ಶೂನ್ಯ * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t ಎಡ_off  = (size_t) addr & (chunk_size - 1);
    uint32_t ಬಲ_off = ಎಡ_off + n;
    ಪ್ರತ್ಯಾಗಮ ಬಲ_off <= chunk_size;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    ಪ್ರತ್ಯಾಗಮ m * ((n + m - 1) / m);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t hex_smin(size_t a, size_t b) {
    ಪ್ರತ್ಯಾಗಮ a < b ? a : b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t hex_smax(size_t a, size_t b) {
    ಪ್ರತ್ಯಾಗಮ a > b ? a : b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_swap_ptr(ಶೂನ್ಯ ** p1, ಶೂನ್ಯ ** p2) {
    ಶೂನ್ಯ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    ಯದಿ (a != 0 && b > SIZE_MAX / a) ಪ್ರತ್ಯಾಗಮ true;
    *out = a * b;
    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    ಯದಿ (a > SIZE_MAX - b) ಪ್ರತ್ಯಾಗಮ true;
    *out = a + b;
    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ಯಾವತ್ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ಪ್ರತ್ಯಾಗಮ a;
}

#endif // HEX_COMMON_H
