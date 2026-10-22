#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

സ്ഥിതം വരിയിൽ uint32_t hex_ceil_pow2(uint32_t x) {
    എങ്കിൽ (x <= 1) { തിരികെ 1; }
    പൂർണ്ണസംഖ്യ p = 2;
    x--;
    ആകുമ്പോൾ (x >>= 1) { p <<= 1; }
    തിരികെ p;
}

സ്ഥിതം വരിയിൽ size_t hmx_ceil_div(size_t അക്കം, size_t den) {
    തിരികെ (അക്കം + den - 1) / den;
}

സ്ഥിതം വരിയിൽ int32_t hex_is_aligned(സ്ഥിരം ശൂന്യം * addr, uint32_t align) {
    തിരികെ ((size_t) addr & (align - 1)) == 0;
}

സ്ഥിതം വരിയിൽ size_t hex_align_up(size_t v, size_t align) {
    തിരികെ hmx_ceil_div(v, align) * align;
}

സ്ഥിതം വരിയിൽ size_t hex_align_down(size_t v, size_t align) {
    തിരികെ (v / align) * align;
}

സ്ഥിതം വരിയിൽ int32_t hex_is_one_chunk(ശൂന്യം * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t ഇടത്_off  = (size_t) addr & (chunk_size - 1);
    uint32_t വലത്_off = ഇടത്_off + n;
    തിരികെ വലത്_off <= chunk_size;
}

സ്ഥിതം വരിയിൽ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    തിരികെ m * ((n + m - 1) / m);
}

സ്ഥിതം വരിയിൽ size_t hex_smin(size_t a, size_t b) {
    തിരികെ a < b ? a : b;
}

സ്ഥിതം വരിയിൽ size_t hex_smax(size_t a, size_t b) {
    തിരികെ a > b ? a : b;
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_swap_ptr(ശൂന്യം ** p1, ശൂന്യം ** p2) {
    ശൂന്യം * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

സ്ഥിതം വരിയിൽ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    എങ്കിൽ (a != 0 && b > SIZE_MAX / a) തിരികെ true;
    *out = a * b;
    തിരികെ false;
}

സ്ഥിതം വരിയിൽ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    എങ്കിൽ (a > SIZE_MAX - b) തിരികെ true;
    *out = a + b;
    തിരികെ false;
}

സ്ഥിതം വരിയിൽ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    ആകുമ്പോൾ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    തിരികെ a;
}

#endif // HEX_COMMON_H
