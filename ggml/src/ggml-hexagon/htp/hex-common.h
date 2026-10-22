#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

સ્થિર અંતઃરેખ uint32_t hex_ceil_pow2(uint32_t x) {
    જો (x <= 1) { પાછા 1; }
    પૂર્ણાંક p = 2;
    x--;
    જ્યારે (x >>= 1) { p <<= 1; }
    પાછા p;
}

સ્થિર અંતઃરેખ size_t hmx_ceil_div(size_t અંક, size_t den) {
    પાછા (અંક + den - 1) / den;
}

સ્થિર અંતઃરેખ int32_t hex_is_aligned(અચલ શૂન્ય * addr, uint32_t align) {
    પાછા ((size_t) addr & (align - 1)) == 0;
}

સ્થિર અંતઃરેખ size_t hex_align_up(size_t v, size_t align) {
    પાછા hmx_ceil_div(v, align) * align;
}

સ્થિર અંતઃરેખ size_t hex_align_down(size_t v, size_t align) {
    પાછા (v / align) * align;
}

સ્થિર અંતઃરેખ int32_t hex_is_one_chunk(શૂન્ય * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t ડાબું_off  = (size_t) addr & (chunk_size - 1);
    uint32_t જમણું_off = ડાબું_off + n;
    પાછા જમણું_off <= chunk_size;
}

સ્થિર અંતઃરેખ uint32_t hex_round_up(uint32_t n, uint32_t m) {
    પાછા m * ((n + m - 1) / m);
}

સ્થિર અંતઃરેખ size_t hex_smin(size_t a, size_t b) {
    પાછા a < b ? a : b;
}

સ્થિર અંતઃરેખ size_t hex_smax(size_t a, size_t b) {
    પાછા a > b ? a : b;
}

સ્થિર અંતઃરેખ શૂન્ય hex_swap_ptr(શૂન્ય ** p1, શૂન્ય ** p2) {
    શૂન્ય * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

સ્થિર અંતઃરેખ bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    જો (a != 0 && b > SIZE_MAX / a) પાછા true;
    *out = a * b;
    પાછા false;
}

સ્થિર અંતઃરેખ bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    જો (a > SIZE_MAX - b) પાછા true;
    *out = a + b;
    પાછા false;
}

સ્થિર અંતઃરેખ uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    જ્યારે (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    પાછા a;
}

#endif // HEX_COMMON_H
