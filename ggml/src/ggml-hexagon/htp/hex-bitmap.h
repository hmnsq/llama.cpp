#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <string.h>

സ്ഥിതം വരിയിൽ ശൂന്യം bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

സ്ഥിതം വരിയിൽ ശൂന്യം bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

സ്ഥിതം വരിയിൽ bool bitmap_test(സ്ഥിരം uint32_t * bitmap, uint32_t idx) {
    തിരികെ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

സ്ഥിതം വരിയിൽ ശൂന്യം bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * വലുപ്പം(uint32_t));
}

#endif // HEX_BITMAP_H
