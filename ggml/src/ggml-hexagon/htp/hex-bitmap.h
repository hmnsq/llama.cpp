#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>

ساكن مضمن فراغ bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساكن مضمن فراغ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساكن مضمن bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    ܦܢܐ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساكن مضمن فراغ bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * حجم(uint32_t));
}

#endif // HEX_BITMAP_H
