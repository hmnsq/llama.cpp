#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <string.h>

સ્થિર અંતઃરેખ શૂન્ય bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

સ્થિર અંતઃરેખ શૂન્ય bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

સ્થિર અંતઃરેખ bool bitmap_test(અચલ uint32_t * bitmap, uint32_t idx) {
    પાછા (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

સ્થિર અંતઃરેખ શૂન્ય bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * કદ(uint32_t));
}

#endif // HEX_BITMAP_H
