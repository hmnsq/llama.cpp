#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool bitmap_test(ಧ್ರುವ uint32_t * bitmap, uint32_t idx) {
    ಪ್ರತ್ಯಾಗಮ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * ಪರಿಮಾಣ(uint32_t));
}

#endif // HEX_BITMAP_H
