#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <string.h>

স্থির অন্তঃস্থ শূন্য bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

স্থির অন্তঃস্থ শূন্য bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

স্থির অন্তঃস্থ bool bitmap_test(ধ্রুৱ uint32_t * bitmap, uint32_t idx) {
    ঘূৰাই (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

স্থির অন্তঃস্থ শূন্য bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * পরিমাণ(uint32_t));
}

#endif // HEX_BITMAP_H
