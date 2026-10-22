#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <string.h>

நிலையான உள்வரி வெறுமை bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

நிலையான உள்வரி வெறுமை bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

நிலையான உள்வரி bool bitmap_test(மாறிலி uint32_t * bitmap, uint32_t idx) {
    திரும்பு (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

நிலையான உள்வரி வெறுமை bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * அளவு(uint32_t));
}

#endif // HEX_BITMAP_H
