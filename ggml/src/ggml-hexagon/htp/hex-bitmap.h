#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#شامل <stdint.h>
#شامل <stdbool.h>
#شامل <string.h>

ایستا درونخطی تهی bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ایستا درونخطی تهی bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ایستا درونخطی bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    بازگشت (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ایستا درونخطی تهی bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * اندازه(uint32_t));
}

#endif // HEX_BITMAP_H
