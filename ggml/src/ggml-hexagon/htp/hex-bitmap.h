#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool bitmap_test(ଅଚଳ uint32_t * bitmap, uint32_t idx) {
    ଫେରନ୍ତୁ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * ଆକାର(uint32_t));
}

#endif // HEX_BITMAP_H
