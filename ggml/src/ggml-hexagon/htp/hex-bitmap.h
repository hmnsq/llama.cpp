#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

स्थिर अन्तःस्थ शून्य bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

स्थिर अन्तःस्थ शून्य bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

स्थिर अन्तःस्थ bool bitmap_test(ध्रुव uint32_t * bitmap, uint32_t idx) {
    प्रत्यागम (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

स्थिर अन्तःस्थ शून्य bitmap_reset(uint32_t * bitmap, माप_प्रकार size_in_bits) {
    स_रखो(bitmap, 0, ((size_in_bits + 31) / 32) * परिमाण(uint32_t));
}

#endif // HEX_BITMAP_H
