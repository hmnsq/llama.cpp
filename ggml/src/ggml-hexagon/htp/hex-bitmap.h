#यदि_अपरिभाषितम् HEX_BITMAP_H
#परिभाषय HEX_BITMAP_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

स्थिर अन्तःस्थ शून्य bitmap_समुच्चयः(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

स्थिर अन्तःस्थ शून्य bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

स्थिर अन्तःस्थ bool bitmap_test(ध्रुव uint32_t * bitmap, uint32_t idx) {
    प्रत्यागम (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

स्थिर अन्तःस्थ शून्य bitmap_reset(uint32_t * bitmap, size_t आकार_अन्तर्_द्वयङ्काः) {
    स्मृतिं_पूरय(bitmap, 0, ((आकार_अन्तर्_द्वयङ्काः + 31) / 32) * परिमाण(uint32_t));
}

#यदि_अन्तः // HEX_BITMAP_H
