#अगर_अघोषित HEX_BITMAP_H
#घोषणा HEX_BITMAP_H

#समावेश <stdint.h>
#समावेश <stdbool.h>
#समावेश <string.h>

अटल अंतरभूत व्योम bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

अटल अंतरभूत व्योम bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

अटल अंतरभूत bool bitmap_test(स्थिर uint32_t * bitmap, uint32_t idx) {
    वापस (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

अटल अंतरभूत व्योम bitmap_reset(uint32_t * bitmap, माप_प्रकार आकार_in_bits) {
    स_रखो(bitmap, 0, ((आकार_in_bits + 31) / 32) * माप(uint32_t));
}

#पूर्ण_अगर // HEX_BITMAP_H
