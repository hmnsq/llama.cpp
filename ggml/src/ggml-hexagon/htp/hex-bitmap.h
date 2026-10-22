#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_BITMAP_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_BITMAP_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bitmap_ಗಣ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool bitmap_test(ಸ್ಥಿರಾಂಕ uint32_t * bitmap, uint32_t idx) {
    ಹಿಂತಿರುಗಿ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bitmap_reset(uint32_t * bitmap, size_t ಗಾತ್ರ_ಒಳ_ಬಿಟ್ಟುಗಳು) {
    memset(bitmap, 0, ((ಗಾತ್ರ_ಒಳ_ಬಿಟ್ಟುಗಳು + 31) / 32) * ಗಾತ್ರ(uint32_t));
}

#ಆದರೆ_ಅಂತ್ಯ // HEX_BITMAP_H
