#جيڪڏهن_وصف_نه_ٿيل HEX_BITMAP_H
#وصف_ڪريو HEX_BITMAP_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>

ساكن مضمن فراغ bitmap_سيٽ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساكن مضمن فراغ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساكن مضمن bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    أرجع (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساكن مضمن فراغ bitmap_reset(uint32_t * bitmap, size_t سائيز_اندريون_بٽ) {
    memset(bitmap, 0, ((سائيز_اندريون_بٽ + 31) / 32) * حجم(uint32_t));
}

#جيڪڏهن_پورو // HEX_BITMAP_H
