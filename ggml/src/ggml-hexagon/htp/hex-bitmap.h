#اگر_تعریف_نشده HEX_BITMAP_H
#تعریف_کن HEX_BITMAP_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>

ساكن مضمن فراغ bitmap_مجموعه(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساكن مضمن فراغ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساكن مضمن bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    أرجع (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساكن مضمن فراغ bitmap_reset(uint32_t * bitmap, size_t اندازه_درون_بیتها) {
    memset(bitmap, 0, ((اندازه_درون_بیتها + 31) / 32) * حجم(uint32_t));
}

#پایان_اگر // HEX_BITMAP_H
