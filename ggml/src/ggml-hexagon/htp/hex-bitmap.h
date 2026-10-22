#که_نه_تعریف_شوی HEX_BITMAP_H
#تعریف_کړه HEX_BITMAP_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>

ساكن مضمن فراغ bitmap_سېټ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساكن مضمن فراغ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساكن مضمن bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    أرجع (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساكن مضمن فراغ bitmap_reset(uint32_t * bitmap, size_t اندازه_دننه_بټونه) {
    memset(bitmap, 0, ((اندازه_دننه_بټونه + 31) / 32) * حجم(uint32_t));
}

#د_که_پای // HEX_BITMAP_H
