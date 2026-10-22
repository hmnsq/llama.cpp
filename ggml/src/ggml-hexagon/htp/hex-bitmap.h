#جے_بنا_تعریف HEX_BITMAP_H
#تعریف_کرو HEX_BITMAP_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>

ساكن مضمن فراغ bitmap_سیٹ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساكن مضمن فراغ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساكن مضمن bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    أرجع (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساكن مضمن فراغ bitmap_reset(uint32_t * bitmap, size_t ناپ_وچ_بٹ) {
    memset(bitmap, 0, ((ناپ_وچ_بٹ + 31) / 32) * حجم(uint32_t));
}

#جے_انت // HEX_BITMAP_H
