#اگر_غیر_تعریف_شدہ HEX_BITMAP_H
#تعریف_کرو HEX_BITMAP_H

#شامل <stdint.h>
#شامل <stdbool.h>
#شامل <string.h>

ساکن اندرونی خالی bitmap_سیٹ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ساکن اندرونی خالی bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ساکن اندرونی bool bitmap_test(ثابت uint32_t * bitmap, uint32_t idx) {
    واپس (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ساکن اندرونی خالی bitmap_reset(uint32_t * bitmap, size_t سائز_اندر_بٹ) {
    memset(bitmap, 0, ((سائز_اندر_بٹ + 31) / 32) * حجم(uint32_t));
}

#اگر_ختم // HEX_BITMAP_H
