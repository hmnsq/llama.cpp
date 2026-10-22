#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_BITMAP_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_BITMAP_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bitmap_ਸੈੱਟ(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool bitmap_test(ਅਚਲ uint32_t * bitmap, uint32_t idx) {
    ਵਾਪਸ (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bitmap_reset(uint32_t * bitmap, size_t ਮਾਪ_ਵਿੱਚ_ਬਿੱਟ) {
    memset(bitmap, 0, ((ਮਾਪ_ਵਿੱਚ_ਬਿੱਟ + 31) / 32) * ਆਕਾਰ(uint32_t));
}

#ਜੇ_ਅੰਤ // HEX_BITMAP_H
