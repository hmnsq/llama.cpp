#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_COMMON_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_COMMON_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SIZE_MAX
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SIZE_MAX ((size_t)-1)
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MAX
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX(a, b) ((a) > (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MIN
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MIN(a, b) ((a) < (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t hex_ceil_pow2(uint32_t x) {
    ਜੇ (x <= 1) { ਵਾਪਸ 1; }
    ਪੂਰਨ p = 2;
    x--;
    ਜਦੋਂ (x >>= 1) { p <<= 1; }
    ਵਾਪਸ p;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hmx_ceil_div(size_t ਅੰਕ, size_t den) {
    ਵਾਪਸ (ਅੰਕ + den - 1) / den;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hex_is_aligned(ਅਚਲ ਖਾਲੀ * addr, uint32_t align) {
    ਵਾਪਸ ((size_t) addr & (align - 1)) == 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hex_align_ਉੱਪਰ(size_t v, size_t align) {
    ਵਾਪਸ hmx_ceil_div(v, align) * align;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hex_align_ਥੱਲੇ(size_t v, size_t align) {
    ਵਾਪਸ (v / align) * align;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hex_is_one_chunk(ਖਾਲੀ * addr, uint32_t n, uint32_t chunk_ਮਾਪ) {
    uint32_t ਖੱਬਾ_off  = (size_t) addr & (chunk_ਮਾਪ - 1);
    uint32_t ਸੱਜਾ_off = ਖੱਬਾ_off + n;
    ਵਾਪਸ ਸੱਜਾ_off <= chunk_ਮਾਪ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t hex_ਗੇੜਾ_ਉੱਪਰ(uint32_t n, uint32_t m) {
    ਵਾਪਸ m * ((n + m - 1) / m);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hex_smin(size_t a, size_t b) {
    ਵਾਪਸ a < b ? a : b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hex_smax(size_t a, size_t b) {
    ਵਾਪਸ a > b ? a : b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_ਅਦਲਾ_ਬਦਲੀ_ptr(ਖਾਲੀ ** p1, ਖਾਲੀ ** p2) {
    ਖਾਲੀ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool hex_mul_ਉਛਾਲ(size_t a, size_t b, size_t *ਬਾਹਰਲਾ) {
    ਜੇ (a != 0 && b > SIZE_MAX / a) ਵਾਪਸ true;
    *ਬਾਹਰਲਾ = a * b;
    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool hex_ਜੋੜੋ_ਉਛਾਲ(size_t a, size_t b, size_t *ਬਾਹਰਲਾ) {
    ਜੇ (a > SIZE_MAX - b) ਵਾਪਸ true;
    *ਬਾਹਰਲਾ = a + b;
    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(uint32_t a, uint32_t b) {
    ਜਦੋਂ (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    ਵਾਪਸ a;
}

#ਜੇ_ਅੰਤ // HEX_COMMON_H
