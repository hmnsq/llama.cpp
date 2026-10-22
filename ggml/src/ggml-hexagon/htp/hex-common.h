#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

ایستا درونخطی uint32_t hex_ceil_pow2(uint32_t x) {
    اگر (x <= 1) { بازگشت 1; }
    صحیح p = 2;
    x--;
    وقتی (x >>= 1) { p <<= 1; }
    بازگشت p;
}

ایستا درونخطی size_t hmx_ceil_div(size_t رقم, size_t den) {
    بازگشت (رقم + den - 1) / den;
}

ایستا درونخطی int32_t hex_is_aligned(ثابت تهی * addr, uint32_t align) {
    بازگشت ((size_t) addr & (align - 1)) == 0;
}

ایستا درونخطی size_t hex_align_up(size_t v, size_t align) {
    بازگشت hmx_ceil_div(v, align) * align;
}

ایستا درونخطی size_t hex_align_down(size_t v, size_t align) {
    بازگشت (v / align) * align;
}

ایستا درونخطی int32_t hex_is_one_chunk(تهی * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t چپ_off  = (size_t) addr & (chunk_size - 1);
    uint32_t راست_off = چپ_off + n;
    بازگشت راست_off <= chunk_size;
}

ایستا درونخطی uint32_t hex_round_up(uint32_t n, uint32_t m) {
    بازگشت m * ((n + m - 1) / m);
}

ایستا درونخطی size_t hex_smin(size_t a, size_t b) {
    بازگشت a < b ? a : b;
}

ایستا درونخطی size_t hex_smax(size_t a, size_t b) {
    بازگشت a > b ? a : b;
}

ایستا درونخطی تهی hex_swap_ptr(تهی ** p1, تهی ** p2) {
    تهی * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ایستا درونخطی bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    اگر (a != 0 && b > SIZE_MAX / a) بازگشت true;
    *out = a * b;
    بازگشت false;
}

ایستا درونخطی bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    اگر (a > SIZE_MAX - b) بازگشت true;
    *out = a + b;
    بازگشت false;
}

ایستا درونخطی uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    وقتی (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    بازگشت a;
}

#endif // HEX_COMMON_H
