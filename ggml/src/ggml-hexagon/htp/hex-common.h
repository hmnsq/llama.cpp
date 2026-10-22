#اگر_غیر_تعریف_شدہ HEX_COMMON_H
#تعریف_کرو HEX_COMMON_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>

#اگر_غیر_تعریف_شدہ SIZE_MAX
#تعریف_کرو SIZE_MAX ((size_t)-1)
#اگر_ختم

#اگر_غیر_تعریف_شدہ MAX
#تعریف_کرو MAX(a, b) ((a) > (b) ? (a) : (b))
#اگر_ختم

#اگر_غیر_تعریف_شدہ MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#اگر_ختم

ساکن اندرونی uint32_t hex_ceil_pow2(uint32_t x) {
    اگر (x <= 1) { واپس 1; }
    صحیح p = 2;
    x--;
    جب (x >>= 1) { p <<= 1; }
    واپس p;
}

ساکن اندرونی size_t hmx_ceil_div(size_t عدد, size_t den) {
    واپس (عدد + den - 1) / den;
}

ساکن اندرونی int32_t hex_is_aligned(ثابت خالی * addr, uint32_t align) {
    واپس ((size_t) addr & (align - 1)) == 0;
}

ساکن اندرونی size_t hex_align_اوپر(size_t v, size_t align) {
    واپس hmx_ceil_div(v, align) * align;
}

ساکن اندرونی size_t hex_align_نیچے_کو(size_t v, size_t align) {
    واپس (v / align) * align;
}

ساکن اندرونی int32_t hex_is_one_chunk(خالی * addr, uint32_t n, uint32_t chunk_سائز) {
    uint32_t بایاں_off  = (size_t) addr & (chunk_سائز - 1);
    uint32_t دایاں_off = بایاں_off + n;
    واپس دایاں_off <= chunk_سائز;
}

ساکن اندرونی uint32_t hex_باری_اوپر(uint32_t n, uint32_t m) {
    واپس m * ((n + m - 1) / m);
}

ساکن اندرونی size_t hex_smin(size_t a, size_t b) {
    واپس a < b ? a : b;
}

ساکن اندرونی size_t hex_smax(size_t a, size_t b) {
    واپس a > b ? a : b;
}

ساکن اندرونی خالی hex_تبادلہ_ptr(خالی ** p1, خالی ** p2) {
    خالی * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساکن اندرونی bool hex_mul_حد_سے_تجاوز(size_t a, size_t b, size_t *برآمد) {
    اگر (a != 0 && b > SIZE_MAX / a) واپس true;
    *برآمد = a * b;
    واپس false;
}

ساکن اندرونی bool hex_جوڑو_حد_سے_تجاوز(size_t a, size_t b, size_t *برآمد) {
    اگر (a > SIZE_MAX - b) واپس true;
    *برآمد = a + b;
    واپس false;
}

ساکن اندرونی uint32_t hex_عاد_اعظم_u32(uint32_t a, uint32_t b) {
    جب (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    واپس a;
}

#اگر_ختم // HEX_COMMON_H
