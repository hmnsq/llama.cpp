#اگر_تعریف_نشده HEX_COMMON_H
#تعریف_کن HEX_COMMON_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#اگر_تعریف_نشده SIZE_MAX
#تعریف_کن SIZE_MAX ((size_t)-1)
#پایان_اگر

#اگر_تعریف_نشده MAX
#تعریف_کن MAX(a, b) ((a) > (b) ? (a) : (b))
#پایان_اگر

#اگر_تعریف_نشده MIN
#تعریف_کن MIN(a, b) ((a) < (b) ? (a) : (b))
#پایان_اگر

ساكن مضمن uint32_t hex_ceil_pow2(uint32_t x) {
    إذا (x <= 1) { أرجع 1; }
    صحيح p = 2;
    x--;
    طالما (x >>= 1) { p <<= 1; }
    أرجع p;
}

ساكن مضمن size_t hmx_ceil_div(size_t رقم, size_t den) {
    أرجع (رقم + den - 1) / den;
}

ساكن مضمن int32_t hex_is_aligned(ثابت فراغ * addr, uint32_t align) {
    أرجع ((size_t) addr & (align - 1)) == 0;
}

ساكن مضمن size_t hex_align_به_بالا(size_t v, size_t align) {
    أرجع hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_پایین(size_t v, size_t align) {
    أرجع (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_اندازه) {
    uint32_t چپ_off  = (size_t) addr & (chunk_اندازه - 1);
    uint32_t راست_off = چپ_off + n;
    أرجع راست_off <= chunk_اندازه;
}

ساكن مضمن uint32_t hex_دور_به_بالا(uint32_t n, uint32_t m) {
    أرجع m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    أرجع a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    أرجع a > b ? a : b;
}

ساكن مضمن فراغ hex_جابجا_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_سرریز(size_t a, size_t b, size_t *بیرون) {
    إذا (a != 0 && b > SIZE_MAX / a) أرجع true;
    *بیرون = a * b;
    أرجع false;
}

ساكن مضمن bool hex_جمع_کن_سرریز(size_t a, size_t b, size_t *بیرون) {
    إذا (a > SIZE_MAX - b) أرجع true;
    *بیرون = a + b;
    أرجع false;
}

ساكن مضمن uint32_t hex_بزرگترین_مقسوم_علیه_u32(uint32_t a, uint32_t b) {
    طالما (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    أرجع a;
}

#پایان_اگر // HEX_COMMON_H
