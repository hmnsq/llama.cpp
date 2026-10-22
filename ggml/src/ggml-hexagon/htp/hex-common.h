#که_نه_تعریف_شوی HEX_COMMON_H
#تعریف_کړه HEX_COMMON_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#که_نه_تعریف_شوی SIZE_MAX
#تعریف_کړه SIZE_MAX ((size_t)-1)
#د_که_پای

#که_نه_تعریف_شوی MAX
#تعریف_کړه MAX(a, b) ((a) > (b) ? (a) : (b))
#د_که_پای

#که_نه_تعریف_شوی MIN
#تعریف_کړه MIN(a, b) ((a) < (b) ? (a) : (b))
#د_که_پای

ساكن مضمن uint32_t hex_ceil_pow2(uint32_t x) {
    إذا (x <= 1) { أرجع 1; }
    صحيح p = 2;
    x--;
    طالما (x >>= 1) { p <<= 1; }
    أرجع p;
}

ساكن مضمن size_t hmx_ceil_div(size_t عدد, size_t den) {
    أرجع (عدد + den - 1) / den;
}

ساكن مضمن int32_t hex_is_aligned(ثابت فراغ * addr, uint32_t align) {
    أرجع ((size_t) addr & (align - 1)) == 0;
}

ساكن مضمن size_t hex_align_پورته(size_t v, size_t align) {
    أرجع hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_ښکته(size_t v, size_t align) {
    أرجع (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_اندازه) {
    uint32_t کیڼ_off  = (size_t) addr & (chunk_اندازه - 1);
    uint32_t ښی_off = کیڼ_off + n;
    أرجع ښی_off <= chunk_اندازه;
}

ساكن مضمن uint32_t hex_پړاو_پورته(uint32_t n, uint32_t m) {
    أرجع m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    أرجع a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    أرجع a > b ? a : b;
}

ساكن مضمن فراغ hex_بدلون_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_ډکېدنه(size_t a, size_t b, size_t *بهر) {
    إذا (a != 0 && b > SIZE_MAX / a) أرجع true;
    *بهر = a * b;
    أرجع false;
}

ساكن مضمن bool hex_جمع_کړه_ډکېدنه(size_t a, size_t b, size_t *بهر) {
    إذا (a > SIZE_MAX - b) أرجع true;
    *بهر = a + b;
    أرجع false;
}

ساكن مضمن uint32_t hex_لوی_ګډ_وېشونکی_u32(uint32_t a, uint32_t b) {
    طالما (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    أرجع a;
}

#د_که_پای // HEX_COMMON_H
