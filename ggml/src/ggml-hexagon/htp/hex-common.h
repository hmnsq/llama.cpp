#جے_بنا_تعریف HEX_COMMON_H
#تعریف_کرو HEX_COMMON_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#جے_بنا_تعریف SIZE_MAX
#تعریف_کرو SIZE_MAX ((size_t)-1)
#جے_انت

#جے_بنا_تعریف MAX
#تعریف_کرو MAX(a, b) ((a) > (b) ? (a) : (b))
#جے_انت

#جے_بنا_تعریف MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#جے_انت

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

ساكن مضمن size_t hex_align_اتے(size_t v, size_t align) {
    أرجع hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_تھلے(size_t v, size_t align) {
    أرجع (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_ناپ) {
    uint32_t بایاں_off  = (size_t) addr & (chunk_ناپ - 1);
    uint32_t دایاں_off = بایاں_off + n;
    أرجع دایاں_off <= chunk_ناپ;
}

ساكن مضمن uint32_t hex_گیڑا_اتے(uint32_t n, uint32_t m) {
    أرجع m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    أرجع a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    أرجع a > b ? a : b;
}

ساكن مضمن فراغ hex_وٹاندرا_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_حد_توں_ودھ(size_t a, size_t b, size_t *باہرلا) {
    إذا (a != 0 && b > SIZE_MAX / a) أرجع true;
    *باہرلا = a * b;
    أرجع false;
}

ساكن مضمن bool hex_جوڑو_حد_توں_ودھ(size_t a, size_t b, size_t *باہرلا) {
    إذا (a > SIZE_MAX - b) أرجع true;
    *باہرلا = a + b;
    أرجع false;
}

ساكن مضمن uint32_t hex_عاد_اعظم_u32(uint32_t a, uint32_t b) {
    طالما (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    أرجع a;
}

#جے_انت // HEX_COMMON_H
