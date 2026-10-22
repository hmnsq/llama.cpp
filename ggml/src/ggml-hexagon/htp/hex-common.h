#جيڪڏهن_وصف_نه_ٿيل HEX_COMMON_H
#وصف_ڪريو HEX_COMMON_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#جيڪڏهن_وصف_نه_ٿيل SIZE_MAX
#وصف_ڪريو SIZE_MAX ((size_t)-1)
#جيڪڏهن_پورو

#جيڪڏهن_وصف_نه_ٿيل MAX
#وصف_ڪريو MAX(a, b) ((a) > (b) ? (a) : (b))
#جيڪڏهن_پورو

#جيڪڏهن_وصف_نه_ٿيل MIN
#وصف_ڪريو MIN(a, b) ((a) < (b) ? (a) : (b))
#جيڪڏهن_پورو

ساكن مضمن uint32_t hex_ceil_pow2(uint32_t x) {
    إذا (x <= 1) { أرجع 1; }
    صحيح p = 2;
    x--;
    طالما (x >>= 1) { p <<= 1; }
    أرجع p;
}

ساكن مضمن size_t hmx_ceil_div(size_t انگ, size_t den) {
    أرجع (انگ + den - 1) / den;
}

ساكن مضمن int32_t hex_is_aligned(ثابت فراغ * addr, uint32_t align) {
    أرجع ((size_t) addr & (align - 1)) == 0;
}

ساكن مضمن size_t hex_align_مٿي(size_t v, size_t align) {
    أرجع hmx_ceil_div(v, align) * align;
}

ساكن مضمن size_t hex_align_هيٺ_ڏانهن(size_t v, size_t align) {
    أرجع (v / align) * align;
}

ساكن مضمن int32_t hex_is_one_chunk(فراغ * addr, uint32_t n, uint32_t chunk_سائيز) {
    uint32_t کاٻو_off  = (size_t) addr & (chunk_سائيز - 1);
    uint32_t ساڄو_off = کاٻو_off + n;
    أرجع ساڄو_off <= chunk_سائيز;
}

ساكن مضمن uint32_t hex_ڦيرو_مٿي(uint32_t n, uint32_t m) {
    أرجع m * ((n + m - 1) / m);
}

ساكن مضمن size_t hex_smin(size_t a, size_t b) {
    أرجع a < b ? a : b;
}

ساكن مضمن size_t hex_smax(size_t a, size_t b) {
    أرجع a > b ? a : b;
}

ساكن مضمن فراغ hex_مٽا_سٽا_ptr(فراغ ** p1, فراغ ** p2) {
    فراغ * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

ساكن مضمن bool hex_mul_حد_کان_مٿي(size_t a, size_t b, size_t *ٻاهريون) {
    إذا (a != 0 && b > SIZE_MAX / a) أرجع true;
    *ٻاهريون = a * b;
    أرجع false;
}

ساكن مضمن bool hex_جوڙيو_حد_کان_مٿي(size_t a, size_t b, size_t *ٻاهريون) {
    إذا (a > SIZE_MAX - b) أرجع true;
    *ٻاهريون = a + b;
    أرجع false;
}

ساكن مضمن uint32_t hex_وڏو_گڏيل_ونڊيندڙ_u32(uint32_t a, uint32_t b) {
    طالما (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    أرجع a;
}

#جيڪڏهن_پورو // HEX_COMMON_H
