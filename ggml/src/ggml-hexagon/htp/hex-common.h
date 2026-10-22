#నిర్వచితం_కాకపోతే HEX_COMMON_H
#నిర్వచించు HEX_COMMON_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>

#నిర్వచితం_కాకపోతే SIZE_MAX
#నిర్వచించు SIZE_MAX ((size_t)-1)
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే MAX
#నిర్వచించు MAX(a, b) ((a) > (b) ? (a) : (b))
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే MIN
#నిర్వచించు MIN(a, b) ((a) < (b) ? (a) : (b))
#అయితే_ముగింపు

స్థిర అంతఃవరుస uint32_t hex_ceil_pow2(uint32_t x) {
    ఒకవేళ (x <= 1) { తిరిగి 1; }
    పూర్ణాంకం p = 2;
    x--;
    అయితే (x >>= 1) { p <<= 1; }
    తిరిగి p;
}

స్థిర అంతఃవరుస size_t hmx_ceil_div(size_t అంకె, size_t den) {
    తిరిగి (అంకె + den - 1) / den;
}

స్థిర అంతఃవరుస int32_t hex_is_aligned(స్థిరరాశి శూన్య * addr, uint32_t align) {
    తిరిగి ((size_t) addr & (align - 1)) == 0;
}

స్థిర అంతఃవరుస size_t hex_align_పైకి(size_t v, size_t align) {
    తిరిగి hmx_ceil_div(v, align) * align;
}

స్థిర అంతఃవరుస size_t hex_align_కిందికి(size_t v, size_t align) {
    తిరిగి (v / align) * align;
}

స్థిర అంతఃవరుస int32_t hex_is_one_chunk(శూన్య * addr, uint32_t n, uint32_t chunk_కొలత) {
    uint32_t ఎడమ_off  = (size_t) addr & (chunk_కొలత - 1);
    uint32_t కుడి_off = ఎడమ_off + n;
    తిరిగి కుడి_off <= chunk_కొలత;
}

స్థిర అంతఃవరుస uint32_t hex_విడత_పైకి(uint32_t n, uint32_t m) {
    తిరిగి m * ((n + m - 1) / m);
}

స్థిర అంతఃవరుస size_t hex_smin(size_t a, size_t b) {
    తిరిగి a < b ? a : b;
}

స్థిర అంతఃవరుస size_t hex_smax(size_t a, size_t b) {
    తిరిగి a > b ? a : b;
}

స్థిర అంతఃవరుస శూన్య hex_మార్పిడి_ptr(శూన్య ** p1, శూన్య ** p2) {
    శూన్య * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

స్థిర అంతఃవరుస bool hex_mul_అధికప్రవాహం(size_t a, size_t b, size_t *బహిర్గతం) {
    ఒకవేళ (a != 0 && b > SIZE_MAX / a) తిరిగి true;
    *బహిర్గతం = a * b;
    తిరిగి false;
}

స్థిర అంతఃవరుస bool hex_కూడు_అధికప్రవాహం(size_t a, size_t b, size_t *బహిర్గతం) {
    ఒకవేళ (a > SIZE_MAX - b) తిరిగి true;
    *బహిర్గతం = a + b;
    తిరిగి false;
}

స్థిర అంతఃవరుస uint32_t hex_గరిష్ఠ_సామాన్య_భాజకం_u32(uint32_t a, uint32_t b) {
    అయితే (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    తిరిగి a;
}

#అయితే_ముగింపు // HEX_COMMON_H
