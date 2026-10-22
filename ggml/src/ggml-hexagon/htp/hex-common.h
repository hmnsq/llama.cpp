#अगर_अघोषित HEX_COMMON_H
#घोषणा HEX_COMMON_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>

#अगर_अघोषित SIZE_MAX
#घोषणा SIZE_MAX ((माप_प्रकार)-1)
#पूर्ण_अगर

#अगर_अघोषित MAX
#घोषणा MAX(a, b) ((a) > (b) ? (a) : (b))
#पूर्ण_अगर

#अगर_अघोषित MIN
#घोषणा MIN(a, b) ((a) < (b) ? (a) : (b))
#पूर्ण_अगर

अटल अंतरभूत uint32_t hex_ceil_pow2(uint32_t x) {
    अगर (x <= 1) { वापस 1; }
    पूर्णांक p = 2;
    x--;
    जबतक (x >>= 1) { p <<= 1; }
    वापस p;
}

अटल अंतरभूत माप_प्रकार hmx_ceil_div(माप_प्रकार अंक, माप_प्रकार den) {
    वापस (अंक + den - 1) / den;
}

अटल अंतरभूत int32_t hex_is_aligned(स्थिर व्योम * addr, uint32_t align) {
    वापस ((माप_प्रकार) addr & (align - 1)) == 0;
}

अटल अंतरभूत माप_प्रकार hex_align_up(माप_प्रकार v, माप_प्रकार align) {
    वापस hmx_ceil_div(v, align) * align;
}

अटल अंतरभूत माप_प्रकार hex_align_down(माप_प्रकार v, माप_प्रकार align) {
    वापस (v / align) * align;
}

अटल अंतरभूत int32_t hex_is_one_chunk(व्योम * addr, uint32_t n, uint32_t chunk_आकार) {
    uint32_t बायाँ_off  = (माप_प्रकार) addr & (chunk_आकार - 1);
    uint32_t दायाँ_off = बायाँ_off + n;
    वापस दायाँ_off <= chunk_आकार;
}

अटल अंतरभूत uint32_t hex_round_up(uint32_t n, uint32_t m) {
    वापस m * ((n + m - 1) / m);
}

अटल अंतरभूत माप_प्रकार hex_smin(माप_प्रकार a, माप_प्रकार b) {
    वापस a < b ? a : b;
}

अटल अंतरभूत माप_प्रकार hex_smax(माप_प्रकार a, माप_प्रकार b) {
    वापस a > b ? a : b;
}

अटल अंतरभूत व्योम hex_swap_ptr(व्योम ** p1, व्योम ** p2) {
    व्योम * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

अटल अंतरभूत bool hex_mul_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    अगर (a != 0 && b > SIZE_MAX / a) वापस true;
    *out = a * b;
    वापस false;
}

अटल अंतरभूत bool hex_add_overflow(माप_प्रकार a, माप_प्रकार b, माप_प्रकार *out) {
    अगर (a > SIZE_MAX - b) वापस true;
    *out = a + b;
    वापस false;
}

अटल अंतरभूत uint32_t hex_gcd_u32(uint32_t a, uint32_t b) {
    जबतक (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    वापस a;
}

#पूर्ण_अगर // HEX_COMMON_H
