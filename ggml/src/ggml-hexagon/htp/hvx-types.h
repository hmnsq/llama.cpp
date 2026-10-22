#اگر_تعریف_نشده HVX_TYPES_H
#تعریف_کن HVX_TYPES_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن <hexagon_types.h>

#تعریف_کن SIZEOF_FP32 (4)
#تعریف_کن SIZEOF_FP16 (2)
#تعریف_کن VLEN        (128)
#تعریف_کن VLEN_FP32   (VLEN / SIZEOF_FP32)
#تعریف_کن VLEN_FP16   (VLEN / SIZEOF_FP16)

تعريفنوع اتحاد {
    HVX_وکتور v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    عائم      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_وکتور_Alias;

تعريفنوع بنية {
    HVX_وکتور v[2];
} HVX_وکتور_x2;

تعريفنوع بنية {
    HVX_وکتور v[4];
} HVX_وکتور_x4;

تعريفنوع بنية {
    HVX_وکتور v[8];
} HVX_وکتور_x8;

#پایان_اگر /* HVX_TYPES_H */
