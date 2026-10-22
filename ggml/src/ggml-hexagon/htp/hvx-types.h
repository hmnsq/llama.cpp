#که_نه_تعریف_شوی HVX_TYPES_H
#تعریف_کړه HVX_TYPES_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن <hexagon_types.h>

#تعریف_کړه SIZEOF_FP32 (4)
#تعریف_کړه SIZEOF_FP16 (2)
#تعریف_کړه VLEN        (128)
#تعریف_کړه VLEN_FP32   (VLEN / SIZEOF_FP32)
#تعریف_کړه VLEN_FP16   (VLEN / SIZEOF_FP16)

تعريفنوع اتحاد {
    HVX_ویکټور v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    عائم      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_ویکټور_Alias;

تعريفنوع بنية {
    HVX_ویکټور v[2];
} HVX_ویکټور_x2;

تعريفنوع بنية {
    HVX_ویکټور v[4];
} HVX_ویکټور_x4;

تعريفنوع بنية {
    HVX_ویکټور v[8];
} HVX_ویکټور_x8;

#د_که_پای /* HVX_TYPES_H */
