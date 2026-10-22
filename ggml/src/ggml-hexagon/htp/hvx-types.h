#اگر_غیر_تعریف_شدہ HVX_TYPES_H
#تعریف_کرو HVX_TYPES_H

#شامل <stdbool.h>
#شامل <stdint.h>

#شامل <hexagon_types.h>

#تعریف_کرو SIZEOF_FP32 (4)
#تعریف_کرو SIZEOF_FP16 (2)
#تعریف_کرو VLEN        (128)
#تعریف_کرو VLEN_FP32   (VLEN / SIZEOF_FP32)
#تعریف_کرو VLEN_FP16   (VLEN / SIZEOF_FP16)

قسمتعریف اتحاد {
    HVX_سمتیہ v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    اعشاری      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_سمتیہ_Alias;

قسمتعریف ساخت {
    HVX_سمتیہ v[2];
} HVX_سمتیہ_x2;

قسمتعریف ساخت {
    HVX_سمتیہ v[4];
} HVX_سمتیہ_x4;

قسمتعریف ساخت {
    HVX_سمتیہ v[8];
} HVX_سمتیہ_x8;

#اگر_ختم /* HVX_TYPES_H */
