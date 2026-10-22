#ifndef HVX_TYPES_H
#define HVX_TYPES_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব <hexagon_types.h>

#define SIZEOF_FP32 (4)
#define SIZEOF_FP16 (2)
#define VLEN        (128)
#define VLEN_FP32   (VLEN / SIZEOF_FP32)
#define VLEN_FP16   (VLEN / SIZEOF_FP16)

প্রকারনির্দেশ সঙ্ঘ {
    HVX_Vector v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    ভিন্ন      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_VectorAlias;

প্রকারনির্দেশ সংরচনা {
    HVX_Vector v[2];
} HVX_Vector_x2;

প্রকারনির্দেশ সংরচনা {
    HVX_Vector v[4];
} HVX_Vector_x4;

প্রকারনির্দেশ সংরচনা {
    HVX_Vector v[8];
} HVX_Vector_x8;

#endif /* HVX_TYPES_H */
