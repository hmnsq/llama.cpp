#ifndef HVX_TYPES_H
#define HVX_TYPES_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ <hexagon_types.h>

#define SIZEOF_FP32 (4)
#define SIZEOF_FP16 (2)
#define VLEN        (128)
#define VLEN_FP32   (VLEN / SIZEOF_FP32)
#define VLEN_FP16   (VLEN / SIZEOF_FP16)

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಙ್ಘ {
    HVX_Vector v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    ಭಿನ್ನ      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_VectorAlias;

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    HVX_Vector v[2];
} HVX_Vector_x2;

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    HVX_Vector v[4];
} HVX_Vector_x4;

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    HVX_Vector v[8];
} HVX_Vector_x8;

#endif /* HVX_TYPES_H */
