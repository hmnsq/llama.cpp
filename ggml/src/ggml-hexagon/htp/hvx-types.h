#अगर_अघोषित HVX_TYPES_H
#घोषणा HVX_TYPES_H

#समावेश <stdbool.h>
#समावेश <stdint.h>

#समावेश <hexagon_types.h>

#घोषणा SIZEOF_FP32 (4)
#घोषणा SIZEOF_FP16 (2)
#घोषणा VLEN        (128)
#घोषणा VLEN_FP32   (VLEN / SIZEOF_FP32)
#घोषणा VLEN_FP16   (VLEN / SIZEOF_FP16)

प्रकार जोड़ {
    HVX_Vector v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    भग्न      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_VectorAlias;

प्रकार काष्ठा {
    HVX_Vector v[2];
} HVX_Vector_x2;

प्रकार काष्ठा {
    HVX_Vector v[4];
} HVX_Vector_x4;

प्रकार काष्ठा {
    HVX_Vector v[8];
} HVX_Vector_x8;

#पूर्ण_अगर /* HVX_TYPES_H */
