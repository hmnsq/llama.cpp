#यदि_अपरिभाषितम् HVX_TYPES_H
#परिभाषय HVX_TYPES_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव <hexagon_types.h>

#परिभाषय SIZEOF_FP32 (4)
#परिभाषय SIZEOF_FP16 (2)
#परिभाषय VLEN        (128)
#परिभाषय VLEN_FP32   (VLEN / SIZEOF_FP32)
#परिभाषय VLEN_FP16   (VLEN / SIZEOF_FP16)

प्रकारनिर्देश सङ्घ {
    HVX_सदिशम् v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    भिन्न      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_सदिशम्_Alias;

प्रकारनिर्देश संरचना {
    HVX_सदिशम् v[2];
} HVX_सदिशम्_x2;

प्रकारनिर्देश संरचना {
    HVX_सदिशम् v[4];
} HVX_सदिशम्_x4;

प्रकारनिर्देश संरचना {
    HVX_सदिशम् v[8];
} HVX_सदिशम्_x8;

#यदि_अन्तः /* HVX_TYPES_H */
