#جيڪڏهن_وصف_نه_ٿيل HVX_TYPES_H
#وصف_ڪريو HVX_TYPES_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن <hexagon_types.h>

#وصف_ڪريو SIZEOF_FP32 (4)
#وصف_ڪريو SIZEOF_FP16 (2)
#وصف_ڪريو VLEN        (128)
#وصف_ڪريو VLEN_FP32   (VLEN / SIZEOF_FP32)
#وصف_ڪريو VLEN_FP16   (VLEN / SIZEOF_FP16)

تعريفنوع اتحاد {
    HVX_ويڪٽر v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    عائم      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_ويڪٽر_Alias;

تعريفنوع بنية {
    HVX_ويڪٽر v[2];
} HVX_ويڪٽر_x2;

تعريفنوع بنية {
    HVX_ويڪٽر v[4];
} HVX_ويڪٽر_x4;

تعريفنوع بنية {
    HVX_ويڪٽر v[8];
} HVX_ويڪٽر_x8;

#جيڪڏهن_پورو /* HVX_TYPES_H */
