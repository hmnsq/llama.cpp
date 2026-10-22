#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_TYPES_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_TYPES_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SIZEOF_FP32 (4)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SIZEOF_FP16 (2)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VLEN        (128)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VLEN_FP32   (VLEN / SIZEOF_FP32)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VLEN_FP16   (VLEN / SIZEOF_FP16)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਸੰਘ {
    HVX_ਸਦਿਸ਼ v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    ਦਸ਼ਮਲਵ      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_ਸਦਿਸ਼_Alias;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    HVX_ਸਦਿਸ਼ v[2];
} HVX_ਸਦਿਸ਼_x2;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    HVX_ਸਦਿਸ਼ v[4];
} HVX_ਸਦਿਸ਼_x4;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    HVX_ਸਦਿਸ਼ v[8];
} HVX_ਸਦਿਸ਼_x8;

#ਜੇ_ਅੰਤ /* HVX_TYPES_H */
