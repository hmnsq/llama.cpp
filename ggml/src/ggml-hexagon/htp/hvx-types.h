#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_TYPES_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_TYPES_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ <hexagon_types.h>

#ವ್ಯಾಖ್ಯಾನಿಸು SIZEOF_FP32 (4)
#ವ್ಯಾಖ್ಯಾನಿಸು SIZEOF_FP16 (2)
#ವ್ಯಾಖ್ಯಾನಿಸು VLEN        (128)
#ವ್ಯಾಖ್ಯಾನಿಸು VLEN_FP32   (VLEN / SIZEOF_FP32)
#ವ್ಯಾಖ್ಯಾನಿಸು VLEN_FP16   (VLEN / SIZEOF_FP16)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಒಕ್ಕೂಟ {
    HVX_ಸದಿಶ v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    ದಶಮಾಂಶ      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_ಸದಿಶ_Alias;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    HVX_ಸದಿಶ v[2];
} HVX_ಸದಿಶ_x2;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    HVX_ಸದಿಶ v[4];
} HVX_ಸದಿಶ_x4;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    HVX_ಸದಿಶ v[8];
} HVX_ಸದಿಶ_x8;

#ಆದರೆ_ಅಂತ್ಯ /* HVX_TYPES_H */
