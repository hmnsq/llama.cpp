#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_DUMP_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_DUMP_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f16_n(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v, uint32_t n) {
    HVX_ಸದಿಶ_Alias u = { .v = v };

    ಸ್ಥಿರಾಂಕ uint32_t n0 = n / 16;
    ಸ್ಥಿರಾಂಕ uint32_t n1 = n % 16;
    ಪೂರ್ಣಾಂಕ            i  = 0;
    ಗಾಗಿ (; i < n0; i++) {
        hex_dump_f16_ಸಾಲು(pref, u.fp16 + (16 * i), 16);
    }
    ಆದರೆ (n1) {
        hex_dump_f16_ಸಾಲು(pref, u.fp16 + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f16(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32_n(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v, uint32_t n) {
    HVX_ಸದಿಶ_Alias u = { .v = v };

    ಸ್ಥಿರಾಂಕ uint32_t n0 = n / 16;
    ಸ್ಥಿರಾಂಕ uint32_t n1 = n % 16;
    ಪೂರ್ಣಾಂಕ            i  = 0;
    ಗಾಗಿ (; i < n0; i++) {
        hex_dump_f32_ಸಾಲು(pref, u.fp32 + (16 * i), 16);
    }
    ಆದರೆ (n1) {
        hex_dump_f32_ಸಾಲು(pref, u.fp32 + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32_hmt(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        ದಶಮಾಂಶ      d[32];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int32(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        int32_t    d[32];
    } u = { .v = v };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32 / 16; i++) {
        hex_dump_int32_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int32_hmt(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int8_hmt(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int8(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        int8_t     d[128];
    } u = { .v = v };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 128 / 16; i++) {
        hex_dump_int8_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_uint8(ಅಕ್ಷರ * pref, HVX_ಸದಿಶ v) {
    ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        uint8_t    d[128];
    } u = { .v = v };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ bool hvx_vec_eq(HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1, size_t n) {
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಒಕ್ಕೂಟ {
        HVX_ಸದಿಶ v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಆದರೆ (u0.d[i] != u1.d[i]) {
            ಹಿಂತಿರುಗಿ false;
        }
    }

    ಹಿಂತಿರುಗಿ true;
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_DUMP_H */
