#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f16_n(ವರ್ಣ * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ಧ್ರುವ uint32_t n0 = n / 16;
    ಧ್ರುವ uint32_t n1 = n % 16;
    ಪೂರ್ಣಾಙ್ಕ            i  = 0;
    ಪ್ರತಿ (; i < n0; i++) {
        hex_dump_f16_ಸಾಲು(pref, u.fp16 + (16 * i), 16);
    }
    ಯದಿ (n1) {
        hex_dump_f16_ಸಾಲು(pref, u.fp16 + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f16(ವರ್ಣ * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32_n(ವರ್ಣ * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ಧ್ರುವ uint32_t n0 = n / 16;
    ಧ್ರುವ uint32_t n1 = n % 16;
    ಪೂರ್ಣಾಙ್ಕ            i  = 0;
    ಪ್ರತಿ (; i < n0; i++) {
        hex_dump_f32_ಸಾಲು(pref, u.fp32 + (16 * i), 16);
    }
    ಯದಿ (n1) {
        hex_dump_f32_ಸಾಲು(pref, u.fp32 + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32_hmt(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        ಭಿನ್ನ      d[32];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_f32(ವರ್ಣ * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int32(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 32 / 16; i++) {
        hex_dump_int32_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int32_hmt(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int8_hmt(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ಮೇಲಿನ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_int8(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 128 / 16; i++) {
        hex_dump_int8_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_vec_dump_uint8(ವರ್ಣ * pref, HVX_Vector v) {
    ಸಙ್ಘ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_ಸಾಲು(pref, u.d + (16 * i), 16);
    }
}

ಸ್ಥಿರ bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    ಪ್ರಕಾರನಿರ್ದೇಶ ಸಙ್ಘ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಯದಿ (u0.d[i] != u1.d[i]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

#endif /* HVX_DUMP_H */
