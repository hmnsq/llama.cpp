#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "hvx-types.h"

স্থির শূন্য hvx_vec_dump_f16_n(অক্ষর * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ধ্রুবক uint32_t n0 = n / 16;
    ধ্রুবক uint32_t n1 = n % 16;
    পূর্ণসংখ্যা            i  = 0;
    জন্য (; i < n0; i++) {
        hex_dump_f16_সারি(pref, u.fp16 + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f16_সারি(pref, u.fp16 + (16 * i), n1);
    }
}

স্থির শূন্য hvx_vec_dump_f16(অক্ষর * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

স্থির শূন্য hvx_vec_dump_f32_n(অক্ষর * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ধ্রুবক uint32_t n0 = n / 16;
    ধ্রুবক uint32_t n1 = n % 16;
    পূর্ণসংখ্যা            i  = 0;
    জন্য (; i < n0; i++) {
        hex_dump_f32_সারি(pref, u.fp32 + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f32_সারি(pref, u.fp32 + (16 * i), n1);
    }
}

স্থির শূন্য hvx_vec_dump_f32_hmt(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        দশমিক      d[32];
    } u = { .v = v };

    FARF(উপরের_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

স্থির শূন্য hvx_vec_dump_f32(অক্ষর * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

স্থির শূন্য hvx_vec_dump_int32(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    জন্য (পূর্ণসংখ্যা i = 0; i < 32 / 16; i++) {
        hex_dump_int32_সারি(pref, u.d + (16 * i), 16);
    }
}

স্থির শূন্য hvx_vec_dump_int32_hmt(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(উপরের_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

স্থির শূন্য hvx_vec_dump_int8_hmt(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(উপরের_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

স্থির শূন্য hvx_vec_dump_int8(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    জন্য (পূর্ণসংখ্যা i = 0; i < 128 / 16; i++) {
        hex_dump_int8_সারি(pref, u.d + (16 * i), 16);
    }
}

স্থির শূন্য hvx_vec_dump_uint8(অক্ষর * pref, HVX_Vector v) {
    সংঘ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    জন্য (পূর্ণসংখ্যা i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_সারি(pref, u.d + (16 * i), 16);
    }
}

স্থির bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    প্রকারলেখ সংঘ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        যদি (u0.d[i] != u1.d[i]) {
            ফেরত false;
        }
    }

    ফেরত true;
}

#endif /* HVX_DUMP_H */
