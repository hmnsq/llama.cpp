#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#অন্তর্ভাৱ <HAP_farf.h>

#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdint.h>

#অন্তর্ভাৱ "hex-utils.h"
#অন্তর্ভাৱ "hvx-types.h"

স্থির শূন্য hvx_vec_dump_f16_n(ৱর্ণ * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ধ্রুৱ uint32_t n0 = n / 16;
    ধ্রুৱ uint32_t n1 = n % 16;
    পূর্ণাঙ্ক            i  = 0;
    বাবে (; i < n0; i++) {
        hex_dump_f16_সাৰি(pref, u.fp16 + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f16_সাৰি(pref, u.fp16 + (16 * i), n1);
    }
}

স্থির শূন্য hvx_vec_dump_f16(ৱর্ণ * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

স্থির শূন্য hvx_vec_dump_f32_n(ৱর্ণ * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ধ্রুৱ uint32_t n0 = n / 16;
    ধ্রুৱ uint32_t n1 = n % 16;
    পূর্ণাঙ্ক            i  = 0;
    বাবে (; i < n0; i++) {
        hex_dump_f32_সাৰি(pref, u.fp32 + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f32_সাৰি(pref, u.fp32 + (16 * i), n1);
    }
}

স্থির শূন্য hvx_vec_dump_f32_hmt(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        ভিন্ন      d[32];
    } u = { .v = v };

    FARF(উপৰেৰ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

স্থির শূন্য hvx_vec_dump_f32(ৱর্ণ * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

স্থির শূন্য hvx_vec_dump_int32(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    বাবে (পূর্ণাঙ্ক i = 0; i < 32 / 16; i++) {
        hex_dump_int32_সাৰি(pref, u.d + (16 * i), 16);
    }
}

স্থির শূন্য hvx_vec_dump_int32_hmt(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(উপৰেৰ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

স্থির শূন্য hvx_vec_dump_int8_hmt(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(উপৰেৰ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

স্থির শূন্য hvx_vec_dump_int8(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    বাবে (পূর্ণাঙ্ক i = 0; i < 128 / 16; i++) {
        hex_dump_int8_সাৰি(pref, u.d + (16 * i), 16);
    }
}

স্থির শূন্য hvx_vec_dump_uint8(ৱর্ণ * pref, HVX_Vector v) {
    সঙ্ঘ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    বাবে (পূর্ণাঙ্ক i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_সাৰি(pref, u.d + (16 * i), 16);
    }
}

স্থির bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    প্রকারনির্দেশ সঙ্ঘ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        যদি (u0.d[i] != u1.d[i]) {
            ঘূৰাই false;
        }
    }

    ঘূৰাই true;
}

#endif /* HVX_DUMP_H */
