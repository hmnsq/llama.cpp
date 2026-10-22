#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-types.h"

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_f16_n(ଅକ୍ଷର * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ଅଚଳ uint32_t n0 = n / 16;
    ଅଚଳ uint32_t n1 = n % 16;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା            i  = 0;
    ପାଇଁ (; i < n0; i++) {
        hex_dump_f16_ଧାଡ଼ି(pref, u.fp16 + (16 * i), 16);
    }
    ଯଦି (n1) {
        hex_dump_f16_ଧାଡ଼ି(pref, u.fp16 + (16 * i), n1);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_f16(ଅକ୍ଷର * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_f32_n(ଅକ୍ଷର * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ଅଚଳ uint32_t n0 = n / 16;
    ଅଚଳ uint32_t n1 = n % 16;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା            i  = 0;
    ପାଇଁ (; i < n0; i++) {
        hex_dump_f32_ଧାଡ଼ି(pref, u.fp32 + (16 * i), 16);
    }
    ଯଦି (n1) {
        hex_dump_f32_ଧାଡ଼ି(pref, u.fp32 + (16 * i), n1);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_f32_hmt(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        ଦଶମିକ      d[32];
    } u = { .v = v };

    FARF(ଉପର_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_f32(ଅକ୍ଷର * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_int32(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 32 / 16; i++) {
        hex_dump_int32_ଧାଡ଼ି(pref, u.d + (16 * i), 16);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_int32_hmt(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ଉପର_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_int8_hmt(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ଉପର_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_int8(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 128 / 16; i++) {
        hex_dump_int8_ଧାଡ଼ି(pref, u.d + (16 * i), 16);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_vec_dump_uint8(ଅକ୍ଷର * pref, HVX_Vector v) {
    ସଂଘ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_ଧାଡ଼ି(pref, u.d + (16 * i), 16);
    }
}

ସ୍ଥିର bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂଘ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ଯଦି (u0.d[i] != u1.d[i]) {
            ଫେରନ୍ତୁ false;
        }
    }

    ଫେରନ୍ତୁ true;
}

#endif /* HVX_DUMP_H */
