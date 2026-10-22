#अगर_अघोषित HVX_DUMP_H
#घोषणा HVX_DUMP_H

#समावेश <HAP_farf.h>

#समावेश <stdbool.h>
#समावेश <stdint.h>

#समावेश "hex-utils.h"
#समावेश "hvx-types.h"

अटल व्योम hvx_vec_dump_f16_n(अक्षर * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    स्थिर uint32_t n0 = n / 16;
    स्थिर uint32_t n1 = n % 16;
    पूर्णांक            i  = 0;
    क्रम (; i < n0; i++) {
        hex_dump_f16_पंक्ति(pref, u.fp16 + (16 * i), 16);
    }
    अगर (n1) {
        hex_dump_f16_पंक्ति(pref, u.fp16 + (16 * i), n1);
    }
}

अटल व्योम hvx_vec_dump_f16(अक्षर * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

अटल व्योम hvx_vec_dump_f32_n(अक्षर * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    स्थिर uint32_t n0 = n / 16;
    स्थिर uint32_t n1 = n % 16;
    पूर्णांक            i  = 0;
    क्रम (; i < n0; i++) {
        hex_dump_f32_पंक्ति(pref, u.fp32 + (16 * i), 16);
    }
    अगर (n1) {
        hex_dump_f32_पंक्ति(pref, u.fp32 + (16 * i), n1);
    }
}

अटल व्योम hvx_vec_dump_f32_hmt(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        भग्न      d[32];
    } u = { .v = v };

    FARF(ऊपरी_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

अटल व्योम hvx_vec_dump_f32(अक्षर * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

अटल व्योम hvx_vec_dump_int32(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    क्रम (पूर्णांक i = 0; i < 32 / 16; i++) {
        hex_dump_int32_पंक्ति(pref, u.d + (16 * i), 16);
    }
}

अटल व्योम hvx_vec_dump_int32_hmt(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ऊपरी_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

अटल व्योम hvx_vec_dump_int8_hmt(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ऊपरी_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

अटल व्योम hvx_vec_dump_int8(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    क्रम (पूर्णांक i = 0; i < 128 / 16; i++) {
        hex_dump_int8_पंक्ति(pref, u.d + (16 * i), 16);
    }
}

अटल व्योम hvx_vec_dump_uint8(अक्षर * pref, HVX_Vector v) {
    जोड़ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    क्रम (पूर्णांक i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_पंक्ति(pref, u.d + (16 * i), 16);
    }
}

अटल bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, माप_प्रकार n) {
    प्रकार जोड़ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    क्रम (पूर्णांक i = 0; i < n; i++) {
        अगर (u0.d[i] != u1.d[i]) {
            वापस false;
        }
    }

    वापस true;
}

#पूर्ण_अगर /* HVX_DUMP_H */
