#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-types.h"

स्थिर शून्य hvx_vec_dump_f16_n(वर्ण * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    अचर uint32_t n0 = n / 16;
    अचर uint32_t n1 = n % 16;
    पूर्णाङ्क            i  = 0;
    लागि (; i < n0; i++) {
        hex_dump_f16_पङ्क्ति(pref, u.fp16 + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f16_पङ्क्ति(pref, u.fp16 + (16 * i), n1);
    }
}

स्थिर शून्य hvx_vec_dump_f16(वर्ण * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

स्थिर शून्य hvx_vec_dump_f32_n(वर्ण * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    अचर uint32_t n0 = n / 16;
    अचर uint32_t n1 = n % 16;
    पूर्णाङ्क            i  = 0;
    लागि (; i < n0; i++) {
        hex_dump_f32_पङ्क्ति(pref, u.fp32 + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f32_पङ्क्ति(pref, u.fp32 + (16 * i), n1);
    }
}

स्थिर शून्य hvx_vec_dump_f32_hmt(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        भिन्न      d[32];
    } u = { .v = v };

    FARF(माथिल्लो_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

स्थिर शून्य hvx_vec_dump_f32(वर्ण * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

स्थिर शून्य hvx_vec_dump_int32(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    लागि (पूर्णाङ्क i = 0; i < 32 / 16; i++) {
        hex_dump_int32_पङ्क्ति(pref, u.d + (16 * i), 16);
    }
}

स्थिर शून्य hvx_vec_dump_int32_hmt(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(माथिल्लो_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

स्थिर शून्य hvx_vec_dump_int8_hmt(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(माथिल्लो_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

स्थिर शून्य hvx_vec_dump_int8(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    लागि (पूर्णाङ्क i = 0; i < 128 / 16; i++) {
        hex_dump_int8_पङ्क्ति(pref, u.d + (16 * i), 16);
    }
}

स्थिर शून्य hvx_vec_dump_uint8(वर्ण * pref, HVX_Vector v) {
    सङ्घ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    लागि (पूर्णाङ्क i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_पङ्क्ति(pref, u.d + (16 * i), 16);
    }
}

स्थिर bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, माप_प्रकार n) {
    प्रकारनिर्देश सङ्घ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        यदि (u0.d[i] != u1.d[i]) {
            फर्काऊ false;
        }
    }

    फर्काऊ true;
}

#endif /* HVX_DUMP_H */
