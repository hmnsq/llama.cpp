#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#شامل <HAP_farf.h>

#شامل <stdbool.h>
#شامل <stdint.h>

#شامل "hex-utils.h"
#شامل "hvx-types.h"

ایستا تهی hvx_vec_dump_f16_n(نویسه * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ثابت uint32_t n0 = n / 16;
    ثابت uint32_t n1 = n % 16;
    صحیح            i  = 0;
    برای (; i < n0; i++) {
        hex_dump_f16_سطر(pref, u.fp16 + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f16_سطر(pref, u.fp16 + (16 * i), n1);
    }
}

ایستا تهی hvx_vec_dump_f16(نویسه * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ایستا تهی hvx_vec_dump_f32_n(نویسه * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    ثابت uint32_t n0 = n / 16;
    ثابت uint32_t n1 = n % 16;
    صحیح            i  = 0;
    برای (; i < n0; i++) {
        hex_dump_f32_سطر(pref, u.fp32 + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f32_سطر(pref, u.fp32 + (16 * i), n1);
    }
}

ایستا تهی hvx_vec_dump_f32_hmt(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        اعشاری      d[32];
    } u = { .v = v };

    FARF(بالا_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ایستا تهی hvx_vec_dump_f32(نویسه * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ایستا تهی hvx_vec_dump_int32(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    برای (صحیح i = 0; i < 32 / 16; i++) {
        hex_dump_int32_سطر(pref, u.d + (16 * i), 16);
    }
}

ایستا تهی hvx_vec_dump_int32_hmt(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(بالا_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ایستا تهی hvx_vec_dump_int8_hmt(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(بالا_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ایستا تهی hvx_vec_dump_int8(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    برای (صحیح i = 0; i < 128 / 16; i++) {
        hex_dump_int8_سطر(pref, u.d + (16 * i), 16);
    }
}

ایستا تهی hvx_vec_dump_uint8(نویسه * pref, HVX_Vector v) {
    اجتماع {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    برای (صحیح i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_سطر(pref, u.d + (16 * i), 16);
    }
}

ایستا bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    تعریفنوع اجتماع {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    برای (صحیح i = 0; i < n; i++) {
        اگر (u0.d[i] != u1.d[i]) {
            بازگشت false;
        }
    }

    بازگشت true;
}

#endif /* HVX_DUMP_H */
