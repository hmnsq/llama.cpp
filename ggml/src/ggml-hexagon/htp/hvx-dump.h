#جے_بنا_تعریف HVX_DUMP_H
#تعریف_کرو HVX_DUMP_H

#ضمن <HAP_farf.h>

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن "hex-utils.h"
#ضمن "hvx-types.h"

ساكن فراغ hvx_vec_dump_f16_n(حرف * pref, HVX_سمتیہ v, uint32_t n) {
    HVX_سمتیہ_Alias u = { .v = v };

    ثابت uint32_t n0 = n / 16;
    ثابت uint32_t n1 = n % 16;
    صحيح            i  = 0;
    لكل (; i < n0; i++) {
        hex_dump_f16_سطر(pref, u.fp16 + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f16_سطر(pref, u.fp16 + (16 * i), n1);
    }
}

ساكن فراغ hvx_vec_dump_f16(حرف * pref, HVX_سمتیہ v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ساكن فراغ hvx_vec_dump_f32_n(حرف * pref, HVX_سمتیہ v, uint32_t n) {
    HVX_سمتیہ_Alias u = { .v = v };

    ثابت uint32_t n0 = n / 16;
    ثابت uint32_t n1 = n % 16;
    صحيح            i  = 0;
    لكل (; i < n0; i++) {
        hex_dump_f32_سطر(pref, u.fp32 + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f32_سطر(pref, u.fp32 + (16 * i), n1);
    }
}

ساكن فراغ hvx_vec_dump_f32_hmt(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        عائم      d[32];
    } u = { .v = v };

    FARF(بلند_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ساكن فراغ hvx_vec_dump_f32(حرف * pref, HVX_سمتیہ v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ساكن فراغ hvx_vec_dump_int32(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        int32_t    d[32];
    } u = { .v = v };

    لكل (صحيح i = 0; i < 32 / 16; i++) {
        hex_dump_int32_سطر(pref, u.d + (16 * i), 16);
    }
}

ساكن فراغ hvx_vec_dump_int32_hmt(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(بلند_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ساكن فراغ hvx_vec_dump_int8_hmt(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(بلند_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ساكن فراغ hvx_vec_dump_int8(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        int8_t     d[128];
    } u = { .v = v };

    لكل (صحيح i = 0; i < 128 / 16; i++) {
        hex_dump_int8_سطر(pref, u.d + (16 * i), 16);
    }
}

ساكن فراغ hvx_vec_dump_uint8(حرف * pref, HVX_سمتیہ v) {
    اتحاد {
        HVX_سمتیہ v;
        uint8_t    d[128];
    } u = { .v = v };

    لكل (صحيح i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_سطر(pref, u.d + (16 * i), 16);
    }
}

ساكن bool hvx_vec_eq(HVX_سمتیہ v0, HVX_سمتیہ v1, size_t n) {
    تعريفنوع اتحاد {
        HVX_سمتیہ v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    لكل (صحيح i = 0; i < n; i++) {
        إذا (u0.d[i] != u1.d[i]) {
            أرجع false;
        }
    }

    أرجع true;
}

#جے_انت /* HVX_DUMP_H */
