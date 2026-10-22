#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#അന്തര്ഭാവ <HAP_farf.h>

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>

#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "hvx-types.h"

സ്ഥിതം ശൂന്യം hvx_vec_dump_f16_n(അക്ഷരം * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    സ്ഥിരം uint32_t n0 = n / 16;
    സ്ഥിരം uint32_t n1 = n % 16;
    പൂർണ്ണസംഖ്യ            i  = 0;
    വേണ്ടി (; i < n0; i++) {
        hex_dump_f16_വരി(pref, u.fp16 + (16 * i), 16);
    }
    എങ്കിൽ (n1) {
        hex_dump_f16_വരി(pref, u.fp16 + (16 * i), n1);
    }
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_f16(അക്ഷരം * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_f32_n(അക്ഷരം * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    സ്ഥിരം uint32_t n0 = n / 16;
    സ്ഥിരം uint32_t n1 = n % 16;
    പൂർണ്ണസംഖ്യ            i  = 0;
    വേണ്ടി (; i < n0; i++) {
        hex_dump_f32_വരി(pref, u.fp32 + (16 * i), 16);
    }
    എങ്കിൽ (n1) {
        hex_dump_f32_വരി(pref, u.fp32 + (16 * i), n1);
    }
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_f32_hmt(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        ദശാംശം      d[32];
    } u = { .v = v };

    FARF(മുകളിലെ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_f32(അക്ഷരം * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_int32(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 32 / 16; i++) {
        hex_dump_int32_വരി(pref, u.d + (16 * i), 16);
    }
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_int32_hmt(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(മുകളിലെ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_int8_hmt(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(മുകളിലെ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_int8(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128 / 16; i++) {
        hex_dump_int8_വരി(pref, u.d + (16 * i), 16);
    }
}

സ്ഥിതം ശൂന്യം hvx_vec_dump_uint8(അക്ഷരം * pref, HVX_Vector v) {
    സംഘം {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_വരി(pref, u.d + (16 * i), 16);
    }
}

സ്ഥിതം bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    തരനിർവചനം സംഘം {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        എങ്കിൽ (u0.d[i] != u1.d[i]) {
            തിരികെ false;
        }
    }

    തിരികെ true;
}

#endif /* HVX_DUMP_H */
