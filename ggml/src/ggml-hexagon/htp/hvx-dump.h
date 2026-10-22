#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#અન્તર્ભાવ <HAP_farf.h>

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "hvx-types.h"

સ્થિર શૂન્ય hvx_vec_dump_f16_n(અક્ષર * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    અચલ uint32_t n0 = n / 16;
    અચલ uint32_t n1 = n % 16;
    પૂર્ણાંક            i  = 0;
    માટે (; i < n0; i++) {
        hex_dump_f16_લીટી(pref, u.fp16 + (16 * i), 16);
    }
    જો (n1) {
        hex_dump_f16_લીટી(pref, u.fp16 + (16 * i), n1);
    }
}

સ્થિર શૂન્ય hvx_vec_dump_f16(અક્ષર * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

સ્થિર શૂન્ય hvx_vec_dump_f32_n(અક્ષર * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    અચલ uint32_t n0 = n / 16;
    અચલ uint32_t n1 = n % 16;
    પૂર્ણાંક            i  = 0;
    માટે (; i < n0; i++) {
        hex_dump_f32_લીટી(pref, u.fp32 + (16 * i), 16);
    }
    જો (n1) {
        hex_dump_f32_લીટી(pref, u.fp32 + (16 * i), n1);
    }
}

સ્થિર શૂન્ય hvx_vec_dump_f32_hmt(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        દશાંશ      d[32];
    } u = { .v = v };

    FARF(ઉપલું_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

સ્થિર શૂન્ય hvx_vec_dump_f32(અક્ષર * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

સ્થિર શૂન્ય hvx_vec_dump_int32(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    માટે (પૂર્ણાંક i = 0; i < 32 / 16; i++) {
        hex_dump_int32_લીટી(pref, u.d + (16 * i), 16);
    }
}

સ્થિર શૂન્ય hvx_vec_dump_int32_hmt(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ઉપલું_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

સ્થિર શૂન્ય hvx_vec_dump_int8_hmt(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ઉપલું_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

સ્થિર શૂન્ય hvx_vec_dump_int8(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    માટે (પૂર્ણાંક i = 0; i < 128 / 16; i++) {
        hex_dump_int8_લીટી(pref, u.d + (16 * i), 16);
    }
}

સ્થિર શૂન્ય hvx_vec_dump_uint8(અક્ષર * pref, HVX_Vector v) {
    સંઘ {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    માટે (પૂર્ણાંક i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_લીટી(pref, u.d + (16 * i), 16);
    }
}

સ્થિર bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    પ્રકારવ્યાખ્યા સંઘ {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        જો (u0.d[i] != u1.d[i]) {
            પાછા false;
        }
    }

    પાછા true;
}

#endif /* HVX_DUMP_H */
