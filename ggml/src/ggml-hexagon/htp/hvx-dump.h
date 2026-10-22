#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_DUMP_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_DUMP_H

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_f16_n(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v, uint32_t n) {
    HVX_ਸਦਿਸ਼_Alias u = { .v = v };

    ਅਚਲ uint32_t n0 = n / 16;
    ਅਚਲ uint32_t n1 = n % 16;
    ਪੂਰਨ            i  = 0;
    ਲਈ (; i < n0; i++) {
        hex_dump_f16_ਸਤਰ(pref, u.fp16 + (16 * i), 16);
    }
    ਜੇ (n1) {
        hex_dump_f16_ਸਤਰ(pref, u.fp16 + (16 * i), n1);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_f16(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_f32_n(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v, uint32_t n) {
    HVX_ਸਦਿਸ਼_Alias u = { .v = v };

    ਅਚਲ uint32_t n0 = n / 16;
    ਅਚਲ uint32_t n1 = n % 16;
    ਪੂਰਨ            i  = 0;
    ਲਈ (; i < n0; i++) {
        hex_dump_f32_ਸਤਰ(pref, u.fp32 + (16 * i), 16);
    }
    ਜੇ (n1) {
        hex_dump_f32_ਸਤਰ(pref, u.fp32 + (16 * i), n1);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_f32_hmt(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        ਦਸ਼ਮਲਵ      d[32];
    } u = { .v = v };

    FARF(ਉੱਚਾ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_f32(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_int32(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        int32_t    d[32];
    } u = { .v = v };

    ਲਈ (ਪੂਰਨ i = 0; i < 32 / 16; i++) {
        hex_dump_int32_ਸਤਰ(pref, u.d + (16 * i), 16);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_int32_hmt(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ਉੱਚਾ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_int8_hmt(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ਉੱਚਾ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_int8(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        int8_t     d[128];
    } u = { .v = v };

    ਲਈ (ਪੂਰਨ i = 0; i < 128 / 16; i++) {
        hex_dump_int8_ਸਤਰ(pref, u.d + (16 * i), 16);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_vec_dump_uint8(ਅੱਖਰ * pref, HVX_ਸਦਿਸ਼ v) {
    ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        uint8_t    d[128];
    } u = { .v = v };

    ਲਈ (ਪੂਰਨ i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_ਸਤਰ(pref, u.d + (16 * i), 16);
    }
}

ਸਥਿਰ bool hvx_vec_eq(HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1, size_t n) {
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਸੰਘ {
        HVX_ਸਦਿਸ਼ v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਜੇ (u0.d[i] != u1.d[i]) {
            ਵਾਪਸ false;
        }
    }

    ਵਾਪਸ true;
}

#ਜੇ_ਅੰਤ /* HVX_DUMP_H */
