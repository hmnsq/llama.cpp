#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hvx-types.h"

நிலையான வெறுமை hvx_vec_dump_f16_n(எழுத்து * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    மாறிலி uint32_t n0 = n / 16;
    மாறிலி uint32_t n1 = n % 16;
    முழு            i  = 0;
    க்கு (; i < n0; i++) {
        hex_dump_f16_வரி(pref, u.fp16 + (16 * i), 16);
    }
    எனில் (n1) {
        hex_dump_f16_வரி(pref, u.fp16 + (16 * i), n1);
    }
}

நிலையான வெறுமை hvx_vec_dump_f16(எழுத்து * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

நிலையான வெறுமை hvx_vec_dump_f32_n(எழுத்து * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    மாறிலி uint32_t n0 = n / 16;
    மாறிலி uint32_t n1 = n % 16;
    முழு            i  = 0;
    க்கு (; i < n0; i++) {
        hex_dump_f32_வரி(pref, u.fp32 + (16 * i), 16);
    }
    எனில் (n1) {
        hex_dump_f32_வரி(pref, u.fp32 + (16 * i), n1);
    }
}

நிலையான வெறுமை hvx_vec_dump_f32_hmt(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        தசம      d[32];
    } u = { .v = v };

    FARF(உயர்வு_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

நிலையான வெறுமை hvx_vec_dump_f32(எழுத்து * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

நிலையான வெறுமை hvx_vec_dump_int32(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    க்கு (முழு i = 0; i < 32 / 16; i++) {
        hex_dump_int32_வரி(pref, u.d + (16 * i), 16);
    }
}

நிலையான வெறுமை hvx_vec_dump_int32_hmt(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(உயர்வு_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

நிலையான வெறுமை hvx_vec_dump_int8_hmt(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(உயர்வு_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

நிலையான வெறுமை hvx_vec_dump_int8(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    க்கு (முழு i = 0; i < 128 / 16; i++) {
        hex_dump_int8_வரி(pref, u.d + (16 * i), 16);
    }
}

நிலையான வெறுமை hvx_vec_dump_uint8(எழுத்து * pref, HVX_Vector v) {
    ஒன்றிப்பு {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    க்கு (முழு i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_வரி(pref, u.d + (16 * i), 16);
    }
}

நிலையான bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    வகைவரை ஒன்றிப்பு {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    க்கு (முழு i = 0; i < n; i++) {
        எனில் (u0.d[i] != u1.d[i]) {
            திரும்பு false;
        }
    }

    திரும்பு true;
}

#endif /* HVX_DUMP_H */
