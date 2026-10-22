#నిర్వచితం_కాకపోతే HVX_DUMP_H
#నిర్వచించు HVX_DUMP_H

#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hvx-types.h"

స్థిర శూన్య hvx_vec_dump_f16_n(అక్షరం * pref, HVX_సదిశ v, uint32_t n) {
    HVX_సదిశ_Alias u = { .v = v };

    స్థిరరాశి uint32_t n0 = n / 16;
    స్థిరరాశి uint32_t n1 = n % 16;
    పూర్ణాంకం            i  = 0;
    కొరకు (; i < n0; i++) {
        hex_dump_f16_పంక్తి(pref, u.fp16 + (16 * i), 16);
    }
    ఒకవేళ (n1) {
        hex_dump_f16_పంక్తి(pref, u.fp16 + (16 * i), n1);
    }
}

స్థిర శూన్య hvx_vec_dump_f16(అక్షరం * pref, HVX_సదిశ v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

స్థిర శూన్య hvx_vec_dump_f32_n(అక్షరం * pref, HVX_సదిశ v, uint32_t n) {
    HVX_సదిశ_Alias u = { .v = v };

    స్థిరరాశి uint32_t n0 = n / 16;
    స్థిరరాశి uint32_t n1 = n % 16;
    పూర్ణాంకం            i  = 0;
    కొరకు (; i < n0; i++) {
        hex_dump_f32_పంక్తి(pref, u.fp32 + (16 * i), 16);
    }
    ఒకవేళ (n1) {
        hex_dump_f32_పంక్తి(pref, u.fp32 + (16 * i), n1);
    }
}

స్థిర శూన్య hvx_vec_dump_f32_hmt(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        దశాంశం      d[32];
    } u = { .v = v };

    FARF(ఎగువ_2, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

స్థిర శూన్య hvx_vec_dump_f32(అక్షరం * pref, HVX_సదిశ v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

స్థిర శూన్య hvx_vec_dump_int32(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        int32_t    d[32];
    } u = { .v = v };

    కొరకు (పూర్ణాంకం i = 0; i < 32 / 16; i++) {
        hex_dump_int32_పంక్తి(pref, u.d + (16 * i), 16);
    }
}

స్థిర శూన్య hvx_vec_dump_int32_hmt(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(ఎగువ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

స్థిర శూన్య hvx_vec_dump_int8_hmt(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(ఎగువ_2, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

స్థిర శూన్య hvx_vec_dump_int8(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        int8_t     d[128];
    } u = { .v = v };

    కొరకు (పూర్ణాంకం i = 0; i < 128 / 16; i++) {
        hex_dump_int8_పంక్తి(pref, u.d + (16 * i), 16);
    }
}

స్థిర శూన్య hvx_vec_dump_uint8(అక్షరం * pref, HVX_సదిశ v) {
    సంఘం {
        HVX_సదిశ v;
        uint8_t    d[128];
    } u = { .v = v };

    కొరకు (పూర్ణాంకం i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_పంక్తి(pref, u.d + (16 * i), 16);
    }
}

స్థిర bool hvx_vec_eq(HVX_సదిశ v0, HVX_సదిశ v1, size_t n) {
    రకనిర్వచనం సంఘం {
        HVX_సదిశ v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        ఒకవేళ (u0.d[i] != u1.d[i]) {
            తిరిగి false;
        }
    }

    తిరిగి true;
}

#అయితే_ముగింపు /* HVX_DUMP_H */
