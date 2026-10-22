#నిర్వచితం_కాకపోతే HEX_DUMP_H
#నిర్వచించు HEX_DUMP_H

#అన్తర్భావ <HAP_farf.h>

స్థిర అంతఃవరుస శూన్య hex_dump_int8_పంక్తి(అక్షరం * pref, స్థిరరాశి int8_t * x, పూర్ణాంకం n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n && p < p_చివర; i++) {
        p += snprintf(p, p_చివర - p, "%d, ", x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_uint8_పంక్తి(అక్షరం * pref, స్థిరరాశి uint8_t * x, uint32_t n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n && p < p_చివర; i++) {
        p += snprintf(p, p_చివర - p, "%d, ", x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_uint32_పంక్తి(అక్షరం * pref, స్థిరరాశి uint32_t * x, uint32_t n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        p += snprintf(p, p_చివర - p, "%u, ", (గుర్తులేని పూర్ణాంకం) x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_int32_పంక్తి(అక్షరం * pref, స్థిరరాశి int32_t * x, uint32_t n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        p += snprintf(p, p_చివర - p, "%d, ", (పూర్ణాంకం) x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_f16_పంక్తి(అక్షరం * pref, స్థిరరాశి __fp16 * x, uint32_t n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        p += snprintf(p, p_చివర - p, "%.6f, ", (దశాంశం) x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_f32_పంక్తి(అక్షరం * pref, స్థిరరాశి దశాంశం * x, uint32_t n) {
    అక్షరం str[1024], *p = str, *p_చివర = str + పరిమాణం(str);
    p += snprintf(p, p_చివర - p, "%s: ", pref);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        p += snprintf(p, p_చివర - p, "%.6f, ", x[i]);
    }
    FARF(ఎగువ_2, "%s\n", str);
}

స్థిర అంతఃవరుస శూన్య hex_dump_f32(అక్షరం * pref, స్థిరరాశి దశాంశం * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    కొరకు (; i < n0; i++) {
        hex_dump_f32_పంక్తి(pref, x + (16 * i), 16);
    }
    ఒకవేళ (n1) {
        hex_dump_f32_పంక్తి(pref, x + (16 * i), n1);
    }
}

స్థిర అంతఃవరుస శూన్య hex_dump_f16(అక్షరం * pref, స్థిరరాశి __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    కొరకు (; i < n0; i++) {
        hex_dump_f16_పంక్తి(pref, x + (16 * i), 16);
    }
    ఒకవేళ (n1) {
        hex_dump_f16_పంక్తి(pref, x + (16 * i), n1);
    }
}

#అయితే_ముగింపు /* HEX_DUMP_H */
