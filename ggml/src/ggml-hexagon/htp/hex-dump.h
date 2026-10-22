#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#অন্তর্ভাব <HAP_farf.h>

স্থির অন্তঃরেখ শূন্য hex_dump_int8_সারি(অক্ষর * pref, ধ্রুবক int8_t * x, পূর্ণসংখ্যা n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_uint8_সারি(অক্ষর * pref, ধ্রুবক uint8_t * x, uint32_t n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_uint32_সারি(অক্ষর * pref, ধ্রুবক uint32_t * x, uint32_t n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned পূর্ণসংখ্যা) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_int32_সারি(অক্ষর * pref, ধ্রুবক int32_t * x, uint32_t n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (পূর্ণসংখ্যা) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_f16_সারি(অক্ষর * pref, ধ্রুবক __fp16 * x, uint32_t n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (দশমিক) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_f32_সারি(অক্ষর * pref, ধ্রুবক দশমিক * x, uint32_t n) {
    অক্ষর str[1024], *p = str, *p_end = str + আকার(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃরেখ শূন্য hex_dump_f32(অক্ষর * pref, ধ্রুবক দশমিক * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    জন্য (; i < n0; i++) {
        hex_dump_f32_সারি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f32_সারি(pref, x + (16 * i), n1);
    }
}

স্থির অন্তঃরেখ শূন্য hex_dump_f16(অক্ষর * pref, ধ্রুবক __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    জন্য (; i < n0; i++) {
        hex_dump_f16_সারি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f16_সারি(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
