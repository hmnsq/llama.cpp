#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#অন্তর্ভাব <HAP_farf.h>

স্থির অন্তঃস্থ শূন্য hex_dump_int8_সারি(বর্ণ * pref, ধ্রুব int8_t * x, পূর্ণাঙ্ক n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_uint8_সারি(বর্ণ * pref, ধ্রুব uint8_t * x, uint32_t n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_uint32_সারি(বর্ণ * pref, ধ্রুব uint32_t * x, uint32_t n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned পূর্ণাঙ্ক) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_int32_সারি(বর্ণ * pref, ধ্রুব int32_t * x, uint32_t n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (পূর্ণাঙ্ক) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f16_সারি(বর্ণ * pref, ধ্রুব __fp16 * x, uint32_t n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (ভিন্ন) x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f32_সারি(বর্ণ * pref, ধ্রুব ভিন্ন * x, uint32_t n) {
    বর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(উপরের_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f32(বর্ণ * pref, ধ্রুব ভিন্ন * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    প্রতি (; i < n0; i++) {
        hex_dump_f32_সারি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f32_সারি(pref, x + (16 * i), n1);
    }
}

স্থির অন্তঃস্থ শূন্য hex_dump_f16(বর্ণ * pref, ধ্রুব __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    প্রতি (; i < n0; i++) {
        hex_dump_f16_সারি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f16_সারি(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
