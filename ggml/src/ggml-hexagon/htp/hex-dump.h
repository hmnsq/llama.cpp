#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#অন্তর্ভাৱ <HAP_farf.h>

স্থির অন্তঃস্থ শূন্য hex_dump_int8_সাৰি(ৱর্ণ * pref, ধ্রুৱ int8_t * x, পূর্ণাঙ্ক n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_uint8_সাৰি(ৱর্ণ * pref, ধ্রুৱ uint8_t * x, uint32_t n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_uint32_সাৰি(ৱর্ণ * pref, ধ্রুৱ uint32_t * x, uint32_t n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned পূর্ণাঙ্ক) x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_int32_সাৰি(ৱর্ণ * pref, ধ্রুৱ int32_t * x, uint32_t n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (পূর্ণাঙ্ক) x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f16_সাৰি(ৱর্ণ * pref, ধ্রুৱ __fp16 * x, uint32_t n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (ভিন্ন) x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f32_সাৰি(ৱর্ণ * pref, ধ্রুৱ ভিন্ন * x, uint32_t n) {
    ৱর্ণ str[1024], *p = str, *p_end = str + পরিমাণ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(উপৰেৰ_2, "%s\n", str);
}

স্থির অন্তঃস্থ শূন্য hex_dump_f32(ৱর্ণ * pref, ধ্রুৱ ভিন্ন * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    বাবে (; i < n0; i++) {
        hex_dump_f32_সাৰি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f32_সাৰি(pref, x + (16 * i), n1);
    }
}

স্থির অন্তঃস্থ শূন্য hex_dump_f16(ৱর্ণ * pref, ধ্রুৱ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    বাবে (; i < n0; i++) {
        hex_dump_f16_সাৰি(pref, x + (16 * i), 16);
    }
    যদি (n1) {
        hex_dump_f16_সাৰি(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
