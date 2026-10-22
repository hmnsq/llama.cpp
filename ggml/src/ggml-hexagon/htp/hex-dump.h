#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include <HAP_farf.h>

static anta_hstha void hex_dump_int8_line(var_na * pref, const int8_t * x, ကိန်းပြည့် n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_uint8_line(var_na * pref, const uint8_t * x, uint32_t n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_uint32_line(var_na * pref, const uint32_t * x, uint32_t n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned ကိန်းပြည့်) x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_int32_line(var_na * pref, const int32_t * x, uint32_t n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (ကိန်းပြည့်) x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_f16_line(var_na * pref, const __fp16 * x, uint32_t n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (float) x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_f32_line(var_na * pref, const float * x, uint32_t n) {
    var_na str[1024], *p = str, *p_end = str + parimaa_na(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    အတွက် (ကိန်းပြည့် i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static anta_hstha void hex_dump_f32(var_na * pref, const float * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    အတွက် (; i < n0; i++) {
        hex_dump_f32_line(pref, x + (16 * i), 16);
    }
    အကယ်၍ (n1) {
        hex_dump_f32_line(pref, x + (16 * i), n1);
    }
}

static anta_hstha void hex_dump_f16(var_na * pref, const __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    အတွက် (; i < n0; i++) {
        hex_dump_f16_line(pref, x + (16 * i), 16);
    }
    အကယ်၍ (n1) {
        hex_dump_f16_line(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
