#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include <HAP_farf.h>

static inline void hex_dump_int8_लिन्या(char * pref, const int8_t * x, int n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_uint8_लिन्या(char * pref, const uint8_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_uint32_लिन्या(char * pref, const uint32_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned int) x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_int32_लिन्या(char * pref, const int32_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (int) x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_f16_लिन्या(char * pref, const __fp16 * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (float) x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_f32_लिन्या(char * pref, const float * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(माताअस_2, "%s\n", str);
}

static inline void hex_dump_f32(char * pref, const float * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        hex_dump_f32_लिन्या(pref, x + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f32_लिन्या(pref, x + (16 * i), n1);
    }
}

static inline void hex_dump_f16(char * pref, const __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        hex_dump_f16_लिन्या(pref, x + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f16_लिन्या(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
