#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_int8_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ int8_t * x, ಪೂರ್ಣಾಙ್ಕ n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_uint8_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ uint8_t * x, uint32_t n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_uint32_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ uint32_t * x, uint32_t n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned ಪೂರ್ಣಾಙ್ಕ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_int32_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ int32_t * x, uint32_t n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (ಪೂರ್ಣಾಙ್ಕ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_f16_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ __fp16 * x, uint32_t n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (ಭಿನ್ನ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_f32_ಸಾಲು(ವರ್ಣ * pref, ಧ್ರುವ ಭಿನ್ನ * x, uint32_t n) {
    ವರ್ಣ str[1024], *p = str, *p_end = str + ಪರಿಮಾಣ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_f32(ವರ್ಣ * pref, ಧ್ರುವ ಭಿನ್ನ * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ಪ್ರತಿ (; i < n0; i++) {
        hex_dump_f32_ಸಾಲು(pref, x + (16 * i), 16);
    }
    ಯದಿ (n1) {
        hex_dump_f32_ಸಾಲು(pref, x + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_dump_f16(ವರ್ಣ * pref, ಧ್ರುವ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ಪ್ರತಿ (; i < n0; i++) {
        hex_dump_f16_ಸಾಲು(pref, x + (16 * i), 16);
    }
    ಯದಿ (n1) {
        hex_dump_f16_ಸಾಲು(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
