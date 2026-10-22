#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_DUMP_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_DUMP_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_int8_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ int8_t * x, ಪೂರ್ಣಾಂಕ n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n && p < p_ಕೊನೆ; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%d, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_uint8_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ uint8_t * x, uint32_t n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n && p < p_ಕೊನೆ; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%d, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_uint32_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ uint32_t * x, uint32_t n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%u, ", (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_int32_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ int32_t * x, uint32_t n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%d, ", (ಪೂರ್ಣಾಂಕ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_f16_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ __fp16 * x, uint32_t n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%.6f, ", (ದಶಮಾಂಶ) x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_f32_ಸಾಲು(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, uint32_t n) {
    ಅಕ್ಷರ str[1024], *p = str, *p_ಕೊನೆ = str + ಗಾತ್ರ(str);
    p += snprintf(p, p_ಕೊನೆ - p, "%s: ", pref);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        p += snprintf(p, p_ಕೊನೆ - p, "%.6f, ", x[i]);
    }
    FARF(ಮೇಲಿನ_2, "%s\n", str);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_f32(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ಗಾಗಿ (; i < n0; i++) {
        hex_dump_f32_ಸಾಲು(pref, x + (16 * i), 16);
    }
    ಆದರೆ (n1) {
        hex_dump_f32_ಸಾಲು(pref, x + (16 * i), n1);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_dump_f16(ಅಕ್ಷರ * pref, ಸ್ಥಿರಾಂಕ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ಗಾಗಿ (; i < n0; i++) {
        hex_dump_f16_ಸಾಲು(pref, x + (16 * i), 16);
    }
    ಆದರೆ (n1) {
        hex_dump_f16_ಸಾಲು(pref, x + (16 * i), n1);
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HEX_DUMP_H */
