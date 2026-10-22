#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#अन्तर्भाव <HAP_farf.h>

स्थिर ओळीत शून्य hex_dump_int8_ओळ(अक्षर * pref, स्थिरांक int8_t * x, पूर्णांक n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_uint8_ओळ(अक्षर * pref, स्थिरांक uint8_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_uint32_ओळ(अक्षर * pref, स्थिरांक uint32_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (अचिन्हित पूर्णांक) x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_int32_ओळ(अक्षर * pref, स्थिरांक int32_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (पूर्णांक) x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_f16_ओळ(अक्षर * pref, स्थिरांक __fp16 * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (दशांश) x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_f32_ओळ(अक्षर * pref, स्थिरांक दशांश * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + आकार(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    साठी (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(वरचा_2, "%s\n", str);
}

स्थिर ओळीत शून्य hex_dump_f32(अक्षर * pref, स्थिरांक दशांश * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    साठी (; i < n0; i++) {
        hex_dump_f32_ओळ(pref, x + (16 * i), 16);
    }
    जर (n1) {
        hex_dump_f32_ओळ(pref, x + (16 * i), n1);
    }
}

स्थिर ओळीत शून्य hex_dump_f16(अक्षर * pref, स्थिरांक __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    साठी (; i < n0; i++) {
        hex_dump_f16_ओळ(pref, x + (16 * i), 16);
    }
    जर (n1) {
        hex_dump_f16_ओळ(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
