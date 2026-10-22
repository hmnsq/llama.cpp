#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#अन्तर्भाव <HAP_farf.h>

स्थिर अन्तःस्थ शून्य hex_dump_int8_पङ्क्ति(वर्ण * pref, अचर int8_t * x, पूर्णाङ्क n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_uint8_पङ्क्ति(वर्ण * pref, अचर uint8_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_uint32_पङ्क्ति(वर्ण * pref, अचर uint32_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (अचिन्हित पूर्णाङ्क) x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_int32_पङ्क्ति(वर्ण * pref, अचर int32_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (पूर्णाङ्क) x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f16_पङ्क्ति(वर्ण * pref, अचर __fp16 * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (भिन्न) x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f32_पङ्क्ति(वर्ण * pref, अचर भिन्न * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_end = str + परिमाण(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(माथिल्लो_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f32(वर्ण * pref, अचर भिन्न * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    लागि (; i < n0; i++) {
        hex_dump_f32_पङ्क्ति(pref, x + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f32_पङ्क्ति(pref, x + (16 * i), n1);
    }
}

स्थिर अन्तःस्थ शून्य hex_dump_f16(वर्ण * pref, अचर __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    लागि (; i < n0; i++) {
        hex_dump_f16_पङ्क्ति(pref, x + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f16_पङ्क्ति(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
