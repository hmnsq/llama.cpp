#यदि_अपरिभाषितम् HEX_DUMP_H
#परिभाषय HEX_DUMP_H

#अन्तर्भाव <HAP_farf.h>

स्थिर अन्तःस्थ शून्य hex_dump_int8_पङ्क्तिः(वर्ण * pref, ध्रुव int8_t * x, पूर्णाङ्क n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n && p < p_अन्तः; i++) {
        p += snprintf(p, p_अन्तः - p, "%d, ", x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_uint8_पङ्क्तिः(वर्ण * pref, ध्रुव uint8_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n && p < p_अन्तः; i++) {
        p += snprintf(p, p_अन्तः - p, "%d, ", x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_uint32_पङ्क्तिः(वर्ण * pref, ध्रुव uint32_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_अन्तः - p, "%u, ", (अचिह्नित पूर्णाङ्क) x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_int32_पङ्क्तिः(वर्ण * pref, ध्रुव int32_t * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_अन्तः - p, "%d, ", (पूर्णाङ्क) x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f16_पङ्क्तिः(वर्ण * pref, ध्रुव __fp16 * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_अन्तः - p, "%.6f, ", (भिन्न) x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f32_पङ्क्तिः(वर्ण * pref, ध्रुव भिन्न * x, uint32_t n) {
    वर्ण str[1024], *p = str, *p_अन्तः = str + परिमाण(str);
    p += snprintf(p, p_अन्तः - p, "%s: ", pref);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        p += snprintf(p, p_अन्तः - p, "%.6f, ", x[i]);
    }
    FARF(उच्चः_2, "%s\n", str);
}

स्थिर अन्तःस्थ शून्य hex_dump_f32(वर्ण * pref, ध्रुव भिन्न * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    प्रति (; i < n0; i++) {
        hex_dump_f32_पङ्क्तिः(pref, x + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f32_पङ्क्तिः(pref, x + (16 * i), n1);
    }
}

स्थिर अन्तःस्थ शून्य hex_dump_f16(वर्ण * pref, ध्रुव __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    प्रति (; i < n0; i++) {
        hex_dump_f16_पङ्क्तिः(pref, x + (16 * i), 16);
    }
    यदि (n1) {
        hex_dump_f16_पङ्क्तिः(pref, x + (16 * i), n1);
    }
}

#यदि_अन्तः /* HEX_DUMP_H */
