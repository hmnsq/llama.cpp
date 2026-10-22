#अगर_अघोषित HEX_DUMP_H
#घोषणा HEX_DUMP_H

#समावेश <HAP_farf.h>

अटल अंतरभूत व्योम hex_dump_int8_पंक्ति(अक्षर * pref, स्थिर int8_t * x, पूर्णांक n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_uint8_पंक्ति(अक्षर * pref, स्थिर uint8_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_uint32_पंक्ति(अक्षर * pref, स्थिर uint32_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (अचिन्हित पूर्णांक) x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_int32_पंक्ति(अक्षर * pref, स्थिर int32_t * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (पूर्णांक) x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_f16_पंक्ति(अक्षर * pref, स्थिर __fp16 * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (भग्न) x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_f32_पंक्ति(अक्षर * pref, स्थिर भग्न * x, uint32_t n) {
    अक्षर str[1024], *p = str, *p_end = str + माप(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    क्रम (पूर्णांक i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(ऊपरी_2, "%s\n", str);
}

अटल अंतरभूत व्योम hex_dump_f32(अक्षर * pref, स्थिर भग्न * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    क्रम (; i < n0; i++) {
        hex_dump_f32_पंक्ति(pref, x + (16 * i), 16);
    }
    अगर (n1) {
        hex_dump_f32_पंक्ति(pref, x + (16 * i), n1);
    }
}

अटल अंतरभूत व्योम hex_dump_f16(अक्षर * pref, स्थिर __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    क्रम (; i < n0; i++) {
        hex_dump_f16_पंक्ति(pref, x + (16 * i), 16);
    }
    अगर (n1) {
        hex_dump_f16_पंक्ति(pref, x + (16 * i), n1);
    }
}

#पूर्ण_अगर /* HEX_DUMP_H */
