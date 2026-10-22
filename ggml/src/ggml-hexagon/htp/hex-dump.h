#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#ضمن <HAP_farf.h>

ساكن مضمن فراغ hex_dump_int8_قۇر(حرف * pref, ثابت int8_t * x, صحيح n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint8_قۇر(حرف * pref, ثابت uint8_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint32_قۇر(حرف * pref, ثابت uint32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned صحيح) x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_int32_قۇر(حرف * pref, ثابت int32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (صحيح) x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f16_قۇر(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (عائم) x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32_قۇر(حرف * pref, ثابت عائم * x, uint32_t n) {
    حرف str[1024], *p = str, *p_end = str + حجم(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(يۇقىرى_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32(حرف * pref, ثابت عائم * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f32_قۇر(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f32_قۇر(pref, x + (16 * i), n1);
    }
}

ساكن مضمن فراغ hex_dump_f16(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f16_قۇر(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f16_قۇر(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
