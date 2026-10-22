#که_نه_تعریف_شوی HEX_DUMP_H
#تعریف_کړه HEX_DUMP_H

#ضمن <HAP_farf.h>

ساكن مضمن فراغ hex_dump_int8_کرښه(حرف * pref, ثابت int8_t * x, صحيح n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_پای_ټکی; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%d, ", x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint8_کرښه(حرف * pref, ثابت uint8_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_پای_ټکی; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%d, ", x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint32_کرښه(حرف * pref, ثابت uint32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%u, ", (بینښه صحيح) x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_int32_کرښه(حرف * pref, ثابت int32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%d, ", (صحيح) x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f16_کرښه(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%.6f, ", (عائم) x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32_کرښه(حرف * pref, ثابت عائم * x, uint32_t n) {
    حرف str[1024], *p = str, *p_پای_ټکی = str + حجم(str);
    p += snprintf(p, p_پای_ټکی - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_پای_ټکی - p, "%.6f, ", x[i]);
    }
    FARF(لوړ_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32(حرف * pref, ثابت عائم * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f32_کرښه(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f32_کرښه(pref, x + (16 * i), n1);
    }
}

ساكن مضمن فراغ hex_dump_f16(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f16_کرښه(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f16_کرښه(pref, x + (16 * i), n1);
    }
}

#د_که_پای /* HEX_DUMP_H */
