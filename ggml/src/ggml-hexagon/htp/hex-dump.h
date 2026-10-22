#جے_بنا_تعریف HEX_DUMP_H
#تعریف_کرو HEX_DUMP_H

#ضمن <HAP_farf.h>

ساكن مضمن فراغ hex_dump_int8_سطر(حرف * pref, ثابت int8_t * x, صحيح n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_سرا; i++) {
        p += snprintf(p, p_سرا - p, "%d, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint8_سطر(حرف * pref, ثابت uint8_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n && p < p_سرا; i++) {
        p += snprintf(p, p_سرا - p, "%d, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_uint32_سطر(حرف * pref, ثابت uint32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_سرا - p, "%u, ", (بنانشانی صحيح) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_int32_سطر(حرف * pref, ثابت int32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_سرا - p, "%d, ", (صحيح) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f16_سطر(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_سرا - p, "%.6f, ", (عائم) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32_سطر(حرف * pref, ثابت عائم * x, uint32_t n) {
    حرف str[1024], *p = str, *p_سرا = str + حجم(str);
    p += snprintf(p, p_سرا - p, "%s: ", pref);
    لكل (صحيح i = 0; i < n; i++) {
        p += snprintf(p, p_سرا - p, "%.6f, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساكن مضمن فراغ hex_dump_f32(حرف * pref, ثابت عائم * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f32_سطر(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f32_سطر(pref, x + (16 * i), n1);
    }
}

ساكن مضمن فراغ hex_dump_f16(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    لكل (; i < n0; i++) {
        hex_dump_f16_سطر(pref, x + (16 * i), 16);
    }
    إذا (n1) {
        hex_dump_f16_سطر(pref, x + (16 * i), n1);
    }
}

#جے_انت /* HEX_DUMP_H */
