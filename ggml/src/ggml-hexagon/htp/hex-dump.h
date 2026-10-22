#اگر_غیر_تعریف_شدہ HEX_DUMP_H
#تعریف_کرو HEX_DUMP_H

#شامل <HAP_farf.h>

ساکن اندرونی خالی hex_dump_int8_سطر(حرف * pref, ثابت int8_t * x, صحیح n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n && p < p_انتہا; i++) {
        p += snprintf(p, p_انتہا - p, "%d, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_uint8_سطر(حرف * pref, ثابت uint8_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n && p < p_انتہا; i++) {
        p += snprintf(p, p_انتہا - p, "%d, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_uint32_سطر(حرف * pref, ثابت uint32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_انتہا - p, "%u, ", (غیرعلامتی صحیح) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_int32_سطر(حرف * pref, ثابت int32_t * x, uint32_t n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_انتہا - p, "%d, ", (صحیح) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_f16_سطر(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_انتہا - p, "%.6f, ", (اعشاری) x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_f32_سطر(حرف * pref, ثابت اعشاری * x, uint32_t n) {
    حرف str[1024], *p = str, *p_انتہا = str + حجم(str);
    p += snprintf(p, p_انتہا - p, "%s: ", pref);
    برائے (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_انتہا - p, "%.6f, ", x[i]);
    }
    FARF(بلند_2, "%s\n", str);
}

ساکن اندرونی خالی hex_dump_f32(حرف * pref, ثابت اعشاری * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    برائے (; i < n0; i++) {
        hex_dump_f32_سطر(pref, x + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f32_سطر(pref, x + (16 * i), n1);
    }
}

ساکن اندرونی خالی hex_dump_f16(حرف * pref, ثابت __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    برائے (; i < n0; i++) {
        hex_dump_f16_سطر(pref, x + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f16_سطر(pref, x + (16 * i), n1);
    }
}

#اگر_ختم /* HEX_DUMP_H */
