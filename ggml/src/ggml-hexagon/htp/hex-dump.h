#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#شامل <HAP_farf.h>

ایستا درونخطی تهی hex_dump_int8_سطر(نویسه * pref, ثابت int8_t * x, صحیح n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_uint8_سطر(نویسه * pref, ثابت uint8_t * x, uint32_t n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_uint32_سطر(نویسه * pref, ثابت uint32_t * x, uint32_t n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned صحیح) x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_int32_سطر(نویسه * pref, ثابت int32_t * x, uint32_t n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (صحیح) x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_f16_سطر(نویسه * pref, ثابت __fp16 * x, uint32_t n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (اعشاری) x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_f32_سطر(نویسه * pref, ثابت اعشاری * x, uint32_t n) {
    نویسه str[1024], *p = str, *p_end = str + اندازه(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    برای (صحیح i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(بالا_2, "%s\n", str);
}

ایستا درونخطی تهی hex_dump_f32(نویسه * pref, ثابت اعشاری * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    برای (; i < n0; i++) {
        hex_dump_f32_سطر(pref, x + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f32_سطر(pref, x + (16 * i), n1);
    }
}

ایستا درونخطی تهی hex_dump_f16(نویسه * pref, ثابت __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    برای (; i < n0; i++) {
        hex_dump_f16_سطر(pref, x + (16 * i), 16);
    }
    اگر (n1) {
        hex_dump_f16_سطر(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
