#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_DUMP_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_DUMP_H

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_int8_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ int8_t * x, ਪੂਰਨ n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n && p < p_ਸਿਰਾ; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%d, ", x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_uint8_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ uint8_t * x, uint32_t n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n && p < p_ਸਿਰਾ; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%d, ", x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_uint32_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ uint32_t * x, uint32_t n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%u, ", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_int32_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ int32_t * x, uint32_t n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%d, ", (ਪੂਰਨ) x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_f16_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ __fp16 * x, uint32_t n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%.6f, ", (ਦਸ਼ਮਲਵ) x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_f32_ਸਤਰ(ਅੱਖਰ * pref, ਅਚਲ ਦਸ਼ਮਲਵ * x, uint32_t n) {
    ਅੱਖਰ str[1024], *p = str, *p_ਸਿਰਾ = str + ਆਕਾਰ(str);
    p += snprintf(p, p_ਸਿਰਾ - p, "%s: ", pref);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        p += snprintf(p, p_ਸਿਰਾ - p, "%.6f, ", x[i]);
    }
    FARF(ਉੱਚਾ_2, "%s\n", str);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_f32(ਅੱਖਰ * pref, ਅਚਲ ਦਸ਼ਮਲਵ * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ਲਈ (; i < n0; i++) {
        hex_dump_f32_ਸਤਰ(pref, x + (16 * i), 16);
    }
    ਜੇ (n1) {
        hex_dump_f32_ਸਤਰ(pref, x + (16 * i), n1);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_dump_f16(ਅੱਖਰ * pref, ਅਚਲ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ਲਈ (; i < n0; i++) {
        hex_dump_f16_ਸਤਰ(pref, x + (16 * i), 16);
    }
    ਜੇ (n1) {
        hex_dump_f16_ਸਤਰ(pref, x + (16 * i), n1);
    }
}

#ਜੇ_ਅੰਤ /* HEX_DUMP_H */
