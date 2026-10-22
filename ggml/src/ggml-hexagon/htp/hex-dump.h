#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#அந்தர்பாவ <HAP_farf.h>

நிலையான உள்வரி வெறுமை hex_dump_int8_வரி(எழுத்து * pref, மாறிலி int8_t * x, முழு n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_uint8_வரி(எழுத்து * pref, மாறிலி uint8_t * x, uint32_t n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_uint32_வரி(எழுத்து * pref, மாறிலி uint32_t * x, uint32_t n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned முழு) x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_int32_வரி(எழுத்து * pref, மாறிலி int32_t * x, uint32_t n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (முழு) x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_f16_வரி(எழுத்து * pref, மாறிலி __fp16 * x, uint32_t n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (தசம) x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_f32_வரி(எழுத்து * pref, மாறிலி தசம * x, uint32_t n) {
    எழுத்து str[1024], *p = str, *p_end = str + அளவு(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    க்கு (முழு i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(உயர்வு_2, "%s\n", str);
}

நிலையான உள்வரி வெறுமை hex_dump_f32(எழுத்து * pref, மாறிலி தசம * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    க்கு (; i < n0; i++) {
        hex_dump_f32_வரி(pref, x + (16 * i), 16);
    }
    எனில் (n1) {
        hex_dump_f32_வரி(pref, x + (16 * i), n1);
    }
}

நிலையான உள்வரி வெறுமை hex_dump_f16(எழுத்து * pref, மாறிலி __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    க்கு (; i < n0; i++) {
        hex_dump_f16_வரி(pref, x + (16 * i), 16);
    }
    எனில் (n1) {
        hex_dump_f16_வரி(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
