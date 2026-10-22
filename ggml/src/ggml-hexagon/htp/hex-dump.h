#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_int8_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ int8_t * x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_uint8_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ uint8_t * x, uint32_t n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_uint32_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ uint32_t * x, uint32_t n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_int32_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ int32_t * x, uint32_t n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_f16_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ __fp16 * x, uint32_t n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (ଦଶମିକ) x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_f32_ଧାଡ଼ି(ଅକ୍ଷର * pref, ଅଚଳ ଦଶମିକ * x, uint32_t n) {
    ଅକ୍ଷର str[1024], *p = str, *p_end = str + ଆକାର(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(ଉପର_2, "%s\n", str);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_f32(ଅକ୍ଷର * pref, ଅଚଳ ଦଶମିକ * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ପାଇଁ (; i < n0; i++) {
        hex_dump_f32_ଧାଡ଼ି(pref, x + (16 * i), 16);
    }
    ଯଦି (n1) {
        hex_dump_f32_ଧାଡ଼ି(pref, x + (16 * i), n1);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_dump_f16(ଅକ୍ଷର * pref, ଅଚଳ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    ପାଇଁ (; i < n0; i++) {
        hex_dump_f16_ଧାଡ଼ି(pref, x + (16 * i), 16);
    }
    ଯଦି (n1) {
        hex_dump_f16_ଧାଡ଼ି(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
