#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#અન્તર્ભાવ <HAP_farf.h>

સ્થિર અંતઃરેખ શૂન્ય hex_dump_int8_લીટી(અક્ષર * pref, અચલ int8_t * x, પૂર્ણાંક n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_uint8_લીટી(અક્ષર * pref, અચલ uint8_t * x, uint32_t n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_uint32_લીટી(અક્ષર * pref, અચલ uint32_t * x, uint32_t n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned પૂર્ણાંક) x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_int32_લીટી(અક્ષર * pref, અચલ int32_t * x, uint32_t n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (પૂર્ણાંક) x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_f16_લીટી(અક્ષર * pref, અચલ __fp16 * x, uint32_t n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (દશાંશ) x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_f32_લીટી(અક્ષર * pref, અચલ દશાંશ * x, uint32_t n) {
    અક્ષર str[1024], *p = str, *p_end = str + કદ(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(ઉપલું_2, "%s\n", str);
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_f32(અક્ષર * pref, અચલ દશાંશ * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    માટે (; i < n0; i++) {
        hex_dump_f32_લીટી(pref, x + (16 * i), 16);
    }
    જો (n1) {
        hex_dump_f32_લીટી(pref, x + (16 * i), n1);
    }
}

સ્થિર અંતઃરેખ શૂન્ય hex_dump_f16(અક્ષર * pref, અચલ __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    માટે (; i < n0; i++) {
        hex_dump_f16_લીટી(pref, x + (16 * i), 16);
    }
    જો (n1) {
        hex_dump_f16_લીટી(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
