#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#അന്തര്ഭാവ <HAP_farf.h>

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_int8_വരി(അക്ഷരം * pref, സ്ഥിരം int8_t * x, പൂർണ്ണസംഖ്യ n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_uint8_വരി(അക്ഷരം * pref, സ്ഥിരം uint8_t * x, uint32_t n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_uint32_വരി(അക്ഷരം * pref, സ്ഥിരം uint32_t * x, uint32_t n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%u, ", (unsigned പൂർണ്ണസംഖ്യ) x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_int32_വരി(അക്ഷരം * pref, സ്ഥിരം int32_t * x, uint32_t n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (പൂർണ്ണസംഖ്യ) x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_f16_വരി(അക്ഷരം * pref, സ്ഥിരം __fp16 * x, uint32_t n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (ദശാംശം) x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_f32_വരി(അക്ഷരം * pref, സ്ഥിരം ദശാംശം * x, uint32_t n) {
    അക്ഷരം str[1024], *p = str, *p_end = str + വലുപ്പം(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(മുകളിലെ_2, "%s\n", str);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_f32(അക്ഷരം * pref, സ്ഥിരം ദശാംശം * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    വേണ്ടി (; i < n0; i++) {
        hex_dump_f32_വരി(pref, x + (16 * i), 16);
    }
    എങ്കിൽ (n1) {
        hex_dump_f32_വരി(pref, x + (16 * i), n1);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hex_dump_f16(അക്ഷരം * pref, സ്ഥിരം __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    വേണ്ടി (; i < n0; i++) {
        hex_dump_f16_വരി(pref, x + (16 * i), 16);
    }
    എങ്കിൽ (n1) {
        hex_dump_f16_വരി(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
