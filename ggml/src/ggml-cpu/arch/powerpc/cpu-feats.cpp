# include "ggml-backend-impl.h"

#यदि defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#यदि defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#endif

#अन्तर्भाव <string>

संरचना powerpc_features {
    std::string platform = "";
    पूर्णाङ्क power_version    = -1;

    तार्किक has_vsx         = असत्य;

    powerpc_features() {
#यदि defined(__linux__)
        अचिन्हित दीर्घ auxval = getauxval(AT_PLATFORM);
        यदि (auxval) {
            platform = std::string(reinterpret_cast<ध्रुव वर्ण*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            यदि (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                पूर्णाङ्क vpos = -1;
                प्रति (पूर्णाङ्क i = platform.length() - 1; i >= 0; i--) {
                    यदि (std::है_अंक(platform[i])) {
                        vpos = i;
                    } अन्यथा {
                        भङ्ग;
                    }
                }
                यदि (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        यदि (power_version >= 9) {
            has_vsx = सत्य;
        }
    }
};

स्थिर पूर्णाङ्क ggml_backend_cpu_powerpc_score() {
    पूर्णाङ्क score = 1;
    powerpc_features pf;

// Platform scores
#यदि defined(GGML_USE_POWER7)
    यदि (pf.power_version < 7) { प्रत्यागम 0; }
    score += 1<<1;
#endif
#यदि defined(GGML_USE_POWER8)
    यदि (pf.power_version < 8) { प्रत्यागम 0; }
    score += 1<<2;
#endif
#यदि defined(GGML_USE_POWER9)
    यदि (pf.power_version < 9) { प्रत्यागम 0; }
    score += 1<<3;
#endif
#यदि defined(GGML_USE_POWER10)
    यदि (pf.power_version < 10) { प्रत्यागम 0; }
    score += 1<<4;
#endif
#यदि defined(GGML_USE_POWER11)
    यदि (pf.power_version < 11) { प्रत्यागम 0; }
    score += 1<<5;
#endif

// Feature scores
#यदि defined(GGML_USE_VSX)
    यदि (!pf.has_vsx) { प्रत्यागम 0; }
    score += 1<<6;
#endif

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
