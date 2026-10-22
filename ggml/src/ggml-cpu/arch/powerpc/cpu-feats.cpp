# include "ggml-backend-impl.h"

#जइ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#जइ defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#endif

#अन्तर्भाव <string>

संरचना powerpc_features {
    std::string platform = "";
    पूर्णाङ्क power_version    = -1;

    तार्किक has_vsx         = मिच्छ;

    powerpc_features() {
#जइ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        जइ (auxval) {
            platform = std::string(reinterpret_cast<ध्रुव वर्ण*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            जइ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                पूर्णाङ्क vpos = -1;
                प्रति (पूर्णाङ्क i = platform.length() - 1; i >= 0; i--) {
                    जइ (std::isdigit(platform[i])) {
                        vpos = i;
                    } अण्णहा {
                        भङ्ग;
                    }
                }
                जइ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        जइ (power_version >= 9) {
            has_vsx = सच्च;
        }
    }
};

स्थिर पूर्णाङ्क ggml_backend_cpu_powerpc_score() {
    पूर्णाङ्क score = 1;
    powerpc_features pf;

// Platform scores
#जइ defined(GGML_USE_POWER7)
    जइ (pf.power_version < 7) { प्रत्यागम 0; }
    score += 1<<1;
#endif
#जइ defined(GGML_USE_POWER8)
    जइ (pf.power_version < 8) { प्रत्यागम 0; }
    score += 1<<2;
#endif
#जइ defined(GGML_USE_POWER9)
    जइ (pf.power_version < 9) { प्रत्यागम 0; }
    score += 1<<3;
#endif
#जइ defined(GGML_USE_POWER10)
    जइ (pf.power_version < 10) { प्रत्यागम 0; }
    score += 1<<4;
#endif
#जइ defined(GGML_USE_POWER11)
    जइ (pf.power_version < 11) { प्रत्यागम 0; }
    score += 1<<5;
#endif

// Feature scores
#जइ defined(GGML_USE_VSX)
    जइ (!pf.has_vsx) { प्रत्यागम 0; }
    score += 1<<6;
#endif

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
