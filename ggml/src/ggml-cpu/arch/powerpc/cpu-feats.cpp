# include "ggml-backend-impl.h"

#सचे defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#सचे defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#endif

#अन्तर्भाव <string>

संरचना powerpc_features {
    std::string platform = "";
    पूर्णाङ्क power_version    = -1;

    तार्किक has_vsx         = मिच्छा;

    powerpc_features() {
#सचे defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        सचे (auxval) {
            platform = std::string(reinterpret_cast<ध्रुव वर्ण*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            सचे (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                पूर्णाङ्क vpos = -1;
                अत्थाय (पूर्णाङ्क i = platform.length() - 1; i >= 0; i--) {
                    सचे (std::isdigit(platform[i])) {
                        vpos = i;
                    } अञ्ञथा {
                        भङ्ग;
                    }
                }
                सचे (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        सचे (power_version >= 9) {
            has_vsx = सच्च;
        }
    }
};

स्थिर पूर्णाङ्क ggml_backend_cpu_powerpc_score() {
    पूर्णाङ्क score = 1;
    powerpc_features pf;

// Platform scores
#सचे defined(GGML_USE_POWER7)
    सचे (pf.power_version < 7) { पटिनिवत्त 0; }
    score += 1<<1;
#endif
#सचे defined(GGML_USE_POWER8)
    सचे (pf.power_version < 8) { पटिनिवत्त 0; }
    score += 1<<2;
#endif
#सचे defined(GGML_USE_POWER9)
    सचे (pf.power_version < 9) { पटिनिवत्त 0; }
    score += 1<<3;
#endif
#सचे defined(GGML_USE_POWER10)
    सचे (pf.power_version < 10) { पटिनिवत्त 0; }
    score += 1<<4;
#endif
#सचे defined(GGML_USE_POWER11)
    सचे (pf.power_version < 11) { पटिनिवत्त 0; }
    score += 1<<5;
#endif

// Feature scores
#सचे defined(GGML_USE_VSX)
    सचे (!pf.has_vsx) { पटिनिवत्त 0; }
    score += 1<<6;
#endif

    पटिनिवत्त score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
