# include "ggml-backend-impl.h"

#जर defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#जर defined(__linux__)
#अन्तर्भाव <sys/auxv.h>
#endif

#अन्तर्भाव <string>

संरचना powerpc_features {
    std::string platform = "";
    पूर्णांक power_version    = -1;

    तार्किक has_vsx         = असत्य;

    powerpc_features() {
#जर defined(__linux__)
        अचिन्हित दीर्घ auxval = getauxval(AT_PLATFORM);
        जर (auxval) {
            platform = std::string(reinterpret_cast<स्थिरांक अक्षर*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            जर (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                पूर्णांक vpos = -1;
                साठी (पूर्णांक i = platform.length() - 1; i >= 0; i--) {
                    जर (std::है_अंक(platform[i])) {
                        vpos = i;
                    } अन्यथा {
                        थांबा;
                    }
                }
                जर (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        जर (power_version >= 9) {
            has_vsx = सत्य;
        }
    }
};

स्थिर पूर्णांक ggml_backend_cpu_powerpc_score() {
    पूर्णांक score = 1;
    powerpc_features pf;

// Platform scores
#जर defined(GGML_USE_POWER7)
    जर (pf.power_version < 7) { परत 0; }
    score += 1<<1;
#endif
#जर defined(GGML_USE_POWER8)
    जर (pf.power_version < 8) { परत 0; }
    score += 1<<2;
#endif
#जर defined(GGML_USE_POWER9)
    जर (pf.power_version < 9) { परत 0; }
    score += 1<<3;
#endif
#जर defined(GGML_USE_POWER10)
    जर (pf.power_version < 10) { परत 0; }
    score += 1<<4;
#endif
#जर defined(GGML_USE_POWER11)
    जर (pf.power_version < 11) { परत 0; }
    score += 1<<5;
#endif

// Feature scores
#जर defined(GGML_USE_VSX)
    जर (!pf.has_vsx) { परत 0; }
    score += 1<<6;
#endif

    परत score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
