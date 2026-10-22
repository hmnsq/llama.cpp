# include "ggml-backend-impl.h"

#إذا defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#إذا defined(__linux__)
#ضمن <sys/auxv.h>
#endif

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_version    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#إذا defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        إذا (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            إذا (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.length() - 1; i >= 0; i--) {
                    إذا (std::isdigit(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                إذا (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        إذا (power_version >= 9) {
            has_vsx = true;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#إذا defined(GGML_USE_POWER7)
    إذا (pf.power_version < 7) { إرجاع 0; }
    score += 1<<1;
#endif
#إذا defined(GGML_USE_POWER8)
    إذا (pf.power_version < 8) { إرجاع 0; }
    score += 1<<2;
#endif
#إذا defined(GGML_USE_POWER9)
    إذا (pf.power_version < 9) { إرجاع 0; }
    score += 1<<3;
#endif
#إذا defined(GGML_USE_POWER10)
    إذا (pf.power_version < 10) { إرجاع 0; }
    score += 1<<4;
#endif
#إذا defined(GGML_USE_POWER11)
    إذا (pf.power_version < 11) { إرجاع 0; }
    score += 1<<5;
#endif

// Feature scores
#إذا defined(GGML_USE_VSX)
    إذا (!pf.has_vsx) { إرجاع 0; }
    score += 1<<6;
#endif

    إرجاع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
