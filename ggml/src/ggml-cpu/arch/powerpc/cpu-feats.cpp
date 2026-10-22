# include "ggml-backend-impl.h"

#אם defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#אם defined(__linux__)
#ضمن <sys/auxv.h>
#endif

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_version    = -1;

    منطقي has_vsx         = שקר;

    powerpc_features() {
#אם defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        אם (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            אם (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                עבור (صحيح i = platform.length() - 1; i >= 0; i--) {
                    אם (std::isdigit(platform[i])) {
                        vpos = i;
                    } אחרת {
                        اكسر;
                    }
                }
                אם (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        אם (power_version >= 9) {
            has_vsx = אמת;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#אם defined(GGML_USE_POWER7)
    אם (pf.power_version < 7) { החזר 0; }
    score += 1<<1;
#endif
#אם defined(GGML_USE_POWER8)
    אם (pf.power_version < 8) { החזר 0; }
    score += 1<<2;
#endif
#אם defined(GGML_USE_POWER9)
    אם (pf.power_version < 9) { החזר 0; }
    score += 1<<3;
#endif
#אם defined(GGML_USE_POWER10)
    אם (pf.power_version < 10) { החזר 0; }
    score += 1<<4;
#endif
#אם defined(GGML_USE_POWER11)
    אם (pf.power_version < 11) { החזר 0; }
    score += 1<<5;
#endif

// Feature scores
#אם defined(GGML_USE_VSX)
    אם (!pf.has_vsx) { החזר 0; }
    score += 1<<6;
#endif

    החזר score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
