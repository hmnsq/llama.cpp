# include "ggml-backend-impl.h"

#ถ้า defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ถ้า defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    จำนวนเต็ม power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ถ้า defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ถ้า (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ถ้า (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                จำนวนเต็ม vpos = -1;
                สำหรับ (จำนวนเต็ม i = platform.length() - 1; i >= 0; i--) {
                    ถ้า (std::isdigit(platform[i])) {
                        vpos = i;
                    } มิฉะนั้น {
                        bha_nga;
                    }
                }
                ถ้า (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ถ้า (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static จำนวนเต็ม ggml_backend_cpu_powerpc_score() {
    จำนวนเต็ม score = 1;
    powerpc_features pf;

// Platform scores
#ถ้า defined(GGML_USE_POWER7)
    ถ้า (pf.power_version < 7) { คืน 0; }
    score += 1<<1;
#endif
#ถ้า defined(GGML_USE_POWER8)
    ถ้า (pf.power_version < 8) { คืน 0; }
    score += 1<<2;
#endif
#ถ้า defined(GGML_USE_POWER9)
    ถ้า (pf.power_version < 9) { คืน 0; }
    score += 1<<3;
#endif
#ถ้า defined(GGML_USE_POWER10)
    ถ้า (pf.power_version < 10) { คืน 0; }
    score += 1<<4;
#endif
#ถ้า defined(GGML_USE_POWER11)
    ถ้า (pf.power_version < 11) { คืน 0; }
    score += 1<<5;
#endif

// Feature scores
#ถ้า defined(GGML_USE_VSX)
    ถ้า (!pf.has_vsx) { คืน 0; }
    score += 1<<6;
#endif

    คืน score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
