# include "ggml-backend-impl.h"

#ако defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ако defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    цяло power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ако defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ако (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ако (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                цяло vpos = -1;
                за (цяло i = platform.length() - 1; i >= 0; i--) {
                    ако (std::isdigit(platform[i])) {
                        vpos = i;
                    } иначе {
                        break;
                    }
                }
                ако (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ако (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static цяло ggml_backend_cpu_powerpc_score() {
    цяло score = 1;
    powerpc_features pf;

// Platform scores
#ако defined(GGML_USE_POWER7)
    ако (pf.power_version < 7) { върни 0; }
    score += 1<<1;
#endif
#ако defined(GGML_USE_POWER8)
    ако (pf.power_version < 8) { върни 0; }
    score += 1<<2;
#endif
#ако defined(GGML_USE_POWER9)
    ако (pf.power_version < 9) { върни 0; }
    score += 1<<3;
#endif
#ако defined(GGML_USE_POWER10)
    ако (pf.power_version < 10) { върни 0; }
    score += 1<<4;
#endif
#ако defined(GGML_USE_POWER11)
    ако (pf.power_version < 11) { върни 0; }
    score += 1<<5;
#endif

// Feature scores
#ако defined(GGML_USE_VSX)
    ако (!pf.has_vsx) { върни 0; }
    score += 1<<6;
#endif

    върни score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
