# include "ggml-backend-impl.h"

#nếu defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#nếu defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    số_nguyên power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#nếu defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        nếu (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            nếu (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                số_nguyên vpos = -1;
                cho (số_nguyên i = platform.length() - 1; i >= 0; i--) {
                    nếu (std::isdigit(platform[i])) {
                        vpos = i;
                    } không_thì {
                        break;
                    }
                }
                nếu (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        nếu (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static số_nguyên ggml_backend_cpu_powerpc_score() {
    số_nguyên score = 1;
    powerpc_features pf;

// Platform scores
#nếu defined(GGML_USE_POWER7)
    nếu (pf.power_version < 7) { trả_về 0; }
    score += 1<<1;
#endif
#nếu defined(GGML_USE_POWER8)
    nếu (pf.power_version < 8) { trả_về 0; }
    score += 1<<2;
#endif
#nếu defined(GGML_USE_POWER9)
    nếu (pf.power_version < 9) { trả_về 0; }
    score += 1<<3;
#endif
#nếu defined(GGML_USE_POWER10)
    nếu (pf.power_version < 10) { trả_về 0; }
    score += 1<<4;
#endif
#nếu defined(GGML_USE_POWER11)
    nếu (pf.power_version < 11) { trả_về 0; }
    score += 1<<5;
#endif

// Feature scores
#nếu defined(GGML_USE_VSX)
    nếu (!pf.has_vsx) { trả_về 0; }
    score += 1<<6;
#endif

    trả_về score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
