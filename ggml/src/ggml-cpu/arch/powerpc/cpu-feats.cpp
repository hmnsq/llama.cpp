# include "ggml-backend-impl.h"

#если defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#если defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    целое power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#если defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        если (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            если (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                целое vpos = -1;
                для (целое i = platform.length() - 1; i >= 0; i--) {
                    если (std::isdigit(platform[i])) {
                        vpos = i;
                    } иначе {
                        break;
                    }
                }
                если (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        если (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static целое ggml_backend_cpu_powerpc_score() {
    целое score = 1;
    powerpc_features pf;

// Platform scores
#если defined(GGML_USE_POWER7)
    если (pf.power_version < 7) { вернуть 0; }
    score += 1<<1;
#endif
#если defined(GGML_USE_POWER8)
    если (pf.power_version < 8) { вернуть 0; }
    score += 1<<2;
#endif
#если defined(GGML_USE_POWER9)
    если (pf.power_version < 9) { вернуть 0; }
    score += 1<<3;
#endif
#если defined(GGML_USE_POWER10)
    если (pf.power_version < 10) { вернуть 0; }
    score += 1<<4;
#endif
#если defined(GGML_USE_POWER11)
    если (pf.power_version < 11) { вернуть 0; }
    score += 1<<5;
#endif

// Feature scores
#если defined(GGML_USE_VSX)
    если (!pf.has_vsx) { вернуть 0; }
    score += 1<<6;
#endif

    вернуть score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
