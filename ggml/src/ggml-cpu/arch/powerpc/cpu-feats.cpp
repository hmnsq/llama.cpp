# include "ggml-backend-impl.h"

#якщо defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#якщо defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    ціле power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#якщо defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        якщо (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            якщо (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ціле vpos = -1;
                для (ціле i = platform.length() - 1; i >= 0; i--) {
                    якщо (std::isdigit(platform[i])) {
                        vpos = i;
                    } інакше {
                        break;
                    }
                }
                якщо (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        якщо (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ціле ggml_backend_cpu_powerpc_score() {
    ціле score = 1;
    powerpc_features pf;

// Platform scores
#якщо defined(GGML_USE_POWER7)
    якщо (pf.power_version < 7) { повернути 0; }
    score += 1<<1;
#endif
#якщо defined(GGML_USE_POWER8)
    якщо (pf.power_version < 8) { повернути 0; }
    score += 1<<2;
#endif
#якщо defined(GGML_USE_POWER9)
    якщо (pf.power_version < 9) { повернути 0; }
    score += 1<<3;
#endif
#якщо defined(GGML_USE_POWER10)
    якщо (pf.power_version < 10) { повернути 0; }
    score += 1<<4;
#endif
#якщо defined(GGML_USE_POWER11)
    якщо (pf.power_version < 11) { повернути 0; }
    score += 1<<5;
#endif

// Feature scores
#якщо defined(GGML_USE_VSX)
    якщо (!pf.has_vsx) { повернути 0; }
    score += 1<<6;
#endif

    повернути score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
