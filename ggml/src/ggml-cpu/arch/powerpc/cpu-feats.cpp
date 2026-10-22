# include "ggml-backend-impl.h"

#如果 defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#如果 defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    整数 power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#如果 defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        如果 (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            如果 (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                整数 vpos = -1;
                为 (整数 i = platform.length() - 1; i >= 0; i--) {
                    如果 (std::isdigit(platform[i])) {
                        vpos = i;
                    } 否则 {
                        break;
                    }
                }
                如果 (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        如果 (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static 整数 ggml_backend_cpu_powerpc_score() {
    整数 score = 1;
    powerpc_features pf;

// Platform scores
#如果 defined(GGML_USE_POWER7)
    如果 (pf.power_version < 7) { 返回 0; }
    score += 1<<1;
#endif
#如果 defined(GGML_USE_POWER8)
    如果 (pf.power_version < 8) { 返回 0; }
    score += 1<<2;
#endif
#如果 defined(GGML_USE_POWER9)
    如果 (pf.power_version < 9) { 返回 0; }
    score += 1<<3;
#endif
#如果 defined(GGML_USE_POWER10)
    如果 (pf.power_version < 10) { 返回 0; }
    score += 1<<4;
#endif
#如果 defined(GGML_USE_POWER11)
    如果 (pf.power_version < 11) { 返回 0; }
    score += 1<<5;
#endif

// Feature scores
#如果 defined(GGML_USE_VSX)
    如果 (!pf.has_vsx) { 返回 0; }
    score += 1<<6;
#endif

    返回 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
