# include "ggml-backend-impl.h"

#もし defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#もし defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    整数 power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#もし defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        もし (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            もし (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                整数 vpos = -1;
                ために (整数 i = platform.length() - 1; i >= 0; i--) {
                    もし (std::isdigit(platform[i])) {
                        vpos = i;
                    } 否则 {
                        break;
                    }
                }
                もし (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        もし (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static 整数 ggml_backend_cpu_powerpc_score() {
    整数 score = 1;
    powerpc_features pf;

// Platform scores
#もし defined(GGML_USE_POWER7)
    もし (pf.power_version < 7) { 戻る 0; }
    score += 1<<1;
#endif
#もし defined(GGML_USE_POWER8)
    もし (pf.power_version < 8) { 戻る 0; }
    score += 1<<2;
#endif
#もし defined(GGML_USE_POWER9)
    もし (pf.power_version < 9) { 戻る 0; }
    score += 1<<3;
#endif
#もし defined(GGML_USE_POWER10)
    もし (pf.power_version < 10) { 戻る 0; }
    score += 1<<4;
#endif
#もし defined(GGML_USE_POWER11)
    もし (pf.power_version < 11) { 戻る 0; }
    score += 1<<5;
#endif

// Feature scores
#もし defined(GGML_USE_VSX)
    もし (!pf.has_vsx) { 戻る 0; }
    score += 1<<6;
#endif

    戻る score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
