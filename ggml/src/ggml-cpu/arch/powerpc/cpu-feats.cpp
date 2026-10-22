# include "ggml-backend-impl.h"

#만약 defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#만약 defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
    std::string platform = "";
    정수 power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#만약 defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        만약 (auxval) {
            platform = std::string(reinterpret_cast<const char*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            만약 (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                정수 vpos = -1;
                위해 (정수 i = platform.length() - 1; i >= 0; i--) {
                    만약 (std::isdigit(platform[i])) {
                        vpos = i;
                    } 아니면 {
                        break;
                    }
                }
                만약 (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        만약 (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static 정수 ggml_backend_cpu_powerpc_score() {
    정수 score = 1;
    powerpc_features pf;

// Platform scores
#만약 defined(GGML_USE_POWER7)
    만약 (pf.power_version < 7) { 반환 0; }
    score += 1<<1;
#endif
#만약 defined(GGML_USE_POWER8)
    만약 (pf.power_version < 8) { 반환 0; }
    score += 1<<2;
#endif
#만약 defined(GGML_USE_POWER9)
    만약 (pf.power_version < 9) { 반환 0; }
    score += 1<<3;
#endif
#만약 defined(GGML_USE_POWER10)
    만약 (pf.power_version < 10) { 반환 0; }
    score += 1<<4;
#endif
#만약 defined(GGML_USE_POWER11)
    만약 (pf.power_version < 11) { 반환 0; }
    score += 1<<5;
#endif

// Feature scores
#만약 defined(GGML_USE_VSX)
    만약 (!pf.has_vsx) { 반환 0; }
    score += 1<<6;
#endif

    반환 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
