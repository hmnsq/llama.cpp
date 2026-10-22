# include "ggml-backend-impl.h"

#නම් defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#නම් defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    පූර්ණ_සංඛ්‍යා power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#නම් defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        නම් (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            නම් (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                පූර්ණ_සංඛ්‍යා vpos = -1;
                සඳහා (පූර්ණ_සංඛ්‍යා i = platform.length() - 1; i >= 0; i--) {
                    නම් (std::isdigit(platform[i])) {
                        vpos = i;
                    } නැතහොත් {
                        bha_nga;
                    }
                }
                නම් (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        නම් (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static පූර්ණ_සංඛ්‍යා ggml_backend_cpu_powerpc_score() {
    පූර්ණ_සංඛ්‍යා score = 1;
    powerpc_features pf;

// Platform scores
#නම් defined(GGML_USE_POWER7)
    නම් (pf.power_version < 7) { ආපසු 0; }
    score += 1<<1;
#endif
#නම් defined(GGML_USE_POWER8)
    නම් (pf.power_version < 8) { ආපසු 0; }
    score += 1<<2;
#endif
#නම් defined(GGML_USE_POWER9)
    නම් (pf.power_version < 9) { ආපසු 0; }
    score += 1<<3;
#endif
#නම් defined(GGML_USE_POWER10)
    නම් (pf.power_version < 10) { ආපසු 0; }
    score += 1<<4;
#endif
#නම් defined(GGML_USE_POWER11)
    නම් (pf.power_version < 11) { ආපසු 0; }
    score += 1<<5;
#endif

// Feature scores
#නම් defined(GGML_USE_VSX)
    නම් (!pf.has_vsx) { ආපසු 0; }
    score += 1<<6;
#endif

    ආපසු score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
