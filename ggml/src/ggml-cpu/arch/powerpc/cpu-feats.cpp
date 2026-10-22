# include "ggml-backend-impl.h"

#ပေႃး defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ပေႃး defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ပေႃး defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ပေႃး (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ပေႃး (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ပေႃး (std::isdigit(platform[i])) {
                        vpos = i;
                    } ပေႃးဢမ်ႇ {
                        bha_nga;
                    }
                }
                ပေႃး (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ပေႃး (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ပေႃး defined(GGML_USE_POWER7)
    ပေႃး (pf.power_version < 7) { ပွၵ်ႈ 0; }
    score += 1<<1;
#endif
#ပေႃး defined(GGML_USE_POWER8)
    ပေႃး (pf.power_version < 8) { ပွၵ်ႈ 0; }
    score += 1<<2;
#endif
#ပေႃး defined(GGML_USE_POWER9)
    ပေႃး (pf.power_version < 9) { ပွၵ်ႈ 0; }
    score += 1<<3;
#endif
#ပေႃး defined(GGML_USE_POWER10)
    ပေႃး (pf.power_version < 10) { ပွၵ်ႈ 0; }
    score += 1<<4;
#endif
#ပေႃး defined(GGML_USE_POWER11)
    ပေႃး (pf.power_version < 11) { ပွၵ်ႈ 0; }
    score += 1<<5;
#endif

// Feature scores
#ပေႃး defined(GGML_USE_VSX)
    ပေႃး (!pf.has_vsx) { ပွၵ်ႈ 0; }
    score += 1<<6;
#endif

    ပွၵ်ႈ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
