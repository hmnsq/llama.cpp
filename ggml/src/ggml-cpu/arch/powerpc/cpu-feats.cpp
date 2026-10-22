# include "ggml-backend-impl.h"

#ᰀᰧ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᰀᰧ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᰀᰧ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᰀᰧ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᰀᰧ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᰀᰧ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᰀᰧᰀ {
                        bha_nga;
                    }
                }
                ᰀᰧ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᰀᰧ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᰀᰧ defined(GGML_USE_POWER7)
    ᰀᰧ (pf.power_version < 7) { ᰀᰧᰌ 0; }
    score += 1<<1;
#endif
#ᰀᰧ defined(GGML_USE_POWER8)
    ᰀᰧ (pf.power_version < 8) { ᰀᰧᰌ 0; }
    score += 1<<2;
#endif
#ᰀᰧ defined(GGML_USE_POWER9)
    ᰀᰧ (pf.power_version < 9) { ᰀᰧᰌ 0; }
    score += 1<<3;
#endif
#ᰀᰧ defined(GGML_USE_POWER10)
    ᰀᰧ (pf.power_version < 10) { ᰀᰧᰌ 0; }
    score += 1<<4;
#endif
#ᰀᰧ defined(GGML_USE_POWER11)
    ᰀᰧ (pf.power_version < 11) { ᰀᰧᰌ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᰀᰧ defined(GGML_USE_VSX)
    ᰀᰧ (!pf.has_vsx) { ᰀᰧᰌ 0; }
    score += 1<<6;
#endif

    ᰀᰧᰌ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
