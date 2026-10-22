# include "ggml-backend-impl.h"

#ᨀᨘ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᨀᨘ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᨀᨘ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᨀᨘ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᨀᨘ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᨀᨘ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᨀᨘᨁ {
                        bha_nga;
                    }
                }
                ᨀᨘ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᨀᨘ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᨀᨘ defined(GGML_USE_POWER7)
    ᨀᨘ (pf.power_version < 7) { ᨀᨘᨒ 0; }
    score += 1<<1;
#endif
#ᨀᨘ defined(GGML_USE_POWER8)
    ᨀᨘ (pf.power_version < 8) { ᨀᨘᨒ 0; }
    score += 1<<2;
#endif
#ᨀᨘ defined(GGML_USE_POWER9)
    ᨀᨘ (pf.power_version < 9) { ᨀᨘᨒ 0; }
    score += 1<<3;
#endif
#ᨀᨘ defined(GGML_USE_POWER10)
    ᨀᨘ (pf.power_version < 10) { ᨀᨘᨒ 0; }
    score += 1<<4;
#endif
#ᨀᨘ defined(GGML_USE_POWER11)
    ᨀᨘ (pf.power_version < 11) { ᨀᨘᨒ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᨀᨘ defined(GGML_USE_VSX)
    ᨀᨘ (!pf.has_vsx) { ᨀᨘᨒ 0; }
    score += 1<<6;
#endif

    ᨀᨘᨒ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
