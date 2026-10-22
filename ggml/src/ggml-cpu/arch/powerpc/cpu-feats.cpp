# include "ggml-backend-impl.h"

#ᬬᬾᬦ᭄ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᬬᬾᬦ᭄ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᬬᬾᬦ᭄ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᬬᬾᬦ᭄ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᬬᬾᬦ᭄ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᬬᬾᬦ᭄ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᬮᬶᬬ {
                        bha_nga;
                    }
                }
                ᬬᬾᬦ᭄ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᬬᬾᬦ᭄ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᬬᬾᬦ᭄ defined(GGML_USE_POWER7)
    ᬬᬾᬦ᭄ (pf.power_version < 7) { ᬯᬮᬶ 0; }
    score += 1<<1;
#endif
#ᬬᬾᬦ᭄ defined(GGML_USE_POWER8)
    ᬬᬾᬦ᭄ (pf.power_version < 8) { ᬯᬮᬶ 0; }
    score += 1<<2;
#endif
#ᬬᬾᬦ᭄ defined(GGML_USE_POWER9)
    ᬬᬾᬦ᭄ (pf.power_version < 9) { ᬯᬮᬶ 0; }
    score += 1<<3;
#endif
#ᬬᬾᬦ᭄ defined(GGML_USE_POWER10)
    ᬬᬾᬦ᭄ (pf.power_version < 10) { ᬯᬮᬶ 0; }
    score += 1<<4;
#endif
#ᬬᬾᬦ᭄ defined(GGML_USE_POWER11)
    ᬬᬾᬦ᭄ (pf.power_version < 11) { ᬯᬮᬶ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᬬᬾᬦ᭄ defined(GGML_USE_VSX)
    ᬬᬾᬦ᭄ (!pf.has_vsx) { ᬯᬮᬶ 0; }
    score += 1<<6;
#endif

    ᬯᬮᬶ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
