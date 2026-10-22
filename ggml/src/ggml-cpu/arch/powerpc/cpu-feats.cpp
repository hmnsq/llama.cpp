# include "ggml-backend-impl.h"

#ᤀᤡ᤺ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᤀᤡ᤺ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᤀᤡ᤺ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᤀᤡ᤺ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᤀᤡ᤺ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᤀᤡ᤺ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᤀᤡ᤺ᤆᤡ {
                        bha_nga;
                    }
                }
                ᤀᤡ᤺ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᤀᤡ᤺ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᤀᤡ᤺ defined(GGML_USE_POWER7)
    ᤀᤡ᤺ (pf.power_version < 7) { ᤀᤡᤴ 0; }
    score += 1<<1;
#endif
#ᤀᤡ᤺ defined(GGML_USE_POWER8)
    ᤀᤡ᤺ (pf.power_version < 8) { ᤀᤡᤴ 0; }
    score += 1<<2;
#endif
#ᤀᤡ᤺ defined(GGML_USE_POWER9)
    ᤀᤡ᤺ (pf.power_version < 9) { ᤀᤡᤴ 0; }
    score += 1<<3;
#endif
#ᤀᤡ᤺ defined(GGML_USE_POWER10)
    ᤀᤡ᤺ (pf.power_version < 10) { ᤀᤡᤴ 0; }
    score += 1<<4;
#endif
#ᤀᤡ᤺ defined(GGML_USE_POWER11)
    ᤀᤡ᤺ (pf.power_version < 11) { ᤀᤡᤴ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᤀᤡ᤺ defined(GGML_USE_VSX)
    ᤀᤡ᤺ (!pf.has_vsx) { ᤀᤡᤴ 0; }
    score += 1<<6;
#endif

    ᤀᤡᤴ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
