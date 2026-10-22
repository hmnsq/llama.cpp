# include "ggml-backend-impl.h"

#ᮚᮦᮔ᮪ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᮚᮦᮔ᮪ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᮚᮦᮔ᮪ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᮚᮦᮔ᮪ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᮚᮦᮔ᮪ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᮚᮦᮔ᮪ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᮜᮤᮚ {
                        bha_nga;
                    }
                }
                ᮚᮦᮔ᮪ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᮚᮦᮔ᮪ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᮚᮦᮔ᮪ defined(GGML_USE_POWER7)
    ᮚᮦᮔ᮪ (pf.power_version < 7) { ᮝᮜᮤ 0; }
    score += 1<<1;
#endif
#ᮚᮦᮔ᮪ defined(GGML_USE_POWER8)
    ᮚᮦᮔ᮪ (pf.power_version < 8) { ᮝᮜᮤ 0; }
    score += 1<<2;
#endif
#ᮚᮦᮔ᮪ defined(GGML_USE_POWER9)
    ᮚᮦᮔ᮪ (pf.power_version < 9) { ᮝᮜᮤ 0; }
    score += 1<<3;
#endif
#ᮚᮦᮔ᮪ defined(GGML_USE_POWER10)
    ᮚᮦᮔ᮪ (pf.power_version < 10) { ᮝᮜᮤ 0; }
    score += 1<<4;
#endif
#ᮚᮦᮔ᮪ defined(GGML_USE_POWER11)
    ᮚᮦᮔ᮪ (pf.power_version < 11) { ᮝᮜᮤ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᮚᮦᮔ᮪ defined(GGML_USE_VSX)
    ᮚᮦᮔ᮪ (!pf.has_vsx) { ᮝᮜᮤ 0; }
    score += 1<<6;
#endif

    ᮝᮜᮤ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
