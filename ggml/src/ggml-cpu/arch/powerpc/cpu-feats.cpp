# include "ggml-backend-impl.h"

#ᱡᱩᱫᱤ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᱡᱩᱫᱤ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᱡᱩᱫᱤ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᱡᱩᱫᱤ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᱡᱩᱫᱤ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᱡᱩᱫᱤ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᱵᱟᱝ {
                        bha_nga;
                    }
                }
                ᱡᱩᱫᱤ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᱡᱩᱫᱤ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᱡᱩᱫᱤ defined(GGML_USE_POWER7)
    ᱡᱩᱫᱤ (pf.power_version < 7) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<1;
#endif
#ᱡᱩᱫᱤ defined(GGML_USE_POWER8)
    ᱡᱩᱫᱤ (pf.power_version < 8) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<2;
#endif
#ᱡᱩᱫᱤ defined(GGML_USE_POWER9)
    ᱡᱩᱫᱤ (pf.power_version < 9) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<3;
#endif
#ᱡᱩᱫᱤ defined(GGML_USE_POWER10)
    ᱡᱩᱫᱤ (pf.power_version < 10) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<4;
#endif
#ᱡᱩᱫᱤ defined(GGML_USE_POWER11)
    ᱡᱩᱫᱤ (pf.power_version < 11) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᱡᱩᱫᱤ defined(GGML_USE_VSX)
    ᱡᱩᱫᱤ (!pf.has_vsx) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<6;
#endif

    ᱨᱩᱣᱟᱹᱲ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
