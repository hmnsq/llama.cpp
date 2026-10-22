# include "ggml-backend-impl.h"

#ᯀᯮᯝ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ᯀᯮᯝ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ᯀᯮᯝ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ᯀᯮᯝ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ᯀᯮᯝ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                for (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ᯀᯮᯝ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ᯀᯮᯝᯂᯪ {
                        bha_nga;
                    }
                }
                ᯀᯮᯝ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ᯀᯮᯝ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ᯀᯮᯝ defined(GGML_USE_POWER7)
    ᯀᯮᯝ (pf.power_version < 7) { ᯀᯮᯞ 0; }
    score += 1<<1;
#endif
#ᯀᯮᯝ defined(GGML_USE_POWER8)
    ᯀᯮᯝ (pf.power_version < 8) { ᯀᯮᯞ 0; }
    score += 1<<2;
#endif
#ᯀᯮᯝ defined(GGML_USE_POWER9)
    ᯀᯮᯝ (pf.power_version < 9) { ᯀᯮᯞ 0; }
    score += 1<<3;
#endif
#ᯀᯮᯝ defined(GGML_USE_POWER10)
    ᯀᯮᯝ (pf.power_version < 10) { ᯀᯮᯞ 0; }
    score += 1<<4;
#endif
#ᯀᯮᯝ defined(GGML_USE_POWER11)
    ᯀᯮᯝ (pf.power_version < 11) { ᯀᯮᯞ 0; }
    score += 1<<5;
#endif

// Feature scores
#ᯀᯮᯝ defined(GGML_USE_VSX)
    ᯀᯮᯝ (!pf.has_vsx) { ᯀᯮᯞ 0; }
    score += 1<<6;
#endif

    ᯀᯮᯞ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
