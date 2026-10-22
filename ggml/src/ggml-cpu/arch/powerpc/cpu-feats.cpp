# include "ggml-backend-impl.h"

#ꦪꦼꦤ꧀ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ꦪꦼꦤ꧀ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    puur_naa_nka power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ꦪꦼꦤ꧀ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ꦪꦼꦤ꧀ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ꦪꦼꦤ꧀ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                puur_naa_nka vpos = -1;
                ꦏꦁꦒꦺꦴ (puur_naa_nka i = platform.length() - 1; i >= 0; i--) {
                    ꦪꦼꦤ꧀ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ꦭꦶꦪ {
                        bha_nga;
                    }
                }
                ꦪꦼꦤ꧀ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ꦪꦼꦤ꧀ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static puur_naa_nka ggml_backend_cpu_powerpc_score() {
    puur_naa_nka score = 1;
    powerpc_features pf;

// Platform scores
#ꦪꦼꦤ꧀ defined(GGML_USE_POWER7)
    ꦪꦼꦤ꧀ (pf.power_version < 7) { ꦮꦭꦶ 0; }
    score += 1<<1;
#endif
#ꦪꦼꦤ꧀ defined(GGML_USE_POWER8)
    ꦪꦼꦤ꧀ (pf.power_version < 8) { ꦮꦭꦶ 0; }
    score += 1<<2;
#endif
#ꦪꦼꦤ꧀ defined(GGML_USE_POWER9)
    ꦪꦼꦤ꧀ (pf.power_version < 9) { ꦮꦭꦶ 0; }
    score += 1<<3;
#endif
#ꦪꦼꦤ꧀ defined(GGML_USE_POWER10)
    ꦪꦼꦤ꧀ (pf.power_version < 10) { ꦮꦭꦶ 0; }
    score += 1<<4;
#endif
#ꦪꦼꦤ꧀ defined(GGML_USE_POWER11)
    ꦪꦼꦤ꧀ (pf.power_version < 11) { ꦮꦭꦶ 0; }
    score += 1<<5;
#endif

// Feature scores
#ꦪꦼꦤ꧀ defined(GGML_USE_VSX)
    ꦪꦼꦤ꧀ (!pf.has_vsx) { ꦮꦭꦶ 0; }
    score += 1<<6;
#endif

    ꦮꦭꦶ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
