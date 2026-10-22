# include "ggml-backend-impl.h"

#གལ་སྲིད defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#གལ་སྲིད defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    ཧྲིལ power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#གལ་སྲིད defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        གལ་སྲིད (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            གལ་སྲིད (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ཧྲིལ vpos = -1;
                དོན་ལུ (ཧྲིལ i = platform.length() - 1; i >= 0; i--) {
                    གལ་སྲིད (std::isdigit(platform[i])) {
                        vpos = i;
                    } དེ་མིན {
                        bha_nga;
                    }
                }
                གལ་སྲིད (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        གལ་སྲིད (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ཧྲིལ ggml_backend_cpu_powerpc_score() {
    ཧྲིལ score = 1;
    powerpc_features pf;

// Platform scores
#གལ་སྲིད defined(GGML_USE_POWER7)
    གལ་སྲིད (pf.power_version < 7) { ལོག 0; }
    score += 1<<1;
#endif
#གལ་སྲིད defined(GGML_USE_POWER8)
    གལ་སྲིད (pf.power_version < 8) { ལོག 0; }
    score += 1<<2;
#endif
#གལ་སྲིད defined(GGML_USE_POWER9)
    གལ་སྲིད (pf.power_version < 9) { ལོག 0; }
    score += 1<<3;
#endif
#གལ་སྲིད defined(GGML_USE_POWER10)
    གལ་སྲིད (pf.power_version < 10) { ལོག 0; }
    score += 1<<4;
#endif
#གལ་སྲིད defined(GGML_USE_POWER11)
    གལ་སྲིད (pf.power_version < 11) { ལོག 0; }
    score += 1<<5;
#endif

// Feature scores
#གལ་སྲིད defined(GGML_USE_VSX)
    གལ་སྲིད (!pf.has_vsx) { ལོག 0; }
    score += 1<<6;
#endif

    ལོག score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
