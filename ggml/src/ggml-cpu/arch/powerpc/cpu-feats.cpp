# include "ggml-backend-impl.h"

#གལ་ཏེ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#གལ་ཏེ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    ཧྲིལ power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#གལ་ཏེ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        གལ་ཏེ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            གལ་ཏེ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ཧྲིལ vpos = -1;
                ཆེད་དུ (ཧྲིལ i = platform.length() - 1; i >= 0; i--) {
                    གལ་ཏེ (std::isdigit(platform[i])) {
                        vpos = i;
                    } དེ་མིན {
                        bha_nga;
                    }
                }
                གལ་ཏེ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        གལ་ཏེ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ཧྲིལ ggml_backend_cpu_powerpc_score() {
    ཧྲིལ score = 1;
    powerpc_features pf;

// Platform scores
#གལ་ཏེ defined(GGML_USE_POWER7)
    གལ་ཏེ (pf.power_version < 7) { ཕྱིར 0; }
    score += 1<<1;
#endif
#གལ་ཏེ defined(GGML_USE_POWER8)
    གལ་ཏེ (pf.power_version < 8) { ཕྱིར 0; }
    score += 1<<2;
#endif
#གལ་ཏེ defined(GGML_USE_POWER9)
    གལ་ཏེ (pf.power_version < 9) { ཕྱིར 0; }
    score += 1<<3;
#endif
#གལ་ཏེ defined(GGML_USE_POWER10)
    གལ་ཏེ (pf.power_version < 10) { ཕྱིར 0; }
    score += 1<<4;
#endif
#གལ་ཏེ defined(GGML_USE_POWER11)
    གལ་ཏེ (pf.power_version < 11) { ཕྱིར 0; }
    score += 1<<5;
#endif

// Feature scores
#གལ་ཏེ defined(GGML_USE_VSX)
    གལ་ཏེ (!pf.has_vsx) { ཕྱིར 0; }
    score += 1<<6;
#endif

    ཕྱིར score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
