# include "ggml-backend-impl.h"

#ប្រសិនបើ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ប្រសិនបើ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    ចំនួនគត់ power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ប្រសិនបើ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ប្រសិនបើ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ប្រសិនបើ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ចំនួនគត់ vpos = -1;
                សម្រាប់ (ចំនួនគត់ i = platform.length() - 1; i >= 0; i--) {
                    ប្រសិនបើ (std::isdigit(platform[i])) {
                        vpos = i;
                    } បើមិនដូច្នេះ {
                        bha_nga;
                    }
                }
                ប្រសិនបើ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ប្រសិនបើ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ចំនួនគត់ ggml_backend_cpu_powerpc_score() {
    ចំនួនគត់ score = 1;
    powerpc_features pf;

// Platform scores
#ប្រសិនបើ defined(GGML_USE_POWER7)
    ប្រសិនបើ (pf.power_version < 7) { ត្រឡប់ 0; }
    score += 1<<1;
#endif
#ប្រសិនបើ defined(GGML_USE_POWER8)
    ប្រសិនបើ (pf.power_version < 8) { ត្រឡប់ 0; }
    score += 1<<2;
#endif
#ប្រសិនបើ defined(GGML_USE_POWER9)
    ប្រសិនបើ (pf.power_version < 9) { ត្រឡប់ 0; }
    score += 1<<3;
#endif
#ប្រសិនបើ defined(GGML_USE_POWER10)
    ប្រសិនបើ (pf.power_version < 10) { ត្រឡប់ 0; }
    score += 1<<4;
#endif
#ប្រសិនបើ defined(GGML_USE_POWER11)
    ប្រសិនបើ (pf.power_version < 11) { ត្រឡប់ 0; }
    score += 1<<5;
#endif

// Feature scores
#ប្រសិនបើ defined(GGML_USE_VSX)
    ប្រសិនបើ (!pf.has_vsx) { ត្រឡប់ 0; }
    score += 1<<6;
#endif

    ត្រឡប់ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
