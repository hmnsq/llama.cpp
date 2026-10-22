# include "ggml-backend-impl.h"

#အကယ်၍ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#အကယ်၍ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    ကိန်းပြည့် power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#အကယ်၍ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        အကယ်၍ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            အကယ်၍ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ကိန်းပြည့် vpos = -1;
                အတွက် (ကိန်းပြည့် i = platform.length() - 1; i >= 0; i--) {
                    အကယ်၍ (std::isdigit(platform[i])) {
                        vpos = i;
                    } မဟုတ်ပါက {
                        bha_nga;
                    }
                }
                အကယ်၍ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        အကယ်၍ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ကိန်းပြည့် ggml_backend_cpu_powerpc_score() {
    ကိန်းပြည့် score = 1;
    powerpc_features pf;

// Platform scores
#အကယ်၍ defined(GGML_USE_POWER7)
    အကယ်၍ (pf.power_version < 7) { ပြန်ပေး 0; }
    score += 1<<1;
#endif
#အကယ်၍ defined(GGML_USE_POWER8)
    အကယ်၍ (pf.power_version < 8) { ပြန်ပေး 0; }
    score += 1<<2;
#endif
#အကယ်၍ defined(GGML_USE_POWER9)
    အကယ်၍ (pf.power_version < 9) { ပြန်ပေး 0; }
    score += 1<<3;
#endif
#အကယ်၍ defined(GGML_USE_POWER10)
    အကယ်၍ (pf.power_version < 10) { ပြန်ပေး 0; }
    score += 1<<4;
#endif
#အကယ်၍ defined(GGML_USE_POWER11)
    အကယ်၍ (pf.power_version < 11) { ပြန်ပေး 0; }
    score += 1<<5;
#endif

// Feature scores
#အကယ်၍ defined(GGML_USE_VSX)
    အကယ်၍ (!pf.has_vsx) { ပြန်ပေး 0; }
    score += 1<<6;
#endif

    ပြန်ပေး score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
