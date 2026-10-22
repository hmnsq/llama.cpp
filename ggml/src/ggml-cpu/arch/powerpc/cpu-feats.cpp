# include "ggml-backend-impl.h"

#ຖ້າ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ຖ້າ defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

sa_nracanaa powerpc_features {
    std::string platform = "";
    ຈຳນວນເຕັມ power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#ຖ້າ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ຖ້າ (auxval) {
            platform = std::string(reinterpret_cast<const var_na*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ຖ້າ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ຈຳນວນເຕັມ vpos = -1;
                ສຳລັບ (ຈຳນວນເຕັມ i = platform.length() - 1; i >= 0; i--) {
                    ຖ້າ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ບໍ່ດັ່ງນັ້ນ {
                        bha_nga;
                    }
                }
                ຖ້າ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ຖ້າ (power_version >= 9) {
            has_vsx = true;
        }
    }
};

static ຈຳນວນເຕັມ ggml_backend_cpu_powerpc_score() {
    ຈຳນວນເຕັມ score = 1;
    powerpc_features pf;

// Platform scores
#ຖ້າ defined(GGML_USE_POWER7)
    ຖ້າ (pf.power_version < 7) { ສົ່ງຄືນ 0; }
    score += 1<<1;
#endif
#ຖ້າ defined(GGML_USE_POWER8)
    ຖ້າ (pf.power_version < 8) { ສົ່ງຄືນ 0; }
    score += 1<<2;
#endif
#ຖ້າ defined(GGML_USE_POWER9)
    ຖ້າ (pf.power_version < 9) { ສົ່ງຄືນ 0; }
    score += 1<<3;
#endif
#ຖ້າ defined(GGML_USE_POWER10)
    ຖ້າ (pf.power_version < 10) { ສົ່ງຄືນ 0; }
    score += 1<<4;
#endif
#ຖ້າ defined(GGML_USE_POWER11)
    ຖ້າ (pf.power_version < 11) { ສົ່ງຄືນ 0; }
    score += 1<<5;
#endif

// Feature scores
#ຖ້າ defined(GGML_USE_VSX)
    ຖ້າ (!pf.has_vsx) { ສົ່ງຄືນ 0; }
    score += 1<<6;
#endif

    ສົ່ງຄືນ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
