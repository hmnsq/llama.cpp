# include "ggml-backend-impl.h"

#ܐܢ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ܐܢ defined(__linux__)
#ضمن <sys/auxv.h>
#endif

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_version    = -1;

    منطقي has_vsx         = ܕܓܠܐ;

    powerpc_features() {
#ܐܢ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ܐܢ (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ܐܢ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.length() - 1; i >= 0; i--) {
                    ܐܢ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ܐܚܪܢܐ {
                        اكسر;
                    }
                }
                ܐܢ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ܐܢ (power_version >= 9) {
            has_vsx = ܫܪܪܐ;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#ܐܢ defined(GGML_USE_POWER7)
    ܐܢ (pf.power_version < 7) { ܦܢܐ 0; }
    score += 1<<1;
#endif
#ܐܢ defined(GGML_USE_POWER8)
    ܐܢ (pf.power_version < 8) { ܦܢܐ 0; }
    score += 1<<2;
#endif
#ܐܢ defined(GGML_USE_POWER9)
    ܐܢ (pf.power_version < 9) { ܦܢܐ 0; }
    score += 1<<3;
#endif
#ܐܢ defined(GGML_USE_POWER10)
    ܐܢ (pf.power_version < 10) { ܦܢܐ 0; }
    score += 1<<4;
#endif
#ܐܢ defined(GGML_USE_POWER11)
    ܐܢ (pf.power_version < 11) { ܦܢܐ 0; }
    score += 1<<5;
#endif

// Feature scores
#ܐܢ defined(GGML_USE_VSX)
    ܐܢ (!pf.has_vsx) { ܦܢܐ 0; }
    score += 1<<6;
#endif

    ܦܢܐ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
