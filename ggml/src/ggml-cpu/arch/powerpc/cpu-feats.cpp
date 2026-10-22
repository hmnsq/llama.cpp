# include "ggml-backend-impl.h"

#𐡄𐡍 defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#𐡄𐡍 defined(__linux__)
#ضمن <sys/auxv.h>
#endif

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_version    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#𐡄𐡍 defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        𐡄𐡍 (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            𐡄𐡍 (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.length() - 1; i >= 0; i--) {
                    𐡄𐡍 (std::isdigit(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                𐡄𐡍 (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        𐡄𐡍 (power_version >= 9) {
            has_vsx = 𐡒𐡔𐡈;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#𐡄𐡍 defined(GGML_USE_POWER7)
    𐡄𐡍 (pf.power_version < 7) { أرجع 0; }
    score += 1<<1;
#endif
#𐡄𐡍 defined(GGML_USE_POWER8)
    𐡄𐡍 (pf.power_version < 8) { أرجع 0; }
    score += 1<<2;
#endif
#𐡄𐡍 defined(GGML_USE_POWER9)
    𐡄𐡍 (pf.power_version < 9) { أرجع 0; }
    score += 1<<3;
#endif
#𐡄𐡍 defined(GGML_USE_POWER10)
    𐡄𐡍 (pf.power_version < 10) { أرجع 0; }
    score += 1<<4;
#endif
#𐡄𐡍 defined(GGML_USE_POWER11)
    𐡄𐡍 (pf.power_version < 11) { أرجع 0; }
    score += 1<<5;
#endif

// Feature scores
#𐡄𐡍 defined(GGML_USE_VSX)
    𐡄𐡍 (!pf.has_vsx) { أرجع 0; }
    score += 1<<6;
#endif

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
