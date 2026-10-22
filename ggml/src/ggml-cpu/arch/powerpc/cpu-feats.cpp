# include "ggml-backend-impl.h"

#𐤀𐤌 defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#𐤀𐤌 defined(__linux__)
#ضمن <sys/auxv.h>
#endif

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_version    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#𐤀𐤌 defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        𐤀𐤌 (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            𐤀𐤌 (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.length() - 1; i >= 0; i--) {
                    𐤀𐤌 (std::isdigit(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                𐤀𐤌 (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        𐤀𐤌 (power_version >= 9) {
            has_vsx = true;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#𐤀𐤌 defined(GGML_USE_POWER7)
    𐤀𐤌 (pf.power_version < 7) { أرجع 0; }
    score += 1<<1;
#endif
#𐤀𐤌 defined(GGML_USE_POWER8)
    𐤀𐤌 (pf.power_version < 8) { أرجع 0; }
    score += 1<<2;
#endif
#𐤀𐤌 defined(GGML_USE_POWER9)
    𐤀𐤌 (pf.power_version < 9) { أرجع 0; }
    score += 1<<3;
#endif
#𐤀𐤌 defined(GGML_USE_POWER10)
    𐤀𐤌 (pf.power_version < 10) { أرجع 0; }
    score += 1<<4;
#endif
#𐤀𐤌 defined(GGML_USE_POWER11)
    𐤀𐤌 (pf.power_version < 11) { أرجع 0; }
    score += 1<<5;
#endif

// Feature scores
#𐤀𐤌 defined(GGML_USE_VSX)
    𐤀𐤌 (!pf.has_vsx) { أرجع 0; }
    score += 1<<6;
#endif

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
