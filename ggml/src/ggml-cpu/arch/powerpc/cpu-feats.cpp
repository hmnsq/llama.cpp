# include "ggml-backend-impl.h"

#إذا defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#إذا defined(__linux__)
#ضمن <sys/auxv.h>
#جے_انت

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_نسخہ    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#إذا defined(__linux__)
        بنانشانی لما auxval = getauxval(AT_PLATFORM);
        إذا (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            إذا (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.لمائی() - 1; i >= 0; i--) {
                    إذا (std::ہندسہ_اے(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                إذا (vpos > -1) {
                    power_نسخہ = std::stoi(platform.substr(vpos));
                }
            }
        }
#جے_انت
        إذا (power_نسخہ >= 9) {
            has_vsx = سچ;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#إذا defined(GGML_USE_POWER7)
    إذا (pf.power_نسخہ < 7) { أرجع 0; }
    score += 1<<1;
#جے_انت
#إذا defined(GGML_USE_POWER8)
    إذا (pf.power_نسخہ < 8) { أرجع 0; }
    score += 1<<2;
#جے_انت
#إذا defined(GGML_USE_POWER9)
    إذا (pf.power_نسخہ < 9) { أرجع 0; }
    score += 1<<3;
#جے_انت
#إذا defined(GGML_USE_POWER10)
    إذا (pf.power_نسخہ < 10) { أرجع 0; }
    score += 1<<4;
#جے_انت
#إذا defined(GGML_USE_POWER11)
    إذا (pf.power_نسخہ < 11) { أرجع 0; }
    score += 1<<5;
#جے_انت

// Feature scores
#إذا defined(GGML_USE_VSX)
    إذا (!pf.has_vsx) { أرجع 0; }
    score += 1<<6;
#جے_انت

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#جے_انت // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
