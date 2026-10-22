# include "ggml-backend-impl.h"

#إذا defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#إذا defined(__linux__)
#ضمن <sys/auxv.h>
#د_که_پای

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_نسخه    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#إذا defined(__linux__)
        بینښه اوږد auxval = getauxval(AT_PLATFORM);
        إذا (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            إذا (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.اوږدتیا() - 1; i >= 0; i--) {
                    إذا (std::که_عدد(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                إذا (vpos > -1) {
                    power_نسخه = std::stoi(platform.substr(vpos));
                }
            }
        }
#د_که_پای
        إذا (power_نسخه >= 9) {
            has_vsx = ریښتیا;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#إذا defined(GGML_USE_POWER7)
    إذا (pf.power_نسخه < 7) { أرجع 0; }
    score += 1<<1;
#د_که_پای
#إذا defined(GGML_USE_POWER8)
    إذا (pf.power_نسخه < 8) { أرجع 0; }
    score += 1<<2;
#د_که_پای
#إذا defined(GGML_USE_POWER9)
    إذا (pf.power_نسخه < 9) { أرجع 0; }
    score += 1<<3;
#د_که_پای
#إذا defined(GGML_USE_POWER10)
    إذا (pf.power_نسخه < 10) { أرجع 0; }
    score += 1<<4;
#د_که_پای
#إذا defined(GGML_USE_POWER11)
    إذا (pf.power_نسخه < 11) { أرجع 0; }
    score += 1<<5;
#د_که_پای

// Feature scores
#إذا defined(GGML_USE_VSX)
    إذا (!pf.has_vsx) { أرجع 0; }
    score += 1<<6;
#د_که_پای

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#د_که_پای // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
