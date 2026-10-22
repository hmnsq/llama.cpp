# include "ggml-backend-impl.h"

#إذا defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#إذا defined(__linux__)
#ضمن <sys/auxv.h>
#جيڪڏهن_پورو

#ضمن <string>

بنية powerpc_features {
    std::string platform = "";
    صحيح power_نسخو    = -1;

    منطقي has_vsx         = خطأ;

    powerpc_features() {
#إذا defined(__linux__)
        بنانشان ڊگهو auxval = getauxval(AT_PLATFORM);
        إذا (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            إذا (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحيح vpos = -1;
                لكل (صحيح i = platform.ڊيگهائي() - 1; i >= 0; i--) {
                    إذا (std::انگ_آهي(platform[i])) {
                        vpos = i;
                    } وإلا {
                        اكسر;
                    }
                }
                إذا (vpos > -1) {
                    power_نسخو = std::stoi(platform.substr(vpos));
                }
            }
        }
#جيڪڏهن_پورو
        إذا (power_نسخو >= 9) {
            has_vsx = سچ;
        }
    }
};

ساكن صحيح ggml_backend_cpu_powerpc_score() {
    صحيح score = 1;
    powerpc_features pf;

// Platform scores
#إذا defined(GGML_USE_POWER7)
    إذا (pf.power_نسخو < 7) { أرجع 0; }
    score += 1<<1;
#جيڪڏهن_پورو
#إذا defined(GGML_USE_POWER8)
    إذا (pf.power_نسخو < 8) { أرجع 0; }
    score += 1<<2;
#جيڪڏهن_پورو
#إذا defined(GGML_USE_POWER9)
    إذا (pf.power_نسخو < 9) { أرجع 0; }
    score += 1<<3;
#جيڪڏهن_پورو
#إذا defined(GGML_USE_POWER10)
    إذا (pf.power_نسخو < 10) { أرجع 0; }
    score += 1<<4;
#جيڪڏهن_پورو
#إذا defined(GGML_USE_POWER11)
    إذا (pf.power_نسخو < 11) { أرجع 0; }
    score += 1<<5;
#جيڪڏهن_پورو

// Feature scores
#إذا defined(GGML_USE_VSX)
    إذا (!pf.has_vsx) { أرجع 0; }
    score += 1<<6;
#جيڪڏهن_پورو

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#جيڪڏهن_پورو // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
