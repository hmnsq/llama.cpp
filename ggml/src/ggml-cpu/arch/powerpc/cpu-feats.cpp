# include "ggml-backend-impl.h"

#اگر defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#اگر defined(__linux__)
#شامل <sys/auxv.h>
#اگر_ختم

#شامل <string>

ساخت powerpc_features {
    std::string platform = "";
    صحیح power_نسخہ    = -1;

    منطقی has_vsx         = جھوٹ;

    powerpc_features() {
#اگر defined(__linux__)
        غیرعلامتی طویل auxval = getauxval(AT_PLATFORM);
        اگر (auxval) {
            platform = std::string(reinterpret_cast<ثابت حرف*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            اگر (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحیح vpos = -1;
                برائے (صحیح i = platform.لمبائی() - 1; i >= 0; i--) {
                    اگر (std::ہندسہ_ہے(platform[i])) {
                        vpos = i;
                    } ورنہ {
                        توڑو;
                    }
                }
                اگر (vpos > -1) {
                    power_نسخہ = std::stoi(platform.substr(vpos));
                }
            }
        }
#اگر_ختم
        اگر (power_نسخہ >= 9) {
            has_vsx = سچ;
        }
    }
};

ساکن صحیح ggml_backend_cpu_powerpc_score() {
    صحیح score = 1;
    powerpc_features pf;

// Platform scores
#اگر defined(GGML_USE_POWER7)
    اگر (pf.power_نسخہ < 7) { واپس 0; }
    score += 1<<1;
#اگر_ختم
#اگر defined(GGML_USE_POWER8)
    اگر (pf.power_نسخہ < 8) { واپس 0; }
    score += 1<<2;
#اگر_ختم
#اگر defined(GGML_USE_POWER9)
    اگر (pf.power_نسخہ < 9) { واپس 0; }
    score += 1<<3;
#اگر_ختم
#اگر defined(GGML_USE_POWER10)
    اگر (pf.power_نسخہ < 10) { واپس 0; }
    score += 1<<4;
#اگر_ختم
#اگر defined(GGML_USE_POWER11)
    اگر (pf.power_نسخہ < 11) { واپس 0; }
    score += 1<<5;
#اگر_ختم

// Feature scores
#اگر defined(GGML_USE_VSX)
    اگر (!pf.has_vsx) { واپس 0; }
    score += 1<<6;
#اگر_ختم

    واپس score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#اگر_ختم // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
