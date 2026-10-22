# include "ggml-backend-impl.h"

#اگر defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#اگر defined(__linux__)
#شامل <sys/auxv.h>
#endif

#شامل <string>

ساختار powerpc_features {
    std::string platform = "";
    صحیح power_version    = -1;

    منطقی has_vsx         = نادرست;

    powerpc_features() {
#اگر defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        اگر (auxval) {
            platform = std::string(reinterpret_cast<ثابت نویسه*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            اگر (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                صحیح vpos = -1;
                برای (صحیح i = platform.length() - 1; i >= 0; i--) {
                    اگر (std::isdigit(platform[i])) {
                        vpos = i;
                    } وگرنه {
                        بشکن;
                    }
                }
                اگر (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        اگر (power_version >= 9) {
            has_vsx = درست;
        }
    }
};

ایستا صحیح ggml_backend_cpu_powerpc_score() {
    صحیح score = 1;
    powerpc_features pf;

// Platform scores
#اگر defined(GGML_USE_POWER7)
    اگر (pf.power_version < 7) { بازگشت 0; }
    score += 1<<1;
#endif
#اگر defined(GGML_USE_POWER8)
    اگر (pf.power_version < 8) { بازگشت 0; }
    score += 1<<2;
#endif
#اگر defined(GGML_USE_POWER9)
    اگر (pf.power_version < 9) { بازگشت 0; }
    score += 1<<3;
#endif
#اگر defined(GGML_USE_POWER10)
    اگر (pf.power_version < 10) { بازگشت 0; }
    score += 1<<4;
#endif
#اگر defined(GGML_USE_POWER11)
    اگر (pf.power_version < 11) { بازگشت 0; }
    score += 1<<5;
#endif

// Feature scores
#اگر defined(GGML_USE_VSX)
    اگر (!pf.has_vsx) { بازگشت 0; }
    score += 1<<6;
#endif

    بازگشت score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
