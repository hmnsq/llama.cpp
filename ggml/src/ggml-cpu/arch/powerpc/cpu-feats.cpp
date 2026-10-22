# include "ggml-backend-impl.h"

#ਜੇ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/auxv.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <string>

ਬਣਤਰ powerpc_features {
    std::string platform = "";
    ਪੂਰਨ power_ਸੰਸਕਰਣ    = -1;

    ਤਾਰਕਿਕ has_vsx         = ਝੂਠ;

    powerpc_features() {
#ਜੇ defined(__linux__)
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ auxval = getauxval(AT_PLATFORM);
        ਜੇ (auxval) {
            platform = std::string(reinterpret_cast<ਅਚਲ ਅੱਖਰ*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ਜੇ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ਪੂਰਨ vpos = -1;
                ਲਈ (ਪੂਰਨ i = platform.ਲੰਮਾਈ() - 1; i >= 0; i--) {
                    ਜੇ (std::ਅੰਕ_ਹੈ(platform[i])) {
                        vpos = i;
                    } ਨਹੀਂਤਾਂ {
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (vpos > -1) {
                    power_ਸੰਸਕਰਣ = std::stoi(platform.substr(vpos));
                }
            }
        }
#ਜੇ_ਅੰਤ
        ਜੇ (power_ਸੰਸਕਰਣ >= 9) {
            has_vsx = ਸੱਚ;
        }
    }
};

ਸਥਿਰ ਪੂਰਨ ggml_backend_cpu_powerpc_score() {
    ਪੂਰਨ score = 1;
    powerpc_features pf;

// Platform scores
#ਜੇ defined(GGML_USE_POWER7)
    ਜੇ (pf.power_ਸੰਸਕਰਣ < 7) { ਵਾਪਸ 0; }
    score += 1<<1;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_USE_POWER8)
    ਜੇ (pf.power_ਸੰਸਕਰਣ < 8) { ਵਾਪਸ 0; }
    score += 1<<2;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_USE_POWER9)
    ਜੇ (pf.power_ਸੰਸਕਰਣ < 9) { ਵਾਪਸ 0; }
    score += 1<<3;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_USE_POWER10)
    ਜੇ (pf.power_ਸੰਸਕਰਣ < 10) { ਵਾਪਸ 0; }
    score += 1<<4;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_USE_POWER11)
    ਜੇ (pf.power_ਸੰਸਕਰਣ < 11) { ਵਾਪਸ 0; }
    score += 1<<5;
#ਜੇ_ਅੰਤ

// Feature scores
#ਜੇ defined(GGML_USE_VSX)
    ਜੇ (!pf.has_vsx) { ਵਾਪਸ 0; }
    score += 1<<6;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#ਜੇ_ਅੰਤ // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
