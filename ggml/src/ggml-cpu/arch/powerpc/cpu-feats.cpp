# include "ggml-backend-impl.h"

#ಆದರೆ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <string>

ರಚನೆ powerpc_features {
    std::string platform = "";
    ಪೂರ್ಣಾಂಕ power_ಆವೃತ್ತಿ    = -1;

    ತಾರ್ಕಿಕ has_vsx         = ಸುಳ್ಳು;

    powerpc_features() {
#ಆದರೆ defined(__linux__)
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ auxval = getauxval(AT_PLATFORM);
        ಆದರೆ (auxval) {
            platform = std::string(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ಆದರೆ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ಪೂರ್ಣಾಂಕ vpos = -1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = platform.ಉದ್ದಳತೆ() - 1; i >= 0; i--) {
                    ಆದರೆ (std::ಅಂಕಿಯೇ(platform[i])) {
                        vpos = i;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (vpos > -1) {
                    power_ಆವೃತ್ತಿ = std::stoi(platform.substr(vpos));
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (power_ಆವೃತ್ತಿ >= 9) {
            has_vsx = ನಿಜ;
        }
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_cpu_powerpc_score() {
    ಪೂರ್ಣಾಂಕ score = 1;
    powerpc_features pf;

// Platform scores
#ಆದರೆ defined(GGML_USE_POWER7)
    ಆದರೆ (pf.power_ಆವೃತ್ತಿ < 7) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<1;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_USE_POWER8)
    ಆದರೆ (pf.power_ಆವೃತ್ತಿ < 8) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<2;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_USE_POWER9)
    ಆದರೆ (pf.power_ಆವೃತ್ತಿ < 9) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<3;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_USE_POWER10)
    ಆದರೆ (pf.power_ಆವೃತ್ತಿ < 10) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<4;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_USE_POWER11)
    ಆದರೆ (pf.power_ಆವೃತ್ತಿ < 11) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<5;
#ಆದರೆ_ಅಂತ್ಯ

// Feature scores
#ಆದರೆ defined(GGML_USE_VSX)
    ಆದರೆ (!pf.has_vsx) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<6;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#ಆದರೆ_ಅಂತ್ಯ // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
