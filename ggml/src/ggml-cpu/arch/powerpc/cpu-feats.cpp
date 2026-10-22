# include "ggml-backend-impl.h"

#ಯದಿ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>
#endif

#ಅನ್ತರ್ಭಾವ <string>

ಸಂರಚನಾ powerpc_features {
    std::string platform = "";
    ಪೂರ್ಣಾಙ್ಕ power_version    = -1;

    ತಾರ್ಕಿಕ has_vsx         = ಅಸತ್ಯ;

    powerpc_features() {
#ಯದಿ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ಯದಿ (auxval) {
            platform = std::string(reinterpret_cast<ಧ್ರುವ ವರ್ಣ*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ಯದಿ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ಪೂರ್ಣಾಙ್ಕ vpos = -1;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = platform.length() - 1; i >= 0; i--) {
                    ಯದಿ (std::isdigit(platform[i])) {
                        vpos = i;
                    } ಅನ್ಯಥಾ {
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ಯದಿ (power_version >= 9) {
            has_vsx = ಸತ್ಯ;
        }
    }
};

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_cpu_powerpc_score() {
    ಪೂರ್ಣಾಙ್ಕ score = 1;
    powerpc_features pf;

// Platform scores
#ಯದಿ defined(GGML_USE_POWER7)
    ಯದಿ (pf.power_version < 7) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<1;
#endif
#ಯದಿ defined(GGML_USE_POWER8)
    ಯದಿ (pf.power_version < 8) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<2;
#endif
#ಯದಿ defined(GGML_USE_POWER9)
    ಯದಿ (pf.power_version < 9) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<3;
#endif
#ಯದಿ defined(GGML_USE_POWER10)
    ಯದಿ (pf.power_version < 10) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<4;
#endif
#ಯದಿ defined(GGML_USE_POWER11)
    ಯದಿ (pf.power_version < 11) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<5;
#endif

// Feature scores
#ಯದಿ defined(GGML_USE_VSX)
    ಯದಿ (!pf.has_vsx) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<6;
#endif

    ಪ್ರತ್ಯಾಗಮ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
