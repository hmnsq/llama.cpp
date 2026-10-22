# include "ggml-backend-impl.h"

#ଯଦି defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <sys/auxv.h>
#endif

#ଅନ୍ତର୍ଭାଵ <string>

ସଂରଚନା powerpc_features {
    std::string platform = "";
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା power_version    = -1;

    ତାର୍କିକ has_vsx         = ମିଥ୍ୟା;

    powerpc_features() {
#ଯଦି defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        ଯଦି (auxval) {
            platform = std::string(reinterpret_cast<ଅଚଳ ଅକ୍ଷର*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ଯଦି (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା vpos = -1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = platform.length() - 1; i >= 0; i--) {
                    ଯଦି (std::isdigit(platform[i])) {
                        vpos = i;
                    } ଅନ୍ୟଥା {
                        ଭାଙ୍ଗ;
                    }
                }
                ଯଦି (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        ଯଦି (power_version >= 9) {
            has_vsx = ସତ୍ୟ;
        }
    }
};

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_cpu_powerpc_score() {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା score = 1;
    powerpc_features pf;

// Platform scores
#ଯଦି defined(GGML_USE_POWER7)
    ଯଦି (pf.power_version < 7) { ଫେରନ୍ତୁ 0; }
    score += 1<<1;
#endif
#ଯଦି defined(GGML_USE_POWER8)
    ଯଦି (pf.power_version < 8) { ଫେରନ୍ତୁ 0; }
    score += 1<<2;
#endif
#ଯଦି defined(GGML_USE_POWER9)
    ଯଦି (pf.power_version < 9) { ଫେରନ୍ତୁ 0; }
    score += 1<<3;
#endif
#ଯଦି defined(GGML_USE_POWER10)
    ଯଦି (pf.power_version < 10) { ଫେରନ୍ତୁ 0; }
    score += 1<<4;
#endif
#ଯଦି defined(GGML_USE_POWER11)
    ଯଦି (pf.power_version < 11) { ଫେରନ୍ତୁ 0; }
    score += 1<<5;
#endif

// Feature scores
#ଯଦି defined(GGML_USE_VSX)
    ଯଦି (!pf.has_vsx) { ଫେରନ୍ତୁ 0; }
    score += 1<<6;
#endif

    ଫେରନ୍ତୁ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
