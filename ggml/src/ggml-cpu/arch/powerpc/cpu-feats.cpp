# include "ggml-backend-impl.h"

#যদি defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#যদি defined(__linux__)
#অন্তর্ভাব <sys/auxv.h>
#endif

#অন্তর্ভাব <string>

কাঠামো powerpc_features {
    std::string platform = "";
    পূর্ণসংখ্যা power_version    = -1;

    যৌক্তিক has_vsx         = মিথ্যা;

    powerpc_features() {
#যদি defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        যদি (auxval) {
            platform = std::string(reinterpret_cast<ধ্রুবক অক্ষর*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            যদি (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                পূর্ণসংখ্যা vpos = -1;
                জন্য (পূর্ণসংখ্যা i = platform.দৈর্ঘ্য() - 1; i >= 0; i--) {
                    যদি (std::isdigit(platform[i])) {
                        vpos = i;
                    } অন্যথা {
                        বিরতি;
                    }
                }
                যদি (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        যদি (power_version >= 9) {
            has_vsx = সত্য;
        }
    }
};

স্থির পূর্ণসংখ্যা ggml_backend_cpu_powerpc_score() {
    পূর্ণসংখ্যা score = 1;
    powerpc_features pf;

// Platform scores
#যদি defined(GGML_USE_POWER7)
    যদি (pf.power_version < 7) { ফেরত 0; }
    score += 1<<1;
#endif
#যদি defined(GGML_USE_POWER8)
    যদি (pf.power_version < 8) { ফেরত 0; }
    score += 1<<2;
#endif
#যদি defined(GGML_USE_POWER9)
    যদি (pf.power_version < 9) { ফেরত 0; }
    score += 1<<3;
#endif
#যদি defined(GGML_USE_POWER10)
    যদি (pf.power_version < 10) { ফেরত 0; }
    score += 1<<4;
#endif
#যদি defined(GGML_USE_POWER11)
    যদি (pf.power_version < 11) { ফেরত 0; }
    score += 1<<5;
#endif

// Feature scores
#যদি defined(GGML_USE_VSX)
    যদি (!pf.has_vsx) { ফেরত 0; }
    score += 1<<6;
#endif

    ফেরত score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
