# include "ggml-backend-impl.h"

#எனில் defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#எனில் defined(__linux__)
#அந்தர்பாவ <sys/auxv.h>
#endif

#அந்தர்பாவ <string>

கட்டமைப்பு powerpc_features {
    std::string platform = "";
    முழு power_version    = -1;

    தருக்க has_vsx         = பொய்;

    powerpc_features() {
#எனில் defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        எனில் (auxval) {
            platform = std::string(reinterpret_cast<மாறிலி எழுத்து*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            எனில் (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                முழு vpos = -1;
                க்கு (முழு i = platform.length() - 1; i >= 0; i--) {
                    எனில் (std::isdigit(platform[i])) {
                        vpos = i;
                    } இல்லையெனில் {
                        நிறுத்து;
                    }
                }
                எனில் (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        எனில் (power_version >= 9) {
            has_vsx = உண்மை;
        }
    }
};

நிலையான முழு ggml_backend_cpu_powerpc_score() {
    முழு score = 1;
    powerpc_features pf;

// Platform scores
#எனில் defined(GGML_USE_POWER7)
    எனில் (pf.power_version < 7) { திரும்பு 0; }
    score += 1<<1;
#endif
#எனில் defined(GGML_USE_POWER8)
    எனில் (pf.power_version < 8) { திரும்பு 0; }
    score += 1<<2;
#endif
#எனில் defined(GGML_USE_POWER9)
    எனில் (pf.power_version < 9) { திரும்பு 0; }
    score += 1<<3;
#endif
#எனில் defined(GGML_USE_POWER10)
    எனில் (pf.power_version < 10) { திரும்பு 0; }
    score += 1<<4;
#endif
#எனில் defined(GGML_USE_POWER11)
    எனில் (pf.power_version < 11) { திரும்பு 0; }
    score += 1<<5;
#endif

// Feature scores
#எனில் defined(GGML_USE_VSX)
    எனில் (!pf.has_vsx) { திரும்பு 0; }
    score += 1<<6;
#endif

    திரும்பு score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
