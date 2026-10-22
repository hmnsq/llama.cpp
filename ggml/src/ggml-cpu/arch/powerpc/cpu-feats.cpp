# include "ggml-backend-impl.h"

#જો defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#જો defined(__linux__)
#અન્તર્ભાવ <sys/auxv.h>
#endif

#અન્તર્ભાવ <string>

સંરચના powerpc_features {
    std::string platform = "";
    પૂર્ણાંક power_version    = -1;

    તાર્કિક has_vsx         = ખોટું;

    powerpc_features() {
#જો defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        જો (auxval) {
            platform = std::string(reinterpret_cast<અચલ અક્ષર*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            જો (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                પૂર્ણાંક vpos = -1;
                માટે (પૂર્ણાંક i = platform.length() - 1; i >= 0; i--) {
                    જો (std::isdigit(platform[i])) {
                        vpos = i;
                    } નહીંતર {
                        તોડો;
                    }
                }
                જો (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        જો (power_version >= 9) {
            has_vsx = સાચું;
        }
    }
};

સ્થિર પૂર્ણાંક ggml_backend_cpu_powerpc_score() {
    પૂર્ણાંક score = 1;
    powerpc_features pf;

// Platform scores
#જો defined(GGML_USE_POWER7)
    જો (pf.power_version < 7) { પાછા 0; }
    score += 1<<1;
#endif
#જો defined(GGML_USE_POWER8)
    જો (pf.power_version < 8) { પાછા 0; }
    score += 1<<2;
#endif
#જો defined(GGML_USE_POWER9)
    જો (pf.power_version < 9) { પાછા 0; }
    score += 1<<3;
#endif
#જો defined(GGML_USE_POWER10)
    જો (pf.power_version < 10) { પાછા 0; }
    score += 1<<4;
#endif
#જો defined(GGML_USE_POWER11)
    જો (pf.power_version < 11) { પાછા 0; }
    score += 1<<5;
#endif

// Feature scores
#જો defined(GGML_USE_VSX)
    જો (!pf.has_vsx) { પાછા 0; }
    score += 1<<6;
#endif

    પાછા score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
