# include "ggml-backend-impl.h"

#എങ്കിൽ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#എങ്കിൽ defined(__linux__)
#അന്തര്ഭാവ <sys/auxv.h>
#endif

#അന്തര്ഭാവ <string>

ഘടന powerpc_features {
    std::string platform = "";
    പൂർണ്ണസംഖ്യ power_version    = -1;

    തർക്കം has_vsx         = അസത്യം;

    powerpc_features() {
#എങ്കിൽ defined(__linux__)
        unsigned long auxval = getauxval(AT_PLATFORM);
        എങ്കിൽ (auxval) {
            platform = std::string(reinterpret_cast<സ്ഥിരം അക്ഷരം*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            എങ്കിൽ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                പൂർണ്ണസംഖ്യ vpos = -1;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = platform.length() - 1; i >= 0; i--) {
                    എങ്കിൽ (std::isdigit(platform[i])) {
                        vpos = i;
                    } അല്ലെങ്കിൽ {
                        നിർത്തുക;
                    }
                }
                എങ്കിൽ (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#endif
        എങ്കിൽ (power_version >= 9) {
            has_vsx = സത്യം;
        }
    }
};

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_cpu_powerpc_score() {
    പൂർണ്ണസംഖ്യ score = 1;
    powerpc_features pf;

// Platform scores
#എങ്കിൽ defined(GGML_USE_POWER7)
    എങ്കിൽ (pf.power_version < 7) { തിരികെ 0; }
    score += 1<<1;
#endif
#എങ്കിൽ defined(GGML_USE_POWER8)
    എങ്കിൽ (pf.power_version < 8) { തിരികെ 0; }
    score += 1<<2;
#endif
#എങ്കിൽ defined(GGML_USE_POWER9)
    എങ്കിൽ (pf.power_version < 9) { തിരികെ 0; }
    score += 1<<3;
#endif
#എങ്കിൽ defined(GGML_USE_POWER10)
    എങ്കിൽ (pf.power_version < 10) { തിരികെ 0; }
    score += 1<<4;
#endif
#എങ്കിൽ defined(GGML_USE_POWER11)
    എങ്കിൽ (pf.power_version < 11) { തിരികെ 0; }
    score += 1<<5;
#endif

// Feature scores
#എങ്കിൽ defined(GGML_USE_VSX)
    എങ്കിൽ (!pf.has_vsx) { തിരികെ 0; }
    score += 1<<6;
#endif

    തിരികെ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
