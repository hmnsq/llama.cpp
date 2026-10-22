# include "ggml-backend-impl.h"

#ఒకవేళ defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#ఒకవేళ defined(__linux__)
#అన్తర్భావ <sys/auxv.h>
#అయితే_ముగింపు

#అన్తర్భావ <string>

నిర్మాణం powerpc_features {
    std::string platform = "";
    పూర్ణాంకం power_సంస్కరణ    = -1;

    తార్కిక has_vsx         = అబద్ధం;

    powerpc_features() {
#ఒకవేళ defined(__linux__)
        గుర్తులేని దీర్ఘ auxval = getauxval(AT_PLATFORM);
        ఒకవేళ (auxval) {
            platform = std::string(reinterpret_cast<స్థిరరాశి అక్షరం*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            ఒకవేళ (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                పూర్ణాంకం vpos = -1;
                కొరకు (పూర్ణాంకం i = platform.నిడివి() - 1; i >= 0; i--) {
                    ఒకవేళ (std::అంకెనా(platform[i])) {
                        vpos = i;
                    } లేకపోతే {
                        ఆపు;
                    }
                }
                ఒకవేళ (vpos > -1) {
                    power_సంస్కరణ = std::stoi(platform.substr(vpos));
                }
            }
        }
#అయితే_ముగింపు
        ఒకవేళ (power_సంస్కరణ >= 9) {
            has_vsx = నిజం;
        }
    }
};

స్థిర పూర్ణాంకం ggml_backend_cpu_powerpc_score() {
    పూర్ణాంకం score = 1;
    powerpc_features pf;

// Platform scores
#ఒకవేళ defined(GGML_USE_POWER7)
    ఒకవేళ (pf.power_సంస్కరణ < 7) { తిరిగి 0; }
    score += 1<<1;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_USE_POWER8)
    ఒకవేళ (pf.power_సంస్కరణ < 8) { తిరిగి 0; }
    score += 1<<2;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_USE_POWER9)
    ఒకవేళ (pf.power_సంస్కరణ < 9) { తిరిగి 0; }
    score += 1<<3;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_USE_POWER10)
    ఒకవేళ (pf.power_సంస్కరణ < 10) { తిరిగి 0; }
    score += 1<<4;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_USE_POWER11)
    ఒకవేళ (pf.power_సంస్కరణ < 11) { తిరిగి 0; }
    score += 1<<5;
#అయితే_ముగింపు

// Feature scores
#ఒకవేళ defined(GGML_USE_VSX)
    ఒకవేళ (!pf.has_vsx) { తిరిగి 0; }
    score += 1<<6;
#అయితే_ముగింపు

    తిరిగి score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#అయితే_ముగింపు // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
