# include "ggml-backend-impl.h"

#अगर defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#अगर defined(__linux__)
#समावेश <sys/auxv.h>
#पूर्ण_अगर

#समावेश <string>

काष्ठा powerpc_features {
    std::string platform = "";
    पूर्णांक power_version    = -1;

    bool has_vsx         = false;

    powerpc_features() {
#अगर defined(__linux__)
        अचिन्हित दीर्घ auxval = getauxval(AT_PLATFORM);
        अगर (auxval) {
            platform = std::string(reinterpret_cast<स्थिर अक्षर*>(auxval));
            // TBD: Do systems exist that return this in uppercase?
            अगर (platform.substr(0, 5) == "power") {
                // Extractt a numeric suffix, if one exists
                पूर्णांक vpos = -1;
                क्रम (पूर्णांक i = platform.लंबाई() - 1; i >= 0; i--) {
                    अगर (std::है_अंक(platform[i])) {
                        vpos = i;
                    } अन्यथा {
                        अवरोध;
                    }
                }
                अगर (vpos > -1) {
                    power_version = std::stoi(platform.substr(vpos));
                }
            }
        }
#पूर्ण_अगर
        अगर (power_version >= 9) {
            has_vsx = true;
        }
    }
};

अटल पूर्णांक ggml_backend_cpu_powerpc_score() {
    पूर्णांक score = 1;
    powerpc_features pf;

// Platform scores
#अगर defined(GGML_USE_POWER7)
    अगर (pf.power_version < 7) { वापस 0; }
    score += 1<<1;
#पूर्ण_अगर
#अगर defined(GGML_USE_POWER8)
    अगर (pf.power_version < 8) { वापस 0; }
    score += 1<<2;
#पूर्ण_अगर
#अगर defined(GGML_USE_POWER9)
    अगर (pf.power_version < 9) { वापस 0; }
    score += 1<<3;
#पूर्ण_अगर
#अगर defined(GGML_USE_POWER10)
    अगर (pf.power_version < 10) { वापस 0; }
    score += 1<<4;
#पूर्ण_अगर
#अगर defined(GGML_USE_POWER11)
    अगर (pf.power_version < 11) { वापस 0; }
    score += 1<<5;
#पूर्ण_अगर

// Feature scores
#अगर defined(GGML_USE_VSX)
    अगर (!pf.has_vsx) { वापस 0; }
    score += 1<<6;
#पूर्ण_अगर

    वापस score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_powerpc_score)

#पूर्ण_अगर // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
