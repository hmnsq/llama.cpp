#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ᱡᱩᱫᱤ defined(__aarch64__) || defined(_M_ARM64)

static puur_naa_nka ggml_backend_cpu_aarch64_score() {
    puur_naa_nka score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ᱡᱩᱫᱤ (!af.has_dotprod) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ᱡᱩᱫᱤ (!af.has_fp16) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ᱡᱩᱫᱤ (!af.has_sve) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ᱡᱩᱫᱤ (!af.has_i8mm) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ᱡᱩᱫᱤ (!af.has_sve2) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ᱡᱩᱫᱤ (!af.has_sme) { ᱨᱩᱣᱟᱹᱲ 0; }
    score += 1<<6;
#endif

    ᱨᱩᱣᱟᱹᱲ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
