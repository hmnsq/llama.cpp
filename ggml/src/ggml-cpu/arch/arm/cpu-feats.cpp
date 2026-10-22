#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ако defined(__aarch64__) || defined(_M_ARM64)

static цяло ggml_backend_cpu_aarch64_score() {
    цяло score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ако (!af.has_dotprod) { върни 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ако (!af.has_fp16) { върни 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ако (!af.has_sve) { върни 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ако (!af.has_i8mm) { върни 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ако (!af.has_sve2) { върни 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ако (!af.has_sme) { върни 0; }
    score += 1<<6;
#endif

    върни score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
