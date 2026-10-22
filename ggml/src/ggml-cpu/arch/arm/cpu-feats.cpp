#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#если defined(__aarch64__) || defined(_M_ARM64)

static целое ggml_backend_cpu_aarch64_score() {
    целое score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    если (!af.has_dotprod) { вернуть 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    если (!af.has_fp16) { вернуть 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    если (!af.has_sve) { вернуть 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    если (!af.has_i8mm) { вернуть 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    если (!af.has_sve2) { вернуть 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    если (!af.has_sme) { вернуть 0; }
    score += 1<<6;
#endif

    вернуть score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
