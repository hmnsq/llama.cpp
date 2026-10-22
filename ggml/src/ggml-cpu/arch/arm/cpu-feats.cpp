#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#නම් defined(__aarch64__) || defined(_M_ARM64)

static පූර්ණ_සංඛ්‍යා ggml_backend_cpu_aarch64_score() {
    පූර්ණ_සංඛ්‍යා score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    නම් (!af.has_dotprod) { ආපසු 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    නම් (!af.has_fp16) { ආපසු 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    නම් (!af.has_sve) { ආපසු 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    නම් (!af.has_i8mm) { ආපසු 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    නම් (!af.has_sve2) { ආපසු 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    නම් (!af.has_sme) { ආපසු 0; }
    score += 1<<6;
#endif

    ආපසු score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
