#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#nếu defined(__aarch64__) || defined(_M_ARM64)

static số_nguyên ggml_backend_cpu_aarch64_score() {
    số_nguyên score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    nếu (!af.has_dotprod) { trả_về 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    nếu (!af.has_fp16) { trả_về 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    nếu (!af.has_sve) { trả_về 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    nếu (!af.has_i8mm) { trả_về 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    nếu (!af.has_sve2) { trả_về 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    nếu (!af.has_sme) { trả_về 0; }
    score += 1<<6;
#endif

    trả_về score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
