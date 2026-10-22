#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#만약 defined(__aarch64__) || defined(_M_ARM64)

static 정수 ggml_backend_cpu_aarch64_score() {
    정수 score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    만약 (!af.has_dotprod) { 반환 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    만약 (!af.has_fp16) { 반환 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    만약 (!af.has_sve) { 반환 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    만약 (!af.has_i8mm) { 반환 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    만약 (!af.has_sve2) { 반환 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    만약 (!af.has_sme) { 반환 0; }
    score += 1<<6;
#endif

    반환 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
