#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#如果 defined(__aarch64__) || defined(_M_ARM64)

static 整数 ggml_backend_cpu_aarch64_score() {
    整数 score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    如果 (!af.has_dotprod) { 返回 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    如果 (!af.has_fp16) { 返回 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    如果 (!af.has_sve) { 返回 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    如果 (!af.has_i8mm) { 返回 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    如果 (!af.has_sve2) { 返回 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    如果 (!af.has_sme) { 返回 0; }
    score += 1<<6;
#endif

    返回 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
