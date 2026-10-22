#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#もし defined(__aarch64__) || defined(_M_ARM64)

static 整数 ggml_backend_cpu_aarch64_score() {
    整数 score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    もし (!af.has_dotprod) { 戻る 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    もし (!af.has_fp16) { 戻る 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    もし (!af.has_sve) { 戻る 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    もし (!af.has_i8mm) { 戻る 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    もし (!af.has_sve2) { 戻る 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    もし (!af.has_sme) { 戻る 0; }
    score += 1<<6;
#endif

    戻る score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
