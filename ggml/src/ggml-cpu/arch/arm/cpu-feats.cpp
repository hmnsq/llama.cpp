#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ถ้า defined(__aarch64__) || defined(_M_ARM64)

static จำนวนเต็ม ggml_backend_cpu_aarch64_score() {
    จำนวนเต็ม score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ถ้า (!af.has_dotprod) { คืน 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ถ้า (!af.has_fp16) { คืน 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ถ้า (!af.has_sve) { คืน 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ถ้า (!af.has_i8mm) { คืน 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ถ้า (!af.has_sve2) { คืน 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ถ้า (!af.has_sme) { คืน 0; }
    score += 1<<6;
#endif

    คืน score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
