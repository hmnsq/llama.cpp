#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#якщо defined(__aarch64__) || defined(_M_ARM64)

static ціле ggml_backend_cpu_aarch64_score() {
    ціле score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    якщо (!af.has_dotprod) { повернути 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    якщо (!af.has_fp16) { повернути 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    якщо (!af.has_sve) { повернути 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    якщо (!af.has_i8mm) { повернути 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    якщо (!af.has_sve2) { повернути 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    якщо (!af.has_sme) { повернути 0; }
    score += 1<<6;
#endif

    повернути score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
