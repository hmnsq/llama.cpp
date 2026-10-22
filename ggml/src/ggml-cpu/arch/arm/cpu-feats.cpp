#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#གལ་ཏེ defined(__aarch64__) || defined(_M_ARM64)

static ཧྲིལ ggml_backend_cpu_aarch64_score() {
    ཧྲིལ score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    གལ་ཏེ (!af.has_dotprod) { ཕྱིར 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    གལ་ཏེ (!af.has_fp16) { ཕྱིར 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    གལ་ཏེ (!af.has_sve) { ཕྱིར 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    གལ་ཏེ (!af.has_i8mm) { ཕྱིར 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    གལ་ཏེ (!af.has_sve2) { ཕྱིར 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    གལ་ཏེ (!af.has_sme) { ཕྱིར 0; }
    score += 1<<6;
#endif

    ཕྱིར score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
