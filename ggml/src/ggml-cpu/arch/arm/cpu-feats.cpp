#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#ប្រសិនបើ defined(__aarch64__) || defined(_M_ARM64)

static ចំនួនគត់ ggml_backend_cpu_aarch64_score() {
    ចំនួនគត់ score = 1;
    const ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();
    GGML_UNUSED(af);

#ifdef GGML_USE_DOTPROD
    ប្រសិនបើ (!af.has_dotprod) { ត្រឡប់ 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    ប្រសិនបើ (!af.has_fp16) { ត្រឡប់ 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    ប្រសិនបើ (!af.has_sve) { ត្រឡប់ 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    ប្រសិនបើ (!af.has_i8mm) { ត្រឡប់ 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    ប្រសិនបើ (!af.has_sve2) { ត្រឡប់ 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    ប្រសិនបើ (!af.has_sme) { ត្រឡប់ 0; }
    score += 1<<6;
#endif

    ត្រឡប់ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
